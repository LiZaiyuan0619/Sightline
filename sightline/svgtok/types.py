"""Token-type table for the OmniSVG 8B vocabulary (197,000), derived from the official
inference-side decoder (refs/OmniSVG/tokenizer.py + config.yaml).

All id ranges below follow the INFERENCE decoder (the one whose rendering works).
The trainer config (refs/OmniSVG-train/configs/tokenization.yaml) has command ids off by
one (writes move=152066 where inference has M=152067); per the design notes the inference
decoder is authoritative.

Token classes and ranges (8B):
  bos 196998, eos 196999, pad 151643
  mask 152065, eom 152066
  commands M/L/C/A/Z = 152067..152071
  coords 152072..192071    (200x200 grid, row-major: token - 152072 = y*200 + x)
  colors 192075..196563    (4098: none, currentColor, then 4096 4-bit RGB colors)
  arc params 196565..196664 (100)
  everything else: base Qwen vocab (< 152064) — text tokens; the model occasionally
  emits one (observed ~1 per 100 streams in T1 data); the official tokenizer drops them
  silently during decoding.

Command grammar (argument counts, from raster_svg — VERIFIED against real T1 streams):
  M(x y)            3 tokens   M <coord> <coord>
  L(x y)            2 tokens   L <coord>
  C(c1 c2 e)        4 tokens   C <coord> <coord> <coord>
  A(r rot la sw e)  6 tokens   A <coord:radius> <coord:rot> <coord:la> <coord:sw> <coord:end>
      (all five A args are coord-class tokens in the stream; raster_svg reads the raw
       ids of args 2-4 and adds PIXEL_OFFSET back — they are grid-range tokens)
  Z               2 tokens   Z <coord>
  color             1 token    terminates the current path (legal ANYWHERE between paths;
                               official raster_svg ends the path at a color and skips to
                               the next command)
A color token inside a command's args also terminates that path (official behavior);
escape tokens (base vocab) are skipped silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ---- special tokens (8B, inference side) ----
BOS = 196998
EOS = 196999
PAD = 151643
MASK = 152065
EOM = 152066
BASE_VOCAB = 152064          # Qwen2.5-VL-7B base vocab; SVG tokens live above this

# ---- command tokens ----
CMD_MOVE = 152067
CMD_LINE = 152068
CMD_CURVE = 152069
CMD_ARC = 152070
CMD_CLOSE = 152071
CMD_IDS = {"M": CMD_MOVE, "L": CMD_LINE, "C": CMD_CURVE, "A": CMD_ARC, "Z": CMD_CLOSE}
CMD_BY_ID = {v: k for k, v in CMD_IDS.items()}

# ---- ranges (inclusive) ----
COORD_START, COORD_END = 152072, 192071      # 40,000 grid coords
COLOR_START, COLOR_END = 192075, 196563      # 4,489 color tokens
ARC_START, ARC_END = 196565, 196664          # 100 arc params
GRID = 200                                   # bbox grid per axis

# argument layout per command, VERIFIED on T1 data (1095 A-commands: coord/arc/arc/arc/coord):
CMD_ARGS = {
    "M": ["coord", "coord"],
    "L": ["coord"],
    "C": ["coord", "coord", "coord"],
    "A": ["coord", "arc", "arc", "arc", "coord"],
    "Z": ["coord"],
}

TYPE_NAMES = ["bos", "eos", "pad", "mask", "eom", "cmd", "coord", "color", "arc", "other"]


def token_type(t: int) -> str:
    """Class of a single token id."""
    if t == BOS:
        return "bos"
    if t == EOS:
        return "eos"
    if t == PAD:
        return "pad"
    if t == MASK:
        return "mask"
    if t == EOM:
        return "eom"
    if CMD_MOVE <= t <= CMD_CLOSE:
        return "cmd"
    if COORD_START <= t <= COORD_END:
        return "coord"
    if COLOR_START <= t <= COLOR_END:
        return "color"
    if ARC_START <= t <= ARC_END:
        return "arc"
    return "other"          # base-vocab text token the model let slip through


def coord_xy(t: int) -> tuple[int, int]:
    """Grid coordinates of a coord token (row-major, y*200+x)."""
    idx = t - COORD_START
    return idx % GRID, idx // GRID


def color_hex(t: int) -> str:
    """Color of a color token (same arithmetic as SVGTokenizer.token_to_color)."""
    RAW = 40010  # colors.color_token_start in config.yaml
    if t == COLOR_START:              # RAW + 0 + BASE? no: token_to_color works on raw ids
        pass
    idx = t - BASE_VOCAB - 40010      # raw color index in [0, 4097]
    if idx == 0:
        return "none"
    if idx == 1:
        return "currentColor"
    ci = idx - 2
    if ci < 0 or ci >= 4096:
        return "#808080"
    r, g, b = (ci >> 8) & 0xF, (ci >> 4) & 0xF, ci & 0xF
    return f"#{(r << 4) | r:02x}{(g << 4) | g:02x}{(b << 4) | b:02x}"


# ---------------------------------------------------------------- grammar state machine

@dataclass
class GrammarState:
    """Snapshot of the parser position (returned by SVGGrammar.state)."""
    cmd: str | None = None
    arg_pos: int = 0
    depth: int = 0


class SVGGrammar:
    """Given a token prefix, what can come next — and is it forced?

    In this grammar no position ever fixes the exact token id: after M the next
    token can be any of 40,000 coords, after A's tag any arc param value. What the
    grammar DOES determine is the token CLASS. We expose:
      expect       -> set of legal classes for the next position
      is_cmd_start -> True if the next position must begin a new command
    A drafter constrained to the legal class still has full freedom over ids.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.cmd = None       # None => next token starts a command
        self.arg_pos = 0
        self.depth = 0        # tokens consumed since the last command start
        self.errors = []      # lenient-mode parse errors (position, message)

    def feed(self, t: int) -> None:
        """Advance the state by one token of a REAL sequence (validates as it goes).

        Official semantics (raster_svg): a color token terminates the current path
        wherever it appears (mid-command args included); the parser then skips to the
        next command token. Escape tokens (base-vocab) are dropped silently.
        feed therefore only accepts: cmd at command start, coord in arg slots,
        color anywhere, eos at end.
        """
        typ = token_type(t)
        if typ == "color":
            # path terminator: reset to command-start state
            self.cmd = None
            self.arg_pos = 0
            self.depth += 1
            return
        if self.cmd is None:
            if typ != "cmd":
                raise ValueError(f"expected command at depth {self.depth}, got {typ} ({t})")
            self.cmd = CMD_BY_ID[t]
            self.arg_pos = 0
            self.depth = 0
        else:
            want = CMD_ARGS[self.cmd][self.arg_pos]
            if typ != want:
                raise ValueError(
                    f"{self.cmd} arg {self.arg_pos}: expected {want}, got {typ} ({t})")
            self.arg_pos += 1
            if self.arg_pos == len(CMD_ARGS[self.cmd]):
                self.cmd = None
                self.arg_pos = 0
        self.depth += 1

    # -- query API for drafting --

    def feed_lenient(self, t: int) -> None:
        """Advance by one token with OFFICIAL lenient semantics: malformed tokens are
        recorded, and the parser resynchronizes by skipping to the next command token
        (exactly what raster_svg's while-loop does: unrecognized tokens advance i by 1)."""
        typ = token_type(t)
        if typ == "color":
            self.cmd = None
            self.arg_pos = 0
            self.depth += 1
            return
        if typ == "other":                 # escape token: dropped silently
            self.depth += 1
            return
        if self.cmd is None:
            if typ == "cmd":
                self.cmd = CMD_BY_ID[t]
                self.arg_pos = 0
                self.depth = 0
            else:
                self.errors.append((self.depth, f"skip {typ} at cmd-start"))
            self.depth += 1
            return
        want = CMD_ARGS[self.cmd][self.arg_pos]
        if typ == want:
            self.arg_pos += 1
            if self.arg_pos == len(CMD_ARGS[self.cmd]):
                self.cmd = None
                self.arg_pos = 0
            self.depth += 1
        elif typ == "cmd":
            # a new command interrupts the unfinished one (official: scanning restarts)
            self.errors.append((self.depth, f"{self.cmd} truncated by new cmd"))
            self.cmd = CMD_BY_ID[t]
            self.arg_pos = 0
            self.depth = 0
            self.depth += 1
        else:
            self.errors.append((self.depth, f"skip {typ} in {self.cmd} arg {self.arg_pos}"))
            self.depth += 1

    # -- query API for drafting --

    def expect(self) -> set[str]:
        """Legal classes for the next position."""
        if self.cmd is None:
            return {"cmd", "color"}          # new command, or a color terminating the path
        # a color may legally appear mid-args too (official semantics), but for drafting
        # purposes the dominant expectation is the grammatical arg class
        return {CMD_ARGS[self.cmd][self.arg_pos], "color"}

    def is_cmd_start(self) -> bool:
        return self.cmd is None

    def legal_token_ids(self) -> list[int]:
        """All token ids legal at the next position (for drafter masking).
        Returns None (represented as []) when the legal set is the whole class range
        and the class has more than 64 members — callers should then use class ranges
        directly instead of enumerating."""
        exp = self.expect()
        if self.cmd is None:
            return [CMD_MOVE, CMD_LINE, CMD_CURVE, CMD_ARC, CMD_CLOSE] + \
                   list(range(COLOR_START, COLOR_END + 1))
        want = CMD_ARGS[self.cmd][self.arg_pos]
        rng = {"coord": (COORD_START, COORD_END), "color": (COLOR_START, COLOR_END),
               "arc": (ARC_START, ARC_END)}[want]
        return list(range(rng[0], rng[1] + 1))


def classify_stream(token_ids) -> dict:
    """Parse a whole generated stream (BOS... [EOS]) and return stats + parse verdict.

    Uses lenient (official) semantics for counting; `parse_ok` reflects STRICT parsing
    (no malformed token anywhere). Errors are recorded per stream, so 100% of streams
    are counted (matching official decoding, which never rejects a stream outright).

    Returns dict with:
      n_tokens, counts per class, n_commands per kind, n_paths (color count),
      parse_ok (strict), n_errors (lenient), first_error,
      escape_tokens (base-vocab tokens the model emitted),
      cmd_starts (list of positions where a command begins)
    """
    g = SVGGrammar()
    counts = {n: 0 for n in TYPE_NAMES}
    cmds = {k: 0 for k in CMD_IDS}
    n_paths = 0
    escapes = []
    cmd_starts = []
    i = 0
    eos_at = None
    for i, t in enumerate(token_ids):
        typ = token_type(t)
        counts[typ] += 1
        if typ == "bos":
            continue
        if typ == "eos":
            eos_at = i
            break
        if typ == "other":
            escapes.append((i, t))
        prev_cmd = g.cmd
        prev_start = g.is_cmd_start()
        g.feed_lenient(t)
        if typ == "cmd" and prev_start:
            cmd_starts.append(i)
            cmds[CMD_BY_ID[t]] += 1
        if typ == "color":
            n_paths += 1
    return {
        "n_tokens": i + 1 if eos_at is None else eos_at + 1,
        "counts": counts,
        "cmds": cmds,
        "n_paths": n_paths,
        "parse_ok": not g.errors,
        "n_errors": len(g.errors),
        "first_error": g.errors[0] if g.errors else None,
        "escape_tokens": escapes,
        "cmd_starts": cmd_starts,
    }


def build_type_table() -> dict:
    """Explicit id -> class map for all SVG-domain ids (compact ranges form also included)."""
    table = {
        "bos": BOS, "eos": EOS, "pad": PAD, "mask": MASK, "eom": EOM,
        "commands": CMD_IDS,
        "coord_range": [COORD_START, COORD_END],
        "color_range": [COLOR_START, COLOR_END],
        "arc_range": [ARC_START, ARC_END],
        "grid": GRID,
        "cmd_args": CMD_ARGS,
    }
    return table


def write_type_table(path: Path):
    table = build_type_table()
    path.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("token_type_table.json")
    write_type_table(out)
    print(f"type table -> {out}")
