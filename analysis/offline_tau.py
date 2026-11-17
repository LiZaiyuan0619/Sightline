"""F2 thermometer: exact offline τ for
training-free drafters on the T1 GREEDY outputs.

Why exact: in greedy decoding the target's output is its argmax sequence. A deterministic
drafter's acceptance against this ground truth can be simulated token-by-token; verification
accepts the longest matching draft prefix plus one bonus token (= the target's own greedy
token at the first mismatch position). This reproduces the online τ with a greedy target.

Drafters (all training-free):
  1. prompt-lookup (n = 2,3,4; K = 8,16,32): copy the K tokens that followed the most
     recent earlier occurrence of the last-n-token n-gram in prompt + generated-so-far.
  2. grammar-fill (K-capped run): fill grammar-DETERMINED tokens only, verified rules
     (measured on T1 greedy data 2026-10-08):
       - fresh path (after BOS/color): next command is M               (100%, 5335/5335)
       - the path's 2nd command is M                                   (100%, 5325/5325)
       - first M of a path: 2nd coord == 1st coord                     (100%, 1973/1973)
       - 2nd M of a path: 1st coord == 1st M's coord                   (~%)
       - after Z: next command is L                                    (79%, 2580/3266)
  3. combined: grammar-fill first; prompt-lookup extends the block.

τ calibers live in analysis/calibers.py (shared with the harness evaluation):
all / eos / trunc / loop × per-stream-mean / round-merged. The 2026-10-09 fix (C1
review §3.1): streams whose loop onset is <= 2 have NO loop-free prefix — they are
excluded from τ_trunc and reported as a separate count instead of silently falling
back to their (loop-inflated) full-stream τ.

Usage:
  python analysis/thermometer.py --cell runs/2026-10-07-omnisvg-t1/greedy-penalty-b1/text2svg-icon
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.svgtok.types import (BOS, CMD_BY_ID, CMD_MOVE, CMD_LINE, CMD_CLOSE,
                          COORD_START, COORD_END, COLOR_START, COLOR_END)
from sightline.svgtok.loops import analyze
from analysis.calibers import (CaliberStream, StreamRecord, caliber_table,
                               format_table, MIN_HEALTHY_ONSET)
from sightline.paths import RUNS

M, L, Z = CMD_MOVE, CMD_LINE, CMD_CLOSE
CMD_ARG_N = {"M": 2, "L": 1, "C": 3, "A": 5, "Z": 1}


# ---------------------------------------------------------------- drafter: prompt lookup

def lookup_draft(context: list[int], end: int, ngram_index: dict, n: int, K: int) -> list[int]:
    """Copy what followed the last earlier occurrence of the n-gram ending at `end`
    (exclusive). context must contain the full sequence; end is the current position."""
    if end < n:
        return []
    key = tuple(context[end - n:end])
    e = ngram_index.get(key)
    if e is None:
        return []
    return context[e:e + K]


# ---------------------------------------------------------------- drafter: grammar fill

def new_state() -> dict:
    return {"cmd": None, "pos": 0, "n_cmds": 0, "last_cmd": None,
            "first_m_coord": None, "prev_coord": None}


def feed(st: dict, t: int) -> None:
    """Advance a (virtual) parse state by one ground-truth/drafted token."""
    if t in CMD_BY_ID:
        name = CMD_BY_ID[t]
        st["cmd"] = name
        st["pos"] = 0
        st["n_cmds"] += 1
        st["last_cmd"] = name
        if name == "M":
            st["first_m_coord"] = None
    elif COORD_START <= t <= COORD_END:
        st["pos"] += 1
        if st["cmd"] == "M" and st["n_cmds"] == 1 and st["pos"] == 1:
            st["first_m_coord"] = t
        st["prev_coord"] = t
    elif COLOR_START <= t <= COLOR_END:
        st.update(cmd=None, pos=0, n_cmds=0, last_cmd="color",
                  first_m_coord=None, prev_coord=None)


def determined_next(st: dict) -> int | None:
    """Grammar-determined next token at state st, or None."""
    cmd, pos, n_cmds = st["cmd"], st["pos"], st["n_cmds"]
    if cmd is not None and pos >= CMD_ARG_N[cmd]:
        cmd = None  # command complete -> at a command-start position
    if cmd is None:
        if st["last_cmd"] in (None, "color") or n_cmds == 1:
            return M              # fresh path or 2nd command of path
        if st["last_cmd"] == "Z":
            return L              # after close comes a line (79%)
        return None
    if cmd == "M" and pos == 1 and n_cmds == 1:
        return st["first_m_coord"]      # 2nd coord == 1st coord (100%)
    if cmd == "M" and pos == 0 and n_cmds >= 2:
        return st["prev_coord"]         # 2nd M's 1st coord == prev endpoint (~%)
    return None


def grammar_draft(st: dict, K: int) -> list[int]:
    """Longest run of grammar-determined tokens, capped at K (K<=0 means unlimited —
    determined runs are short anyway). Uses a virtual state; st is not mutated."""
    virt = dict(st)
    out = []
    while K <= 0 or len(out) < K:
        nxt = determined_next(virt)
        if nxt is None:
            break
        out.append(nxt)
        feed(virt, nxt)
    return out


# ---------------------------------------------------------------- simulation

def simulate(prompt_ids: list[int], gen_ids: list[int], drafter: str, K: int, n: int,
             loop_onset: int | None) -> dict:
    """Exact greedy-speculation simulation over one stream.

    gen_ids[0] is the SVG BOS (given). Returns per-stream round/emitted counts for the
    full stream and for the loop segment (loop_onset in gen_ids coordinates)."""
    context = list(prompt_ids) + list(gen_ids)
    P = len(prompt_ids)
    n_gen = len(gen_ids)

    idx: dict = {}
    if drafter in ("lookup", "combined"):
        for e in range(n, P + 1):        # n-gram ends available after BOS
            idx[tuple(context[e - n:e])] = e

    state = new_state()
    pos = 1
    steps = 0
    loop_steps = 0
    loop_emitted = 0

    while pos < n_gen:
        ctx_end = P + pos              # exclusive end of drafter-visible context
        if drafter == "lookup":
            draft = lookup_draft(context, ctx_end, idx, n, K)
        elif drafter == "grammar":
            draft = grammar_draft(state, K)
        else:
            g = grammar_draft(state, K)
            if len(g) < K:
                g += lookup_draft(context, ctx_end, idx, n, K - len(g))
            draft = g
        acc = 0
        for d in draft:
            if pos + acc < n_gen and d == gen_ids[pos + acc]:
                acc += 1
            else:
                break
        bonus = 1 if pos + acc < n_gen else 0
        emitted = acc + bonus
        in_loop = loop_onset is not None and pos >= loop_onset
        if in_loop:
            loop_steps += 1
            loop_emitted += emitted
        steps += 1
        new_pos = pos + emitted
        for t in gen_ids[pos:new_pos]:   # advance parse state
            feed(state, t)
        if drafter in ("lookup", "combined"):
            for e in range(P + pos, P + new_pos):
                idx[tuple(context[e - n:e])] = e
        pos = new_pos

    emitted_total = n_gen - 1
    return {"steps": steps, "emitted": emitted_total,
            "loop_steps": loop_steps, "loop_emitted": loop_emitted}


# ---------------------------------------------------------------- per-stream bundle

def stream_bundle(prompt_ids: list[int], gen_ids: list[int], eos: bool,
                  drafter: str, K: int, n: int, onset: int | None) -> CaliberStream:
    """Simulate one stream and bundle it into calibers.CaliberStream.

    trunc: no loop (onset None) -> full stream; onset > MIN_HEALTHY_ONSET ->
    re-simulated loop-free prefix; onset <= MIN_HEALTHY_ONSET -> None (EXCLUDED,
    counted separately — the 2026-10-09 fix; the old code fell back to the
    full-stream τ here and inflated τ_trunc)."""
    sim = simulate(prompt_ids, gen_ids, drafter, K, n, onset)
    full = StreamRecord(steps=sim["steps"], emitted=sim["emitted"])
    loop = (StreamRecord(steps=sim["loop_steps"], emitted=sim["loop_emitted"])
            if sim["loop_steps"] > 0 else None)
    if onset is None:
        trunc = full
    elif onset > MIN_HEALTHY_ONSET:
        sim_t = simulate(prompt_ids, gen_ids[:onset], drafter, K, n, None)
        trunc = StreamRecord(steps=sim_t["steps"], emitted=sim_t["emitted"])
    else:
        trunc = None
    return CaliberStream(full=full, eos=eos, onset=onset, trunc=trunc, loop=loop)


def run_cell(cell_dir: Path, out_dir: Path, limit: int | None = None, configs=None):
    rows = [json.loads(l) for l in (cell_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    if configs is None:
        configs = [("lookup", K, n) for n in (2, 3, 4) for K in (8, 16, 32)]
        configs += [("grammar", 0, 0)]
        configs += [("combined", K, 3) for K in (8, 16, 32)]

    results = []
    for drafter, K, n in configs:
        bundles = []
        for r in rows:
            c = r["candidates"][0]
            gen = c["raw_token_ids"]
            eos = c["eos_found"]
            info = analyze(c["token_ids"], eos)
            bundles.append(stream_bundle(r["prompt_input_ids"], gen, eos,
                                          drafter, K, n, info.onset))
        table = caliber_table(bundles)
        results.append({
            "drafter": f"{drafter}" + (f"-n{n}" if drafter != "grammar" else ""),
            "K": K or None, "n": n or None,
            **table,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (cell_dir.parent.name + "--" + cell_dir.name + ".json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"== {cell_dir.parent.name}/{cell_dir.name}")
    for x in results:
        print(f"  [{x['drafter']} K={x['K']}]")
        print("  " + format_table(x).replace("\n", "\n  "))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=(RUNS / "2026-10-09-thermometer-fixed"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    run_cell(args.cell, args.out, args.limit)
