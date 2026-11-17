"""Loop detector for OmniSVG generated streams.

Classifies each generated stream into one of four classes and, for loops, locates the
onset. Rules (tuned on T1 data, 2026-10-08; thresholds are explicit and configurable):

  1. EOS: stream ends with EOS (no loop by definition)
  2. capped: no EOS and length == the generation cap (1536)
  3. strict-loop: capped AND the tail window (last W tokens) matches some period p
                        with rate >= STRICT_RATE (0.90)
  4. approx-loop: capped, not strict, but tail window best-period rate >= APPROX_RATE (0.70)
  5. no-period: capped and best rate < APPROX_RATE — drifting or legitimately long

Onset: the earliest position s such that the periodic match rate on [s, n) stays
>= ONSET_RATE (0.95) with allowance (mismatch budget 5% + 2 tokens), scanning backward
from the end; reported onset is s - p (the loop probably starts at a period boundary).

Design notes vs the coordinator's quick check (runs/2026-10-08-coordinator-loopcheck):
  - same core scan, but thresholds named, configurable, and recorded in the output;
  - four-way classification with an explicit approx-loop band (0.70–0.90);
  - returns per-stream records (jsonl-able) instead of only aggregates;
  - onset search guards against tiny prefixes (min 2 full periods).

Validation requirement: sample prefixes-before-onset and render them
for eyeball checks — see render_check and the report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

STRICT_RATE = 0.90
APPROX_RATE = 0.70
ONSET_RATE = 0.95
ONSET_BUDGET = 0.05      # +2 tokens absolute
TAIL_WINDOW = 384
MAX_PERIOD = 256
MIN_PERIODS_FOR_ONSET = 2


@dataclass
class LoopInfo:
    label: str            # eos | strict-loop | approx-loop | no-period | short
    best_rate: float      # tail-window best periodic match rate
    period: int | None    # best period (tokens)
    onset: int | None     # estimated loop start index (in token_ids incl. BOS)
    loop_tokens: int      # tokens from onset to end (0 if no loop)


def analyze(tokens: list[int], eos: bool, cap: int = 1536,
            strict_rate: float = STRICT_RATE, approx_rate: float = APPROX_RATE,
            tail_window: int = TAIL_WINDOW, max_period: int = MAX_PERIOD) -> LoopInfo:
    n = len(tokens)
    if eos:
        return LoopInfo("eos", 0.0, None, None, 0)
    if n < cap:
        return LoopInfo("short", 0.0, None, None, 0)

    # ---- best period in the tail window ----
    win = min(tail_window, n)
    best_rate, best_p = 0.0, None
    for p in range(1, min(max_period, n // 2) + 1):
        lo = max(p, n - win)
        m = sum(1 for i in range(lo, n) if tokens[i] == tokens[i - p])
        r = m / (n - lo)
        if r > best_rate + 1e-9:
            best_rate, best_p = r, p
        if r >= 0.999:
            break

    # ---- classification ----
    if best_p is not None and best_rate >= strict_rate:
        label = "strict-loop"
    elif best_p is not None and best_rate >= approx_rate:
        label = "approx-loop"
    else:
        label = "no-period"

    # ---- onset (only meaningful for loops) ----
    onset = None
    if label in ("strict-loop", "approx-loop") and best_p is not None:
        p = best_p
        mism, s = 0, n
        for i in range(n - 1, p - 1, -1):
            if tokens[i] != tokens[i - p]:
                mism += 1
            if mism > ONSET_BUDGET * (n - i) + 2:
                break
            s = i
        # require at least MIN_PERIODS_FOR_ONSET full periods in the loop segment
        if n - s >= MIN_PERIODS_FOR_ONSET * p:
            onset = max(0, s - p)
        else:
            onset = None
            label = "no-period" if label == "approx-loop" else label
            # (a "loop" shorter than 2 periods is not worth calling a loop)

    loop_tokens = (n - onset) if onset is not None else 0
    return LoopInfo(label, best_rate, best_p, onset, loop_tokens)


def classify_cell(cell_dir: Path, out_path: Path | None = None) -> dict:
    """Run the detector over one T1 cell's results.jsonl; write per-stream jsonl if asked."""
    rows = [json.loads(l) for l in (cell_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    records = []
    for r in rows:
        c = r["candidates"][0]
        info = analyze(c["token_ids"], c["eos_found"])
        records.append({"id": r["id"], **asdict(info), "length": len(c["token_ids"])})
    agg = {
        "cell": str(cell_dir),
        "n": len(records),
        "eos": sum(x["label"] == "eos" for x in records),
        "strict": sum(x["label"] == "strict-loop" for x in records),
        "approx": sum(x["label"] == "approx-loop" for x in records),
        "no_period": sum(x["label"] == "no-period" for x in records),
        "short": sum(x["label"] == "short" for x in records),
        "loop_token_share": (
            sum(x["loop_tokens"] for x in records) /
            sum(x["length"] for x in records) if records else 0.0),
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for x in records:
                f.write(json.dumps(x) + "\n")
    return agg


def render_check(records: list[dict], cell_dir: Path, out_dir: Path, k: int = 5,
                 render_fn=None) -> list[Path]:
    """Sample k strict-loop streams, decode prefix-before-onset and full stream, render
    both, so the loop can be eyeballed. render_fn(raw_ids, subtype) -> (svg, image)."""
    import random
    rng = random.Random(0)
    loops = [x for x in records if x["label"] == "strict-loop" and x["onset"] is not None]
    rng.shuffle(loops)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    rows = {json.loads(l)["id"]: json.loads(l)
            for l in (cell_dir / "results.jsonl").read_text().splitlines() if l.strip()}
    for x in loops[:k]:
        r = rows[x["id"]]
        c = r["candidates"][0]
        for name, ids in (("prefix", c["token_ids"][:x["onset"]]),
                          ("full", c["token_ids"])):
            if render_fn is None:
                continue
            svg, img = render_fn(c["raw_token_ids"] if name == "full" else ids, r.get("subtype", "icon"))
            if svg:
                (out_dir / f"{x['id'][:8]}-{name}.svg").write_text(svg)
            if img is not None:
                img.save(out_dir / f"{x['id'][:8]}-{name}.png")
            saved.append(out_dir / f"{x['id'][:8]}-{name}.png")
    return saved
