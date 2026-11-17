"""P5 round 2 data: the verification-window sweep on both heads (SGLang 0.5.22, InternSVG-8B text-to-SVG, one H200).

Source: runs/2026-11-04/b2w/b2w.json (cells[mode][arm][concurrency] -> tok_per_s, latency_s, speedup, accept, shard,
factor), harvested by runs/2026-11-04/b2w/report_b2w.py. The window-8 / window-24 shards are mapped onto
the published one-node curve by their own in-shard block-16 re-run (factor).

NOT runs/2026-11-04/figdata_b3/p5_sglang.json: that is the curve with four arms only (no window 8 / 24 on either
head) and it carries the conclusion the sweep overturned ("block 16 wins by 4-9% at high load").
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import json
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[4]
_D = json.loads((RUNS / "2026-11-04/b2w/b2w.json").read_text())["cells"]
MODES = ["greedy", "sample"]
CONC = [1, 2, 4, 8, 16, 32, 64]
OURS = {32: "b32", 24: "b32w24", 16: "b32w16", 8: "b32w8"}       # our block-32 head, by verification window
BASE = {16: "b16", 8: "b16w8"}                                    # DFlash's block-16 head, by verification window


def cell(mode, arm, c):
    return _D[mode][arm][str(c)]


def spd(mode, arm, c):
    return cell(mode, arm, c)["speedup"]


def best(mode, head, c):
    """(speed-up, window) of the head's best window at concurrency c."""
    arms = OURS if head == "ours" else BASE
    return max((spd(mode, a, c), w) for w, a in arms.items())


def lead(mode, c):
    return best(mode, "ours", c)[0] / best(mode, "base", c)[0] - 1.0


# published numbers (report_b2w.md, logs/11-04-night-figure-session-summary.md SS4.1) -------------------------------
assert round(spd("greedy", "b32", 1), 2) == 9.29 and round(spd("greedy", "b16", 1), 2) == 7.22
assert (round(spd("greedy", "b32w8", 1), 2), round(spd("greedy", "b16w8", 1), 2)) == (4.65, 4.67)   # window decides
assert [round(100 * lead("greedy", c)) for c in CONC] == [29, 24, 19, -9, -6, -1, -2]
assert [round(100 * lead("sample", c)) for c in CONC] == [24, 26, -4, -6, -3, 3, 1]
assert [best("greedy", "base", c)[1] for c in CONC] == [16, 16, 16, 16, 16, 8, 8]               # the baseline tunes too
assert [best("greedy", "ours", c)[1] for c in CONC] == [32, 24, 24, 16, 8, 8, 8]
# by load (the handoff's "5 of 7 within 3%" was loose; these are the exact cells):
#   low load, 1-2 concurrent: ours ahead by % in both modes (greedy 4 concurrent still +19%)
#   high load, concurrent: within 3.1% in 5 of the 6 settings; greedy 16 concurrent -6%
#   the crossover, 4-8 concurrent: greedy 8 -9% (the worst cell), sampling 4 / 8 -4% / -6%
_lo = [lead(m, c) for m in MODES for c in (1, 2)]
_hi = [lead(m, c) for m in MODES for c in (16, 32, 64)]
assert 0.23 < min(_lo) and max(_lo) < 0.29, _lo
assert sum(abs(v) <= 0.031 for v in _hi) == 5 and round(min(_hi), 3) == -0.063, _hi
assert round(min(lead(m, c) for m in MODES for c in CONC), 3) == -0.092

if __name__ == "__main__":
    for m in MODES:
        print(m, [(c, round(best(m, 'ours', c)[0], 2), best(m, 'ours', c)[1], round(best(m, 'base', c)[0], 2),
                   best(m, 'base', c)[1], f"{100 * lead(m, c):+.0f}%") for c in CONC])
