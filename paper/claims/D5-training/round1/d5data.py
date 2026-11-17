"""D5 data: how much training the draft head needs, and where it starts from. Same caliber as D1-D3 (loop-free tokens per round,
split by runs/2026-11-04/figdata_b4.split_tau), read from the result rows.

  epochs   OmniSVG-8B text-to-SVG, block 32 x 300k, the five intermediate checkpoints re-run with the top-16 tree at
           127 nodes ($SIGHTLINE_RUNS/r1030/scale/ep1..5, ) + epoch 6 = table 1's own head ($SIGHTLINE_RUNS/tableA/t2-own/ours)
  start    OmniSVG-8B image-to-SVG, block 32 x 100k x 3 epochs: the text-to-SVG head used as is (zero-shot), trained
           from scratch, and warm-started from the text-to-SVG head ($SIGHTLINE_RUNS/r1029/ladder/{zeroshot,scratch,warm}, same job's AR)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import sys
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(RUNS / "2026-11-04"))
import figdata_b4 as B  # noqa: E402

DES = "D1-top16-N127"
EPOCHS = {e: f"r1030/scale/ep{e}" for e in range(1, 6)}
EPOCHS[6] = "tableA/t2-own/ours"
START = {"zero-shot": "r1029/ladder/zeroshot", "from scratch": "r1029/ladder/scratch", "warm start": "r1029/ladder/warm"}


def arm(root, mode="greedy"):
    rs = B.rows(root, DES, mode)
    assert rs, root
    return dict(**B.split_tau(rs), **B.speed(rs))


EP = {e: arm(r) for e, r in EPOCHS.items()}
ST = {k: arm(r) for k, r in START.items()}

# published: epoch tokens per round (all tokens) 12.59 -> 14.64 -> 15.56 -> 16.13 -> 16.16 (runs/2026-10-31/cap_scale.md); epoch 6 is
# table 1's head (loop-free 12.37 in D2); the ladder 11.89 / 11.48 / 13.30 (logs/10-30-final-session-summary.md).
# Loop-free tokens per round here uses the D1-D3 split (figdata_b4.split_tau, the caliber script); cap_scale.md's "loop-free tokens per round"
# came from the decoder's rec_trunc record and reads 0.04-0.08 lower (9.14 / 10.99 / 11.76 / 12.12 / 12.38).
assert [round(EP[e]["tau_all"], 2) for e in range(1, 6)] == [12.59, 14.64, 15.56, 16.13, 16.16], [EP[e]["tau_all"] for e in EP]
assert [round(EP[e]["tau_healthy"], 2) for e in range(1, 7)] == [9.18, 11.05, 11.83, 12.20, 12.46, 12.37]
assert [round(ST[k]["tau_all"], 2) for k in START] == [11.89, 11.48, 13.30], [ST[k]["tau_all"] for k in START]

if __name__ == "__main__":
    for e, a in EP.items():
        print("epoch", e, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items() if k in ("tau_all", "tau_healthy", "tau_loop", "speedup", "ms_round", "rows", "ar_source")})
    for k, a in ST.items():
        print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in a.items() if kk in ("tau_all", "tau_healthy", "tau_loop", "speedup", "ms_round", "rows", "ar_source")})
