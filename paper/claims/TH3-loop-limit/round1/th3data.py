"""Data layer for the Proposition 3 (loop limit) figure, round 1 (main line).

ONE dataset for every candidate: the SArena full test sets of table 2 (4 model x task blocks x icon / illustration),
DDTree and ours, each lossless and with loop control (escape then stop), same job per pair. Recomputed with the
experiment line's loop_sarena.py logic (read-only inputs, nothing fitted):
  phi       share of the output's tokens the loop control removed = 1 - tokens(loop control) / tokens(lossless)
  tau_h/l   tokens per round on loop-free / loop rounds for that block and method (table-1 rows, greedy; the sampling
            values bracket the same verdict, see the README)
  r         tau_l / tau_h -- how much more a round writes inside a loop than outside it
  predicted speed-up of stopping the loop = 1 / (1 - psi), psi = loop share of verify ROUNDS
            = phi / (phi + r (1 - phi))   =>   speed-up = 1 + phi / (r (1 - phi)).  The original model has r = 1.
  measured seconds(lossless) / seconds(loop control), same cell, same job.
Asserted against the numbers runs/2026-11-05/theory/loop_sarena.md printed (median error 6.1%, 8/8 leads kept).
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import numpy as np
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
LR = json.loads((RUNS / "2026-11-04/figdata_b3/l_rules.json").read_text())
M2 = json.loads((RUNS / "2026-11-03/figdata_b1/m2_healthy.json").read_text())["blocks"]
METHODS = ("DDTree", "Ours")
MODE = "greedy"
CELLS = {(r["block"], r["subset"], r["method"]): r for r in LR["sarena"]}


def _tau_pair(e):
    h, a = e["healthy"], e["all"]
    rh, ra = h["rounds"], a["rounds"]
    return h["tau"], (a["tau"] * ra - h["tau"] * rh) / (ra - rh)


def speedup(phi, r):
    return 1 + phi / (r * (1 - phi))


ROWS = []
for (blk, sub, meth), c in sorted(CELLS.items()):
    if meth not in METHODS:
        continue
    f = CELLS.get((blk, sub, meth + " (fast)"))
    if not f or blk not in M2:
        continue
    phi = 1 - f["pred_tokens"] / c["pred_tokens"]
    th, tl = _tau_pair(M2[blk][MODE][meth])
    r = tl / th
    ROWS.append(dict(block=blk, subset=sub, method=meth, phi=phi, tau_h=th, tau_l=tl, r=r,
                     predicted=speedup(phi, r), measured=c["seconds"] / f["seconds"]))

LEADS = []
for (blk, sub, meth), c in sorted(CELLS.items()):
    if meth != "Ours":
        continue
    dd, of, ddf = CELLS.get((blk, sub, "DDTree")), CELLS.get((blk, sub, "Ours (fast)")), CELLS.get((blk, sub, "DDTree (fast)"))
    if dd and of and ddf:
        LEADS.append(dict(block=blk, subset=sub, lossless=dd["seconds"] / c["seconds"],
                          loop_control=ddf["seconds"] / of["seconds"]))

R_MED = {m: float(np.median([x["r"] for x in ROWS if x["method"] == m])) for m in METHODS}

# The lead once both decoders stop loops, predicted from the lossless lead and the two cells' predicted loop-stop
# speed-ups: lead x s_ours / s_window16.  NOTE loop_sarena.md printed lead x s_window16 / s_ours (ratio inverted), which
# is where its "median error 18%, all 8 high" came from; the correct direction gives median 6.8%, 4/8 high.
_P = {(x["block"], x["subset"], x["method"]): x for x in ROWS}
for l in LEADS:
    l["predicted"] = l["lossless"] * _P[(l["block"], l["subset"], "Ours")]["predicted"] / \
        _P[(l["block"], l["subset"], "DDTree")]["predicted"]


def psi(phi, r):
    """Share of verify rounds spent inside loops: stopping them saves exactly these rounds, speed-up 1 / (1 - psi)."""
    return phi / (phi + r * (1 - phi))


def short(block):
    m, t = block.split(", ")
    return f"{m.split('-')[0]} {'text' if t.startswith('text') else 'image'}"


def check():
    assert len(ROWS) == 16 and len(LEADS) == 8, (len(ROWS), len(LEADS))
    err = sorted(abs(x["predicted"] / x["measured"] - 1) for x in ROWS)
    assert abs(err[len(err) // 2] - 0.061) < 0.0015, err[len(err) // 2]          # loop_sarena.md: median 6.1%
    one = [x for x in ROWS if x["block"] == "OmniSVG-8B, text-to-SVG" and x["subset"] == "illustration"
           and x["method"] == "DDTree"][0]
    assert abs(one["predicted"] - 1.702) < 0.0015 and abs(one["measured"] - 1.587) < 0.0015
    assert all(l["loop_control"] > 1 for l in LEADS)                               # lead kept in 8/8
    lo = [l["lossless"] for l in LEADS]
    lc = [l["loop_control"] for l in LEADS]
    assert abs(min(lo) - 1.129) < 0.002 and abs(max(lo) - 1.413) < 0.002, (min(lo), max(lo))
    assert abs(min(lc) - 1.049) < 0.002 and abs(max(lc) - 1.187) < 0.002, (min(lc), max(lc))
    le = [l["predicted"] / l["loop_control"] - 1 for l in LEADS]
    assert abs(float(np.median(np.abs(le))) - 0.068) < 0.002 and sum(e > 0 for e in le) == 4, le
    assert all(l["predicted"] < l["lossless"] for l in LEADS)                     # the law: the lead shrinks, 8/8
    assert sum(l["loop_control"] < l["lossless"] for l in LEADS) == 7             # measured: shrinks in 7/8
    for x in ROWS:                                                                 # psi form == closed form
        assert abs(1 / (1 - psi(x["phi"], x["r"])) - x["predicted"]) < 1e-12
    return True


if __name__ == "__main__":
    check()
    for x in ROWS:
        print(f"{short(x['block']):14} {x['subset']:12} {x['method']:6} phi {x['phi']:.3f} r {x['r']:.2f} "
              f"pred x{x['predicted']:.3f} meas x{x['measured']:.3f}")
    print("median r:", {m: round(v, 2) for m, v in R_MED.items()})
    for l in LEADS:
        print(f"lead {short(l['block']):14} {l['subset']:12} lossless x{l['lossless']:.3f} -> loop control "
              f"x{l['loop_control']:.3f} (predicted x{l['predicted']:.3f})")
    print("check: ok")
