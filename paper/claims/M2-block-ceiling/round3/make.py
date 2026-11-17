"""M2 (+ M3) round 3 --  (B line): B8 + C2 in one pass, the round-2 pick m2b_survival only.

B8 (X4): the InternSVG-8B text-to-SVG panels now read TABLE 1's heads (tableA/isvg-mix); round 2 read the heads
before the continued training. Data: runs/2026-11-06/chores/b8/figdata/m2_healthy.json (fix_isvg.py: the four
other blocks are identical to runs/2026-11-03/figdata_b1/m2_healthy.json; InternSVG text recomputed on table 1's rows).
What moves (loop-free content, share of rounds that fill the window): ours 11.5 -> 14.3% greedy, 5.5 -> 6.9% sampling;
DDTree 40.0 -> 44.8 / 21.9 -> 25.9%; EAGLE-3 47.9 -> 53.2 / 30.0 -> 33.8%; DFlash 23.6 -> 27.5 / 11.9 -> 14.2%.
The claim keeps its direction in every block and mode; its ranges become ours 4-14% (was 4-12%) and
DDTree / DARTree / EAGLE-3 % (was %).
C2 (style only): house.apply; the method legend by house.legend in the same place; the plot area starts 2 pt
further right so the three-line y title clears the canvas edge (house.check: 'runs off the figure'); house.save.
Run: python paper/claims/M2-block-ceiling/round3/make.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import matplotlib.pyplot as plt
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(RUNS / "2026-11-06/chores/c2"))
import c2lib as C  # noqa: E402

H = C.H
spec = importlib.util.spec_from_file_location("m2r2", HERE.parent / "round2/make.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
H.apply()
OLD = r2.DATA
NEW = json.loads((RUNS / "2026-11-06/chores/b8/figdata/m2_healthy.json").read_text())["blocks"]
ISVG = "InternSVG-8B, text-to-SVG"
for b in OLD:
    assert (NEW[b] == OLD[b]) == (b != ISVG), b                     # only the InternSVG text block changed
DATA = NEW


def m2b_survival():
    """Round 2's m2b_survival on the B8 data; left margin + legend are the only layout changes."""
    Ht = 3.85
    fig = plt.figure(figsize=(r2.W, Ht))
    left, right, gap = 0.094, 0.006, 0.012
    rects = r2.t1.grid_rects(left, right, 0.40, Ht, gap, 1.12, 0.36)
    for r, mode in enumerate(r2.MODES):
        for c, blk in enumerate(r2.BLOCKS):
            ax = fig.add_axes(rects[mode][c])
            for m in r2.METHODS:
                d = DATA[blk][mode][m]
                s = d["healthy"]["survival"]
                ks = np.array(sorted(int(k) for k in s))
                ys = np.array([s[str(k)] for k in ks])
                ours = m == "Ours"
                ax.plot(ks, ys, color=r2.COLOR[m], lw=2.0 if ours else 1.15, zorder=5 if ours else 3,
                        drawstyle="steps-post")
                ax.plot([ks[-1], ks[-1]], [ys[-1], 0], color=r2.COLOR[m], lw=2.0 if ours else 1.15,
                        zorder=5 if ours else 3, linestyle=(0, (2, 1.5)))
            ax.set_xlim(1, 33)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([1, 16, 32])
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(axis="both", pad=1.5)
            if c == 0:
                ax.set_ylabel(f"share of rounds\nemitting ≥ k,\n{r2.MODE_TXT[mode]}", fontsize=H.SZ["label"],
                              labelpad=2, linespacing=1.05)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("tokens per round k", fontsize=H.SZ["label"], labelpad=1.5)
            if r == 0:
                ax.set_title(r2.BLOCK_LABEL[blk], fontsize=H.SZ["header"], pad=3, linespacing=1.08)
    C.legend_rect(fig, [left, 0.03 / Ht, 1 - left - right, 0.21 / Ht],
                  [H.item(m, r2.COLOR[m]) for m in r2.METHODS], gap_cap_frac=0.05)
    H.save(fig, str(HERE / "m2b_survival"), width="full")


def ranges(D):
    ours = [D[b][m]["Ours"]["healthy"]["limit_share"] for b in D for m in r2.MODES]
    base = [D[b][m][x]["healthy"]["limit_share"] for b in D for m in r2.MODES for x in ("DDTree", "DARTree", "EAGLE-3")]
    return min(ours), max(ours), min(base), max(base)


def main():
    o, n = ranges(OLD), ranges(NEW)
    print("healthy limit-hit, ours %.1f-%.1f%% -> %.1f-%.1f%%; DDTree/DARTree/EAGLE-3 %.1f-%.1f%% -> %.1f-%.1f%%" % (
        100 * o[0], 100 * o[1], 100 * n[0], 100 * n[1], 100 * o[2], 100 * o[3], 100 * n[2], 100 * n[3]))
    for b in r2.BLOCKS:                                            # the claim, block by block
        for m in r2.MODES:
            ours = NEW[b][m]["Ours"]["healthy"]["limit_share"]
            base = min(NEW[b][m][x]["healthy"]["limit_share"] for x in ("DDTree", "DARTree", "EAGLE-3"))
            assert ours < base, (b, m, ours, base)
    m2b_survival()
    print("wrote m2b_survival")


if __name__ == "__main__":
    main()
