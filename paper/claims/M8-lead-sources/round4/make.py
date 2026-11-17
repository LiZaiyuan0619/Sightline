"""M8 round 4 --  (B line, C2): round 3's m8f_bars moved onto the house style. Style only: same data
(round 1's loader), same bars, same dots (greedy left / official sampling right, filled / hollow as ENCODING.md SS2).
What changes: both legend rows (bars; dots) by house.legend in the old places; house.apply; house.save; the bars
start at the axis floor (0.9) instead of 0 -- the same picture, but no clipped-away path for the PDF audit to
find under the tick labels (round 3 had 10 such text-fill-edge warnings).
Run: python paper/claims/M8-lead-sources/round4/make.py
"""
import importlib.util
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
spec = importlib.util.spec_from_file_location("m8r3", HERE.parent / "round3/make.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
r2 = r3.r2
H.apply()


def m8f_bars(tau):
    R = r2.ratios(tau)
    fig = plt.figure(figsize=(r2.CW, 2.75))
    ax = fig.add_axes([0.19, 0.205, 0.79, 0.60])
    X, Y = [], []
    for i, b in enumerate(r2.BLK):
        for j, (seg, col) in enumerate((("healthy", r2.LIGHT), ("loop", r2.COLOR["Ours"]))):
            v = [R[(b, md)][seg] for md in r2.MODES]
            x = i + (j - 0.5) * 0.38
            ax.bar(x, np.mean(v) - 0.9, bottom=0.9, width=0.36, color=col, linewidth=0, zorder=2)  # no clipped path
            for k, (md, y) in enumerate(zip(r2.MODES, v)):
                xd = x + (k - 0.5) * 0.15
                ax.scatter([xd], [y], s=12, facecolor="#1A1A1A" if md == "greedy" else "white",
                           edgecolor="#1A1A1A", linewidths=0.7, zorder=4)
                X.append(xd)
                Y.append(y)
    H.ref_line(ax, y=1.0, lw=0.7, zorder=3)
    ax.set_xlim(-0.6, len(r2.BLK) - 0.4)
    ax.set_ylim(0.9, 1.85)
    r2.fs.assert_inside(ax, X, Y)
    ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_yticklabels(["×1.0", "×1.2", "×1.4", "×1.6", "×1.8"])
    ax.set_ylabel("ours ÷ DDTree,\ntokens per round", fontsize=H.SZ["label"], labelpad=2, linespacing=1.05)
    ax.set_xticks(range(len(r2.BLK)))
    ax.set_xticklabels([r2.SHORT[b] for b in r2.BLK], fontsize=7.5, linespacing=1.12)
    ax.tick_params(axis="x", length=0, pad=2)
    bot = C.legend_rect(fig, [0.19, 0.905, 0.80, 0.07], C.patch_items(
        [(r2.LIGHT, "loop-free content"), (r2.COLOR["Ours"], "inside loops")]), gap_cap_frac=0.05)
    H.legend(fig, [H.item("greedy", "#1A1A1A", kind="marker", ms=4.2, mew=0.8),
                   H.item("official sampling", "#1A1A1A", kind="marker", hollow=True, ms=4.2, mew=0.8)],
             x_pt=0.19 * r2.CW * 72, top_pt=bot + 0.5, gap_pt=9.4)
    H.save(fig, str(HERE / "m8f_bars"), width="col")


def main():
    tau, est, fast, lossless = r2.r1.load()
    m8f_bars(tau)
    print("wrote m8f_bars")


if __name__ == "__main__":
    main()
