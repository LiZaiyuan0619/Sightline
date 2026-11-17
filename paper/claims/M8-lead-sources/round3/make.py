"""M8 round 3 -- the user picked m8f (bars); two fixes, nothing else changes.

User on round 2: (1) the OmniSVG-8B image-to-SVG "inside loops" bar seemed to miss its black dot -- it was there,
hidden: greedy 1.657 and sampling 1.659 sit on top of each other, so the white dot covered the black one; the two dots
are now side by side (greedy left, sampling right) and can never hide each other. (2) Say what the dots are: a legend
entry each (filled = greedy, hollow = official sampling).
Run: python3 paper/claims/M8-lead-sources/round3/make.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m8r2", HERE.parent / "round2/make.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
fs, SZ, COLOR, BLK, MODES, CW, LIGHT = r2.fs, r2.SZ, r2.COLOR, r2.BLK, r2.MODES, r2.CW, r2.LIGHT


def m8f_bars(tau):
    R = r2.ratios(tau)
    fig = plt.figure(figsize=(CW, 2.75))
    ax = fig.add_axes([0.19, 0.205, 0.79, 0.60])
    X, Y = [], []
    for i, b in enumerate(BLK):
        for j, (seg, col) in enumerate((("healthy", LIGHT), ("loop", COLOR["Ours"]))):
            v = [R[(b, md)][seg] for md in MODES]
            x = i + (j - 0.5) * 0.38
            ax.bar(x, np.mean(v), width=0.36, color=col, linewidth=0, zorder=2)
            for k, (md, y) in enumerate(zip(MODES, v)):
                xd = x + (k - 0.5) * 0.15                  # greedy left, sampling right: never on top of each other
                ax.scatter([xd], [y], s=12, facecolor="#1A1A1A" if md == "greedy" else "white",
                           edgecolor="#1A1A1A", linewidths=0.7, zorder=4)
                X.append(xd)
                Y.append(y)
    ax.axhline(1.0, color=fs.REF, lw=0.7, linestyle=(0, (3, 2)), zorder=3)
    ax.set_xlim(-0.6, len(BLK) - 0.4)
    ax.set_ylim(0.9, 1.85)
    fs.assert_inside(ax, X, Y)
    ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8])
    ax.set_yticklabels(["×1.0", "×1.2", "×1.4", "×1.6", "×1.8"])
    ax.set_ylabel("ours ÷ DDTree,\ntokens per round", fontsize=SZ["label"], labelpad=2, linespacing=1.05)
    ax.set_xticks(range(len(BLK)))
    ax.set_xticklabels([r2.SHORT[b] for b in BLK], fontsize=7.5, linespacing=1.12)
    ax.tick_params(axis="x", length=0, pad=2)
    r2.r1.m2.legend_row(fig, [0.19, 0.905, 0.80, 0.07], [(LIGHT, "loop-free content"), (COLOR["Ours"], "inside loops")])
    h = [Line2D([], [], marker="o", ls="", ms=3.8, mfc="#1A1A1A", mec="#1A1A1A"),
         Line2D([], [], marker="o", ls="", ms=3.8, mfc="white", mec="#1A1A1A", mew=0.8)]
    fig.legend(h, ["greedy", "official sampling"], loc="upper left", bbox_to_anchor=(0.185, 0.885), ncol=2,
               fontsize=SZ["legend"], frameon=False, handletextpad=0.2, columnspacing=1.0)
    fs.save(fig, str(HERE / "m8f_bars"), width=CW)


def main():
    tau, est, fast, lossless = r2.r1.load()
    R = r2.ratios(tau)
    d = abs(R[("OmniSVG-8B image", "greedy")]["loop"] - R[("OmniSVG-8B image", "sample")]["loop"])
    print(f"OmniSVG-8B image, inside loops: greedy and sampling differ by {d:.4f} (they overlapped in round 2)")
    m8f_bars(tau)
    print("wrote m8f_bars")


if __name__ == "__main__":
    main()
