"""Proposition 2 (horizon in decisions) evidence figure, round 1 (main line). Single-column candidates.

Claim (one sentence): a block drafter writes every position of a block from the same prefix, so a decision crossed
inside the block costs only the positions that depend on it -- after two or more decisions the median prompt still
gets about nine in ten command tokens right but fewer than half of its coordinates.

Every candidate reads the SAME data (thhdata.py): our block-32 head, the same 277 prompts as Proposition 1's figure,
loop-free, greedy, decision point = target entropy > 0.2 nats, offsets 1-8 of the block (where distance alone costs
almost nothing), per-prompt cells with >= 30 scored positions. q1e adds the farther offsets of the same prompts.
Run:  python3 paper/claims/TH2-horizon/round1/make.py [names...]   (STRICT=0 to draft)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import thhdata as T  # noqa: E402

T.check()
H.apply()
SZ, C, INK, MUTED = H.SZ, H.C, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W_PT = H.COL_W * 72
LEG_FS = 7.8
# token classes as in M1 (structure blue-grey, numbers amber), darker for marks
COL = {"cmd": "#566F8E", "coord": "#C98A1E"}
LIGHT = {"cmd": "#CED7E2", "coord": "#F0DCB4"}
NAME = {"cmd": "commands", "coord": "coordinates"}
XLAB = ["0", "1", "2 or more"]


def _fig(h_pt):
    return plt.figure(figsize=(W_PT / 72, h_pt / 72))


def _pct_axis(ax, lo=0, hi=100, ticks=(0, 25, 50, 75, 100)):
    ax.set_ylim(lo, hi)
    ax.set_yticks(list(ticks))


# =============================================================================================== candidates
def q1a_slopes():
    """Per prompt, first-guess accuracy after 0, 1 and 2+ decisions crossed inside the block (faint lines), medians in
    bold, for commands and coordinates."""
    fig = _fig(182)
    items = [H.item(NAME["cmd"], COL["cmd"], kind="line", lw=2.0), H.item(NAME["coord"], COL["coord"], kind="line", lw=2.0)]
    bot = H.legend(fig, items, x_pt=40, top_pt=2, width_pt=W_PT - 42, fontsize=LEG_FS, gap_pt=12)
    ax = H.axes_pt(fig, 40, 30, W_PT - 40 - 34, 182 - 30 - bot - 4)
    X = np.array([0, 1, 2])
    for cl, dx in (("coord", 0.03), ("cmd", -0.03)):
        M = 100 * T.lines(cl)
        for row in M:
            ln, = ax.plot(X + dx, row, color=COL[cl], lw=0.45, alpha=0.10, zorder=1)
            ln.set_rasterized(True)
            ln.set_gid("bg")
        med = np.median(M, axis=0)
        ax.plot(X + dx, med, color=COL[cl], lw=2.2, marker="o", ms=4.6, zorder=4,
                path_effects=[pe.withStroke(linewidth=3.6, foreground="white")])
        ax.text(2.12, med[-1], f"{med[-1]:.0f}%", ha="left", va="center", fontsize=SZ["legend"], color=COL[cl],
                fontweight="bold")
    ax.set_xlim(-0.2, 2.2)
    ax.set_xticks(X)
    ax.set_xticklabels(XLAB)
    _pct_axis(ax)
    ax.set_xlabel("decisions crossed inside the block", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("first guess right (%)", fontsize=SZ["label"], labelpad=2)
    return fig


def q1b_paired():
    """Per prompt after two or more decisions crossed: x = coordinates right, y = commands right. Above the diagonal,
    the prompt kept its commands better than its coordinates; the cross marks the medians."""
    fig = _fig(208)
    ax = H.axes_pt(fig, 40, 30, W_PT - 40 - 12, 208 - 30 - 10)
    x, y = (100 * v for v in T.paired_c2())
    tri = ax.fill_between([0, 100], [0, 100], [100, 100], color="#EEF1F5", lw=0, zorder=0)
    tri.set_gid("bg")
    ln, = ax.plot([0, 100], [0, 100], color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref")
    ax.text(86, 74, "same\naccuracy", ha="left", va="top", fontsize=SZ["tick"] - 0.9, color=MUTED, linespacing=1.1)
    sc = ax.scatter(x, y, s=10, color=COL["cmd"], alpha=0.55, lw=0, zorder=3)
    sc.set_rasterized(True)
    mx, my = float(np.median(x)), float(np.median(y))
    for a, b in (([mx, mx], [0, my]), ([0, mx], [my, my])):
        l2, = ax.plot(a, b, color=INK, lw=0.8, ls=(0, (1, 1.5)), zorder=2)
        l2.set_gid("ref")
    ax.plot([mx], [my], marker="D", ms=6.4, mfc="white", mec=INK, mew=1.5, zorder=5)
    t = ax.text(mx + 3, my + 3, f"median prompt\ncoordinates {mx:.0f}%, commands {my:.0f}%", ha="left",
                va="bottom", fontsize=SZ["tick"] - 0.9, color=INK, linespacing=1.15,
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    t.set_gid("free")
    share = 100 * float(np.mean(y > x))
    ax.text(96, 5, f"{share:.0f}% of prompts\nabove the line", ha="right", va="bottom", fontsize=SZ["tick"] - 0.4,
            color=COL["cmd"], fontweight="bold", linespacing=1.12)
    ax.set_xlim(0, 100)
    _pct_axis(ax)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_aspect("auto")
    ax.set_xlabel("coordinates right after 2+ decisions (%)", fontsize=SZ["label"] - 0.3, labelpad=1.5)
    ax.set_ylabel("commands right after 2+ decisions (%)", fontsize=SZ["label"] - 0.3, labelpad=2)
    return fig


def q1c_swarm():
    """Violin + one dot per prompt for each class and number of decisions crossed; median bar, value on top."""
    fig = _fig(184)
    ax = H.axes_pt(fig, 40, 40, W_PT - 40 - 4, 184 - 40 - 6)
    rng = np.random.default_rng(1106)
    pos, xs, labs, spans = 0.0, [], [], {}
    for cl in ("cmd", "coord"):
        start = pos
        for c in range(3):
            v = 100 * T.cell(cl, 0, c)
            v = v[~np.isnan(v)]
            vp = ax.violinplot([v], positions=[pos], widths=0.78, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(LIGHT[cl])
                b.set_edgecolor("none")
                b.set_alpha(1.0)
                b.set_gid("bg")
            sc = ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)), v, s=2.2, color=COL[cl], alpha=0.45, lw=0, zorder=3)
            sc.set_rasterized(True)
            med = float(np.median(v))
            ax.plot([pos - 0.3, pos + 0.3], [med, med], color=COL[cl], lw=1.9, zorder=4, solid_capstyle="butt",
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])
            ax.text(pos, 104, f"{med:.0f}%", ha="center", va="bottom", fontsize=SZ["tick"] - 0.4, color=COL[cl],
                    fontweight="bold" if c == 2 else "normal")
            xs.append(pos)
            labs.append(XLAB[c].replace(" or more", "+"))
            pos += 1.0
        spans[cl] = (start, pos - 1.0)
        pos += 0.6
    ax.set_xlim(-0.6, pos - 1.0)
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    for cl, (a, b) in spans.items():
        ax.plot([a - 0.34, b + 0.34], [-0.115, -0.115], transform=ax.get_xaxis_transform(), clip_on=False,
                color=COL[cl], lw=0.9)
        ax.text((a + b) / 2, -0.135, NAME[cl], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=SZ["tick"] - 0.2, color=COL[cl])
    ax.text(-0.04, -0.085, "decisions\ncrossed", transform=ax.transAxes, ha="right", va="center",
            fontsize=SZ["tick"] - 1.0, color=MUTED, linespacing=1.1)
    ax.set_ylabel("first guess right (%)", fontsize=SZ["label"], labelpad=2)
    return fig


def q1d_drop():
    """Per-prompt loss from crossing two or more decisions (accuracy at none minus at 2+), as the share of prompts
    that lose at most that much: commands stay near the left edge, coordinates sit far to the right."""
    fig = _fig(170)
    items = [H.item(NAME["cmd"], COL["cmd"], kind="line", lw=2.0), H.item(NAME["coord"], COL["coord"], kind="line", lw=2.0)]
    bot = H.legend(fig, items, x_pt=40, top_pt=2, width_pt=W_PT - 42, fontsize=LEG_FS, gap_pt=12)
    ax = H.axes_pt(fig, 40, 30, W_PT - 40 - 8, 170 - 30 - bot - 4)
    for cl in ("cmd", "coord"):
        d = np.sort(T.drop(cl))
        yy = 100 * np.arange(1, len(d) + 1) / len(d)
        ax.step(np.concatenate([[d[0]], d]), np.concatenate([[0], yy]), where="post", color=COL[cl], lw=2.0, zorder=3)
        m = float(np.median(d))
        ax.plot([m], [50], "o", ms=4.6, color=COL[cl], zorder=5)
        ax.text(m + (3 if cl == "coord" else 3), 46, f"median {m:.0f} points", ha="left", va="top",
                fontsize=SZ["tick"] - 0.6, color=COL[cl], fontweight="bold")
    ln = ax.axhline(50, color=H.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref")
    ax.set_xlim(-10, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    _pct_axis(ax)
    ax.set_xlabel("accuracy lost after 2+ decisions (points)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("prompts losing at most\nthis much (%)", fontsize=SZ["label"], labelpad=2, linespacing=1.18)
    return fig


def q1e_bands():
    """The same prompts at farther offsets: median prompt accuracy by offset band, no decision crossed (solid) vs two
    or more (dashed). The gap opened by decisions stays wide at every distance; distance alone costs little."""
    fig = _fig(176)
    items = [H.item(NAME["cmd"], COL["cmd"], kind="line", lw=2.0), H.item(NAME["coord"], COL["coord"], kind="line", lw=2.0),
             H.item("no decision crossed", INK, kind="line", lw=1.4), H.item("2 or more", INK, kind="line", lw=1.4,
                                                                             ls=(0, (3, 2)))]
    bot = H.legend(fig, items, x_pt=40, top_pt=2, width_pt=W_PT - 42, fontsize=LEG_FS, gap_pt=9)
    ax = H.axes_pt(fig, 40, 30, W_PT - 40 - 30, 176 - 30 - bot - 4)
    X = np.arange(3)
    for cl in ("cmd", "coord"):
        for c, ls in ((0, "-"), (2, (0, (3, 2)))):
            v = [100 * float(np.nanmedian(T.cell(cl, b, c))) for b in range(3)]
            ax.plot(X, v, color=COL[cl], lw=2.0, ls=ls, marker="o", ms=4.0, mfc=COL[cl] if c == 0 else "white",
                    mew=1.3, zorder=4)
            if c == 2:
                ax.text(2.1, v[-1], f"{v[-1]:.0f}%", ha="left", va="center", fontsize=SZ["tick"] - 0.4,
                        color=COL[cl], fontweight="bold")
    ax.set_xlim(-0.25, 2.25)
    ax.set_xticks(X)
    ax.set_xticklabels(["1–8", "9–16", "17–31"])
    _pct_axis(ax)
    ax.set_xlabel("offset inside the block", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("first guess right (%)", fontsize=SZ["label"], labelpad=2)
    return fig


CANDIDATES = {"q1a_slopes": q1a_slopes, "q1b_paired": q1b_paired, "q1c_swarm": q1c_swarm, "q1d_drop": q1d_drop,
              "q1e_bands": q1e_bands}


def main(names):
    outs = []
    for n in names or list(CANDIDATES):
        fig = CANDIDATES[n]()
        H.save(fig, str(HERE / n), width="col", strict=STRICT)
        plt.close(fig)
        outs.append(str(HERE / f"{n}.png"))
    if not names:
        H.sheet(outs, str(HERE / "_sheet.png"), cols=3,
                title="TH2 horizon in decisions, round 1: one dataset (block-32 head, 277 prompts, offsets 1-8)")


if __name__ == "__main__":
    main(sys.argv[1:])
