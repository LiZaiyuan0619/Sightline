"""Where the ceiling goes (TH5), round 1 (night, main line). Replaces the round-1 funnel of
TH1-run-length, which used the stationary ceiling (91-101) and so was not on the same footing as its other levels.

Claim (plain words). The three draft heads share a ceiling near 80 tokens per round, whatever block length they were
trained on. What separates them is the verification window: verifying 16, 32 or 64 positions keeps 14, 26 or 41 of
those tokens. Past that the drafter's own accuracy over distance (horizon) and the 127-node tree budget take over,
and they pull the block-64 head back down to the level of the block-32 head. This is why the block stops at 32.
The chain on the same head is the control: its ceiling is only 14, so its window hardly costs it anything. Widening
the candidate set is what makes the window the binding level, which is Proposition 1.

Four increasingly real models, NOT three additive losses (theory blueprint v2 section 2.3):
  ceiling  W = infinity, offset-1 labels, 16 candidates per position
  window   W finite, same labels
  horizon round-by-round replay, labels follow the round start, 16 candidates at every depth
  budget   round-by-round replay, best-first tree of 127 nodes
  measured the D2 runs on the same rows (plotted as an open ring on the budget point, they agree to 0.2%)

Data: runs/2026-11-05/theory/stream_gaps_{b64-3ep,b32-6ep,b16-6ep}{,-sample}.json, key E. Every level is asserted
against the numbers published in the theory blueprint before anything is drawn. Greedy in the figure, official
sampling has the same structure and goes to the appendix (printed by report).

Run:  python3 paper/claims/TH5-ceiling-losses/round1/make.py [names...]   (STRICT=0 to draft)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402
from sightline.paths import RUNS

H.apply()
SZ, C, INK, MUTED = H.SZ, H.C, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W_PT = H.COL_W * 72
BLUE, BLUE_MID, GREY = C["ours"], C["ours_mid"], "#8C8C8C"
THEORY = RUNS / "2026-11-05/theory"
LEVELS = ["ceiling", "window", "horizon", "budget"]
LEVEL_NAME = ["no limit", "+ window", "+ horizon", "+ budget"]
HEADS = [("b64-3ep", "blocks of 64", 64), ("b32-6ep", "blocks of 32", 32), ("b16-6ep", "blocks of 16", 16)]
# published in docs/writing/04-theory-section-v2.md section 2.3; the figure must reproduce these
PUBLISHED = {
    ("b64-3ep", ""): [79.00, 41.13, 20.58, 12.23, 12.24],
    ("b32-6ep", ""): [82.91, 25.83, 18.50, 12.35, 12.37],
    ("b16-6ep", ""): [86.04, 14.45, 12.73, 10.37, 10.36],
    ("b64-3ep", "-sample"): [62.12, 36.62, 17.45, 10.71, 10.71],
    ("b32-6ep", "-sample"): [64.61, 24.16, 16.87, 11.58, 11.61],
    ("b16-6ep", "-sample"): [64.30, 13.86, 11.75, 9.63, 9.63],
}


def load(sub=""):
    out = {}
    for h, name, blk in HEADS:
        d = json.loads((THEORY / f"stream_gaps_{h}{sub}.json").read_text())
        tree, chain = d["E"]["D1-top16-N127"], d["E"]["D0-chain"]
        row = [tree[k] for k in LEVELS] + [tree["measured"]]
        ref = PUBLISHED[(h, sub)]
        for got, want in zip(row, ref):
            assert abs(got - want) < 0.015, (h, sub, got, want)
        out[h] = dict(name=name, block=blk, tree=row,
                      chain=[chain[k] for k in LEVELS] + [chain["measured"]])
    return out


D, DS = load(), load("-sample")
CHAIN_HEAD = "b32-6ep"      # the chain control is the same head as our block-32 tree


def _fig(h_pt):
    return plt.figure(figsize=(W_PT / 72, h_pt / 72))


def _tau_axis(ax, lo=7.8, hi=105):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator([10, 20, 40, 80]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))


def _level_axis(ax):
    ax.set_xlim(-0.35, 3.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(LEVEL_NAME, fontsize=SZ["tick"] - 0.5)
    ax.tick_params(axis="x", length=0, pad=3)


def c1_models():
    """The four models as one line per head, with the chain on the same head as a control."""
    H_PT = 176
    fig = _fig(H_PT)
    L, B = 34, 30
    ax = H.axes_pt(fig, L, B, W_PT - L - 8, H_PT - B - 14)
    xs = np.arange(4)
    shades = {64: BLUE_MID, 32: BLUE, 16: "#9FB8D4"}
    ch = D[CHAIN_HEAD]["chain"]
    ax.plot(xs, ch[:4], color=GREY, lw=1.4, ls=(0, (3, 1.6)), marker="o", ms=3.0, mew=0, zorder=2)
    for h, name, blk in HEADS:
        row = D[h]["tree"]
        ax.plot(xs, row[:4], color=shades[blk], lw=2.0 if blk == 32 else 1.6, marker="o", ms=3.6, mew=0,
                zorder=4 if blk == 32 else 3)
        ax.plot([3], [row[4]], marker="o", ms=7.0, mfc="none", mec=shades[blk], mew=1.1, zorder=5)
    # the three window values (14, 26, 41) go in the caption: on a falling curve every label position
    # sits on a line, and a white box behind text is not allowed by the house check
    # key in the empty upper right (every curve is below 22 from the horizon level on)
    for n, (blk, txt) in enumerate([(64, "blocks of 64"), (32, "blocks of 32"), (16, "blocks of 16")]):
        y = 90 * (0.845 ** n)
        ln, = ax.plot([2.02, 2.32], [y, y], color=shades[blk], lw=2.0 if blk == 32 else 1.6, zorder=6)
        ln.set_gid("free")
        t = ax.text(2.42, y, txt, ha="left", va="center", fontsize=SZ["legend"] - 1.0, color=shades[blk],
                    fontweight="bold" if blk == 32 else "normal")
        t.set_gid("free")
    y = 90 * (0.845 ** 3)
    ln, = ax.plot([2.02, 2.32], [y, y], color=GREY, lw=1.4, ls=(0, (3, 1.6)), zorder=6)
    ln.set_gid("free")
    t = ax.text(2.42, y, "chain, same head", ha="left", va="center", fontsize=SZ["legend"] - 1.0, color=GREY)
    t.set_gid("free")
    _tau_axis(ax)
    _level_axis(ax)
    ax.set_ylabel("tokens per round", fontsize=SZ["label"] - 0.5, labelpad=2)
    return fig


def c2_slope():
    """What each level keeps of the one before it, in percent, so the three heads are comparable."""
    H_PT = 176
    fig = _fig(H_PT)
    L, B = 40, 30
    ax = H.axes_pt(fig, L, B, W_PT - L - 26, H_PT - B - 16)
    shades = {64: BLUE_MID, 32: BLUE, 16: "#9FB8D4"}
    for h, name, blk in HEADS:
        row = D[h]["tree"]
        keep = [100 * row[i + 1] / row[i] for i in range(3)]
        ax.plot(range(3), keep, color=shades[blk], lw=2.0 if blk == 32 else 1.6, marker="o", ms=3.6, mew=0,
                zorder=4 if blk == 32 else 3)
        t = ax.text(2.08, keep[2], f"{blk}", ha="left", va="center", fontsize=SZ["tick"] - 0.3, color=shades[blk],
                    fontweight="bold" if blk == 32 else "normal")
        t.set_gid("free")
    ch = D[CHAIN_HEAD]["chain"]
    ax.plot(range(3), [100 * ch[i + 1] / ch[i] for i in range(3)], color=GREY, lw=1.4, ls=(0, (3, 1.6)),
            marker="o", ms=3.0, mew=0, zorder=2)
    ax.set_xlim(-0.3, 2.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["window", "horizon", "tree budget"], fontsize=SZ["tick"] - 0.5)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("kept from the level before (%)", fontsize=SZ["label"] - 0.5, labelpad=2)
    return fig


def c3_bars():
    """The same four models as grouped bars on a log axis."""
    H_PT = 176
    fig = _fig(H_PT)
    L, B = 34, 30
    ax = H.axes_pt(fig, L, B, W_PT - L - 10, H_PT - B - 16)
    shades = {64: BLUE_MID, 32: BLUE, 16: "#9FB8D4"}
    w = 0.26
    for j, (h, name, blk) in enumerate(HEADS):
        row = D[h]["tree"]
        ax.bar(np.arange(4) + (j - 1) * w, row[:4], width=w * 0.92, color=shades[blk], lw=0, zorder=3,
               bottom=8.5)
    _tau_axis(ax)
    _level_axis(ax)
    ax.set_ylabel("tokens per round", fontsize=SZ["label"] - 0.5, labelpad=2)
    return fig


def c4_share():
    """The share of the ceiling each level still holds. Honest, but it puts "we reach 15% of the ideal" on the page
    (figure guide 2.2 warns about ceiling lines in the main text), so it is a fallback, not the recommendation."""
    H_PT = 176
    fig = _fig(H_PT)
    L, B = 40, 30
    ax = H.axes_pt(fig, L, B, W_PT - L - 26, H_PT - B - 16)
    shades = {64: BLUE_MID, 32: BLUE, 16: "#9FB8D4"}
    for h, name, blk in HEADS:
        row = D[h]["tree"]
        ax.plot(range(4), [100 * v / row[0] for v in row[:4]], color=shades[blk], lw=2.0 if blk == 32 else 1.6,
                marker="o", ms=3.6, mew=0, zorder=4 if blk == 32 else 3)
    ch = D[CHAIN_HEAD]["chain"]
    ax.plot(range(4), [100 * v / ch[0] for v in ch[:4]], color=GREY, lw=1.4, ls=(0, (3, 1.6)), marker="o",
            ms=3.0, mew=0, zorder=2)
    _level_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(8, 140)
    ax.yaxis.set_major_locator(FixedLocator([10, 25, 50, 100]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}%"))
    # key in the empty lower left (the three tree curves are above 17% until the horizon level)
    for n, (blk, txt) in enumerate([(64, "blocks of 64"), (32, "blocks of 32"), (16, "blocks of 16"),
                                    (None, "chain, same head")]):
        y = 15.0 * (0.86 ** n)
        col = GREY if blk is None else shades[blk]
        ln, = ax.plot([0.06, 0.36], [y, y], color=col, lw=1.4 if blk is None else (2.0 if blk == 32 else 1.6),
                      ls=(0, (3, 1.6)) if blk is None else "-", zorder=6)
        ln.set_gid("free")
        t = ax.text(0.46, y, txt, ha="left", va="center", fontsize=SZ["legend"] - 1.2, color=col,
                    fontweight="bold" if blk == 32 else "normal")
        t.set_gid("free")
    ax.set_ylabel("share of the ceiling kept", fontsize=SZ["label"] - 0.5, labelpad=2)
    return fig


def report():
    print("greedy (300 rows), tree of 127 nodes, loop-free content")
    for h, name, blk in HEADS:
        r = D[h]["tree"]
        print(f"  {name:14s} ceiling {r[0]:6.2f}  window {r[1]:6.2f}  horizon {r[2]:6.2f}  "
              f"budget {r[3]:6.2f}  measured {r[4]:6.2f}  ({100*(r[3]-r[4])/r[4]:+.1f}%)")
    r = D[CHAIN_HEAD]["chain"]
    print(f"  chain, {CHAIN_HEAD:9s} ceiling {r[0]:6.2f}  window {r[1]:6.2f}  horizon {r[2]:6.2f}  budget {r[3]:6.2f}")
    print("official sampling (100 rows), for the appendix")
    for h, name, blk in HEADS:
        r = DS[h]["tree"]
        print(f"  {name:14s} " + "  ".join(f"{v:6.2f}" for v in r))


CANDIDATES = {"c1_models": c1_models, "c2_slope": c2_slope, "c3_bars": c3_bars, "c4_share": c4_share}

if __name__ == "__main__":
    report()
    names = sys.argv[1:] or list(CANDIDATES)
    outs = []
    for n in names:
        fig = CANDIDATES[n]()
        H.save(fig, str(HERE / n), width="col", strict=STRICT)
        plt.close(fig)
        outs.append(str(HERE / f"{n}.png"))
    if len(names) > 1:
        H.sheet(outs, str(HERE / "_sheet.png"), cols=2, title="TH5 where the ceiling goes, round 1 (11-07)")
