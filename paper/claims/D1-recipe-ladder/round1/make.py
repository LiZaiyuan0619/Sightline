"""D1 -- the recipe is a 2 x 2 and the two factors are complementary. Round 1, evening (house style).

Claim: our recipe changes exactly two things about a block drafter -- the draft block goes from 16 to 32 tokens,
and the block is verified as a top-16 tree instead of a chain. On loop-free content each factor alone buys
x1.07-1.13 (longer block) and x1.37-1.49 (tree) more tokens per verification round; together they buy x1.67-1.73,
MORE than the product of the two (x1.55-1.59). A longer block gives the tree more to catch.

Caliber: loop-free content -- rounds that start before the row's loop onset. 300 MMSVG-Bench prompts, greedy.
Node hygiene: the four arms did not all run on one node, so the figure never shows the raw speed-up column; it
shows tokens per round and the decoder's own token rate (tokens per round / ms per round), both free of the target's node speed. The data
pack asserts that the rate reproduces the published ladder on OmniSVG text (x1.16 / x1.23 / x1.48).

Candidates, one chart family each (guide SS3.2 / SS7.3):
  a1_interaction interaction plot: chain and tree as two lines over block length, one panel per block
  a2_waterfall    waterfall from DFlash's recipe to ours: + tree, + longer block, + what the two add together
  a3_grid         the 2 x 2 itself, four cells per block with the two factor arrows written on
  a4_cost_gain    cost against gain: ms per round on x, tokens per round on y, the four arms joined

Data: runs/2026-11-04/figdata_b4/d1_ladder.json (runs/2026-11-04/figdata_b4.py).
Run:  python3 paper/claims/D1-recipe-ladder/round1/make.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402
from sightline.paths import RUNS

H.apply()
SZ, C = H.SZ, H.C
D = json.loads((RUNS / "2026-11-04/figdata_b4/d1_ladder.json").read_text())["blocks"]
BLOCKS = ["OmniSVG-8B, text-to-SVG", "InternSVG-8B, text-to-SVG", "OmniSVG-8B, image-to-SVG"]
ARMS = ["16-chain", "16-tree", "32-chain", "32-tree"]
NAME = {"16-chain": "block 16, chain", "16-tree": "block 16, tree", "32-chain": "block 32, chain",
        "32-tree": "block 32, tree"}
# colours: the two baselines keep their own (block 16 chain = DFlash, block 16 tree = DDTree); our arm is ours
ARM_C = {"16-chain": H.METHOD["DFlash"], "16-tree": H.METHOD["DDTree"], "32-chain": "#9AAEC8",
         "32-tree": C["ours"]}


def tau(b, a):
    return D[b][a]["tau_healthy"]


def rate(b, a):
    return D[b][a]["tok_per_s_healthy"]


def gains(b, key=tau):
    base = key(b, "16-chain")
    return {a: key(b, a) / base for a in ARMS}


def check():
    for b in BLOCKS:
        g = gains(b)
        prod = g["16-tree"] * g["32-chain"]
        assert g["32-tree"] > prod + 0.07, (b, g["32-tree"], prod)
    t = gains(BLOCKS[0])
    assert (round(t["16-tree"], 2), round(t["32-chain"], 2), round(t["32-tree"], 2)) == (1.37, 1.13, 1.70), t
    print("D1: tree x%.2f-%.2f, longer block x%.2f-%.2f, together x%.2f-%.2f (product would be x%.2f-%.2f)" % (
        min(gains(b)["16-tree"] for b in BLOCKS), max(gains(b)["16-tree"] for b in BLOCKS),
        min(gains(b)["32-chain"] for b in BLOCKS), max(gains(b)["32-chain"] for b in BLOCKS),
        min(gains(b)["32-tree"] for b in BLOCKS), max(gains(b)["32-tree"] for b in BLOCKS),
        min(gains(b)["16-tree"] * gains(b)["32-chain"] for b in BLOCKS),
        max(gains(b)["16-tree"] * gains(b)["32-chain"] for b in BLOCKS)))


def block_title(b):
    mdl, task = b.split(", ")
    return f"{mdl}, {task}"


# ------------------------------------------------------------------------------------------------------- a1
def a1_interaction():
    """Two lines per panel. If the factors were independent the lines would be parallel; the tree line is steeper."""
    W = H.FULL_W
    items = [H.item("chain, as DFlash releases it", H.METHOD["DFlash"], kind="line", lw=1.4, marker="o",
                    with_marker=True, ms=4.6),
             H.item("top-16 tree, as DDTree releases it", H.METHOD["DDTree"], kind="line", lw=1.4, marker="o",
                    with_marker=True, ms=4.6)]
    leg = H.legend_height(items, W, x_pt=50, width_pt=W * 72 - 54, gap_pt=14)
    left, gap_x, right, bottom, ph, title_h = 50.0, 50.0, 6.0, 30.0, 104.0, 15.0
    pw = (W * 72 - left - right - 2 * gap_x) / 3
    Ht = bottom + ph + title_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht / 72))
    H.legend(fig, items, x_pt=50, top_pt=0, width_pt=W * 72 - 54, gap_pt=14)
    axes = [H.axes_pt(fig, left + i * (pw + gap_x), bottom, pw, ph) for i in range(3)]
    for ax, b in zip(axes, BLOCKS):
        for draft, col in (("chain", H.METHOD["DFlash"]), ("tree", H.METHOD["DDTree"])):
            ys = [tau(b, f"16-{draft}"), tau(b, f"32-{draft}")]
            ours = draft == "tree"
            ax.plot([0, 1], ys, "-o", color=C["ours"] if ours else col, lw=2.0 if ours else 1.4,
                    ms=5.4 if ours else 4.6, mfc=C["ours"] if ours else col, mec="white", mew=0.6, zorder=4 if ours else 3)
            if ours:    # the released tree arm is DDTree's colour, our arm is ours; recolour the first point
                ax.plot([0], [ys[0]], "o", ms=4.6, mfc=col, mec="white", mew=0.6, zorder=5)
        ax.set_xlim(-0.28, 1.32)
        ax.set_xticks([0, 1], ["16", "32"])
        ax.set_xlabel("draft block length", fontsize=SZ["label"], labelpad=2)
        ax.set_ylabel("tokens accepted\nper round", fontsize=SZ["label"], labelpad=2)
        lo = min(tau(b, a) for a in ARMS)
        hi = max(tau(b, a) for a in ARMS)
        ax.set_ylim(lo - 0.22 * (hi - lo), hi + 0.30 * (hi - lo))
        H.fs.assert_inside(ax, [0, 1, 0, 1], [tau(b, a) for a in ARMS])
        ax.set_title(block_title(b), fontsize=SZ["header"], pad=4, color=H.INK)
        # the gain of each factor, written where it happens
        ax.annotate(f"×{gains(b)['32-tree'] / gains(b)['16-tree']:.2f}", (1, tau(b, "32-tree")),
                    textcoords="offset points", xytext=(4, 0), ha="left", va="center", fontsize=SZ["legend"],
                    color=C["ours"], fontweight="bold")
        ax.annotate(f"×{gains(b)['32-chain']:.2f}", (1, tau(b, "32-chain")), textcoords="offset points",
                    xytext=(4, 0), ha="left", va="center", fontsize=SZ["legend"], color=H.MUTED)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.21, 0.5)
    H.save(fig, str(HERE / "a1_interaction"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- a2
def a2_waterfall():
    """From DFlash's released recipe to ours in three steps, the third being what the two factors add together."""
    W = H.FULL_W
    items = [H.item("the top-16 tree", H.METHOD["DDTree"]), H.item("the longer draft block", "#9AAEC8"),
             H.item("what the two add together", C["ours"])]
    leg = H.legend_height(items, W, x_pt=50, width_pt=W * 72 - 54, gap_pt=14)
    left, gap_x, right, bottom, ph, title_h = 50.0, 50.0, 6.0, 48.0, 104.0, 15.0
    pw = (W * 72 - left - right - 2 * gap_x) / 3
    Ht = bottom + ph + title_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht / 72))
    H.legend(fig, items, x_pt=50, top_pt=0, width_pt=W * 72 - 54, gap_pt=14)
    axes = [H.axes_pt(fig, left + i * (pw + gap_x), bottom, pw, ph) for i in range(3)]
    for ax, b in zip(axes, BLOCKS):
        base = tau(b, "16-chain")
        d_tree = tau(b, "16-tree") - base
        d_blk = tau(b, "32-chain") - base
        d_both = tau(b, "32-tree") - base - d_tree - d_blk
        steps = [("DFlash\nrecipe", 0, base, "#C9C9C9"), ("+ tree", base, d_tree, H.METHOD["DDTree"]),
                 ("+ block 32", base + d_tree, d_blk, "#9AAEC8"),
                 ("+ together", base + d_tree + d_blk, d_both, C["ours"]),
                 ("Ours", 0, tau(b, "32-tree"), C["ours"])]
        for i, (lab, bot, h, col) in enumerate(steps):
            ax.bar(i, h, bottom=bot, width=0.66, color=col, lw=0, zorder=3)
            ax.text(i, bot + h + 0.12, f"+{h:.1f}" if 0 < i < 4 else f"{bot + h:.1f}", ha="center", va="bottom",
                    fontsize=SZ["legend"], color=col if i else H.INK,
                    fontweight="bold" if i == 4 else "normal")
        for i in range(4):
            y = steps[i][1] + steps[i][2]
            ln, = ax.plot([i + 0.33, i + 1 - 0.33], [y, y], color="#C0C0C0", lw=0.7, zorder=2)
            ln.set_gid("ref")
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(0, max(tau(b, a) for a in ARMS) * 1.22)
        ax.set_xticks([])
        for i, (lab, *_rest) in enumerate(steps):
            ax.text(i, -0.035 if i % 2 == 0 else -0.145, lab, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=7.4, color=C["ours"] if lab == "Ours" else H.INK, linespacing=1.0,
                    fontweight="bold" if lab == "Ours" else "normal")
        ax.set_ylabel("tokens accepted\nper round", fontsize=SZ["label"], labelpad=2)
        ax.set_title(block_title(b), fontsize=SZ["header"], pad=4, color=H.INK)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.165, 0.5)
    H.save(fig, str(HERE / "a2_waterfall"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- a3
def a3_grid():
    """The 2 x 2 as a 2 x 2: each cell is an arm, the arrows between them carry the factor's gain."""
    W = H.FULL_W
    items = [H.item("released recipes", H.METHOD["DDTree"]), H.item("Ours", C["ours"])]
    leg = H.legend_height(items, W, x_pt=50, width_pt=W * 72 - 54, gap_pt=14)
    left, gap_x, right, bottom, ph, title_h = 50.0, 54.0, 6.0, 34.0, 104.0, 15.0
    pw = (W * 72 - left - right - 2 * gap_x) / 3
    Ht = bottom + ph + title_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht / 72))
    H.legend(fig, items, x_pt=50, top_pt=0, width_pt=W * 72 - 54, gap_pt=14)
    axes = [H.axes_pt(fig, left + i * (pw + gap_x), bottom, pw, ph) for i in range(3)]
    pos = {"16-chain": (0, 0), "32-chain": (1, 0), "16-tree": (0, 1), "32-tree": (1, 1)}
    for ax, b in zip(axes, BLOCKS):
        ax.set_xlim(-0.95, 1.95)
        ax.set_ylim(-0.75, 1.75)
        ax.set_xticks([0, 1], ["16", "32"])
        ax.set_yticks([0, 1], ["chain", "tree"])
        ax.tick_params(length=0, pad=3)
        ax.set_xlabel("draft block length", fontsize=SZ["label"], labelpad=2)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        for a, (x, y) in pos.items():
            ours = a == "32-tree"
            col = C["ours"] if ours else H.METHOD["DDTree"]
            ax.add_patch(Rectangle((x - 0.34, y - 0.235), 0.68, 0.47, facecolor=col if ours else "white",
                                   edgecolor=col, lw=1.2, zorder=3))
            ax.text(x, y, f"{tau(b, a):.1f}", ha="center", va="center", fontsize=10.5, zorder=4, gid="free",
                    color="white" if ours else H.INK, fontweight="bold" if ours else "normal")
        for (f, t_, dx, dy, lab) in (("16-chain", "32-chain", 0, -0.0, None), ("16-chain", "16-tree", 0, 0, None),
                                     ("32-chain", "32-tree", 0, 0, None), ("16-tree", "32-tree", 0, 0, None)):
            (x0, y0), (x1, y1) = pos[f], pos[t_]
            ax.add_patch(FancyArrowPatch((x0 + (0.37 if x1 > x0 else 0), y0 + (0.27 if y1 > y0 else 0)),
                                         (x1 - (0.37 if x1 > x0 else 0), y1 - (0.27 if y1 > y0 else 0)),
                                         arrowstyle="-|>,head_length=3.4,head_width=2.1", mutation_scale=1,
                                         color="#9A9A9A", lw=0.9, zorder=2))
            g = tau(b, t_) / tau(b, f)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if x1 > x0:                      # the longer-block arrows: label above / below the row
                ax.text(mx, my + (0.30 if y0 == 1 else -0.30), f"×{g:.2f}", ha="center",
                        va="bottom" if y0 == 1 else "top", fontsize=SZ["legend"], color=H.INK, zorder=4)
            else:                            # the tree arrows: label outside the column
                ax.text(mx + (0.40 if x0 == 1 else -0.40), my, f"×{g:.2f}",
                        ha="left" if x0 == 1 else "right", va="center", fontsize=SZ["legend"], color=H.INK,
                        zorder=4)
        ax.set_title(block_title(b), fontsize=SZ["header"], pad=4, color=H.INK)
    H.save(fig, str(HERE / "a3_grid"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- a4
def a4_cost_gain():
    """What each factor costs and what it buys. Grey rays are equal token rate; up and left is better."""
    W = H.FULL_W
    items = [H.item(NAME[a], ARM_C[a], kind="marker", ms=5.6 if a == "32-tree" else 5.0) for a in ARMS]
    leg = H.legend_height(items, W, x_pt=50, width_pt=W * 72 - 54, gap_pt=12)
    left, gap_x, right, bottom, ph, title_h = 50.0, 50.0, 6.0, 30.0, 104.0, 15.0
    pw = (W * 72 - left - right - 2 * gap_x) / 3
    Ht = bottom + ph + title_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht / 72))
    H.legend(fig, items, x_pt=50, top_pt=0, width_pt=W * 72 - 54, gap_pt=12)
    axes = [H.axes_pt(fig, left + i * (pw + gap_x), bottom, pw, ph) for i in range(3)]
    for ax, b in zip(axes, BLOCKS):
        xs = [D[b][a]["ms_round"] for a in ARMS]
        ys = [tau(b, a) for a in ARMS]
        x0, x1 = min(xs) * 0.90, max(xs) * 1.10
        y0, y1 = 0, max(ys) * 1.22
        for r in (300, 450, 600, 750):
            ln, = ax.plot([x0, x1], [r * x0 / 1000, r * x1 / 1000], color="#E6E6E6", lw=0.7, zorder=1)
            ln.set_gid("ref")
        ax.plot([D[b][a]["ms_round"] for a in ("16-chain", "16-tree")],
                [tau(b, a) for a in ("16-chain", "16-tree")], color="#C6C6C6", lw=0.9, zorder=2)
        ax.plot([D[b][a]["ms_round"] for a in ("32-chain", "32-tree")],
                [tau(b, a) for a in ("32-chain", "32-tree")], color="#C6C6C6", lw=0.9, zorder=2)
        for a in ARMS:
            ours = a == "32-tree"
            ax.plot([D[b][a]["ms_round"]], [tau(b, a)], "o", ms=6.4 if ours else 5.4, mfc=ARM_C[a],
                    mec="white", mew=0.7, zorder=5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        H.fs.assert_inside(ax, xs, ys)
        ax.set_xlabel("ms per verification round", fontsize=SZ["label"], labelpad=2)
        ax.set_ylabel("tokens accepted\nper round", fontsize=SZ["label"], labelpad=2)
        ax.set_title(block_title(b), fontsize=SZ["header"], pad=4, color=H.INK)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.21, 0.5)
    H.save(fig, str(HERE / "a4_cost_gain"), width="full")
    plt.close(fig)


if __name__ == "__main__":
    check()
    for fn in (a1_interaction, a2_waterfall, a3_grid, a4_cost_gain):
        try:
            fn()
            print("wrote", fn.__name__)
        except AssertionError as e:
            print("CHECK", fn.__name__, e)
    H.sheet([str(HERE / f"{n}.png") for n in ("a1_interaction", "a2_waterfall", "a3_grid", "a4_cost_gain")
             if (HERE / f"{n}.png").exists()], str(HERE / "_sheet.png"), cols=1,
            title="D1 round 1 - the recipe is a 2 x 2 (double column, print scale)")
