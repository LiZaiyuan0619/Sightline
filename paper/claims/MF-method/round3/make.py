"""Method figure (main.pdf Figure 2), round 3 (main-line session 8): single-column candidates around the trunk.

User 11-09/ the double-column round 2 (mfb_pipeline) is too empty and unclear; make it single column and direct,
put the trunk first and keep details few ("draw only the core idea and the trunk first", too many details made the old one messy);
draw several versions before deciding. No seconds on the figure. Plain words, no symbols.
The round is real: round 0 of OmniSVG-8B text-to-SVG bench row 150 (runs/2026-10-29/analysis/trees/tree-gidx150.json,
our block-32 head, top-16 candidates, 127-node tree, through../round1/mfdata.py): 21 drafted tokens accepted, then a
bonus token; the 127 nodes reach position 22; positions 11 and 15 accept the draft's second candidate.
A window of 16 verifies at most 15 drafted positions plus a bonus token, so on the same drafts it stops after 15.
  v1_tree        the real tree alone, with the two windows
  v2_abstract    a clean chain with a few branches (DDTree's style), the path taking second candidates at 11 and 15
  v3_rulers      two rows of slots: what a window of 16 and a window of 32 keep from this round
  v4_tree_tiers v1 with loop control and best of 8 as two strips under it (abstract marks)
  v5_rulers_tiers v3 with the two strips side by side
Run: python3 paper/claims/MF-method/round3/make.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE.parent / "round1"))
import house as H  # noqa: E402
import mfdata as MD  # noqa: E402

H.apply()
INK, MUTED, REF = H.INK, H.MUTED, H.REF
OURS, OURS_L = H.METHOD["Ours"], H.C["ours_light"]
NODE = "#B9B9B9"
W = 234.0
FS = dict(lab=7.5, small=7.0, key=8.0)
T = MD.tree()
F = T["facts"]
ACC, BONUS = F["accepted"], F["accepted"] + 1          # 21 accepted, bonus at position 22
assert (ACC, F["max_depth"], F["off_chain"]) == (21, 22, [(11, 2), (15, 2)])


def fig_pt(h):
    return plt.figure(figsize=(W / 72, h / 72))


def overlay(fig):
    return H._overlay(fig)


def save(fig, name):
    H.save(fig, str(HERE / name), width=W / 72, strict=True)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- the real tree (v1, v4)
def draw_tree(fig, x0, y0, w, h, show_axis=True):
    """The real round in a box (points from the figure's bottom-left): depth on x (positions 1-32), the tidy layout of
    round 1 on y, the accepted path straight along the middle in our blue, the bonus token as a ring."""
    ax = H.axes_pt(fig, x0, y0, w, h)
    ax.set_xlim(0.5, 32.5)
    ylo, yhi = F["y_lo"] - 0.8, F["y_hi"] + 0.8
    ax.set_ylim(ylo, yhi)
    # what the longer window adds: a light band between the two window ends
    ax.axvspan(15.5, 31.5, color="#EEF3F9", zorder=0, lw=0).set_gid("bg")
    for (xa, ya), (xb, yb), acc in T["edges"]:
        if not acc:
            ax.plot([xa, xb], [ya, yb], color=NODE, lw=0.45, zorder=2, solid_capstyle="round")
    xs = [d for d, y, a, r in T["nodes"] if not a]
    ys = [y for d, y, a, r in T["nodes"] if not a]
    ax.scatter(xs, ys, s=4.0, color=NODE, edgecolors="none", zorder=3)
    path = sorted((d, y) for d, y, a, r in T["nodes"] if a)
    ax.plot([p[0] for p in path], [p[1] for p in path], color=OURS, lw=1.6, zorder=4, solid_capstyle="round")
    ax.scatter([p[0] for p in path], [p[1] for p in path], s=9.0, color=OURS, edgecolors="none", zorder=5)
    ax.plot([path[-1][0], BONUS], [0, 0], color=OURS, lw=1.0, ls=(0, (1.5, 1.2)), zorder=4)
    ax.scatter([BONUS], [0], s=16, facecolors="white", edgecolors=OURS, linewidths=1.0, zorder=6)
    for xv, col, ls in ((15.5, MUTED, (0, (3, 2))), (31.5, OURS, "-")):
        ax.plot([xv, xv], [ylo, yhi - 0.6], color=col, lw=0.9, ls=ls, zorder=1)[0].set_gid("ref")
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    if show_axis:
        ax.set_xticks([1, 8, 16, 24, 31])
        ax.tick_params(axis="x", labelsize=FS["small"], length=2.0, pad=1.5)
        ax.set_xlabel("drafted position", fontsize=FS["lab"], labelpad=1.0)
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    return ax


def tree_labels(fig, ax):
    tr = ax.transData
    def at(x, y, s, **kw):
        t = ax.text(x, y, s, fontsize=kw.pop("fs", FS["lab"]), **kw)
        return t
    yhi = ax.get_ylim()[1]
    at(15.5, yhi, "window of 16", ha="center", va="bottom", color=MUTED)
    at(31.5, yhi, "window of 32", ha="right", va="bottom", color=OURS, fontweight="bold")
    at(BONUS + 0.75, 0.0, f"{ACC} accepted,\nthen a bonus", ha="left", va="center", color=OURS, fs=FS["small"],
       linespacing=1.05)
    at(1.0, F["y_lo"] + 0.3, "127-node draft tree,\n16 candidates per position", ha="left", va="bottom", color=MUTED,
       fs=FS["small"], linespacing=1.05)


def v1_tree(name="v1_tree"):
    Ht = 112.0
    fig = fig_pt(Ht)
    ax = draw_tree(fig, 6, 22, W - 10, Ht - 22 - 12)
    tree_labels(fig, ax)
    save(fig, name)


# ------------------------------------------------------------------------------------------ v2: the abstract chain
def draw_abstract(fig, x0, y0, w, h, show_axis=True):
    """A clean schematic of the same round, drawn as a fishbone: every alternative candidate branches off the accepted
    path at the position before it. At positions 11 and 15 the draft's first candidate (drawn above, grey) was rejected
    and its second candidate (on the path) accepted. Empty dots run on to the window of 32."""
    ax = H.axes_pt(fig, x0, y0, w, h)
    ax.set_xlim(0.3, 32.5)
    ax.set_ylim(-2.4, 2.8)
    ax.axvspan(15.5, 31.5, color="#EEF3F9", zorder=0, lw=0).set_gid("bg")
    second = [d for d, _ in F["off_chain"]]                  # 11 and 15
    widths = {d: (1 if d < 15 else 2) for d in range(7, BONUS + 1)}
    for d, k in widths.items():
        ys = [1.15] if k == 1 else [1.15, -1.15]
        if d in second:
            ys = [1.15] + ([-1.15] if k == 2 else [])
        for y in ys:
            first = d in second and y > 0
            ax.plot([d - 1, d], [0, y], color=NODE if not first else "#8C8C8C", lw=0.6, zorder=2)
            ax.scatter([d], [y], s=12, facecolors="#8C8C8C" if first else "white", edgecolors="#8C8C8C" if first else NODE,
                       linewidths=0.8, zorder=3)
    ax.plot(range(1, ACC + 1), [0] * ACC, color=OURS, lw=1.6, zorder=4)
    ax.scatter(range(1, ACC + 1), [0] * ACC, s=15, color=OURS, edgecolors="none", zorder=5)
    ax.scatter(second, [0, 0], s=34, facecolors="none", edgecolors=OURS, linewidths=0.9, zorder=6)
    ax.plot([ACC, BONUS], [0, 0], color=OURS, lw=1.0, ls=(0, (1.5, 1.2)), zorder=4)
    ax.scatter([BONUS], [0], s=20, facecolors="white", edgecolors=OURS, linewidths=1.1, zorder=6)
    ax.scatter(range(BONUS + 1, 32), [0] * (31 - BONUS), s=6, color="#D5D5D5", edgecolors="none", zorder=3)
    for xv, col, ls in ((15.5, MUTED, (0, (3, 2))), (31.5, OURS, "-")):
        ax.plot([xv, xv], [-2.4, 2.25], color=col, lw=0.9, ls=ls, zorder=1)[0].set_gid("ref")
    for s_ in ("left", "right", "top"):
        ax.spines[s_].set_visible(False)
    ax.set_yticks([])
    if show_axis:
        ax.set_xticks([1, 8, 16, 24, 31])
        ax.tick_params(axis="x", labelsize=FS["small"], length=2.0, pad=1.5)
        ax.set_xlabel("drafted position", fontsize=FS["lab"], labelpad=1.0)
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    return ax


def v2_abstract(name="v2_abstract"):
    Ht = 100.0
    fig = fig_pt(Ht)
    ax = draw_abstract(fig, 6, 22, W - 10, Ht - 22 - 12)
    yhi = ax.get_ylim()[1]
    ax.text(15.5, yhi, "window of 16", ha="center", va="bottom", color=MUTED, fontsize=FS["lab"])
    ax.text(31.5, yhi, "window of 32", ha="right", va="bottom", color=OURS, fontsize=FS["lab"], fontweight="bold")
    ax.text(BONUS + 0.4, -0.55, f"{ACC} accepted,\nthen a bonus", ha="left", va="top", color=OURS, fontsize=FS["small"],
            linespacing=1.05)
    # one label for each kind of mark at 11 and 15, with a short leader to each
    bx, by = 9.2, -1.25
    ax.text(bx, by, "second candidates, accepted", ha="center", va="top", color=OURS, fontsize=FS["small"])
    for d in (11, 15):
        ax.plot([d, bx + (d - 13) * 0.45], [-0.34, by + 0.28], color=OURS, lw=0.5, zorder=2)
    tx, ty = 8.6, 2.05
    ax.text(tx, ty, "first candidates, rejected", ha="center", va="bottom", color="#7A7A7A", fontsize=FS["small"])
    for d in (11, 15):
        ax.plot([d, tx + (d - 13) * 0.45], [1.42, ty - 0.28], color="#7A7A7A", lw=0.5, zorder=2)
    save(fig, name)


# ---------------------------------------------------------------------------------------------- v3: two rulers
def draw_rulers(fig, x0, y0, w, h):
    """Two rows of token slots for this round, on the same drafts: a window of 16 fills its 16 slots (15 accepted
    drafts and the bonus token); a window of 32 takes 21 accepted drafts and the bonus, 22 tokens, and leaves 10 slots."""
    ax = H.axes_pt(fig, x0, y0, w, h)
    ax.set_xlim(0.4, 32.6)
    ax.set_ylim(-0.12, 2.25)
    rows = [(1.35, 16, MUTED, "#9E9E9E", "window of 16, 16 tokens this round"),
            (0.0, 32, OURS, OURS, f"window of 32, {BONUS} tokens this round")]
    for y, win, col, fill, title in rows:
        n_acc = min(ACC, win - 1)
        for k in range(1, win + 1):
            if k <= n_acc:
                ax.add_patch(Rectangle((k - 0.42, y), 0.84, 0.62, facecolor=fill, edgecolor="none", zorder=3))
            elif k == n_acc + 1:
                ax.add_patch(Rectangle((k - 0.42, y), 0.84, 0.62, facecolor="white", edgecolor=fill, lw=1.0, zorder=4))
            else:
                ax.add_patch(Rectangle((k - 0.42, y), 0.84, 0.62, facecolor="white", edgecolor="#D3D3D3", lw=0.5,
                                       ls=(0, (1.5, 1.0)), zorder=3))
        ax.plot([win + 0.5, win + 0.5], [y - 0.08, y + 0.70], color=col, lw=1.0, zorder=5,
                ls="-" if win == 32 else (0, (3, 2)))[0].set_gid("ref")
        ax.text(0.58, y + 0.70, title, ha="left", va="bottom", fontsize=FS["lab"], color=col,
                fontweight="bold" if win == 32 else "normal")
    for s_ in ("left", "right", "top"):
        ax.spines[s_].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.tick_params(axis="x", labelsize=FS["small"], length=2.0, pad=1.5)
    ax.set_xlabel("token of the round", fontsize=FS["lab"], labelpad=1.0)
    return ax


def ruler_keys(fig, top_pt):
    items = [H.item("accepted draft", OURS), H.item("bonus token", OURS, kind="marker", marker="s", hollow=True, ms=4.6),
             H.item("unused", "#BDBDBD", kind="marker", marker="s", hollow=True, ms=4.6)]
    return H.legend(fig, items, x_pt=4, top_pt=top_pt, width_pt=W - 8, fontsize=FS["small"], gap_pt=8.0)


def v3_rulers(name="v3_rulers"):
    Ht = 86.0
    fig = fig_pt(Ht)
    lh = H.legend_height([H.item("x", OURS)], W / 72, x_pt=4, fontsize=FS["small"])
    ax = draw_rulers(fig, 6, 22, W - 10, Ht - 22 - lh - 4)
    ruler_keys(fig, 1.0)
    save(fig, name)


# --------------------------------------------------------------------------------------- tiers (v4, v5): abstract
LOOP_C, LOOP_L = H.C["loop"], H.C["loop_light"]


def tier_axes(fig, x0, y0, w, h):
    """An axes whose data units are points, for hand-placed marks."""
    ax = H.axes_pt(fig, x0, y0, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return ax


def draw_loop(ax, x, y, sq=4.0, gap=0.7, block_gap=2.2):
    """Abstract marks: three copies of one cycle, an escape, three copies again, a stop. Returns the right edge."""
    cols = [LOOP_C, "#E2B866", LOOP_L]

    def copies(x0):
        for k in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((x0 + j * (sq + gap), y - sq * 0.7), sq, sq * 1.4, facecolor=cols[j],
                                       edgecolor="none", zorder=3))
            x0 += 3 * sq + 2 * gap + block_gap
        return x0 - block_gap
    x1 = copies(x)
    ax.annotate("", xy=(x1 + 13, y), xytext=(x1 + 2, y),
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=0.9, mutation_scale=6, shrinkA=0, shrinkB=0))
    ax.text(x1 + 7.5, y - sq * 0.7 - 1.5, "escape", ha="center", va="top", fontsize=FS["small"], color=OURS)
    x2 = copies(x1 + 15)
    ax.plot([x2 + 2.5, x2 + 2.5], [y - sq * 0.9, y + sq * 0.9], color=INK, lw=1.6, zorder=4)
    ax.text(x2 + 2.5, y - sq * 0.7 - 1.5, "stop", ha="center", va="top", fontsize=FS["small"], color=INK)
    return x2 + 4


def draw_bo8(ax, x, y, tile=9.0, gap=1.8, best=5):
    """Eight samples of one batch as tiles; each is rendered and scored, and the kept one is framed in our blue."""
    tone = ["#D9D9D9", "#C8C8C8", "#E3E3E3", "#BDBDBD", "#D0D0D0", "#C2C2C2", "#DADADA", "#CBCBCB"]
    for k in range(8):
        xk = x + k * (tile + gap)
        ax.add_patch(Rectangle((xk, y - tile / 2), tile, tile, facecolor=tone[k], edgecolor=OURS if k == best else "none",
                               lw=1.4 if k == best else 0, zorder=3))
    xe = x + 8 * tile + 7 * gap
    ax.text(x, y - tile / 2 - 1.5, "render each, score it, keep the best", ha="left", va="top", fontsize=FS["small"],
            color=INK)
    return xe


def v4_tree_tiers(name="v4_tree_tiers"):
    tree_h, strip_h, gap = 84.0, 22.0, 6.0
    lx, tx = 2.0, 13.0                       # panel letter, tier titles
    cx = 62.0                                # where the tier marks start
    Ht = 12 + tree_h + 22 + 2 * strip_h + gap + 4
    fig = fig_pt(Ht)
    y_tree = 2 * strip_h + gap + 4 + 22
    ax = draw_tree(fig, 14, y_tree, W - 18, tree_h)
    tree_labels(fig, ax)
    H.letter(fig, lx, 2, "a")
    for k, (title, letter) in enumerate((("loop control", "b"), ("best of 8", "c"))):
        y0 = 4 + (1 - k) * (strip_h + gap)
        a = tier_axes(fig, 0, y0, W, strip_h)
        ymid = strip_h * 0.62
        a.text(tx, ymid, title, ha="left", va="center", fontsize=FS["lab"], fontweight="bold", color=INK)
        if k == 0:
            draw_loop(a, cx, ymid)
        else:
            draw_bo8(a, cx, ymid)
        H.letter(fig, lx, Ht - y0 - strip_h + 2, letter)
    save(fig, name)


def v5_rulers_tiers(name="v5_rulers_tiers"):
    rul_h, cell_h = 60.0, 34.0
    lh = H.legend_height([H.item("x", OURS)], W / 72, x_pt=4, fontsize=FS["small"])
    Ht = lh + 4 + rul_h + 22 + 10 + cell_h
    fig = fig_pt(Ht)
    y_r = cell_h + 10 + 22
    ax = draw_rulers(fig, 14, y_r, W - 18, rul_h)
    ruler_keys(fig, 1.0)
    H.letter(fig, 2, lh + 4, "a")
    half = (W - 14 - 8) / 2
    for k, (title, letter) in enumerate((("loop control", "b"), ("best of 8", "c"))):
        x0 = 14 + k * (half + 8)
        a = tier_axes(fig, x0, 2, half, cell_h)
        a.text(0, cell_h - 1, title, ha="left", va="top", fontsize=FS["lab"], fontweight="bold", color=INK)
        ymid = cell_h * 0.48
        if k == 0:
            draw_loop(a, 1, ymid, sq=2.6, gap=0.5, block_gap=1.6)
        else:
            draw_bo8(a, 1, ymid, tile=8.5, gap=1.6)
        H.letter(fig, 2 if k == 0 else x0 - 10, Ht - 2 - cell_h + 1, letter)
    save(fig, name)

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for k, fn in (("1", v1_tree), ("2", v2_abstract), ("3", v3_rulers), ("4", v4_tree_tiers), ("5", v5_rulers_tiers)):
        if k in which:
            try:
                fn()
            except AssertionError as e:
                print("CHECK FAILED", e)
