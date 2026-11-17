"""Panels for the merged design-choice row. Each candidate draws itself into a box given in points, so the same
function makes the stand-alone candidate (box = whole figure) and its cell in the assembled row.

Row geometry (double column 6.875 in = 495 pt): three boxes of 165 pt; every axes shares the same top and
bottom edge across the row (nature-figure alignment gate), letters a / b / c sit at the box's left edge, level
with the axes top (guide SS5.2).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import ddata as dd  # noqa: E402

H.apply()
SZ, C, M = H.SZ, H.C, H.METHOD
INK, MUTED = H.INK, H.MUTED

BOX_W, BOX_H = 165.0, 150.0          # one cell of the row, points
AX_BOTTOM, AX_TOP = 31.0, 124.0      # shared by every cell (alignment); 26 pt legend strip on top
LEG_FS = 7.6
HEAD_CHAIN = "#93A9C6"               # our block-32 head decoded as a chain (a lighter ours, not a method colour)


AX_LEFT, AX_RIGHT = 43.0, 4.0        # same in every cell, so the three axes have one width


def _ax(fig, box, left=AX_LEFT, right=AX_RIGHT, bottom=AX_BOTTOM, top=AX_TOP):
    x0, y0, w, h = box
    return H.axes_pt(fig, x0 + left, y0 + bottom, w - left - right, top - bottom)


def _legend(fig, box, items, left=6.0, gap=7.0, fs=None, width=None):
    """House legend in the strip above the axes, laid out in points from the box's top-left corner."""
    x0, y0, w, h = box
    top = fig.get_figheight() * 72 - (y0 + h) + 1.0
    return H.legend(fig, items, x_pt=x0 + left, top_pt=top, width_pt=width or (w - left - 2), fontsize=fs or LEG_FS,
                    gap_pt=gap, row_pt=(fs or LEG_FS) * 1.38)


def _ref(artist):
    artist.set_gid("ref")
    return artist


def finish(fig):
    """Call before saving: every multi-line string at line spacing >= 1.12."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if "\n" in t.get_text() and getattr(t, "_linespacing", 1.2) < 1.12:
            t.set_linespacing(1.12)


def _times(v, _pos=None):
    return f"×{v:g}"


# =========================================================================================================== D1
YL_GAIN = "tokens per round\n× DFlash's recipe"
TEAL_TXT = "#4F8C8C"     # DFlash's light teal is too pale for text; same hue, darker


def _gain_axis(ax, lo, hi, ticks):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(_times)


def d1_interaction(fig, box):
    """Interaction plot, one pair of lines per block, the three blocks side by side on one gain axis.
    Parallel lines would mean the two changes are independent. The dashed line is that 'independent' case
    (the tree's start, climbing by the chain's own block gain); the tree lands above it -- the light wedge."""
    _legend(fig, box, [H.item("chain", M["DFlash"], kind="line", lw=1.4),
                       H.item("top-16 tree", C["ours"], kind="line", lw=1.7),
                       H.item("if independent", C["ours"], kind="line", lw=0.8, ls=(0, (2.0, 1.5))),
                       H.item("DFlash", M["DFlash"], kind="marker", ms=4.4),
                       H.item("DDTree", M["DDTree"], kind="marker", ms=4.4),
                       H.item("Ours", C["ours"], kind="marker", ms=4.8)], gap=6.0)
    ax = _ax(fig, box)
    _gain_axis(ax, 0.94, 1.95, [1.0, 1.2, 1.4, 1.6, 1.8])
    xs, ys = [], []
    for i, b in enumerate(dd.BLOCKS):
        g = dd.G1[b]
        x0, x1 = 3 * i, 3 * i + 1.55
        wedge = Polygon([[x0, g["16-tree"]], [x1, g["product"]], [x1, g["32-tree"]]], closed=True,
                        facecolor=C["ours_light"], edgecolor="none", zorder=1)
        wedge.set_gid("bg")
        ax.add_patch(wedge)
        _ref(ax.plot([x0, x1], [g["16-tree"], g["product"]], color=C["ours"], lw=0.8, ls=(0, (2.0, 1.5)),
                     zorder=2)[0])
        ax.plot([x0, x1], [1.0, g["32-chain"]], color=M["DFlash"], lw=1.4, zorder=3, solid_capstyle="round")
        ax.plot([x0, x1], [g["16-tree"], g["32-tree"]], color=C["ours"], lw=1.7, zorder=4, solid_capstyle="round")
        for x, y, col, ms in ((x0, 1.0, M["DFlash"], 4.4), (x1, g["32-chain"], HEAD_CHAIN, 4.0),
                              (x0, g["16-tree"], M["DDTree"], 4.6), (x1, g["32-tree"], C["ours"], 5.2)):
            ax.plot([x], [y], "o", ms=ms, mfc=col, mec="white", mew=0.5, zorder=5)
            xs.append(x), ys.append(y)
        ax.text(x1, g["32-tree"] * 1.03, f"×{g['32-tree']:.2f}", ha="center", va="bottom",
                fontsize=SZ["legend"] - 0.5, color=C["ours"], fontweight="bold")
    ax.set_xlim(-0.6, 8.25)
    pos = [3 * i + d for i in range(3) for d in (0, 1.55)]
    H.fs.two_row_xaxis(ax, pos, ["16", "32"] * 3,
                       [("OmniSVG\ntext", [0, 1.55]), ("InternSVG\ntext", [3, 4.55]),
                        ("OmniSVG\nimage", [6, 7.55])], line_down_pt=11.0, name_down_pt=12.0, pad=0.3,
                       fs_group=SZ["tick"] - 0.6)
    for t in ax.texts:
        if "\n" in t.get_text():
            t.set_linespacing(0.92)
    ax.set_ylabel(YL_GAIN, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    H.fs.assert_inside(ax, xs, ys)
    return [ax]


def d1_mutual(fig, box):
    """Slope chart of the interaction itself: each change is worth more once the other is in.
    Left pair: what the top-16 tree adds, at block 16 vs at block 32. Right pair: what the longer block adds,
    to a chain vs to a tree. Every line climbs, in all three blocks."""
    _legend(fig, box, [H.item("one model × task", "#8A8A8A", kind="line", lw=1.2, marker="o", with_marker=True,
                              ms=3.6),
                       H.item("with the other change in", C["ours"], kind="marker", ms=4.6)], gap=8.0)
    ax = _ax(fig, box)
    _gain_axis(ax, 0.97, 1.75, [1.0, 1.2, 1.4, 1.6])
    pairs = [((0, 1), ("16-tree", "tree_on_long"), M["DDTree"]), ((2.5, 3.5), ("32-chain", "block_on_tree"),
                                                                    HEAD_CHAIN)]
    xs, ys = [], []
    for (xa, xb), (ka, kb), col in pairs:
        for b in dd.BLOCKS:
            ya, yb = dd.G1[b][ka], dd.G1[b][kb]
            ax.plot([xa, xb], [ya, yb], color=col, lw=1.3, zorder=3, solid_capstyle="round")
            ax.plot([xa], [ya], "o", ms=4.0, mfc=col, mec="white", mew=0.5, zorder=4)
            ax.plot([xb], [yb], "o", ms=4.8, mfc=C["ours"], mec="white", mew=0.5, zorder=5)
            xs += [xa, xb]
            ys += [ya, yb]
        low_a = min(dd.G1[b][ka] for b in dd.BLOCKS)
        top_b = max(dd.G1[b][kb] for b in dd.BLOCKS)
        ax.text(xa, low_a / 1.03, "×" + dd.rng(ka), ha="center", va="top", fontsize=SZ["legend"] - 1.5, color=INK)
        ax.text(xb, top_b * 1.025, "×" + dd.rng(kb), ha="center", va="bottom", fontsize=SZ["legend"] - 1.5,
                color=C["ours"], fontweight="bold")
    ax.set_xlim(-0.75, 4.25)
    H.fs.two_row_xaxis(ax, [0, 1, 2.5, 3.5], ["at 16", "at 32", "chain", "tree"],
                       [("the tree's gain", [0, 1]), ("block 32's gain", [2.5, 3.5])],
                       line_down_pt=11.0, name_down_pt=12.0, pad=0.42, fs_group=SZ["tick"] - 0.4)
    for t in ax.get_xticklabels():
        t.set_fontsize(SZ["tick"] - 0.6)
    ax.set_ylabel("gain in tokens\nper round", fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    H.fs.assert_inside(ax, xs, ys)
    return [ax]


def d1_plane(fig, box):
    """Each block's head is one point: x = what it accepts drafted as a chain, y = as a top-16 tree. Doubling the
    block moves it up and right, across the grey 'tree adds xr' guides -- the tree's edge over the chain grows
    with the longer block."""
    _legend(fig, box, [H.item("block 16", M["DDTree"], kind="marker", ms=4.6),
                       H.item("block 32 (Ours)", C["ours"], kind="marker", ms=5.0, bold=True,
                              text_color=C["ours"]),
                       H.item("tree = 1.4, 1.5, 1.6 × chain", "#BDBDBD", kind="line", lw=0.8,
                              ls=(0, (2.5, 1.8)))], gap=7.0)
    ax = _ax(fig, box)
    xs = [dd.tau1(b, a) for b in dd.BLOCKS for a in ("16-chain", "32-chain")]
    ys = [dd.tau1(b, a) for b in dd.BLOCKS for a in ("16-tree", "32-tree")]
    X0, X1, Y0, Y1 = 5.55, 8.95, 8.25, 13.0
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    for r in (1.4, 1.5, 1.6):
        xx = np.array([5.0, 9.5])
        _ref(ax.plot(xx, r * xx, color="#CFCFCF", lw=0.7, ls=(0, (2.5, 1.8)), zorder=1)[0])
    for r, xt in ((1.6, 6.05), (1.5, 7.05), (1.4, 7.6)):
        ax.text(xt, r * xt, f"×{r:g}", fontsize=SZ["tick"] - 0.8, color="#8F8F8F", ha="center", va="center",
                rotation=0, bbox=dict(facecolor="white", edgecolor="none", pad=0.4), zorder=2)
    for b in dd.BLOCKS:
        p0 = (dd.tau1(b, "16-chain"), dd.tau1(b, "16-tree"))
        p1 = (dd.tau1(b, "32-chain"), dd.tau1(b, "32-tree"))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>,head_length=4.0,head_width=2.3", mutation_scale=1,
                                     color=C["ours"], lw=1.3, shrinkA=2.6, shrinkB=3.2, zorder=4))
        ax.plot(*p0, "o", ms=4.8, mfc=M["DDTree"], mec="white", mew=0.5, zorder=5)
        ax.plot(*p1, "o", ms=5.4, mfc=C["ours"], mec="white", mew=0.5, zorder=5)
    lab = {"OmniSVG-8B, text-to-SVG": ("32", -0.13, 0.0, "right", "OmniSVG\ntext"),
           "InternSVG-8B, text-to-SVG": ("32", -0.11, 0.10, "right", "InternSVG\ntext"),
           "OmniSVG-8B, image-to-SVG": ("16", 0.11, -0.10, "left", "OmniSVG\nimage")}
    for b, (blk, dx, dy, ha, txt) in lab.items():
        x, y = dd.tau1(b, f"{blk}-chain"), dd.tau1(b, f"{blk}-tree")
        ax.text(x + dx, y + dy, txt, fontsize=SZ["legend"] - 1.3, color=INK, ha=ha, va="center", linespacing=0.92,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3), zorder=3)
    ax.set_xlabel("tokens per round, as a chain", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("as a top-16 tree", fontsize=SZ["label"], labelpad=2)
    H.fs.assert_inside(ax, xs, ys)
    return [ax]


def d1_waterfall(fig, box):
    """The round-1 waterfall folded into one panel: bars are the median block, white dots the three blocks, on
    the gain scale so the blocks share one axis. Steps: + tree, + longer block, + what the two add together."""
    _legend(fig, box, [H.item("median of the three model × task blocks", "#BDBDBD"),
                       H.item("each block", INK, kind="marker", hollow=True, ms=3.4, mew=0.7)], gap=7.0)
    ax = _ax(fig, box)
    lv = {b: [1.0, dd.G1[b]["16-tree"], dd.G1[b]["product"], dd.G1[b]["32-tree"]] for b in dd.BLOCKS}
    med = np.median(np.array([lv[b] for b in dd.BLOCKS]), axis=0)
    cols = [M["DDTree"], HEAD_CHAIN, C["ours_mid"]]
    ax.bar(0, med[0], width=0.64, color="#CFCFCF", lw=0, zorder=2)
    for i in range(1, 4):
        ax.bar(i, med[i] - med[i - 1], bottom=med[i - 1], width=0.64, color=cols[i - 1], lw=0, zorder=2)
        _ref(ax.plot([i - 1 + 0.32, i - 0.32], [med[i - 1]] * 2, color="#BDBDBD", lw=0.6, zorder=1)[0])
    ax.bar(4, med[3], width=0.64, color=C["ours"], lw=0, zorder=2)
    _ref(ax.plot([3 + 0.32, 4 - 0.32], [med[3]] * 2, color="#BDBDBD", lw=0.6, zorder=1)[0])
    for b in dd.BLOCKS:
        for i, y in enumerate(lv[b][1:], start=1):
            ax.plot([i], [y], "o", ms=2.8, mfc="white", mec=INK, mew=0.55, zorder=4)
    incs = [med[1] / med[0], med[2] / med[1], med[3] / med[2]]
    for i, f in enumerate(incs, start=1):
        ax.text(i, max(lv[b][i] for b in dd.BLOCKS) + 0.05, f"×{f:.2f}", ha="center", va="bottom",
                fontsize=SZ["legend"] - 1.4, color=[TEAL_TXT, "#5F7DA6", C["ours_mid"]][i - 1])
    ax.text(4, med[3] + 0.05, f"×{med[3]:.2f}", ha="center", va="bottom", fontsize=SZ["legend"], color=C["ours"],
            fontweight="bold")
    ax.set_ylim(0, 2.05)
    ax.set_xlim(-0.55, 4.55)
    ax.set_xticks([])
    for i, lab in enumerate(["DFlash", "+ tree", "+ block 32", "+ together", "Ours"]):
        ax.text(i, -0.03 if i % 2 == 0 else -0.135, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=SZ["tick"] - 0.6, color=C["ours"] if lab == "Ours" else INK,
                fontweight="bold" if lab == "Ours" else "normal")
    ax.yaxis.set_major_formatter(_times)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylabel(YL_GAIN, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    return [ax]


# =========================================================================================================== D2
ALL_C = "#7D7D7D"        # all tokens, as table 1 counts them (a neutral; the original model's grey is darker)
X2 = {16: 0.0, 32: 1.0, 64: 2.0}


def _pct(a, b):
    return 100.0 * (a / b - 1.0)


def _sg(v, nd=0):
    s = f"{v:+.{nd}f}%"
    return s.replace("-", "\u2212")


YL_TAU = "tokens accepted\nper round"


def d2_gap(fig, box):
    """Two lines over the block length: all tokens (table 1's count) and loop-free content. The amber gap between
    them is what repetition loops add. From 32 to 64 the all-token line still climbs; the loop-free line does not."""
    _legend(fig, box, [H.item("greedy", INK, kind="line", lw=1.2, marker="o", with_marker=True, ms=3.6),
                       H.item("official sampling", INK, kind="line", lw=1.0, ls=(0, (3, 1.6)), marker="o",
                              with_marker=True, hollow=True, ms=3.6),
                       H.item("added inside loops", C["loop_light"], kind="band")], gap=6.5)
    ax = _ax(fig, box)
    xs = [X2[L] for L in dd.LENGTHS]
    g_all = [dd.tau2(L, "greedy", "all") for L in dd.LENGTHS]
    g_h = [dd.tau2(L, "greedy", "healthy") for L in dd.LENGTHS]
    s_all = [dd.tau2(L, "sample", "all") for L in dd.LENGTHS]
    s_h = [dd.tau2(L, "sample", "healthy") for L in dd.LENGTHS]
    band = ax.fill_between(xs, g_h, g_all, color=C["loop_light"], lw=0, zorder=1)
    band.set_gid("bg")
    for ys, col, ls, hollow, lw in ((s_all, ALL_C, (0, (3, 1.6)), True, 1.0), (s_h, C["ours"], (0, (3, 1.6)), True, 1.2),
                                    (g_all, ALL_C, "-", False, 1.3), (g_h, C["ours"], "-", False, 1.8)):
        ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=3 if hollow else 4, solid_capstyle="round")
        ax.plot(xs, ys, "o", ms=4.0 if hollow else 4.6, mfc="white" if hollow else col, mec=col,
                mew=0.9 if hollow else 0.5, zorder=5)
    fz = SZ["legend"] - 0.9
    ax.text(2.13, g_all[2], f"all tokens\n{_sg(_pct(g_all[2], g_all[1]))}", color=ALL_C, fontsize=fz, ha="left",
            va="center", linespacing=0.95)
    ax.text(2.13, g_h[2] + 0.45, f"loop-free\n{_sg(_pct(g_h[2], g_h[1]))}", color=C["ours"], fontsize=fz,
            ha="left", va="center", fontweight="bold", linespacing=0.95)
    ax.text(2.13, s_all[2] + 0.1, _sg(_pct(s_all[2], s_all[1])), color=ALL_C, fontsize=fz - 0.3, ha="left",
            va="center")
    ax.text(2.13, s_h[2] - 0.5, _sg(_pct(s_h[2], s_h[1])), color=C["ours"], fontsize=fz - 0.3, ha="left",
            va="center")
    ax.set_xlim(-0.2, 2.82)
    ax.set_ylim(8.6, 17.9)
    ax.set_xticks(xs, ["16", "32", "64"])
    for t in ax.get_xticklabels():
        if t.get_text() == "32":
            t.set_color(C["ours"]), t.set_fontweight("bold")
    ax.set_xlabel("draft block length", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel(YL_TAU, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    H.fs.assert_inside(ax, xs * 4, g_all + g_h + s_all + s_h)
    return [ax]


def d2_pivot(fig, box):
    """Everything relative to block 32 (ours). Halving the block costs everywhere; doubling it pays only inside
    repetition loops -- loop-free content stays put (greedy solid, official sampling dashed)."""
    _legend(fig, box, [H.item("greedy", INK, kind="line", lw=1.2, marker="o", with_marker=True, ms=3.6),
                       H.item("official sampling", INK, kind="line", lw=1.0, ls=(0, (3, 1.6)), marker="o",
                              with_marker=True, hollow=True, ms=3.6)], gap=8.0)
    ax = _ax(fig, box)
    xs = [X2[L] for L in dd.LENGTHS]
    H.ref_line(ax, y=0)
    ax.axvline(1.0, color="#E4E4E4", lw=4.5, zorder=0).set_gid("bg")
    series = [("loop", C["loop"], "inside loops"), ("all", ALL_C, "all tokens"), ("healthy", C["ours"], "loop-free")]
    allv = []
    for kind, col, _ in series:
        for mode, ls, hollow in (("sample", (0, (3, 1.6)), True), ("greedy", "-", False)):
            ys = [_pct(dd.tau2(L, mode, kind), dd.tau2(32, mode, kind)) for L in dd.LENGTHS]
            allv += ys
            ax.plot(xs, ys, color=col, lw=1.7 if kind == "healthy" and not hollow else 1.2, ls=ls,
                    zorder=4 if kind == "healthy" else 3, solid_capstyle="round")
            ax.plot(xs, ys, "o", ms=3.8 if hollow else 4.4, mfc="white" if hollow else col, mec=col,
                    mew=0.9 if hollow else 0.5, zorder=5)
    fz = SZ["legend"] - 1.0
    for kind, col, name in series:
        g = _pct(dd.tau2(64, "greedy", kind), dd.tau2(32, "greedy", kind))
        s = _pct(dd.tau2(64, "sample", kind), dd.tau2(32, "sample", kind))
        y = g + (1.5 if kind == "all" else (-3.5 if kind == "healthy" else 0))
        ax.text(2.1, y, _sg(g), color=col, fontsize=fz, ha="left", va="center",
                fontweight="bold" if kind == "healthy" else "normal")
    for k, (kind, col, name) in enumerate(series):
        ax.text(0.02, 0.97 - 0.105 * k, name, transform=ax.transAxes, color=col, fontsize=fz + 0.3, ha="left",
                va="top", fontweight="bold" if kind == "healthy" else "normal")
    ax.set_xlim(-0.2, 2.55)
    ax.set_ylim(-46, 36)
    ax.set_yticks([-40, -20, 0, 20])
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:+.0f}".replace("-", "\u2212") if v else "0")
    ax.set_xticks(xs, ["16", "32", "64"])
    for t in ax.get_xticklabels():
        if t.get_text() == "32":
            t.set_color(C["ours"]), t.set_fontweight("bold")
    ax.set_xlabel("draft block length", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens per round,\nvs block 32 (%)", fontsize=SZ["label"], labelpad=1, linespacing=1.0)
    H.fs.assert_inside(ax, xs, allv)
    return [ax]


def d2_doublings(fig, box):
    """Dot plot of the two doublings. 16 -> 32 helps everywhere; 32 -> 64 helps only inside loops."""
    _legend(fig, box, [H.item("16 → 32", "#A3A3A3", kind="marker", ms=4.6),
                       H.item("32 → 64", C["ours"], kind="marker", ms=4.6),
                       H.item("greedy", INK, kind="marker", ms=4.0),
                       H.item("official sampling", INK, kind="marker", hollow=True, ms=4.0, mew=0.9)], gap=6.0)
    ax = _ax(fig, box)
    H.ref_line(ax, y=0)
    cats = [("healthy", "loop-free"), ("all", "all tokens"), ("loop", "inside\nloops")]
    allv = []
    for i, (kind, _) in enumerate(cats):
        for j, (a, b, col) in enumerate(((16, 32, "#A3A3A3"), (32, 64, C["ours"]))):
            for k, (mode, hollow) in enumerate((("greedy", False), ("sample", True))):
                v = _pct(dd.tau2(b, mode, kind), dd.tau2(a, mode, kind))
                allv.append(v)
                x = i + (-0.17 if j == 0 else 0.17) + (-0.06 if k == 0 else 0.06)
                ax.plot([x], [v], "o", ms=5.0 if not hollow else 4.6, mfc="white" if hollow else col, mec=col,
                        mew=1.0 if hollow else 0.5, zorder=4)
        g = _pct(dd.tau2(64, "greedy", kind), dd.tau2(32, "greedy", kind))
        if kind == "healthy":
            ax.text(i + 0.36, g - 2, "≈ 0", fontsize=SZ["legend"] - 0.6, color=C["ours"], fontweight="bold",
                    ha="left", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-12, 80)
    ax.set_yticks([0, 25, 50, 75])
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:+.0f}" if v else "0")
    ax.set_xticks(range(3), [c[1] for c in cats])
    for t in ax.get_xticklabels():
        t.set_linespacing(0.92)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("tokens per round,\nper doubling (%)", fontsize=SZ["label"], labelpad=1, linespacing=1.0)
    H.fs.assert_inside(ax, [0, 2], allv)
    return [ax]


def d2_stack(fig, box):
    """Round 2's merged stack (b2_merged) at a third of the width: loop-free content in blue, the part only loops
    add in light amber; the dashed line is block 32's loop-free level."""
    _legend(fig, box, [H.item("loop-free content", C["ours"]), H.item("added inside loops", C["loop_light"])],
            gap=8.0)
    ax = _ax(fig, box)
    pos = {("greedy", 16): 0, ("greedy", 32): 1, ("greedy", 64): 2, ("sample", 16): 3.5, ("sample", 32): 4.5,
           ("sample", 64): 5.5}
    for (mode, L), x in pos.items():
        h, a = dd.tau2(L, mode, "healthy"), dd.tau2(L, mode, "all")
        ax.bar(x, h, width=0.72, color=C["ours"] if L == 32 else "#5F84B3", lw=0, zorder=2)
        ax.bar(x, a - h, bottom=h, width=0.72, color=C["loop_light"], lw=0, zorder=2)
        ax.text(x, a + 0.25, f"{a:.1f}", ha="center", va="bottom", fontsize=SZ["tick"] - 1.0, color=INK)
        t = ax.text(x, h - 0.5, f"{h:.1f}", ha="center", va="top", fontsize=SZ["tick"] - 1.0, color="white",
                    rotation=90)
        t.set_gid("free")
    for mode, x0 in (("greedy", 1), ("sample", 4.5)):
        lvl = dd.tau2(32, mode, "healthy")
        _ref(ax.plot([x0 - 0.36, x0 + 1.36], [lvl, lvl], color="#6F6F6F", lw=0.8, ls=(0, (2.4, 1.6)), zorder=3)[0])
    ax.set_xlim(-0.6, 6.1)
    ax.set_ylim(0, 19.5)
    H.fs.two_row_xaxis(ax, list(pos.values()), ["16", "32", "64"] * 2,
                       [("greedy", [0, 1, 2]), ("official sampling", [3.5, 4.5, 5.5])], line_down_pt=11.0,
                       name_down_pt=12.0, pad=0.36, fs_group=SZ["tick"] - 0.4)
    for t in ax.get_xticklabels():
        if t.get_text() == "32":
            t.set_color(C["ours"]), t.set_fontweight("bold")
    ax.set_ylabel(YL_TAU, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    return [ax]


# =========================================================================================================== D3
def _cum(S, A):
    n, a = S.sum(0), A.sum(0)
    return np.concatenate([[0], np.cumsum(n)]), np.concatenate([[0], np.cumsum(a)])


def d3_return(fig, box):
    """Return on the node budget. Walk down the draft block from the first position: x = tree nodes spent so far,
    y = draft tokens accepted so far (per round). Both trees spend the same 127 nodes. Up to position 15 ours has
    spent 83 of them and accepts about what DDTree does with all 127; its last 44 nodes sit at positions,
    out of a 16-token block's reach, and buy 5.4 more tokens per round."""
    _legend(fig, box, [H.item("DDTree, block 16", M["DDTree"], kind="line", lw=1.6),
                       H.item("Ours, block 32", C["ours"], kind="line", lw=1.9, bold=True, text_color=C["ours"]),
                       H.item("every 5th position", INK, kind="marker", ms=3.4)], gap=6.0)
    ax = _ax(fig, box)
    xo, yo = _cum(dd.S_OURS, dd.A_OURS)
    xd, yd = _cum(dd.S_DDT, dd.A_DDT)
    band = ax.fill_between(xo[15:], 0, yo[15:], color=C["ours_light"], lw=0, zorder=1, alpha=0.75)
    band.set_gid("bg")
    ax.plot(xd, yd, color=M["DDTree"], lw=1.6, zorder=3, solid_capstyle="round")
    ax.plot(xo, yo, color=C["ours"], lw=1.9, zorder=4, solid_capstyle="round")
    for p in (5, 10, 15):
        ax.plot([xd[p]], [yd[p]], "o", ms=3.4, mfc=M["DDTree"], mec="white", mew=0.5, zorder=5)
    for p in (5, 10, 15, 20, 25, 31):
        ax.plot([xo[p]], [yo[p]], "o", ms=3.6, mfc=C["ours"], mec="white", mew=0.5, zorder=6)
    fz = SZ["legend"] - 1.0
    tgt = yd[15]
    i = int(np.searchsorted(yo, tgt))
    xc = xo[i - 1] + (tgt - yo[i - 1]) / (yo[i] - yo[i - 1]) * (xo[i] - xo[i - 1])
    _ref(ax.plot([xc, xd[15]], [tgt, tgt], color=M["DDTree"], lw=0.8, ls=(0, (1.2, 1.4)), zorder=2)[0])
    _ref(ax.plot([xc, xc], [0, tgt], color=C["ours"], lw=0.7, ls=(0, (1.2, 1.4)), zorder=2)[0])
    ax.text(xc - 2.5, 0.5, f"{xc:.0f} nodes", ha="right", va="bottom", fontsize=fz, color=C["ours"])
    ax.text(xo[31] - 1.5, yo[31] + 0.35, f"{yo[31]:.1f}", ha="right", va="bottom", fontsize=SZ["legend"] - 0.4,
            color=C["ours"], fontweight="bold")
    ax.text(xd[15] + 2.0, yd[15] - 0.15, f"{yd[15]:.1f}", ha="left", va="center", fontsize=SZ["legend"] - 0.4,
            color=M["DDTree"], fontweight="bold")
    ax.text((xo[15] + xo[31]) / 2 + 4, 2.6, "positions\n16–31", ha="center", va="bottom", fontsize=fz,
            color=C["ours"], linespacing=0.95)
    assert abs(xc - 86.9) < 0.2, xc
    ax.set_xlim(0, 142)
    ax.set_ylim(0, 16.2)
    ax.set_xticks([0, 40, 80, 127])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("tree nodes, cumulative", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel(YL_TAU, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    H.fs.assert_inside(ax, list(xo) + list(xd), list(yo) + list(yd))
    return [ax]


def d3_yield(fig, box):
    """What one node buys at each position of the draft block: accepted draft tokens per tree node. Past the first
    few positions our nodes buy 2-3x more, and our tree keeps buying after DDTree's 16-token block has ended."""
    _legend(fig, box, [H.item("DDTree, block 16", M["DDTree"], kind="line", lw=1.5),
                       H.item("Ours, block 32", C["ours"], kind="line", lw=1.8, bold=True, text_color=C["ours"]),
                       H.item("beyond a 16-token block", C["ours_light"], kind="band")], gap=6.0)
    ax = _ax(fig, box)
    yo = dd.A_OURS.sum(0) / dd.S_OURS.sum(0)
    yd = dd.A_DDT.sum(0) / dd.S_DDT.sum(0)
    ax.axvspan(15.5, 31.5, color=C["ours_light"], lw=0, zorder=0, alpha=0.6).set_gid("bg")
    po, pd_ = np.arange(1, 32), np.arange(1, 16)
    ax.plot(pd_, yd, color=M["DDTree"], lw=1.5, zorder=3, solid_capstyle="round")
    ax.plot(pd_[-1:], yd[-1:], "o", ms=3.8, mfc=M["DDTree"], mec="white", mew=0.5, zorder=5)
    ax.plot(po, yo, color=C["ours"], lw=1.8, zorder=4, solid_capstyle="round")
    fz = SZ["legend"] - 1.0
    ax.text(23.5, 0.215, f"{100 * dd.BEYOND16:.0f}% of our\naccepted tokens", ha="center", va="bottom", fontsize=fz,
            color=C["ours"], linespacing=0.95)
    ax.text(15.9, yd[-1] + 0.004, "block 16\nends", ha="left", va="center", fontsize=fz, color=M["DDTree"],
            linespacing=0.95)
    ax.set_xlim(0.5, 31.5)
    ax.set_ylim(0, 0.37)
    ax.set_xticks([1, 8, 16, 24, 31])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel("position in the draft block", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens accepted\nper tree node", fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    H.fs.assert_inside(ax, po, list(yo) + list(yd))
    return [ax]


def d3_heat(fig, box):
    """Where the 127 nodes sit, by position in the block (x) and the drafter's rank among siblings (rows); dot area
    is what they accept. Both trees fan out over 2nd-9th choices in the first few positions and accept almost only
    first choices; DDTree's budget then piles up against the end of its 16-token block, ours runs on to 31."""
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    groups = [("top-1", [0]), ("2nd", [1]), ("3rd–8th", [2, 3, 4, 5, 6, 7]), ("9th+", [8])]

    def collapse(Mx):
        return np.array([Mx[idx].sum(0) for _, idx in groups])

    cmap = LinearSegmentedColormap.from_list("nodes", ["#FFFFFF", "#D9D9D9", "#8C8C8C", "#3F3F3F"])
    norm = PowerNorm(0.6, vmin=0, vmax=10)
    x0, y0, w, h = box
    left, right = AX_LEFT, AX_RIGHT
    gap = 9.0
    hh = (AX_TOP - AX_BOTTOM - gap) / 2
    axes = []
    for k, (name, S, A, col, npos) in enumerate((("Ours", dd.S_OURS, dd.A_OURS, C["ours"], 31),
                                                ("DDTree", dd.S_DDT, dd.A_DDT, M["DDTree"], 15))):
        bot = AX_BOTTOM + (hh + gap if k == 0 else 0)
        ax = H.axes_pt(fig, x0 + left, y0 + bot, w - left - right, hh)
        Sg, Ag = collapse(S), collapse(A)
        ax.imshow(Sg, aspect="auto", cmap=cmap, norm=norm, extent=(0.5, npos + 0.5, len(groups) - 0.5, -0.5),
                  interpolation="nearest", zorder=1)
        r, p = np.nonzero(Ag > 0.02)
        sizes = 26 * Ag[r, p] / 1.0
        ax.scatter(p + 1, r, s=sizes, c=col, edgecolors="white", linewidths=0.3, zorder=3)
        ax.set_xlim(0.5, 31.5)
        ax.set_ylim(len(groups) - 0.5, -0.5)
        ax.set_yticks(range(len(groups)), [g for g, _ in groups], fontsize=7.0)
        ax.tick_params(axis="y", length=0, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(31.3, len(groups) - 0.62, name, ha="right", va="bottom", fontsize=SZ["legend"] - 0.8, color=col,
                fontweight="bold")
        if k == 0:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xticks([1, 8, 16, 24, 31])
            ax.set_xlabel("position in the draft block", fontsize=SZ["label"], labelpad=1.5)
        axes.append(ax)
    fig.text((x0 + 3) / (fig.get_figwidth() * 72), (y0 + (AX_TOP + AX_BOTTOM) / 2) / (fig.get_figheight() * 72),
             "sibling rank", rotation=90, ha="left", va="center", fontsize=SZ["label"], color=INK)
    # legend strip: node shade and dot area
    _legend(fig, box, [H.item("nodes per round (shade)", "#8C8C8C"),
                       H.item("tokens accepted (dot area)", INK, kind="marker", ms=4.0)], gap=6.0)
    return axes
