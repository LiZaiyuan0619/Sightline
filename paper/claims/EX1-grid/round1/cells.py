"""Candidate cells for the first experiments composite (main.pdf Figures 5, 7, 8, 9, 10), round 1.

User, before compositing, look at each figure's own looks and change its form where it is ugly (Figures 8
and 9 named); then a 2 x 3 composite with Figure 5 as one column of two stacked panels. Every candidate here draws
itself into a box given in points (the TH6 / D-group pattern), at the size it will print inside the composite:
a right-hand cell is CELL_W x CELL_H, Figure 5's column TALL_W x TALL_H. Data: data.py (signed-off loaders, asserted).

  Figure 8 (structure is easy, numbers are hard)
    c8a_share_bars   numbers: share of the output vs share of the draft's misses, per block (round 2's m1g, restyled)
    c8b_arrows       the same two shares as one arrow per block (output share -> miss share), blocks as rows
    c8c_classes      every token class of every block: share of the output (x) vs share of the misses (y), diagonal
    c8d_wrong_bars   how often the draft is wrong at structure vs numbers, per block (main.pdf's m1e, restyled)
  Figure 9 (published decoders stop at their own window)
    c9a_dots         share of rounds that fill the window, one dot per block and decoding, methods on x
    c9b_ridges       tokens per round, pooled over the five blocks (greedy), one ridge per method
    c9c_bars         the mean over the ten settings as a bar, the ten settings as dots
    c9d_diagonal     each learned baseline (x) against ours (y) in the same setting; the diagonal = equal
  Figure 10 (loops are common)
    c10a_stacked     main.pdf's stacked bars (finished / length cap / inside a loop), restyled to one cell
    c10b_dots        outputs that end inside a loop, greedy (filled) and official sampling (hollow), targets as rows
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import data as D  # noqa: E402

H.apply()
TICK, LAB, NOTE, KEY = 7.5, 8.0, 7.2, 7.4      # a cell one third of the page wide (TH6); floor 7 pt (guide 4.1)
plt.rcParams.update({"xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.labelsize": LAB})
C, M, INK, MUTED = H.C, H.METHOD, H.INK, H.MUTED
STRUCT_C, NUM_C, NUM_LIGHT, COLOUR_C = "#8E9EB0", C["loop"], C["loop_light"], "#B48DAA"

CELL_W, CELL_H = 155.0, 150.0
LEG_DX = 0.0          # make room for the panel letter at the top left of a cell (set by the composite)
TALL_W, TALL_H = 185.0, 300.0

# block order with a family-grouped axis: OmniSVG (8B text, 4B text, 8B image) | InternSVG (8B text, 8B image)
ORDER = [(D.BLOCKS[0], "OmniSVG", "8B", "text"), (D.BLOCKS[1], "OmniSVG", "4B", "text"),
         (D.BLOCKS[3], "OmniSVG", "8B", "image"), (D.BLOCKS[2], "InternSVG", "8B", "text"),
         (D.BLOCKS[4], "InternSVG", "8B", "image")]
ROWNAME = {b: f"{fam}-{size}\n{task}" for b, fam, size, task in ORDER}


def top_of(fig, box):
    """Distance in points from the figure's top edge to the top of the box."""
    return fig.get_figheight() * 72 - (box[1] + box[3])


def ax_in(fig, box, l, b, r, t):
    x0, y0, w, h = box
    return H.axes_pt(fig, x0 + l, y0 + b, w - l - r, h - b - t)


def free(*artists):
    for a in artists:
        a.set_gid("free")
    return artists[0] if len(artists) == 1 else artists


def ref(ax, y=None, x=None):
    return H.ref_line(ax, y=y, x=x, lw=0.7)


def _group_lines(ax, y_off=-24.5):
    """Draw the family group lines as real lines in offset points (annotate arrows are skipped by house.check)."""
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory
    fig = ax.figure
    tr = blended_transform_factory(ax.transData, ax.transAxes) + ScaledTranslation(0, y_off / 72, fig.dpi_scale_trans)
    for lo, hi in ((0, 2), (3, 4)):
        ln = ax.plot([lo - 0.34, hi + 0.34], [0, 0], color=H.GROUP_LINE, lw=0.6, transform=tr, clip_on=False)[0]
        ln.set_gid("ref")


def block_axis2(ax):
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{s}\n{t}" for _, _, s, t in ORDER], fontsize=7.0, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    _group_lines(ax)
    for fam, lo, hi in (("OmniSVG", 0, 2), ("InternSVG", 3, 4)):
        ax.annotate(fam, xy=((lo + hi) / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -26.5),
                    textcoords="offset points", ha="center", va="top", fontsize=7.0, annotation_clip=False)


def method_axis(ax, methods, y_rows=(-2.0, -10.5)):
    """x = methods, names staggered on two rows (guide 3.1: wrap or stagger, do not move variants to y)."""
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=2, pad=1)
    for i, m in enumerate(methods):
        ours = m == "Ours"
        ax.annotate(m, xy=(i, 0), xycoords=("data", "axes fraction"), xytext=(0, y_rows[i % 2]),
                    textcoords="offset points", ha="center", va="top", fontsize=7.0,
                    color=M["Ours"] if ours else INK, fontweight="bold" if ours else "normal", annotation_clip=False)


# ================================================================================================================
# Figure 8: structure is easy, numbers are hard (ours, greedy, loop-free positions; data.classes / numbers_share)
# ================================================================================================================
def c8a_share_bars(fig, box):
    ax = ax_in(fig, box, l=31, b=36, r=3, t=20)
    w = 0.36
    for i, (b, *_r) in enumerate(ORDER):
        out, miss = D.numbers_share(b)
        for dx, v, col in ((-w / 2 - 0.01, out, NUM_LIGHT), (w / 2 + 0.01, miss, NUM_C)):
            ax.bar(i + dx, 100 * v, width=w, color=col, edgecolor="none", zorder=3)
            free(ax.text(i + dx, 100 * v + 2, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=7.0))
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))
    ax.set_ylabel("numbers (%)", labelpad=2)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    block_axis2(ax)
    H.legend(fig, [H.item("of the output", NUM_LIGHT), H.item("of the draft's misses", NUM_C)],
             x_pt=box[0] + 4 + LEG_DX, top_pt=top_of(fig, box) + 2, width_pt=box[2] - 6 - LEG_DX, fontsize=KEY)
    return ax


def c8b_arrows(fig, box):
    ax = ax_in(fig, box, l=46, b=25, r=10, t=20)
    rows = ORDER
    n = len(rows)
    for j, (b, *_r) in enumerate(rows):
        y = n - 1 - j
        out, miss = (100 * v for v in D.numbers_share(b))
        ax.annotate("", xy=(miss - 1.2, y), xytext=(out + 1.2, y),
                    arrowprops=dict(arrowstyle="-|>", color=NUM_C, lw=1.1, mutation_scale=7, shrinkA=0, shrinkB=0),
                    zorder=3)
        ax.scatter([out], [y], s=22, facecolor="white", edgecolor=NUM_C, linewidths=1.1, zorder=4)
        ax.scatter([miss], [y], s=24, color=NUM_C, edgecolor="white", linewidths=0.4, zorder=4)
        free(ax.text(out - 2.6, y, f"{out:.0f}", ha="right", va="center", fontsize=7.0))
        free(ax.text(miss + 2.6, y, f"{miss:.0f}", ha="left", va="center", fontsize=7.0))
    ax.set_yticks(range(n))
    ax.set_yticklabels([ROWNAME[b] for b, *_ in rows][::-1], fontsize=7.0, linespacing=1.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(48, 100)
    ax.xaxis.set_major_locator(FixedLocator([50, 75, 100]))
    ax.set_xlabel("numbers, share (%)", labelpad=1.5)
    H.legend(fig, [H.item("of the output", NUM_C, kind="marker", hollow=True, ms=4.8),
                   H.item("of the draft's misses", NUM_C, kind="marker", ms=4.8)],
             x_pt=box[0] + 4 + LEG_DX, top_pt=top_of(fig, box) + 2, width_pt=box[2] - 6 - LEG_DX, fontsize=KEY)
    return ax


def c8c_classes(fig, box):
    ax = ax_in(fig, box, l=31, b=26, r=6, t=20)
    gc = {"structure": STRUCT_C, "numbers": NUM_C, "colour": COLOUR_C}
    for b, *_r in ORDER:
        for k, g, share, rej, acc in D.classes(b):
            if k == "eos" or share < 0.003:
                continue
            ax.scatter([100 * share], [100 * max(rej, 1e-4)], s=13, color=gc[g], edgecolor="white", linewidths=0.35,
                       zorder=3 if g != "numbers" else 4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = 0.3, 60
    ax.plot([lo, hi], [lo, hi], color=H.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)[0].set_gid("ref")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0.05, hi)
    ax.xaxis.set_major_locator(FixedLocator([1, 10]))
    ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_formatter(NullFormatter())
        a.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("share of the output (%)", labelpad=1.5)
    ax.set_ylabel("share of the misses (%)", labelpad=1)
    free(ax.text(0.4, 25, "missed more\nthan its share", fontsize=7.0, color=MUTED, va="top", linespacing=1.0))
    H.legend(fig, [H.item("structure", STRUCT_C, kind="marker", ms=4.6), H.item("numbers", NUM_C, kind="marker", ms=4.6),
                   H.item("colors", COLOUR_C, kind="marker", ms=4.6)],
             x_pt=box[0] + 4 + LEG_DX, top_pt=top_of(fig, box) + 2, width_pt=box[2] - 6 - LEG_DX, fontsize=KEY)
    return ax


def c8d_wrong_bars(fig, box):
    ax = ax_in(fig, box, l=31, b=36, r=3, t=20)
    w = 0.36
    for i, (b, *_r) in enumerate(ORDER):
        r = D.wrong_rate(b)
        for dx, key, col in ((-w / 2 - 0.01, "structure", STRUCT_C), (w / 2 + 0.01, "numbers", NUM_C)):
            v = 100 * r[key]
            ax.bar(i + dx, v, width=w, color=col, edgecolor="none", zorder=3)
            s = f"{v:.1f}" if v < 2 else f"{v:.0f}"
            free(ax.text(i + dx, v + 0.3, s, ha="center", va="bottom", fontsize=7.0))
    ax.set_ylim(0, 12.5)
    ax.yaxis.set_major_locator(FixedLocator([0, 5, 10]))
    ax.set_ylabel("draft wrong (%)", labelpad=2)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    block_axis2(ax)
    H.legend(fig, [H.item("structure", STRUCT_C), H.item("numbers", NUM_C)],
             x_pt=box[0] + 4 + LEG_DX, top_pt=top_of(fig, box) + 2, width_pt=box[2] - 6 - LEG_DX, fontsize=KEY)
    return ax


# ================================================================================================================
# Figure 9: published decoders stop at their own window (loop-free rounds; data.window)
# ================================================================================================================
WM = D.WIN_METHODS
LEARNED = ["EAGLE-3", "DFlash", "DARTree", "DDTree"]


def _fills(m):
    return [(i, md, 100 * D.window(b, md, m)["fill"]) for i, (b, *_r) in enumerate(ORDER) for md in D.MODES]


def _greedy_sample_legend(fig, box, color=INK):
    H.legend(fig, [H.item("greedy", color, kind="marker", ms=4.6), H.item("official sampling", color, kind="marker",
                                                                         hollow=True, ms=4.6)],
             x_pt=box[0] + 4 + LEG_DX, top_pt=top_of(fig, box) + 2, width_pt=box[2] - 6 - LEG_DX, fontsize=KEY)


def c9a_dots(fig, box):
    ax = ax_in(fig, box, l=37, b=24, r=3, t=20)
    for j, m in enumerate(WM):
        col = M[m]
        for i, md, v in _fills(m):
            dx = -0.28 + 0.14 * i
            ax.scatter([j + dx], [v], s=11 if m != "Ours" else 13, marker="o",
                       facecolor=col if md == "greedy" else "white", edgecolor=col,
                       linewidths=0.8, zorder=4 if m == "Ours" else 3)
    ax.set_xlim(-0.6, len(WM) - 0.4)
    ax.set_ylim(0, 58)
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40]))
    ax.set_ylabel("rounds that fill\nthe window (%)", labelpad=2.5, linespacing=1.2)
    method_axis(ax, WM)
    _greedy_sample_legend(fig, box)
    return ax


def c9b_ridges(fig, box):
    ax = ax_in(fig, box, l=33, b=25, r=4, t=6)
    n = len(WM)
    xs = np.arange(1, 33)
    for j, m in enumerate(WM):
        base = n - 1 - j
        h = np.zeros(32)
        for b, *_r in ORDER:
            hist = D.window(b, "greedy", m)["hist"]
            h += np.array([hist.get(int(k), 0.0) for k in xs])
        h /= len(ORDER)
        hh = 0.86 * h / h.max()
        col = M[m]
        ax.bar(xs, hh, bottom=base, width=1.0, color=col, edgecolor="none", alpha=0.95 if m == "Ours" else 0.85,
               zorder=3)
        ln = ax.plot([0.5, 32.5], [base, base], color="#D7D7D7", lw=0.5, zorder=2)[0]
        ln.set_gid("ref")
        cap = D.CAP[m]
        ax.plot([cap, cap], [base - 0.08, base], color=INK, lw=0.8, zorder=4, clip_on=False)
    ax.set_yticks(np.arange(n) + 0.3)
    ax.set_yticklabels(WM[::-1], fontsize=7.0)
    for t in ax.get_yticklabels():
        if t.get_text() == "Ours":
            t.set_color(M["Ours"])
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.12, n)
    ax.set_xlim(0.5, 32.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 16, 32]))
    ax.set_xlabel("tokens per round", labelpad=1.5)
    return ax


def c9c_bars(fig, box):
    ax = ax_in(fig, box, l=37, b=24, r=3, t=20)
    for j, m in enumerate(WM):
        vals = [v for _i, _md, v in _fills(m)]
        col = M[m]
        ax.bar(j, np.mean(vals), width=0.62, color=col, alpha=0.35 if m != "Ours" else 0.45, edgecolor="none", zorder=2)
        for i, md, v in _fills(m):
            dx = -0.22 + 0.11 * i
            ax.scatter([j + dx], [v], s=9, facecolor=col if md == "greedy" else "white", edgecolor=col,
                       linewidths=0.7, zorder=4)
    ax.set_xlim(-0.6, len(WM) - 0.4)
    ax.set_ylim(0, 58)
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40]))
    ax.set_ylabel("rounds that fill\nthe window (%)", labelpad=2.5, linespacing=1.2)
    method_axis(ax, WM)
    _greedy_sample_legend(fig, box)
    return ax


def c9d_diagonal(fig, box):
    ax = ax_in(fig, box, l=31, b=26, r=6, t=18)
    for m in LEARNED:
        col = M[m]
        for b, *_r in ORDER:
            for md in D.MODES:
                x = 100 * D.window(b, md, m)["fill"]
                y = 100 * D.window(b, md, "Ours")["fill"]
                ax.scatter([x], [y], s=13, facecolor=col if md == "greedy" else "white", edgecolor=col,
                           linewidths=0.8, zorder=3)
    ax.plot([0, 56], [0, 56], color=H.REF, lw=0.7, ls=(0, (3, 2)), zorder=1)[0].set_gid("ref")
    ax.set_xlim(0, 56)
    ax.set_ylim(0, 56)
    ax.xaxis.set_major_locator(FixedLocator([0, 25, 50]))
    ax.yaxis.set_major_locator(FixedLocator([0, 25, 50]))
    ax.set_xlabel("baseline fills its window (%)", labelpad=1.5)
    ax.set_ylabel("ours fills its window (%)", labelpad=1.5)
    free(ax.text(29, 38, "equal", fontsize=7.0, color=MUTED, rotation=45, ha="center", va="center"))
    H.legend(fig, [H.item(m, M[m], kind="marker", ms=4.2) for m in LEARNED],
             x_pt=box[0] + 2 + LEG_DX, top_pt=top_of(fig, box) + 2, width_pt=box[2] - 3 - LEG_DX, fontsize=7.0, gap_pt=4.5, pad_pt=2.0)
    return ax


# ================================================================================================================
# Figure 10: loops are common (data.loops)
# ================================================================================================================
TGT = D.LOOP_TARGETS
TGT_NAME = {t: t.split(", ")[0] + "\n" + t.split(", ")[1].replace("-to-SVG", "") for t in TGT}


def c10a_stacked(fig, box):
    ax = ax_in(fig, box, l=47, b=23, r=9, t=20)
    n = len(TGT)
    hgt = 0.34
    for j, t in enumerate(TGT):
        y0 = n - 1 - j
        for k, md in enumerate(("greedy", "sample")):
            y = y0 + (0.19 if k == 0 else -0.19)
            s = D.loops(t, md)
            left = 0.0
            for key, col in (("finished", "#C9D3DD"), ("cap", "#E9E1D2"), ("loop", NUM_C)):
                ax.barh(y, 100 * s[key], left=left, height=hgt, color=col, edgecolor="none", zorder=3)
                left += 100 * s[key]
    ax.set_yticks(range(n))
    ax.set_yticklabels([TGT_NAME[t] for t in TGT][::-1], fontsize=7.0, linespacing=1.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, n - 0.45)
    ax.xaxis.set_major_locator(FixedLocator([0, 50, 100]))
    ax.set_xlabel("outputs (%)", labelpad=1.5)
    H.legend(fig, [H.item("finished", "#C9D3DD"), H.item("length cap", "#E9E1D2"), H.item("in a loop", NUM_C)],
             x_pt=box[0] + 4 + LEG_DX, top_pt=top_of(fig, box) + 2, width_pt=box[2] - 6 - LEG_DX, fontsize=KEY)
    return ax


def c10b_dots(fig, box):
    ax = ax_in(fig, box, l=47, b=24, r=12, t=20)
    n = len(TGT)
    for j, t in enumerate(TGT):
        y = n - 1 - j
        g, s = (100 * D.loops(t, md)["loop"] for md in ("greedy", "sample"))
        ln = ax.plot([min(g, s), max(g, s)], [y, y], color=NUM_LIGHT, lw=2.2, solid_capstyle="round", zorder=2)[0]
        ln.set_gid("bg")
        ax.scatter([s], [y], s=22, facecolor="white", edgecolor=NUM_C, linewidths=1.0, zorder=4)
        ax.scatter([g], [y], s=24, color=NUM_C, edgecolor="white", linewidths=0.4, zorder=5)
        free(ax.text(max(g, s) + 2.4, y, f"{g:.0f}", ha="left", va="center", fontsize=7.0, color=INK))
    ax.set_yticks(range(n))
    ax.set_yticklabels([TGT_NAME[t] for t in TGT][::-1], fontsize=7.0, linespacing=1.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 52)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_locator(FixedLocator([0, 25, 50]))
    ax.set_xlabel("outputs ending in a loop (%)", labelpad=1.5)
    _greedy_sample_legend(fig, box, color=NUM_C)
    return ax


# ================================================================================================================
# Figure 7: the lead holds across six model families (data.T3: ext_v4.json; radius / x = baseline speed-up / ours)
# ================================================================================================================
T = D.T3
BASES = ["PLD", "EAGLE-3", "DFlash", "DARTree", "DDTree"]
SPOKES = [("greedy", 0), ("greedy", 1), ("sample", 0), ("sample", 1)]   # (decoding, subset index)


def _ratio(f, m, mode, k):
    s = T.SUBSETS[f][k]
    return T.spd(f, m, mode, s) / T.spd(f, "Ours", mode, s)


def _t3_legend(fig, box, rows_gap=4.5, with_ours=False):
    items = [H.item(m, M[m], kind="marker", ms=4.0) for m in BASES] + \
            ([H.item("Ours", C["ours"], kind="line", lw=1.5)] if with_ours else []) + \
            [H.item("our lead", C["ours_light"], kind="band")]
    return H.legend(fig, items, x_pt=box[0] + 2 + LEG_DX, top_pt=top_of(fig, box) + 1, width_pt=box[2] - 3 - LEG_DX, fontsize=7.0,
                    gap_pt=rows_gap, pad_pt=2.0)


def c7a_radar(fig, box):
    bot = _t3_legend(fig, box, with_ours=True)
    x0, y0, w, h = box
    Dd = 86.0
    top_used = bot - top_of(fig, box)                       # legend height inside the box
    cy = y0 + (h - top_used) / 2 + 1
    W_, Hf = fig.get_figwidth() * 72, fig.get_figheight() * 72
    ax = fig.add_axes([(x0 + (w - Dd) / 2) / W_, (cy - Dd / 2) / Hf, Dd / W_, Dd / Hf], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    sector, step = 60.0, 12.0
    th, meta = [], []
    for i, f in enumerate(T.FAMILIES):
        for j, (mo, k) in enumerate(SPOKES):
            th.append(np.deg2rad(i * sector + (j - 1.5) * step))
            meta.append((f, mo, k))
    th = np.array(th)
    R = {m: np.array([_ratio(f, m, mo, k) for f, mo, k in meta]) for m in BASES}
    for i in range(len(T.FAMILIES)):
        idx = np.arange(4 * i, 4 * i + 4)
        tt = np.linspace(th[idx[0]] - np.deg2rad(step / 2), th[idx[-1]] + np.deg2rad(step / 2), 80)
        base = np.interp(tt, th[idx], R["DDTree"][idx])
        ax.fill_between(tt, np.minimum(base, 1.0), 1.0, color=C["ours_light"], lw=0, zorder=1).set_gid("bg")
        ax.plot(tt, base, color=M["DDTree"], lw=0.9, zorder=3)
    for r in (0.5, 0.75):
        ax.plot(np.linspace(0, 2 * np.pi, 361), [r] * 361, color="#E4E4E4", lw=0.5, zorder=0)[0].set_gid("ref")
    ax.plot(np.linspace(0, 2 * np.pi, 361), [1.0] * 361, color=C["ours"], lw=1.5, zorder=5)
    for m in BASES:
        if m == "DDTree":
            continue
        ax.scatter(th, R[m], s=5.5, color=M[m], edgecolor="none", zorder=4)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    for i, f in enumerate(T.FAMILIES):
        ang = i * sector
        name = f.replace("StarVector-", "StarVector\n")
        ha = "center" if ang in (0, 180) else ("left" if ang < 180 else "right")
        rr = 1.12 if ang in (0, 180) else 1.15
        free(ax.text(np.deg2rad(ang), rr, name, ha=ha, va="bottom" if ang == 0 else ("top" if ang == 180 else "center"),
                     fontsize=7.0, color=INK, linespacing=1.0))
    return ax


def c7b_rows(fig, box):
    bot = _t3_legend(fig, box)
    t = bot - top_of(fig, box) + 3
    ax = ax_in(fig, box, l=49, b=24, r=5, t=t)
    fams = T.FAMILIES
    n = len(fams)
    offs = [0.27, 0.09, -0.09, -0.27]
    for i, f in enumerate(fams):
        yc = n - 1 - i
        for j, (mo, k) in enumerate(SPOKES):
            y = yc + offs[j]
            dd = _ratio(f, "DDTree", mo, k)
            if dd < 1:
                ax.add_patch(Rectangle((dd, y - 0.075), 1 - dd, 0.15, color=C["ours_light"], lw=0, zorder=1, gid="bg"))
            for m in BASES:
                ax.scatter([_ratio(f, m, mo, k)], [y], s=6.5, color=M[m], edgecolor="none",
                           zorder=4 if m == "DDTree" else 3)
        ln = ax.plot([0.08, 1.0], [yc - 0.5, yc - 0.5], color="#EDEDED", lw=0.5, zorder=0)[0]
        ln.set_gid("ref")
    ln = ax.axvline(1.0, color=C["ours"], lw=1.5, zorder=5)
    ln.set_gid("ref")
    ax.set_yticks(range(n))
    ax.set_yticklabels(fams[::-1], fontsize=7.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(0.08, 1.06)
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 0.75, 1.0]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("speed-up, share of ours", labelpad=1.5)
    free(ax.text(1.0, n - 0.42, "Ours", fontsize=7.0, color=C["ours"], fontweight="bold", ha="right", va="bottom"))
    return ax


# ================================================================================================================
# Figure 5: speed and quality of SVG generators (SArena-Icon text-to-SVG; data.OTHERS / data.OURS), one tall column
# ================================================================================================================
CAT_COL = {"optimisation": C["optim"], "LLM, reported": C["other_llm"], "LLM, measured": C["other_llm"]}
XL = (0.5, 450.0)
OURS2 = C["ours_mid"]
# label placement per panel: name -> (dx_pt, dy_pt, ha, va); measured against the print size of this cell
LAB_FID = {"VectorFusion": (-4, 0, "right", "center"), "SVGDreamer": (4, -5, "right", "top"),
           "LLM4SVG-7B": (-4, 0, "right", "center"), "OmniSVG-3B": (-4, 0, "right", "center"),
           "SVGen-3B": (0, -5, "center", "top"), "SVGen-7B": (4, 0, "left", "center"),
           "HiVG-3B": (4, 0, "left", "center"), "InternSVG-8B": (0, 5, "center", "bottom")}
LAB_CLIP = {"VectorFusion": (4, 0, "left", "center"), "SVGDreamer": (-4, 0, "right", "center"),
            "LLM4SVG-7B": (4, 0, "left", "center"), "OmniSVG-3B": (-4, 0, "right", "center"),
            "SVGen-3B": (0, -5, "center", "top"), "SVGen-7B": (4, 1, "left", "bottom"),
            "HiVG-3B": (4, 3, "left", "bottom"), "InternSVG-8B": (4, 4, "left", "bottom")}


def _land_panel(ax, key, labels, better_low):
    for o in D.OTHERS:
        ax.scatter([o["s"]], [o[key]], s=24, color=CAT_COL[o["cat"]], edgecolor="white", linewidths=0.5, zorder=4)
        if labels and o["name"] in labels:
            dx, dy, ha, va = labels[o["name"]]
            free(ax.annotate(o["name"], (o["s"], o[key]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                             fontsize=7.0, color=INK))
    for model, mk in (("OmniSVG-8B", "o"), ("InternSVG-8B", "s")):
        for a, b, col in (("AR", "Ours", C["ours"]), ("AR x8", "Ours x8", OURS2)):
            pa, pb = D.OURS[(model, a)], D.OURS[(model, b)]
            ax.annotate("", xy=(pb["s"] * 1.13, pb[key]), xytext=(pa["s"] / 1.12, pa[key]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.85, mutation_scale=6, shrinkA=0, shrinkB=0),
                        zorder=3)
            ax.scatter([pa["s"]], [pa[key]], s=18, marker=mk, facecolor="white", edgecolor=C["ar"], linewidths=0.9,
                       zorder=5)
            ax.scatter([pb["s"]], [pb[key]], s=22, marker=mk, color=col, edgecolor="white", linewidths=0.4, zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(*XL)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))


def _land_legend(fig, box):
    items = [H.item("OmniSVG-8B", INK, kind="marker", marker="o", hollow=True, ms=4.4),
             H.item("InternSVG-8B", INK, kind="marker", marker="s", hollow=True, ms=4.2),
             H.item("original decoding", C["ar"], kind="marker", marker="o", hollow=True, ms=4.4),
             H.item("ours", C["ours"], kind="marker", ms=4.6, bold=False, text_color=INK),
             H.item("ours, best of 8", OURS2, kind="marker", ms=4.6, bold=False, text_color=INK),
             H.item("optimization-based", C["optim"], kind="marker", ms=4.6),
             H.item("other SVG LLMs", C["other_llm"], kind="marker", ms=4.6)]
    return H.legend(fig, items, x_pt=box[0] + 2 + LEG_DX, top_pt=top_of(fig, box) + 1, width_pt=box[2] - 3 - LEG_DX, fontsize=7.0,
                    gap_pt=5.0, pad_pt=2.0)


def _landscape(fig, box, clip_labels):
    bot = _land_legend(fig, box)
    t = bot - top_of(fig, box) + 4
    x0, y0, w, h = box
    l, r, b, gap = 31, 6, 25, 8
    ph = (h - t - b - gap)
    h1, h2 = 0.56 * ph, 0.44 * ph
    ax1 = H.axes_pt(fig, x0 + l, y0 + b + h2 + gap, w - l - r, h1)
    ax2 = H.axes_pt(fig, x0 + l, y0 + b, w - l - r, h2)
    _land_panel(ax1, "fid", LAB_FID, True)
    ax1.set_ylim(37, 6.5)
    ax1.yaxis.set_major_locator(FixedLocator([10, 20, 30]))
    ax1.set_ylabel("FID", labelpad=2)
    ax1.tick_params(axis="x", labelbottom=False)
    free(ax1.text(0.98, 0.97, "lower is better", transform=ax1.transAxes, ha="right", va="top", fontsize=7.0,
                  color=MUTED))
    _land_panel(ax2, "clip", LAB_CLIP if clip_labels else None, False)
    ax2.set_ylim(18.8, 26.0)
    ax2.yaxis.set_major_locator(FixedLocator([20, 22, 24]))
    ax2.set_ylabel("CLIP-T2I", labelpad=2)
    ax2.set_xlabel("seconds per SVG (log scale)", labelpad=1.5)
    free(ax2.text(0.98, 0.97, "higher is better", transform=ax2.transAxes, ha="right", va="top", fontsize=7.0,
                  color=MUTED))
    return ax1, ax2


def c5a_tall(fig, box):
    return _landscape(fig, box, clip_labels=True)


def c5b_tall_toplabels(fig, box):
    return _landscape(fig, box, clip_labels=False)


CANDIDATES = {
    "c8a_share_bars": (c8a_share_bars, "cell"), "c8b_arrows": (c8b_arrows, "cell"),
    "c8c_classes": (c8c_classes, "cell"), "c8d_wrong_bars": (c8d_wrong_bars, "cell"),
    "c9a_dots": (c9a_dots, "cell"), "c9b_ridges": (c9b_ridges, "cell"), "c9c_bars": (c9c_bars, "cell"),
    "c9d_diagonal": (c9d_diagonal, "cell"),
    "c10a_stacked": (c10a_stacked, "cell"), "c10b_dots": (c10b_dots, "cell"),
    "c7a_radar": (c7a_radar, "cell"), "c7b_rows": (c7b_rows, "cell"),
    "c5a_tall": (c5a_tall, "tall"), "c5b_tall_toplabels": (c5b_tall_toplabels, "tall"),
}
