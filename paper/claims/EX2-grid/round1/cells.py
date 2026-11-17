"""The second experiments composite (main.pdf Figures 6, 7, 8, 9 as of night), round 1: one 2 x 3 figure.

User, compose these four next; the forms stay (each was signed off on its own), only the cells change size.
Grid = the D-group cell (165 x 150 pt, axes 43 / 4 / 31 / 124, a 26 pt legend strip on top), so all six axes share
their edges by row and by column.
  a main.pdf Figure 6, the two sources of the lead   M8 round 4's bars, redrawn here at the cell size; the InternSVG text
                                                      block now on table 1's heads (m8fix.py; B8 had fixed M1 / M2 only)
  b main.pdf Figure 7, the speed identity            P1 round 2's rays, redrawn here at the cell size
  c main.pdf Figure 9, training length, warm start   D5 round 1's d5a_twopart, called as is (drawn for this cell size)
  d main.pdf Figure 8a, the recipe                   D-group round 1's d1_interaction, as is
  e main.pdf Figure 8b, block length                 D-group round 1's d2_gap, as is
  f main.pdf Figure 8c, the same 127 nodes           D-group round 1's d3_return, as is
Figure 8's three cells stay one row (they were designed as one); Figure 9 moves into the first row, so the paragraph
that cites them says training first, then the design.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

HERE = Path(__file__).resolve().parent
CLAIMS = HERE.parents[1]
sys.path.insert(0, str(CLAIMS / "D-group/round1"))
sys.path.insert(0, str(HERE))
import dpanels as P  # noqa: E402  (imports house and ddata, which asserts its published numbers)
import m8fix  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D5 = _load("d5r1", CLAIMS / "D5-training/round1/make.py")            # d5data asserts the published tokens per round
P1 = _load("p1r2", CLAIMS / "P1-speed-identity/round2/make.py")      # its round 1 load asserts the identity
H = P.H
C, M, INK, MUTED = P.C, P.M, P.INK, P.MUTED
BOX_W, BOX_H = P.BOX_W, P.BOX_H

# ---------------------------------------------------------------------------------------------------------- a
ORDER = [("OmniSVG-8B text", "8B", "text"), ("OmniSVG-4B text", "4B", "text"), ("OmniSVG-8B image", "8B", "image"),
         ("InternSVG-8B text", "8B", "text"), ("InternSVG-8B image", "8B", "image")]       # the order of Figure 5c


def _block_axis(ax, fig, line_down=18.0, name_down=19.5):
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([f"{s}\n{t}" for _, s, t in ORDER], fontsize=7.0, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for fam, lo, hi in (("OmniSVG", 0, 2), ("InternSVG", 3, 4)):
        ln = ax.plot([lo - 0.34, hi + 0.34], [0, 0], color=H.GROUP_LINE, lw=0.6, clip_on=False,
                     transform=tr + ScaledTranslation(0, -line_down / 72, fig.dpi_scale_trans))[0]
        ln.set_gid("ref")
        ax.annotate(fam, xy=((lo + hi) / 2, 0), xycoords=tr, xytext=(0, -name_down), textcoords="offset points",
                    ha="center", va="top", fontsize=7.0, annotation_clip=False)


def a_lead_sources(fig, box):
    P._legend(fig, box, [H.item("loop-free content", C["ours_light"]), H.item("inside loops", C["ours"]),
                         H.item("greedy", INK, kind="marker", ms=4.4),
                         H.item("official sampling", INK, kind="marker", hollow=True, ms=4.4)], gap=6.0)
    ax = P._ax(fig, box)
    for i, (b, *_r) in enumerate(ORDER):
        for j, (seg, col) in enumerate((("healthy", C["ours_light"]), ("loop", C["ours"]))):
            v = [m8fix.TAU[(b, md, "Ours")][seg] / m8fix.TAU[(b, md, "DDTree")][seg] for md in ("greedy", "sample")]
            x = i + (j - 0.5) * 0.38
            ax.bar(x, np.mean(v) - 0.9, bottom=0.9, width=0.36, color=col, linewidth=0, zorder=2)
            for k, y in enumerate(v):
                ax.scatter([x + (k - 0.5) * 0.15], [y], s=10, facecolor=INK if k == 0 else "white",
                           edgecolor=INK, linewidths=0.7, zorder=4)
    H.ref_line(ax, y=1.0, lw=0.7)
    ax.set_ylim(0.9, 1.85)
    ax.yaxis.set_major_locator(FixedLocator([1.0, 1.2, 1.4, 1.6, 1.8]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"×{v:.1f}"))
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    _block_axis(ax, fig)
    ax.set_ylabel("ours ÷ DDTree,\ntokens per round", fontsize=H.SZ["label"], labelpad=2, linespacing=1.12)
    return [ax]


# ---------------------------------------------------------------------------------------------------------- b
XMAX, YMAX = 40.0, 21.5


def b_speed_identity(fig, box):
    rows, ar = P1.r1.load()
    P._legend(fig, box, [H.item("equal speed-up over the original model", "#C8C8C8", kind="line", lw=0.9)], gap=6.0)
    ax = P._ax(fig, box)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    texts = []
    for s in (2, 4, 6, 8, 10):
        xe, ye = 39.6, s * 39.6 / ar
        if ye > 20.6:
            xe, ye = 20.6 * ar / s, 20.6
        texts.append(ax.text(xe, ye, f"{s}×", fontsize=7.0, color="#9A9A9A", ha="right" if ye < 20 else "center",
                             va="bottom"))
    lab = {"PLD": (-4, -3, "right", "center"), "REST": (4, 3, "left", "center"),
           "EAGLE-3": (-4, 0, "right", "center"), "DFlash": (-4, 0, "right", "center"),
           "DARTree": (4, 2, "left", "center"), "DDTree": (-4, 3, "right", "bottom"),
           "Ours": (4, 0, "left", "center")}
    for m in P1.r1.METHODS:
        r = rows[m]
        ours = m == "Ours"
        ax.scatter(r["ms"], r["tau"], s=30 if ours else 17, color=M[m], edgecolor="white", linewidths=0.5, zorder=4)
        dx, dy, ha, va = lab[m]
        texts.append(ax.annotate(m, (r["ms"], r["tau"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                 fontsize=7.4, color=M["Ours"] if ours else INK, fontweight="bold" if ours else "normal"))
    ax.xaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 40]))
    ax.yaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20]))
    ax.set_xlabel("ms per round", fontsize=H.SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens per round", fontsize=H.SZ["label"], labelpad=2)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    pad = P1.CLEAR_PT * fig.dpi / 72
    boxes = []
    for t in texts:
        bb = Text.get_window_extent(t, rend)
        (bx0, by0), (bx1, by1) = inv.transform([(bb.x0 - pad, bb.y0 - pad), (bb.x1 + pad, bb.y1 + pad)])
        boxes.append((bx0, by0, bx1, by1))
    for s in (2, 4, 6, 8, 10):
        x1 = min(XMAX, YMAX * ar / s)
        y1 = s * x1 / ar
        for a_, b_ in P1._cut(0.0, 0.0, x1, y1, boxes):
            ln, = ax.plot([a_ * x1, b_ * x1], [a_ * y1, b_ * y1], color="#DADADA", lw=0.7, zorder=0)
            ln.set_gid("ref")
    return [ax]


CELLS = {"a": a_lead_sources, "b": b_speed_identity, "c": D5.d5a_twopart,
         "d": P.d1_interaction, "e": P.d2_gap, "f": P.d3_return}
