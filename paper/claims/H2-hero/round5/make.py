"""Hero (Figure 1), round 5 (main-line session 8): round 4's h4a rose, larger, with the paper's type sizes.

User, on round 4: pick h4a (the ten-petal rose) without drawings ("they add nothing"); the rose is too small inside the
single column, with blank space left and right while the legend row stretches the width, and the ring labels (8x, 12x)
may be too small to read in print; Figure 4 stays in the main text for now (the hero summarises Figure 4a).
Changes from round 4 (same data, same encoding):
  - the decoder legend leaves the bottom row and fills the two empty top corners (guide: legends into empty corners,
    no blank edges), so the rose can take the width;
  - the rose's radius is the largest that keeps every label inside the column (searched, not guessed);
  - type at the house sizes signed off on pair names 8.5 pt, our speed-ups 9.5 pt bold, ring labels 8 pt,
    legend 8.5 pt (round 4 used 7 / 8 / 7 / 7.5).
Candidate: r5a (legend in the top corners); round 4's h4a is kept on the page mock-up for comparison.
Data:../round4/data4.py (asserts the published numbers). Run: python3 paper/claims/H2-hero/round5/make.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
R4 = HERE.parent / "round4"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(R4))
import house as H  # noqa: E402
import data4 as D  # noqa: E402

H.apply()
INK, MUTED, REF = H.INK, H.MUTED, H.REF
COL = dict(H.METHOD, **{"original model": H.C["ar"]})
W = 234.0
FS = dict(pair=8.5, lead=9.5, ring=8.0, legend=8.5)
MARK = "#5A5A5A"                       # tip markers: filled = greedy, hollow = official sampling
SEC_GAP, MODE_GAP = 7.0, 2.5           # degrees, as round 4
RMAX = 15.6                            # data units at the axes edge


def geometry():
    petal = (72 - SEC_GAP - MODE_GAP) / 2
    mids = {p: 90 + 72 * (2 - k) for k, p in enumerate(D.PAIRS)}      # InternSVG-8B text-to-SVG at the top
    centre = {}
    for p in D.PAIRS:
        centre[(p, "greedy")] = mids[p] + (MODE_GAP + petal) / 2
        centre[(p, "sample")] = mids[p] - (MODE_GAP + petal) / 2
    return centre, petal, mids


def means(p, md):
    out = {m: float(np.mean(D.cells(p, md, m))) for m in D.DECODERS if None not in D.cells(p, md, m)}
    out["original model"] = 1.0
    return out


def all_text_boxes(fig, pad_pt=0.0, skip=()):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    out = []
    for t in fig.findobj(Text):
        if t in skip or not t.get_visible() or not t.get_text().strip():
            continue
        b = Text.get_window_extent(t, rend)
        out.append(Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad))
    return out


def rings(fig, ax, radii, color="#E1E1E1", lw=0.55, pad_pt=1.8):
    """Faint rings, broken wherever they would cross any text of the figure."""
    boxes = all_text_boxes(fig, pad_pt)
    th = np.radians(np.arange(0, 360.01, 0.25))
    for r in radii:
        xy = ax.transData.transform(np.column_stack([th, np.full_like(th, r)]))
        ok = [not any(b.contains(x, y) for b in boxes) for x, y in xy] + [False]
        s0 = None
        for i, k in enumerate(ok):
            if k and s0 is None:
                s0 = i
            if not k and s0 is not None:
                if i - s0 > 1:
                    ax.plot(th[s0:i], np.full(i - s0, r), color=color, lw=lw, zorder=1)[0].set_gid("ref")
                s0 = None


def place_label(fig, ax, p, th_mid, r0, taken, canvas, span=30.0):
    """The pair's name near its sector's middle ray: try small angular offsets first, then larger radii; keep it inside
    the canvas (2 pt margin) and clear of every other text (3 pt)."""
    rend = fig.canvas.get_renderer()
    for dth in (0, 6, -6, 12, -12, 18, -18, 24, -24, span, -span):
        th = math.radians(th_mid + dth)
        c, s_ = math.cos(th), math.sin(th)
        ha = "center" if abs(c) < 0.3 else ("left" if c > 0 else "right")
        va = "bottom" if s_ > 0.3 else ("top" if s_ < -0.3 else "center")
        r = r0
        while r < r0 + 6:
            t = ax.text(th, r, D.PAIR_LINES[p], fontsize=FS["pair"], ha=ha, va=va, color=INK, linespacing=1.12, zorder=8)
            bb = Text.get_window_extent(t, rend)
            inside = bb.x0 >= canvas.x0 and bb.x1 <= canvas.x1 and bb.y0 >= canvas.y0 and bb.y1 <= canvas.y1
            if inside and not any(bb.overlaps(o) for o in taken):
                return t, bb
            t.remove()
            r += 0.25
    return None, None


def legend_columns(fig, top_pt, scale_k):
    """Decoder legend split into the two top corners, five entries each (families kept together)."""
    left = [H.item(m, COL[m]) for m in ["original model", "PLD", "REST", "SpS", "EAGLE-3"]]
    right = [H.item(m, COL[m]) for m in ["ViSpec", "DFlash", "DARTree", "DDTree", "Ours"]]
    row = FS["legend"] * 1.3
    for i, it in enumerate(left):
        H.legend(fig, [it], x_pt=2.5, top_pt=top_pt + i * row, width_pt=80, fontsize=FS["legend"], row_pt=row)
    xr = W - 2.5 - max(text_width(fig, it["label"], it["bold"]) for it in right) - 7.0 - 3.0
    for i, it in enumerate(right):
        H.legend(fig, [it], x_pt=xr, top_pt=top_pt + i * row, width_pt=W - xr, fontsize=FS["legend"], row_pt=row)
    return top_pt + 5 * row, xr


MODE_ITEMS = [H.item("greedy", MARK, kind="marker", marker="o", ms=4.0, mew=0.9),
              H.item("official sampling", MARK, kind="marker", marker="o", hollow=True, ms=4.0, mew=0.9)]
MODE_ROW = FS["legend"] * 1.45


def text_width(fig, s, bold=False):
    t = fig.text(0, 0, s, fontsize=FS["legend"], fontweight="bold" if bold else "normal")
    fig.canvas.draw()
    w = Text.get_window_extent(t, fig.canvas.get_renderer()).width * 72 / fig.dpi
    t.remove()
    return w


def build(R, Ht, cy, ring_scale=False, final=False):
    """Returns the figure and the vertical extent of its content around the centre (points), or None if a label did
    not fit."""
    fig = plt.figure(figsize=(W / 72, Ht / 72))
    cx = W / 2
    ax = fig.add_axes([(cx - R) / W, (cy - R) / Ht, 2 * R / W, 2 * R / Ht], projection="polar")
    ax.set_facecolor("none")
    ax.set_ylim(0, RMAX)
    ax.set_axis_off()
    centre, petal, mids = geometry()
    tips = {}
    for (p, md), a in centre.items():
        mv = means(p, md)
        th = math.radians(a)
        for m, v in sorted(mv.items(), key=lambda kv: -kv[1]):
            w = petal * (1.0 if m == "Ours" else 0.86)
            ax.bar(th, v, width=math.radians(w), bottom=0.0, color=COL[m], edgecolor="white" if m != "Ours" else "none",
                   lw=0.3, zorder=3, align="center")
        tips[(p, md)] = mv["Ours"]
        r_m = mv["Ours"] + 0.7
        ax.plot([th], [r_m], marker="o", ms=4.0, mew=0.9, mec=MARK, mfc=MARK if md == "greedy" else "white", ls="none",
                zorder=8, clip_on=False)
        c, s_ = math.cos(th), math.sin(th)
        ha = "center" if abs(c) < 0.45 else ("left" if c > 0 else "right")
        va = "bottom" if s_ > 0.45 else ("top" if s_ < -0.45 else "center")
        ax.text(th, r_m + 0.65, f"{mv['Ours']:.1f}×", fontsize=FS["lead"], fontweight="bold", color=COL["Ours"],
                ha=ha, va=va, zorder=8)
    gap = math.radians(mids[D.PAIRS[2]] - 36)            # between InternSVG-8B text and OmniSVG-8B image
    for r in ((4, 8, 12) if ring_scale else (8, 12)):
        ax.text(gap, r, f"{r}×", fontsize=FS["ring"], color=MUTED, ha="center", va="center", zorder=2)
    fig.canvas.draw()
    k = fig.dpi / 72
    canvas = Bbox.from_extents(2 * k, 2 * k, (W - 2) * k, (Ht - 2) * k)
    taken = all_text_boxes(fig, 3.0)
    # the top pair first (its place sets the legend's top), then the rest
    order = [D.PAIRS[2]] + [p for p in D.PAIRS if p != D.PAIRS[2]]
    top_pair_bb = None
    for i, p in enumerate(order):
        if i == 1:                                        # legends go in after the top label is placed
            leg_top_pt = Ht - top_pair_bb.y1 / k
            leg_bot, xr = legend_columns(fig, leg_top_pt, k)
            taken = all_text_boxes(fig, 3.0)
            taken.append(Bbox.from_extents(0, (Ht - leg_bot - 1) * k, 76 * k, (Ht - leg_top_pt) * k))
            taken.append(Bbox.from_extents((xr - 2) * k, (Ht - leg_bot - 1) * k, W * k, (Ht - leg_top_pt) * k))
        r0 = max(tips[(p, "greedy")], tips[(p, "sample")]) + 2.0
        t, bb = place_label(fig, ax, p, mids[p], r0, taken, canvas)
        if t is None:
            plt.close(fig)
            return None, None
        if i == 0:
            top_pair_bb = bb
        taken.append(Bbox.from_extents(bb.x0 - 3 * k, bb.y0 - 3 * k, bb.x1 + 3 * k, bb.y1 + 3 * k))
    rings(fig, ax, (4, 8, 12))
    # vertical extent of everything drawn so far, around the centre; the two decodings go in one row under it
    fig.canvas.draw()
    ys = [b.y0 for b in all_text_boxes(fig)] + [b.y1 for b in all_text_boxes(fig)]
    ys += [(cy - 12 / RMAX * R) * k, (cy + 12 / RMAX * R) * k]       # the 12x ring
    lo, hi = min(ys) / k, max(ys) / k
    H.legend(fig, MODE_ITEMS, x_pt=2, top_pt=Ht - lo + 1.5, width_pt=W - 4, fontsize=FS["legend"], gap_pt=10.0,
             align="center", row_pt=MODE_ROW)
    lo -= MODE_ROW + 1.5
    return fig, (cy - lo, hi - cy)


def make(name, ring_scale=False):
    for R in np.arange(116, 84, -2.0):
        fig, ext = build(R, 420.0, 210.0, ring_scale)
        if fig is None:
            continue
        plt.close(fig)
        below, above = ext
        Ht = math.ceil(below + above + 5)
        fig, ext2 = build(R, Ht, below + 2.5, ring_scale)
        if fig is None:                                   # the tight canvas changed a placement; give it room
            continue
        print(f"{name}: radius {R:.0f} pt (round 4: 76), canvas {Ht} pt = {Ht / 72:.2f} in")
        H.save(fig, str(HERE / name), width=W / 72, strict=True)
        plt.close(fig)
        return
    raise RuntimeError("no radius fits")


if __name__ == "__main__":
    make("r5a_rose")
