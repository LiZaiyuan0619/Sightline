"""House style for the SVG speculative-decoding paper -- started 11-04.

A default to start from, not a rule book. User, 11-04: "the shared style is only a reference, to be applied
flexibly; I do not want to be bound by it in advance". Any figure may override anything here; say why in that round's README.
The paper-wide meaning of each colour and mark is in ENCODING.md next to this file.

What it gives:
  COL_W, FULL_W     two-column page widths in inches (3.25, 6.875; checked against the template 10-30)
  SZ                the type sizes signed off on 11-03 (tick 8, axis title 9, legend 8.5, leads 9-9.5)
  C, METHOD         paper-wide colours: the nine methods of table 1, the original model, the loop accent, ...
  apply()           figstyle's Times + STIX + these sizes
  legend(...)       a legend laid out in points, so it looks the same at 3.25 in and 6.875 in; handles are
                    drawn as the marks they stand for (a hollow circle for a hollow circle, not a square)
  check(fig)        geometry checks measured inside matplotlib, aimed at the blind spots of the PDF audit
                    (guide SS9): neighbouring strings on one line, rotated text, text on data lines / markers /
                    bars, text cut by the canvas edge, legend handles touching their text
  save(fig, ...)    figstyle.save (print width, optional audits) + check(); FAILs raise
  sheet(...)        one contact sheet of a round's PNGs at print scale, for the visual pass

Marking artists for check():  gid "ref" (reference / zero / diagonal lines), "bg" (background bands), "free"
(a text that may sit on marks, e.g. a number inside its own bar), "run" (pieces of one string set token by token,
which abut on purpose). Text over thin reference lines and light
bands is fine by the user (guide SS4.3), so those are never obstacles.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox


import figstyle as fs  # noqa: E402

# --- sizes ---------------------------------------------------------------------------------------------
COL_W, FULL_W = 3.25, 6.875
SZ = dict(tick=8.0, label=9.0, legend=8.5, header=8.5, note=8.5, lead=9.0, bar_lead=9.5, letter=12.0, min=7.0)

# --- colours (ENCODING.md says what each one means) ----------------------------------------------------
INK = "#2E2E2E"          # all text
MUTED = "#6B6B6B"        # secondary text (units, 'lower is better')
REF = "#8F8F8F"          # reference lines: 1x, zero, diagonal
GROUP_LINE = "#AEB5B5"   # the thin line over a group name on a two-row axis
METHOD = {
    "PLD": "#D8C6AC", "REST": "#C3A77C", "SpS": "#A98A5B",          # training-free (browns)
    "EAGLE-3": "#B4A8D4", "ViSpec": "#8C7CB8",                       # layer-serial drafting (purples)
    "DFlash": "#9FC8C8", "DARTree": "#67A3A3", "DDTree": "#2F6D6D",  # block / tree drafting (teals)
    "Ours": "#1D4E89",
}
METHODS = list(METHOD)   # table 1 order: families together, DDTree (second almost everywhere) next to us
C = dict(
    ours=METHOD["Ours"],
    ours_light="#BFD0E6",   # our lead over the best baseline (table 1 figure), our secondary state
    ours_mid="#6E9AC4",     # a second variant of ours, when one panel needs two
    ar="#7A7A7A",           # the original model decoding on its own (AR). Grey: it is the reference, not a rival
    ar_light="#C9C9C9",
    loop="#C98A1E",         # loops: the loop share, loop tokens, the draft's misses (M1, M5)
    loop_light="#EBD3A4",
    gain="#4E8A55",         # significantly better (red / green only for direction, guide SS4.2)
    loss="#B0524A",         # significantly worse
    same="#9A9A9A",         # no significant change
    other_llm="#9C8CC0",    # other SVG LLMs (landscape figure)
    optim="#C9A27E",        # optimisation-based SVG generators (landscape figure)
)


def apply() -> None:
    fs.apply_style(tick=SZ["tick"], label=SZ["label"])
    plt.rcParams.update({"xtick.labelsize": SZ["tick"], "ytick.labelsize": SZ["tick"],
                         "legend.fontsize": SZ["legend"], "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK})


# --- legend laid out in points ---------------------------------------------------------------------------
def _overlay(fig):
    """A full-figure axes in point coordinates for legends and free text; created once per figure."""
    for a in fig.axes:
        if a.get_label() == "_house_overlay":
            return a
    w, h = fig.get_figwidth() * 72, fig.get_figheight() * 72
    a = fig.add_axes([0, 0, 1, 1], label="_house_overlay", zorder=10)
    a.set_xlim(0, w)
    a.set_ylim(0, h)
    a.axis("off")
    a.patch.set_alpha(0)
    return a


def item(label, color, kind="patch", marker="o", hollow=False, ms=5.5, lw=1.6, ls="-", bold=None,
         text_color=None, mew=1.1, with_marker=False):
    """One legend entry. kind: 'patch' (filled square), 'band' (wide light rectangle), 'marker', 'line'.
    with_marker draws the marker on top of a line handle, for a line-with-points series."""
    is_ours = label == "Ours"
    return dict(label=label, color=color, kind=kind, marker=marker, hollow=hollow, ms=ms, lw=lw, ls=ls,
                bold=is_ours if bold is None else bold, with_marker=with_marker,
                text_color=text_color or (METHOD["Ours"] if is_ours else INK), mew=mew)


def legend(fig, items, x_pt, top_pt, width_pt=None, fontsize=None, gap_pt=9.0, pad_pt=3.0, handle_pt=None,
           row_pt=None, align="left"):
    """Lay the entries out left to right in rows that fit width_pt, starting at (x_pt, top_pt) measured from
    the figure's left and top edges. Spacing is in points, so a single-column figure gets the same handle-to-
    text distance as a double-column one (the 11-04 helpers used fractions of the width and broke at 3.25 in).
    Returns the legend's bottom edge, in points from the top, so the axes can start right under it."""
    fontsize = fontsize or SZ["legend"]
    ov = _overlay(fig)
    W, H = fig.get_figwidth() * 72, fig.get_figheight() * 72
    width_pt = width_pt or (W - x_pt - 2)
    row_pt = row_pt or fontsize * 1.45
    rend = fig.canvas.get_renderer()
    # measure the labels
    widths = []
    for it in items:
        t = ov.text(0, 0, it["label"], fontsize=fontsize, fontweight="bold" if it["bold"] else "normal")
        bb = t.get_window_extent(rend)
        widths.append(bb.width * 72 / fig.dpi)
        t.remove()
    hw = [handle_pt or (11.0 if it["kind"] in ("line", "band") else 7.0) for it in items]
    rows, cur, cur_w = [], [], 0.0
    for i, it in enumerate(items):
        w = hw[i] + pad_pt + widths[i]
        if cur and cur_w + gap_pt + w > width_pt:
            rows.append((cur, cur_w))
            cur, cur_w = [], 0.0
        cur_w += (gap_pt if cur else 0) + w
        cur.append(i)
    if cur:
        rows.append((cur, cur_w))
    y = H - top_pt - row_pt / 2
    for idx, rw in rows:
        x = x_pt + {"left": 0, "center": (width_pt - rw) / 2, "right": width_pt - rw}[align]
        for i in idx:
            it = items[i]
            cx, cy = x + hw[i] / 2, y
            col = it["color"]
            if it["kind"] == "patch":
                s = 6.2
                ov.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, facecolor=col, edgecolor="none"))
            elif it["kind"] == "band":
                ov.add_patch(Rectangle((x, cy - 3.6), hw[i], 7.2, facecolor=col, edgecolor="none"))
            elif it["kind"] == "marker":
                ov.plot([cx], [cy], marker=it["marker"], ms=it["ms"], mew=it["mew"], mec=col,
                        mfc="white" if it["hollow"] else col, ls="none")
            elif it["kind"] == "line":
                ov.plot([x, x + hw[i]], [cy, cy], color=col, lw=it["lw"], ls=it["ls"], solid_capstyle="butt")
                if it.get("marker") and it["kind"] == "line" and it.get("with_marker"):
                    ov.plot([cx], [cy], marker=it["marker"], ms=it["ms"], mec=col,
                            mfc="white" if it["hollow"] else col, ls="none")
            ov.text(x + hw[i] + pad_pt, cy, it["label"], fontsize=fontsize, va="center", ha="left",
                    color=it["text_color"], fontweight="bold" if it["bold"] else "normal")
            x += hw[i] + pad_pt + widths[i] + gap_pt
        y -= row_pt
    return top_pt + row_pt * len(rows)


def legend_height(items, width_in, x_pt, width_pt=None, **kw):
    """Height in points that legend() will take, measured on a scratch figure of the same width (so the real
    figure can be created at its final height)."""
    f = plt.figure(figsize=(width_in, 3))
    bot = legend(f, items, x_pt=x_pt, top_pt=0, width_pt=width_pt, **kw)
    plt.close(f)
    return bot


def axes_pt(fig, left_pt, bottom_pt, width_pt, height_pt):
    """add_axes in points from the figure's left and bottom edges."""
    W, H = fig.get_figwidth() * 72, fig.get_figheight() * 72
    return fig.add_axes([left_pt / W, bottom_pt / H, width_pt / W, height_pt / H])


def letter(fig, x_pt, top_pt, s):
    fs.letter_at(fig, x_pt, top_pt, s, fs=SZ["letter"])


def ref_line(ax, y=None, x=None, **kw):
    kw = dict(color=REF, lw=0.8, ls=(0, (3, 2)), zorder=1) | kw
    ln = ax.axhline(y, **kw) if y is not None else ax.axvline(x, **kw)
    ln.set_gid("ref")
    return ln


# --- geometry check ------------------------------------------------------------------------------------
def _texts(fig):
    out = []

    def add(t):
        if isinstance(t, Text) and t.get_visible() and t.get_text().strip() and t.get_alpha() != 0:
            out.append(t)

    for t in fig.texts:
        add(t)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in ax.texts:
            add(t)
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            add(t)
        for axis in (ax.xaxis, ax.yaxis):
            if not axis.get_visible():
                continue
            lo, hi = sorted(axis.get_view_interval())
            span = hi - lo
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-6 * span <= loc <= hi + 1e-6 * span):
                    continue
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible():
                        add(lab)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                add(t)
    for lg in fig.legends:
        for t in lg.get_texts():
            add(t)
    return out


def _px(fig, pt):
    return pt * fig.dpi / 72.0


def _obstacles(fig, rend):
    """(kind, artist, geometry) for the marks a text must not sit on. geometry: ('path', Path in display,
    halfwidth_px, filled) or ('dots', centers Nx2, radii N)."""
    out = []
    for ax in fig.axes:
        arts = list(ax.lines) + list(ax.collections) + list(ax.patches)
        for a in arts:
            if not a.get_visible() or a.get_gid() in ("ref", "bg", "free"):
                continue
            clip = a.get_clip_box() if a.get_clip_on() else None
            if isinstance(a, Line2D):
                xy = a.get_xydata()
                if len(xy) == 0:
                    continue
                tr = a.get_transform()
                if a.get_linestyle() not in ("None", "none", "", " ") and a.get_linewidth() > 0 and len(xy) > 1:
                    p = tr.transform_path(a.get_path())
                    out.append(("line", a, ("path", p, _px(fig, a.get_linewidth()) / 2, False), clip))
                if a.get_marker() not in (None, "None", "none", "", " ") and a.get_markersize() > 0:
                    cen = tr.transform(xy)
                    r = np.full(len(cen), _px(fig, a.get_markersize()) / 2)
                    out.append(("marker", a, ("dots", cen, r), clip))
            elif isinstance(a, PathCollection):
                offs = a.get_offsets()
                if len(offs) == 0:
                    continue
                cen = a.get_offset_transform().transform(offs)
                s = np.asarray(a.get_sizes(), float)
                s = np.broadcast_to(s if s.size else np.array([36.0]), (len(cen),))
                out.append(("marker", a, ("dots", cen, _px(fig, np.sqrt(s)) / 2), clip))
            elif isinstance(a, LineCollection):
                tr = a.get_transform()
                lws = np.atleast_1d(a.get_linewidths())
                for k, seg in enumerate(a.get_segments()):
                    if len(seg) < 2:
                        continue
                    p = tr.transform_path(MPath(seg))
                    out.append(("line", a, ("path", p, _px(fig, lws[k % len(lws)]) / 2, False), clip))
            elif isinstance(a, PolyCollection):
                continue      # filled bands (fill_between): text over light bands is fine (guide SS4.3)
            elif isinstance(a, FancyArrowPatch):
                p = a.get_path()                         # already in display coordinates
                out.append(("arrow", a, ("path", p, _px(fig, a.get_linewidth()) / 2, True), clip))
            elif isinstance(a, Patch):
                if isinstance(a, Rectangle) and a.get_alpha() is not None and a.get_alpha() < 0.35:
                    continue
                p = a.get_transform().transform_path(a.get_path())
                out.append(("bar" if isinstance(a, Rectangle) else "patch", a,
                            ("path", p, _px(fig, a.get_linewidth()) / 2, a.get_fill()), clip))
    return out


def _hit(geom, bb):
    if geom[0] == "path":
        _, p, hw, filled = geom
        b = Bbox.from_extents(bb.x0 - hw, bb.y0 - hw, bb.x1 + hw, bb.y1 + hw)
        return p.intersects_bbox(b, filled=filled)
    _, cen, r = geom
    inside = ((cen[:, 0] + r > bb.x0) & (cen[:, 0] - r < bb.x1) & (cen[:, 1] + r > bb.y0) & (cen[:, 1] - r < bb.y1))
    return bool(inside.any())


def check(fig, same_line_gap_pt=1.5, verbose=True, name=""):
    """Return a list of findings ('FAIL ...'). Measured on the drawn figure, before export."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = _texts(fig)
    # Text.get_window_extent, not Annotation's: an annotation's own extent includes its arrow
    boxes = [Text.get_window_extent(t, rend) for t in texts]
    found = []
    lab = lambda t: repr(t.get_text().replace("\n", " ")[:40])
    # 1. text cut by the canvas. matplotlib's text box is a little tighter than the glyph boxes the PDF audit reads
    #    (descenders of an x-axis title came out 'clipped' there), so keep 1 pt clear of every edge
    m = _px(fig, 1.0)
    for t, bb in zip(texts, boxes):
        if bb.x0 < m or bb.y0 < m or bb.x1 > W - m or bb.y1 > H - m:
            found.append(f"FAIL text-canvas {lab(t)} runs off the figure (or is within 1 pt of its edge)")
    # 2. text on text: two horizontal strings on one line need a gap (catches "unchanged7 of 8"); stacked
    #    lines may touch but not overlap; a rotated string (axis titles) only fails on a real overlap
    g = _px(fig, same_line_gap_pt) / 2
    v = _px(fig, 1.0)
    rot = [abs(t.get_rotation() % 180) > 1 for t in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i].get_gid() == "run" and texts[j].get_gid() == "run":
                continue      # one string set token by token (the SVG source sample, M1): pieces abut on purpose
            a, b = boxes[i], boxes[j]
            gi = 0 if (rot[i] or rot[j]) else g
            vi = v if not (rot[i] or rot[j]) else v * 0.8
            ax_ = Bbox.from_extents(a.x0 - gi + (vi if gi == 0 else 0), a.y0 + vi, a.x1 + gi - (vi if gi == 0 else 0), a.y1 - vi)
            bx_ = Bbox.from_extents(b.x0 - gi + (vi if gi == 0 else 0), b.y0 + vi, b.x1 + gi - (vi if gi == 0 else 0), b.y1 - vi)
            if ax_.overlaps(bx_):
                found.append(f"FAIL text-text {lab(texts[i])} x {lab(texts[j])}")
    # 3. text on marks
    obs = _obstacles(fig, rend)
    for t, bb in zip(texts, boxes):
        if t.get_gid() == "free":
            continue
        own = t.arrow_patch if isinstance(t, Annotation) else None
        shrunk = Bbox.from_extents(bb.x0 + v * 0.6, bb.y0 + v * 0.6, bb.x1 - v * 0.6, bb.y1 - v * 0.6)
        for kind, art, geom, clip in obs:
            if art is own:
                continue
            b = shrunk
            if clip is not None:
                b = Bbox.intersection(shrunk, clip)
                if b is None or b.width <= 0 or b.height <= 0:
                    continue
            if _hit(geom, b):
                who = art.get_label() if not str(art.get_label()).startswith("_") else type(art).__name__
                if isinstance(art, Line2D) and len(art.get_xydata()):
                    xy = art.get_xydata()
                    who += f" (data {xy[0][0]:.3g},{xy[0][1]:.3g} .. {xy[-1][0]:.3g},{xy[-1][1]:.3g})"
                found.append(f"FAIL text-{kind} {lab(t)} sits on {who}")
    if verbose:
        head = f"house.check {name}: " if name else "house.check: "
        print(head + ("clean" if not found else f"{len(found)} finding(s)"))
        for f in found[:30]:
            print("   ", f)
    return found


def save(fig, out_base, width="col", strict=True, **kw):
    """figstyle.save (print width + optional audits) after house.check; a FAIL raises when strict."""
    name = Path(out_base).name
    found = check(fig, name=name)
    target = {"col": COL_W, "full": FULL_W}.get(width, width)
    paths = fs.save(fig, out_base, width=target, **kw)
    Path(f"{out_base}.qa.txt").write_text("\n".join(found) + ("\n" if found else "clean\n"))
    if strict and found:
        raise AssertionError(f"{name}: {len(found)} geometry finding(s), see {out_base}.qa.txt")
    return paths


# --- contact sheet ---------------------------------------------------------------------------------------
def sheet(pngs, out, cols=2, dpi=600, show_dpi=220, title=None):
    """Put a round's PNGs on one page at a common print scale (each image keeps its printed size)."""
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for p in pngs]
    k = show_dpi / dpi
    ims = [im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS) for im in ims]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    cap = 34
    rows = math.ceil(len(ims) / cols)
    cw = max(im.width for im in ims) + 40
    rh = [max(ims[r * cols + c].height for c in range(cols) if r * cols + c < len(ims)) + cap + 30
          for r in range(rows)]
    top = 50 if title else 10
    page = Image.new("RGB", (cw * cols, top + sum(rh)), "white")
    d = ImageDraw.Draw(page)
    if title:
        d.text((20, 12), title, fill="black", font=font)
    y = top
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if i >= len(ims):
                continue
            x = c * cw + 20
            d.text((x, y), Path(pngs[i]).stem, fill="black", font=font)
            page.paste(ims[i], (x, y + cap))
            d.rectangle([x - 1, y + cap - 1, x + ims[i].width, y + cap + ims[i].height], outline="#DDDDDD")
        y += rh[r]
    page.save(out)
    return out
