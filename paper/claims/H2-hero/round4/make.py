"""Hero (Figure 1), round 4 (main-line session 8): data-only single-column candidates.

User, on round 3: the case-study forms are cluttered and read as appendix material; drop them and use experiment
data only; to avoid showing one batch of data twice, the hero may take a main-text figure's data and that figure leaves
the main text. Decided: no best of 8 on the hero (DFlash x8 is faster there), no an interactive GPU node, draw A-D plus A with
drawings. Every candidate carries main.pdf Figure 4's speed-ups for all 60 settings and every published decoder; once
one is picked, Figure 4 leaves the main text (C also takes Figure 5d, D also Figure 4b's tokens per round).
  h4a_rose        60 settings around a rose, greedy in the upper half and official sampling in the lower half; each
                  petal overlays the decoders sorted by speed-up, so the outer colour of a petal is the fastest decoder
  h4a2_rose_draw h4a with one finished drawing of each model-task pair beside its name
  h4b_fan         one line per setting across the decoders, the last step (to ours) drawn in our blue
  h4c_wall        per decoder: tokens per round on loop-free content with its window (Figure 5d) | its 60 speed-ups
  h4d_cloud       speed-up against tokens per round, every decoder and setting, greedy and official sampling
Data: data4.py (asserts the published numbers). Nothing re-run. Run: python3 paper/claims/H2-hero/round4/make.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import data4 as D  # noqa: E402
from sightline.paths import RUNS

H.apply()
INK, MUTED, REF = H.INK, H.MUTED, H.REF
COL = dict(H.METHOD, **{"original model": H.C["ar"]})
W = 234.0                                    # 3.25 in, in points
NAME = {m: m for m in D.DECODERS} | {"original model": "original model"}
FS = dict(tick=7.0, label=7.5, legend=7.5, pair=7.0)


def fig_pt(w, h):
    return plt.figure(figsize=(w / 72, h / 72))


def save(fig, name):
    H.save(fig, str(HERE / name), width=W / 72, strict=True)
    plt.close(fig)


def legend_items(order=("original model",) + tuple(D.DECODERS)):
    return [H.item(NAME[m], COL[m]) for m in order]


# ------------------------------------------------------------------------------------------------------------- A
# Five sectors, one per model-task pair, clockwise in Figure 4's order with InternSVG-8B text-to-SVG (the tallest) at
# the top. Inside a sector, clockwise: the six benchmarks under greedy decoding, a small gap, the same six under official
# sampling. A bezel outside the petals marks the two decodings (dark: greedy, light: official sampling).
SEC_GAP, MODE_GAP, FILL = 5.0, 2.0, 0.84        # degrees; petal width as a share of its slot
BEZEL = ("#6F6F6F", "#C4C4C4")                 # greedy, official sampling
HOLE = 0.0


def rose_geometry():
    slot = (360 - 5 * SEC_GAP - 5 * MODE_GAP) / 60
    mids = {p: 90 + 72 * (2 - k) for k, p in enumerate(D.PAIRS)}        # pair 3 (index 2) at the top
    ang, groups = {}, {}
    for p in D.PAIRS:
        a = mids[p] + 36 - SEC_GAP / 2                                    # counter-clockwise edge of the sector
        for md in D.MODES:
            a0 = a
            for j in range(6):
                ang[(p, md, j)] = a - slot / 2
                a -= slot
            groups[(p, md)] = (a, a0)                                     # angular span of the group (deg)
            a -= MODE_GAP
    return ang, groups, slot, mids


def draw_rose(ax, rmax, bezel_r=(14.35, 14.85)):
    ang, groups, slot, mids = rose_geometry()
    for (p, md, j), a in ang.items():
        vals = [(D.value(p, md, j, m), m) for m in D.DECODERS if D.value(p, md, j, m) is not None]
        vals.append((1.0, "original model"))
        th = math.radians(a)
        for v, m in sorted(vals, reverse=True):          # largest at the back, so every decoder shows its band
            w = slot * (FILL if m == "Ours" else FILL * 0.78)
            ax.bar(th, v, width=math.radians(w), bottom=HOLE, color=COL[m], edgecolor="none", zorder=3,
                   align="center")
    t = np.linspace(0, 2 * np.pi, 721)
    for r in (4, 8, 12):
        ax.plot(t, np.full_like(t, HOLE + r), color="#E1E1E1", lw=0.5, zorder=1)[0].set_gid("ref")
    ax.plot(t, np.full_like(t, HOLE + 1.0), color=REF, lw=0.6, ls=(0, (2, 1.5)), zorder=6)[0].set_gid("ref")
    for (p, md), (a1, a0) in groups.items():
        tt = np.radians(np.linspace(a1 + 0.4, a0 - 0.4, 40))
        ax.fill_between(tt, bezel_r[0], bezel_r[1], color=BEZEL[0 if md == "greedy" else 1], lw=0, zorder=2)
    ax.set_ylim(0, rmax)
    ax.set_axis_off()
    return ang, groups, slot, mids


def scale_labels(ax, groups):
    """8x and 12x on their rings, inside the official-sampling group of InternSVG-8B image-to-SVG (petals below 6)."""
    a1, a0 = groups[(D.PAIRS[4], "sample")]
    th = math.radians((a1 + a0) / 2)
    for r in (8, 12):
        ax.text(th, HOLE + r, f"{r}×", fontsize=FS["tick"], color=MUTED, ha="center", va="center", zorder=7)


def pair_labels(ax, mids, r_lab):
    out = []
    for p in D.PAIRS:
        th = mids[p]
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        ha = "center" if abs(c) < 0.3 else ("left" if c > 0 else "right")
        va = "bottom" if s > 0.3 else ("top" if s < -0.3 else "center")
        t = ax.text(math.radians(th), r_lab, D.PAIR_LINES[p], fontsize=FS["pair"], ha=ha, va=va, color=INK,
                    linespacing=1.0, zorder=8)
        out.append((p, t, th))
    return out


def h4a_rose60(name="h4a_rose60", drawings=False):
    """All 60 settings as thin petals (six benchmarks per group); markers and our group means as in h4a_rose."""
    rmax = 15.6
    R = 76.0
    leg_items = legend_items()
    mode_items = [H.item("greedy", MARK, kind="marker", marker="o", ms=3.4, mew=0.8),
                  H.item("official sampling", MARK, kind="marker", marker="o", hollow=True, ms=3.4, mew=0.8)]
    leg_h = H.legend_height(leg_items, W / 72, x_pt=3, width_pt=W - 6, fontsize=FS["legend"], gap_pt=6.5)
    top_h, bot_h = 102.0, 72.0
    Ht = top_h + bot_h + leg_h + 2
    fig = fig_pt(W, Ht)
    cx, cy = W / 2, leg_h + 2 + bot_h
    ax = fig.add_axes([(cx - R) / W, (cy - R) / Ht, 2 * R / W, 2 * R / Ht], projection="polar")
    ax.set_facecolor("none")
    ang, groups, slot, mids = rose_geometry()
    for (p, md, j), a in ang.items():
        vals = [(D.value(p, md, j, m), m) for m in D.DECODERS if D.value(p, md, j, m) is not None]
        vals.append((1.0, "original model"))
        for v, m in sorted(vals, reverse=True):
            w = slot * (FILL if m == "Ours" else FILL * 0.78)
            ax.bar(math.radians(a), v, width=math.radians(w), bottom=0.0, color=COL[m], edgecolor="none", zorder=3,
                   align="center")
    tips = {}
    for (p, md), (a1, a0) in groups.items():
        th = math.radians((a1 + a0) / 2)
        top = max(D.cells(p, md, "Ours"))
        mean = float(np.mean(D.cells(p, md, "Ours")))
        tips[(p, md)] = top
        r_m = top + 0.75
        ax.plot([th], [r_m], marker="o", ms=3.4, mew=0.8, mec=MARK, mfc=MARK if md == "greedy" else "white", ls="none",
                zorder=8, clip_on=False)
        c, s_ = math.cos(th), math.sin(th)
        ha = "center" if abs(c) < 0.45 else ("left" if c > 0 else "right")
        va = "bottom" if s_ > 0.45 else ("top" if s_ < -0.45 else "center")
        ax.text(th, r_m + 0.75, f"{mean:.1f}×", fontsize=8.0, fontweight="bold", color=COL["Ours"], ha=ha, va=va, zorder=8)
    gap_th = math.radians(mids[D.PAIRS[2]] - 36)
    for r in (8, 12):
        ax.text(gap_th, r, f"{r}×", fontsize=FS["tick"], color=MUTED, ha="center", va="center", zorder=2)
    ax.set_ylim(0, rmax)
    ax.set_axis_off()
    labs = place_pair_labels(fig, ax, mids, tips)
    rings_around_text(fig, ax, (4, 8, 12))
    if drawings:
        add_drawings(fig, ax, labs, cx, cy, R, rmax, Ht)
    H.legend(fig, leg_items, x_pt=3, top_pt=Ht - leg_h - 1, width_pt=W - 6, fontsize=FS["legend"], gap_pt=6.5,
             align="center")
    for k, it in enumerate(mode_items):
        H.legend(fig, [it], x_pt=3, top_pt=2 + k * 10.5, width_pt=90, fontsize=FS["legend"])
    save(fig, name)


# --- A, bold: ten petals (pair x decoding), each decoder at its mean over the six benchmarks -----------------------------
B_SEC_GAP, B_MODE_GAP = 7.0, 2.5
MARK = "#5A5A5A"                               # tip markers: filled = greedy, hollow = official sampling


def bold_geometry():
    petal = (72 - B_SEC_GAP - B_MODE_GAP) / 2
    mids = {p: 90 + 72 * (2 - k) for k, p in enumerate(D.PAIRS)}
    centre = {}
    for p in D.PAIRS:
        centre[(p, "greedy")] = mids[p] + (B_MODE_GAP + petal) / 2
        centre[(p, "sample")] = mids[p] - (B_MODE_GAP + petal) / 2
    return centre, petal, mids


def means(p, md):
    out = {m: float(np.mean(D.cells(p, md, m))) for m in D.DECODERS if None not in D.cells(p, md, m)}
    out["original model"] = 1.0
    return out


def h4a_rose(name="h4a_rose", drawings=False):
    rmax = 15.6
    R = 76.0
    leg_items = legend_items()
    mode_items = [H.item("greedy", MARK, kind="marker", marker="o", ms=3.4, mew=0.8),
                  H.item("official sampling", MARK, kind="marker", marker="o", hollow=True, ms=3.4, mew=0.8)]
    leg_h = H.legend_height(leg_items, W / 72, x_pt=3, width_pt=W - 6, fontsize=FS["legend"], gap_pt=6.5)
    top_h, bot_h = 92.0, 66.0
    Ht = top_h + bot_h + leg_h + 2
    fig = fig_pt(W, Ht)
    cx, cy = W / 2, leg_h + 2 + bot_h
    ax = fig.add_axes([(cx - R) / W, (cy - R) / Ht, 2 * R / W, 2 * R / Ht], projection="polar")
    ax.set_facecolor("none")
    centre, petal, mids = bold_geometry()
    t = np.linspace(0, 2 * np.pi, 721)
    tips = {}
    for (p, md), a in centre.items():
        mv = means(p, md)
        th = math.radians(a)
        for m, v in sorted(mv.items(), key=lambda kv: -kv[1]):
            w = petal * (1.0 if m == "Ours" else 0.86)
            ax.bar(th, v, width=math.radians(w), bottom=0.0, color=COL[m], edgecolor="white" if m != "Ours" else "none",
                   lw=0.25, zorder=3, align="center")
        tips[(p, md)] = mv["Ours"]
        # tip marker (filled = greedy, hollow = official sampling) and our mean speed-up
        r_m = mv["Ours"] + 0.75
        ax.plot([th], [r_m], marker="o", ms=3.4, mew=0.8, mec=MARK, mfc=MARK if md == "greedy" else "white",
                ls="none", zorder=8, clip_on=False)
        c, s_ = math.cos(th), math.sin(th)
        ha = "center" if abs(c) < 0.45 else ("left" if c > 0 else "right")
        va = "bottom" if s_ > 0.45 else ("top" if s_ < -0.45 else "center")
        ax.text(th, r_m + 0.75, f"{mv['Ours']:.1f}×", fontsize=8.0, fontweight="bold", color=COL["Ours"], ha=ha, va=va,
                zorder=8)
    # 8x and 12x where their rings cross open space: the gap between InternSVG-8B text-to-SVG and OmniSVG-8B image-to-SVG
    gap_th = math.radians(mids[D.PAIRS[2]] - 36)
    for r in (8, 12):
        ax.text(gap_th, r, f"{r}×", fontsize=FS["tick"], color=MUTED, ha="center", va="center", zorder=2)
    ax.set_ylim(0, rmax)
    ax.set_axis_off()
    labs = place_pair_labels(fig, ax, mids, tips)
    rings_around_text(fig, ax, (4, 8, 12))
    if drawings:
        add_drawings(fig, ax, labs, cx, cy, R, rmax, Ht)
    H.legend(fig, leg_items, x_pt=3, top_pt=Ht - leg_h - 1, width_pt=W - 6, fontsize=FS["legend"], gap_pt=6.5,
             align="center")
    for k, it in enumerate(mode_items):                  # the two decodings, in the empty top-left corner
        H.legend(fig, [it], x_pt=3, top_pt=2 + k * 10.5, width_pt=90, fontsize=FS["legend"])
    save(fig, name)


def rings_around_text(fig, ax, radii, color="#E3E3E3", lw=0.5, pad_pt=1.6):
    """Faint rings at the given radii, broken wherever they would cross a text (drawn after all texts)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    boxes = [t.get_window_extent(rend) for t in ax.texts + fig.texts if t.get_text().strip()]
    boxes = [b.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad) for b in boxes]
    th = np.radians(np.arange(0, 360.01, 0.25))
    for r in radii:
        xy = ax.transData.transform(np.column_stack([th, np.full_like(th, r)]))
        ok = np.array([not any(b.contains(x, y) for b in boxes) for x, y in xy])
        start = None
        for i, k in enumerate(list(ok) + [False]):
            if k and start is None:
                start = i
            if not k and start is not None:
                if i - start > 1:
                    ax.plot(th[start:i], np.full(i - start, r), color=color, lw=lw, zorder=1)[0].set_gid("ref")
                start = None


def place_pair_labels(fig, ax, mids, tips, step=0.25, start=2.2):
    """Each pair's name on its sector's middle ray, pushed outwards until it clears every text drawn so far."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pad = 3.0 * fig.dpi / 72

    def grow(bb):
        return bb.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)

    taken = [grow(t.get_window_extent(rend)) for t in ax.texts]
    labs = []
    for p in D.PAIRS:
        th = math.radians(mids[p])
        c, s_ = math.cos(th), math.sin(th)
        ha = "center" if abs(c) < 0.3 else ("left" if c > 0 else "right")
        va = "bottom" if s_ > 0.3 else ("top" if s_ < -0.3 else "center")
        r = max(tips[(p, "greedy")], tips[(p, "sample")]) + start
        while True:
            t = ax.text(th, r, D.PAIR_LINES[p], fontsize=FS["pair"], ha=ha, va=va, color=INK, linespacing=1.0, zorder=8)
            bb = t.get_window_extent(rend)
            if not any(bb.overlaps(o) for o in taken) or r > 30:
                break
            t.remove()
            r += step
        taken.append(grow(bb))
        labs.append((p, t, th))
    return labs


B3 = RUNS / "2026-11-06/chores/b3/hero"
BLOCK_KEY = dict(zip(D.PAIRS, ["t2", "4b", "isvg", "i2", "i2s"]))


def drawing(p):
    """B3's rule-picked prompt of each pair (the three decoders write it token for token alike, at the median speed-up
    reading; runs/2026-11-06/chores/b3/README.md), finished drawing as rendered by the official renderer."""
    import json
    cases = json.loads((B3 / "cases.json").read_text())
    k = BLOCK_KEY[p]
    return Image.open(B3 / "finals" / k / f"{cases[k]['pick']}.png").convert("RGB")


def add_drawings(fig, ax, labs, cx, cy, R, rmax, Ht, size=19.0, gap=2.5):
    """One finished drawing beside each pair's name, on the side away from the rose."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    k = 72 / fig.dpi
    for p, t, th in labs:
        bb = t.get_window_extent(rend)
        x0, x1, y0, y1 = bb.x0 * k, bb.x1 * k, bb.y0 * k, bb.y1 * k
        ha = t.get_ha()
        if ha == "right":
            x = x0 - gap - size
            sz = min(size, x0 - gap - 2.5)
            x = x0 - gap - sz
        else:
            x = x1 + gap
            sz = min(size, W - 2.5 - x)
        size_ = sz
        y = (y0 + y1) / 2 - size_ / 2
        a = fig.add_axes([x / W, y / Ht, size_ / W, size_ / Ht])
        a.imshow(np.asarray(drawing(p)), interpolation="lanczos")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("#CFCFCF")
            sp.set_linewidth(0.4)
            sp.set_visible(True)


# ------------------------------------------------------------------------------------------------------------- B
FAN_ORDER = ["original model", "SpS", "REST", "PLD", "ViSpec", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]  # by mean speed-up
FAN_TICK = {"original model": "original model"}
LINE_C = {"greedy": "#9B9B9B", "sample": "#CDCDCD"}


def draw_fan(fig, ax, ytop=14.4, label_ours=True):
    xs = {m: i for i, m in enumerate(FAN_ORDER)}
    rng = np.random.default_rng(0)
    for md in ("sample", "greedy"):                         # greedy on top
        for p, md_, j in D.settings():
            if md_ != md:
                continue
            pts = [(xs["original model"], 1.0)] + [(xs[m], D.value(p, md, j, m)) for m in D.DECODERS
                                                  if D.value(p, md, j, m) is not None]
            x, y = zip(*pts)
            ax.plot(x[:-1], y[:-1], color=LINE_C[md], lw=0.45, zorder=2, solid_capstyle="round")
            ax.plot(x[-2:], y[-2:], color=COL["Ours"] if md == "greedy" else H.C["ours_mid"], lw=0.7, zorder=4,
                    solid_capstyle="round")
    for m in FAN_ORDER:
        vals = [(D.value(p, md, j, m) if m != "original model" else 1.0, md) for p, md, j in D.settings()
                if m == "original model" or D.value(p, md, j, m) is not None]
        for v, md in vals:
            ax.plot([xs[m]], [v], marker="o", ms=2.6 if m != "Ours" else 3.2, mew=0.6, mec=COL[m],
                    mfc=COL[m] if md == "greedy" else "white", ls="none", zorder=5 if m == "Ours" else 3)
    ax.set_xlim(-0.35, len(FAN_ORDER) - 0.65)
    ax.set_ylim(0, ytop)
    ax.set_yticks([1, 4, 8, 12])
    ax.set_yticklabels(["1×", "4×", "8×", "12×"])
    ax.set_xticks(range(len(FAN_ORDER)))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=2.0)
    import matplotlib.transforms as mtrans
    for i, m in enumerate(FAN_ORDER):                    # two staggered rows, so neighbours never touch
        off = mtrans.offset_copy(ax.get_xaxis_transform(), fig=fig, x=0, y=-(4.0 if i % 2 == 0 else 12.5), units="points")
        ax.text(i, 0, FAN_TICK.get(m, m), transform=off, fontsize=FS["tick"], ha="center", va="top",
                color=COL["Ours"] if m == "Ours" else INK, fontweight="bold" if m == "Ours" else "normal")
    ax.tick_params(axis="y", labelsize=FS["tick"], length=2.0, pad=1.5)
    ax.set_ylabel("speed-up over the original model", fontsize=FS["label"], labelpad=2)
    H.ref_line(ax, y=1.0)
    if label_ours:
        o = [D.value(p, md, j, "Ours") for p, md, j in D.settings()]
        for v in (max(o), min(o)):
            t_ = ax.text(len(FAN_ORDER) - 1 + 0.28, v, f"{v:.1f}×", fontsize=8.0, fontweight="bold", color=COL["Ours"],
                         ha="left", va="center", zorder=6, clip_on=False)
    return xs


def fan_legend():
    return [H.item("greedy", LINE_C["greedy"], kind="line", lw=1.2, with_marker=True, marker="o", ms=2.6),
            H.item("official sampling", LINE_C["sample"], kind="line", lw=1.2, with_marker=True, marker="o", hollow=True,
                   ms=2.6)]


def h4b_fan(name="h4b_fan"):
    Ht = 180.0
    fig = fig_pt(W, Ht)
    items = fan_legend()
    lh = H.legend_height(items, W / 72, x_pt=30, width_pt=W - 34, fontsize=FS["legend"], gap_pt=8.0)
    ax = H.axes_pt(fig, 30, 25, W - 30 - 24, Ht - 25 - lh - 3)
    draw_fan(fig, ax)
    H.legend(fig, items, x_pt=30, top_pt=1.5, width_pt=W - 34, fontsize=FS["legend"], gap_pt=8.0)
    save(fig, name)


# ------------------------------------------------------------------------------------------------------------- C
WALL_ROWS = ["PLD", "REST", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]


def draw_wall(fig, ax):
    """Figure 5d's data, drawn for the hero: per decoder, loop-free rounds by the tokens they collect (greedy, pooled over
    the five pairs, each ridge scaled to its own peak); the dark tick is the decoder's window, and the number above it
    is the share of rounds that stop exactly there."""
    n = len(WALL_ROWS)
    xs = np.arange(1, 33)
    for j, m in enumerate(WALL_ROWS):
        base = n - 1 - j
        h = np.array(D.pooled_hist(m)[1:33])
        hh = 0.80 * h / h.max()
        cap = D.CAP[m]
        cols = [COL[m]] * 32
        ax.bar(xs, hh, bottom=base, width=0.92, color=cols, edgecolor="none", zorder=3)
        ax.plot([0.4, 32.6], [base, base], color="#DADADA", lw=0.5, zorder=2)[0].set_gid("ref")
        ax.plot([cap + 0.5, cap + 0.5], [base - 0.12, base + 0.86], color=INK, lw=0.9, zorder=4)
        share = 100 * D.pooled_hist(m)[cap]
        t = ax.text(cap + 0.95, base + 0.5, f"{share:.0f}%", fontsize=7.0, color=COL["Ours"] if m == "Ours" else INK,
                    fontweight="bold" if m in ("Ours", "DDTree") else "normal", ha="left", va="center", zorder=5)
    ax.set_yticks(np.arange(n) + 0.32)
    ax.set_yticklabels(WALL_ROWS[::-1], fontsize=FS["tick"])
    for tl in ax.get_yticklabels():
        if tl.get_text() == "Ours":
            tl.set_color(COL["Ours"])
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.15, n - 0.05)
    ax.set_xlim(0.4, 36.0)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.tick_params(axis="x", labelsize=FS["tick"], length=2.0, pad=1.5)
    ax.set_xlabel("tokens per round, loop-free content", fontsize=FS["label"], labelpad=1.5)


def h4c_wall(name="h4c_wall"):
    top_h, gap, fan_h = 84.0, 27.0, 70.0
    items = [H.item("its window, and the share of rounds that stop there", INK, kind="marker", marker="|", ms=6.0)]
    items2 = fan_legend()
    lh = H.legend_height(items, W / 72, x_pt=40, width_pt=W - 44, fontsize=FS["legend"])
    lh2 = H.legend_height(items2, W / 72, x_pt=30, width_pt=W - 34, fontsize=FS["legend"])
    Ht = lh + 2 + top_h + gap + lh2 + 1 + fan_h + 25
    fig = fig_pt(W, Ht)
    ax1 = H.axes_pt(fig, 40, Ht - lh - 2 - top_h, W - 44, top_h)
    draw_wall(fig, ax1)
    H.legend(fig, items, x_pt=40, top_pt=1.0, width_pt=W - 44, fontsize=FS["legend"])
    y2 = 25
    ax2 = H.axes_pt(fig, 30, y2, W - 30 - 24, fan_h)
    draw_fan(fig, ax2, ytop=14.4)
    ax2.set_ylabel("speed-up", fontsize=FS["label"], labelpad=2)
    H.legend(fig, items2, x_pt=30, top_pt=Ht - y2 - fan_h - lh2 - 1, width_pt=W - 34, fontsize=FS["legend"])
    H.letter(fig, 2, 2, "a")
    H.letter(fig, 2, Ht - y2 - fan_h - lh2 - 2, "b")
    save(fig, name)


# ------------------------------------------------------------------------------------------------------------- D
CLOUD_ORDER = ["PLD", "REST", "SpS", "EAGLE-3", "ViSpec", "DFlash", "DARTree", "DDTree", "Ours"]


def draw_cloud(fig, ax, md, ytop, xtop=26.5):
    for m in CLOUD_ORDER:
        pts = [(D.value(p, md_, j, m, "tau"), D.value(p, md_, j, m)) for p, md_, j in D.settings()
               if md_ == md and D.value(p, md_, j, m) is not None]
        x, y = zip(*pts)
        ax.scatter(x, y, s=19 if m == "Ours" else 9, color=COL[m], edgecolors="white", linewidths=0.25,
                   zorder=5 if m == "Ours" else 3)
    ax.scatter([1], [1], s=14, color=COL["original model"], marker="D", edgecolors="none", zorder=4)
    ax.set_xlim(0, xtop)
    ax.set_ylim(0, ytop)
    H.ref_line(ax, y=1.0)
    ax.tick_params(labelsize=FS["tick"], length=2.0, pad=1.5)


def cloud_labels(fig, ax, md):
    """Each decoder's name beside its cloud: right of the cloud's highest point, nudged up until clear."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pad = 1.5 * fig.dpi / 72
    occupied = []
    for coll in ax.collections:
        off = coll.get_offsets()
        for x, y in ax.transData.transform(off):
            occupied.append((x - 4, y - 4, x + 4, y + 4))
    from matplotlib.transforms import Bbox
    occ = [Bbox.from_extents(*b) for b in occupied]
    placed = []
    spec = {m: None for m in CLOUD_ORDER}
    for m in ["Ours", "DDTree", "DARTree", "DFlash", "EAGLE-3", "ViSpec", "SpS", "REST", "PLD", "original model"]:
        if m == "original model":
            pts = [(1.0, 1.0)]
        else:
            pts = [(D.value(p, md_, j, m, "tau"), D.value(p, md_, j, m)) for p, md_, j in D.settings()
                   if md_ == md and D.value(p, md_, j, m) is not None]
        xs, ys = zip(*pts)
        cands = []
        top = max(pts, key=lambda q: q[1])
        right = max(pts, key=lambda q: q[0])
        cands += [(top[0], top[1], "center", "bottom", 0, 3), (right[0], right[1], "left", "center", 4, 0),
                  (max(xs), float(np.median(ys)), "left", "center", 4, 0), (min(xs), float(np.median(ys)), "right", "center", -4, 0),
                  (float(np.median(xs)), min(ys), "center", "top", 0, -3), (top[0], top[1], "left", "bottom", 3, 2),
                  (top[0], top[1], "right", "bottom", -3, 2)]
        import matplotlib.transforms as mtrans
        done = None
        for (x, y, ha, va, dx, dy) in cands:
            tr = mtrans.offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
            t = ax.text(x, y, m, transform=tr, fontsize=7.0, ha=ha, va=va, zorder=8,
                        color=COL["Ours"] if m == "Ours" else (COL[m] if m in ("DDTree", "DARTree") else INK),
                        fontweight="bold" if m == "Ours" else "normal")
            bb = t.get_window_extent(rend)
            bbp = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
            axbb = ax.get_window_extent(rend)
            inside = bb.x0 >= axbb.x0 + 1 and bb.x1 <= axbb.x1 - 1 and bb.y0 >= axbb.y0 + 1 and bb.y1 <= axbb.y1 - 1
            if inside and not any(bbp.overlaps(o) for o in occ + placed):
                done = t
                placed.append(bbp)
                break
            t.remove()
        if done is None:
            print("cloud label not placed:", m, md)


def h4d_cloud(name="h4d_cloud"):
    ph, gap = 80.0, 10.0
    items = [H.item(NAME[m], COL[m], kind="marker", marker="D" if m == "original model" else "o", ms=4.0)
             for m in ["original model"] + CLOUD_ORDER]
    lh = H.legend_height(items, W / 72, x_pt=3, width_pt=W - 6, fontsize=FS["legend"], gap_pt=6.0)
    Ht = lh + 4 + ph + gap + ph + 24
    fig = fig_pt(W, Ht)
    H.legend(fig, items, x_pt=3, top_pt=1.0, width_pt=W - 6, fontsize=FS["legend"], gap_pt=6.0, align="center")
    axes = []
    for k, (md, ytop) in enumerate((("greedy", 14.6), ("sample", 7.6))):
        y0 = 24 + (1 - k) * (ph + gap)
        ax = H.axes_pt(fig, 30, y0, W - 34, ph)
        draw_cloud(fig, ax, md, ytop)
        ax.set_yticks([1, 4, 8, 12] if md == "greedy" else [1, 2, 4, 6])
        ax.set_yticklabels([f"{v}×" for v in ([1, 4, 8, 12] if md == "greedy" else [1, 2, 4, 6])])
        ax.text(0.02, 0.97, D.MODE_NAME[md], transform=ax.transAxes, fontsize=FS["label"], ha="left", va="top", color=INK)
        if k == 1:
            ax.set_xlabel("tokens per round", fontsize=FS["label"], labelpad=1.5)
        else:
            ax.set_xticklabels([])
        axes.append(ax)
    W_ = fig.get_figwidth() * 72
    fig.text(3 / W_, (24 + ph + gap / 2) / Ht, "speed-up over the original model", rotation=90, fontsize=FS["label"],
             ha="left", va="center")
    save(fig, name)


# ------------------------------------------------------------------------------------------------------- sheets
def pages(entries, out):
    """Each candidate in the right column of a mock first page (8.5 x 11 in, 3.25 in columns), as round 3 did."""
    from PIL import ImageDraw, ImageFont
    dpi = 64
    pw, ph, tm = int(8.5 * dpi), int(11 * dpi), int(1.0 * dpi)
    cw, gap, lm = int(3.25 * dpi), int(0.375 * dpi), int(0.8125 * dpi)
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
    img = Image.new("RGB", (len(entries) * (pw + 16) + 16, ph + 40), "white")
    d0 = ImageDraw.Draw(img)
    for i, (n, png) in enumerate(entries):
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        x_r = lm + cw + gap
        y = tm + int(0.9 * dpi)
        d.rectangle([lm + int(1.2 * dpi), tm, pw - lm - int(1.2 * dpi), tm + 10], fill="#7A7A7A")
        im = Image.open(png).convert("RGB")
        h = int(round(im.height / im.width * 3.25 * dpi))
        page.paste(im.resize((cw, h), Image.LANCZOS), (x_r, y))
        yy = y + h + int(0.1 * dpi)
        for k in range(3):
            d.rectangle([x_r, yy, x_r + (cw if k < 2 else int(cw * 0.6)), yy + 3], fill="#9A9A9A")
            yy += int(0.15 * dpi)
        yy += int(0.12 * dpi)
        while yy < ph - tm:
            d.rectangle([x_r, yy, x_r + cw, yy + 3], fill="#D0D0D0")
            yy += int(0.152 * dpi)
        yl = y
        while yl < ph - tm:
            d.rectangle([lm, yl, lm + cw, yl + 3], fill="#D0D0D0")
            yl += int(0.152 * dpi)
        d.rectangle([0, 0, pw - 1, ph - 1], outline="#BBBBBB")
        x = 16 + i * (pw + 16)
        img.paste(page, (x, 34))
        d0.text((x, 8), f"{n}  {h / dpi:.2f} in", fill="black", font=font)
    img.save(out)


ALL = ["h4a_rose", "h4a2_rose_draw", "h4a_rose60", "h4b_fan", "h4c_wall", "h4d_cloud"]


if __name__ == "__main__":
    import sys as _s
    which = _s.argv[1:] or ["a", "a2", "a60", "b", "c", "d", "sheet"]
    if "a" in which:
        h4a_rose()
    if "a2" in which:
        h4a_rose(name="h4a2_rose_draw", drawings=True)
    if "a60" in which:
        h4a_rose60()
    if "b" in which:
        h4b_fan()
    if "c" in which:
        h4c_wall()
    if "d" in which:
        h4d_cloud()
    if "sheet" in which:
        H.sheet([str(HERE / f"{n}.png") for n in ALL], str(HERE / "_sheet.png"), cols=3,
                title="Hero round 4 (11-10): data-only single-column candidates at print scale")
        pages([("now in the draft", str(HERE.parent / "round2/c1_teaser.png"))] + [(n, str(HERE / f"{n}.png")) for n in ALL],
              HERE / "_pages.png")
