"""Round 2 candidates for turning table 1 (lossless speed-up) into a figure --  evening.

What round 1 got back from the user:
  - c1 / c2 / c4 pass. c1 and c2: the y axis is too tall, there is white space even above our bar
    (suggested: official sampling up to 10, greedy up to 12.5). c1 and c2 also hide the benchmarks: a reader
    cannot tell that every bar stands for six benchmarks without the appendix table. c4 shows the benchmarks
    but leaves white space too (sampling row) and is less pretty than c1 / c2.
  - "Can it be a radar chart? What else can show 2 modes x 5 blocks x 6 benchmarks x 9 methods?"
  - tokens per round goes into the figure if it is positive. It is: ours has the longest tokens per round in 60 / 60 benchmark cells.

Limits: greedy 0-14, sampling 0-7.5. The data maxima are 13.83 and 7.07, so the suggested 12.5 would cut
three InternSVG-8B text-to-SVG points; every hand-set limit below is asserted (guide SS8).
Round 1 never called fs.apply_style, so it printed in DejaVu Sans with a full frame; round 2 is in Times.

Candidates, one chart family each (guide SS3.2, SS7.3):
  r2a_bars_tight   c1 with tight per-row limits and the six benchmarks as six glyphs inside every bar
  r2b_radar        two radars (greedy | sampling), 30 spokes = 5 blocks x 6 benchmarks, one ring per method
  r2c_radar_rel    one radar, 60 spokes, every value divided by ours in the same cell: ours is the unit circle
  r2d_ticks        the 30 cells side by side per mode, one tick per method, our lead shaded; no lines
  r2e_swarm        the 396 comparisons themselves: baseline / ours per cell, one dot each, all below 1
  r2f_tau_speed    tokens per round on x, speed-up on y, 2 x 5 small multiples, one dot per method x benchmark

Data: runs/2026-11-02/figdata/table1.json (table 1's own caliber: all tokens).
Run:  python3 paper/claims/T1-main-table/round2/make.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import figstyle as fs  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment, PanelAlignmentError  # noqa: E402   # stand-in in paper/_house
from sightline.paths import RUNS

fs.apply_style()

COL_W_IN, FULL_W_IN = 3.25, 6.875           # two-column page, checked against the template on
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = RUNS / "2026-11-02/figdata/table1.json"

OURS = "Ours (lossless)"
# same order and colours as round 1: families together, DDTree (second almost everywhere) next to us
METHODS = ["PLD", "REST", "SpS", "EAGLE-3", "ViSpec", "DFlash", "DARTree", "DDTree", OURS]
SHORT = {m: m for m in METHODS} | {OURS: "Ours"}
COLOR = {
    "PLD": "#D8C6AC", "REST": "#C3A77C", "SpS": "#A98A5B",          # training-free
    "EAGLE-3": "#B4A8D4", "ViSpec": "#8C7CB8",                       # layer-serial drafting
    "DFlash": "#9FC8C8", "DARTree": "#67A3A3", "DDTree": "#2F6D6D",  # block / tree drafting
    OURS: "#1D4E89",
}
OURS_FILL = "#BFD0E6"
BLOCKS = ["OmniSVG-8B, text-to-SVG", "OmniSVG-4B, text-to-SVG", "InternSVG-8B, text-to-SVG",
          "OmniSVG-8B, image-to-SVG", "InternSVG-8B, image-to-SVG"]
BLOCK_LABEL = {b: b.replace(", ", "\n") for b in BLOCKS}
BENCH = ["MMSVG-Icon", "MMSVG-Illus.", "MMSVG-2M-Icon", "MMSVG-2M-Illus.", "SArena-Icon", "SArena-Illus."]
DATASETS = ["MMSVG", "MMSVG-2M", "SArena"]
MODES = ("greedy", "sample")
MODE_LABEL = {"greedy": "greedy", "sample": "official sampling"}
YMAX = {"greedy": 14.0, "sample": 7.5}
YTICKS = {"greedy": [0, 4, 8, 12], "sample": [0, 2, 4, 6]}


# --------------------------------------------------------------------------------------------------
def load():
    """-> sp[(block, mode)][method] and tokens per round[(block, mode)][method]: arrays of 6 (one per benchmark, in BENCH
    order), NaN where the method does not apply (SpS outside OmniSVG-8B text, ViSpec outside image blocks)."""
    raw = json.loads(DATA.read_text())
    assert "all tokens" in raw["meta"]["speedup"], raw["meta"]["speedup"]
    sp, tau = {}, {}
    for b in raw["blocks"]:
        names = [c[0].replace("†", "") for c in b["columns"]]
        assert names == BENCH, names
        key = (b["block"], b["mode"])
        sp[key] = {m: np.array([c["speedup"] if c else np.nan for c in v], float) for m, v in b["rows"].items()}
        tau[key] = {m: np.array([c["tau"] if c else np.nan for c in v], float) for m, v in b["rows"].items()}
    assert sorted(sp) == sorted((blk, md) for blk in BLOCKS for md in MODES)
    return sp, tau


def applies(sp, key, m):
    return bool(np.isfinite(sp[key][m]).any())


def best_baseline(arrs, key):
    """Per benchmark, the best baseline value in this block and mode."""
    return np.nanmax(np.vstack([arrs[key][m] for m in METHODS if m != OURS]), axis=0)


def lead(sp, key):
    """Block-level lead, as in round 1: our mean over the 6 benchmarks / the best baseline's mean."""
    return np.nanmean(sp[key][OURS]) / max(np.nanmean(sp[key][m]) for m in METHODS
                                           if m != OURS and applies(sp, key, m))


def check_claims(sp, tau):
    wins, cells, lo, hi, tlo, thi = 0, 0, 9, 0, 9, 0
    for key in sp:
        for m in METHODS:
            if m == OURS or not applies(sp, key, m):
                continue
            ok = np.isfinite(sp[key][m])
            wins += int((sp[key][OURS][ok] > sp[key][m][ok]).sum())
            cells += int(ok.sum())
        r = sp[key][OURS] / best_baseline(sp, key)
        t = tau[key][OURS] / best_baseline(tau, key)
        lo, hi, tlo, thi = min(lo, r.min()), max(hi, r.max()), min(tlo, t.min()), max(thi, t.max())
    print(f"pairwise wins (block x mode x benchmark x baseline): {wins} / {cells}")
    print(f"speed-up lead over the best baseline, per benchmark cell: x{lo:.3f} to x{hi:.3f}")
    print(f"tau lead over the best baseline, per benchmark cell: x{tlo:.3f} to x{thi:.3f}")
    assert wins == cells == 396, (wins, cells)  # 60 cells x the 6-7 baselines that apply (SpS: 1 block, ViSpec: 2); round 1 said 480 = 60 x 8
    assert lo > 1 and tlo > 1


def align(fig, name):
    """nature-figure's render-time alignment gate. These are round-2 sketches (guide SS7.1: rough is fine),
    so a finding is printed for the record instead of blocking the export."""
    try:
        require_matplotlib_panel_alignment(fig, json_out=str(OUT / f"{name}.alignment.json"))
        print(f"  alignment: PASS ({name})")
    except PanelAlignmentError as e:
        print(f"  alignment: {str(e).splitlines()[0]} ({name})")


def legend_methods(fig, rect, ncol=9, methods=METHODS):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    nrow = int(np.ceil(len(methods) / ncol))
    for i, m in enumerate(methods):
        col, row = i % ncol, i // ncol
        x = 0.004 + col * (1.0 / ncol)
        y = 1 - (row + 0.5) / nrow
        ax.add_patch(Rectangle((x, y - 0.13 / nrow * 2), 0.018 * 9 / ncol, 0.26 / nrow * 2,
                               color=COLOR[m], linewidth=0))
        ax.text(x + 0.024 * 9 / ncol, y, SHORT[m], fontsize=7.0, va="center", ha="left",
                color=COLOR[m] if m == OURS else "#3A3A3A",
                fontweight="bold" if m == OURS else "normal")
    return ax


# --------------------------------------------------------------------------------------------------
GLYPH = ["o", "s", "^"]                      # MMSVG, MMSVG-2M, SArena; filled = icon, hollow = illustration


def glyph_kw(j):
    d, s = j // 2, j % 2
    if s == 0:
        return dict(marker=GLYPH[d], facecolor="#262626", edgecolor="#FFFFFF")
    return dict(marker=GLYPH[d], facecolor="#FFFFFF", edgecolor="#262626")


def r2a_bars_tight(sp, tau):
    """Round 1's c1 with (i) tight limits per row and (ii) the six benchmarks drawn as six glyphs inside
    every bar, always in the same left-to-right order, so each bar visibly stands for six benchmarks."""
    fig = plt.figure(figsize=(FULL_W_IN, 3.05))
    left, right, top, bot, hgap, vgap = 0.066, 0.004, 0.082, 0.148, 0.011, 0.075
    pw = (1 - left - right - 4 * hgap) / 5
    ph = (1 - top - bot - vgap) / 2
    for r, mode in enumerate(MODES):
        for c, blk in enumerate(BLOCKS):
            ax = fig.add_axes([left + c * (pw + hgap), bot + (1 - r) * (ph + vgap), pw, ph])
            key = (blk, mode)
            ms = [m for m in METHODS if applies(sp, key, m)]
            xs_all, ys_all = [], []
            for i, m in enumerate(ms):
                v = sp[key][m]
                ax.bar(i, np.nanmean(v), width=0.84, color=COLOR[m], linewidth=0, zorder=2)
                for j in range(6):
                    x = i + (j - 2.5) * 0.118
                    ax.scatter(x, v[j], s=4.2, linewidths=0.32, zorder=5, **glyph_kw(j))
                    xs_all.append(x)
                    ys_all.append(v[j])
            ax.axhline(1.0, color=fs.REF, linewidth=0.55, linestyle=(0, (3, 2)), zorder=1)
            ax.set_xlim(-0.62, len(ms) - 0.38)
            ax.set_ylim(0, YMAX[mode])
            fs.assert_inside(ax, xs_all, ys_all)
            ax.set_xticks([])
            ax.set_yticks(YTICKS[mode])
            ax.tick_params(axis="y", pad=1.5)
            if c == 0:
                ax.set_ylabel(f"speed-up, {MODE_LABEL[mode]}", fontsize=fs.SIZE["label"], labelpad=2)
            else:
                ax.set_yticklabels([])
            ax.text(0.035, 0.965, f"×{lead(sp, key):.2f}", transform=ax.transAxes, ha="left",
                    va="top", fontsize=8.0, color=COLOR[OURS])
            ax.text(0.035, 0.80, "over best\nbaseline", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.0, color="#6A6A6A", linespacing=1.0)
            if r == 0:
                ax.set_title(BLOCK_LABEL[blk], fontsize=fs.SIZE["tick"], pad=2.5, color=fs.INK,
                             linespacing=1.1)
    legend_methods(fig, [left, 0.055, 1 - left - right, 0.05])
    # glyph key: which benchmark each of the six glyphs is
    ax = fig.add_axes([left, 0.004, 1 - left - right, 0.048])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x = 0.004
    for d, name in enumerate(DATASETS):
        ax.scatter(x + 0.006, 0.5, s=9, linewidths=0.4, **glyph_kw(2 * d))
        ax.scatter(x + 0.020, 0.5, s=9, linewidths=0.4, **glyph_kw(2 * d + 1))
        ax.text(x + 0.031, 0.5, name, fontsize=7.0, va="center", ha="left", color="#3A3A3A")
        x += 0.125
    ax.text(x + 0.01, 0.5, "filled = icon, hollow = illustration; glyphs left to right in this order",
            fontsize=7.0, va="center", ha="left", color="#6A6A6A")
    align(fig, "r2a_bars_tight")
    fs.save(fig, str(OUT / "r2a_bars_tight"), width=FULL_W_IN)


# --------------------------------------------------------------------------------------------------
def spoke_angles(n_sector, per, margin_deg, sub=None):
    """Clockwise from 12 o'clock, in radians, shape (n_sector, per). ``sub`` = (n_first, gap_deg) splits each
    sector into two halves with an extra gap (used by r2c: greedy half, sampling half)."""
    sector = 360.0 / n_sector
    gaps = 0.0 if sub is None else sub[1]
    step = (sector - 2 * margin_deg - gaps) / (per - 1)
    out = np.zeros((n_sector, per))
    for b in range(n_sector):
        for k in range(per):
            extra = gaps if (sub is not None and k >= sub[0]) else 0.0
            out[b, k] = b * sector + margin_deg + k * step + extra
    return np.deg2rad(out)


def pol(r, th):
    return r * np.sin(th), r * np.cos(th)


def radar_frame(ax, rmax, rings, ring_label=None, ref=None):
    th = np.linspace(0, 2 * np.pi, 361)
    for r in rings:
        ax.plot(*pol(r, th), color="#E4E4E4", linewidth=0.45, zorder=0)
    ax.plot(*pol(rmax, th), color="#BDBDBD", linewidth=0.6, zorder=0)
    if ref is not None:
        ax.plot(*pol(ref, th), color=fs.REF, linewidth=0.55, linestyle=(0, (3, 2)), zorder=1)
    if ring_label is not None:
        for r in rings:
            ax.text(0.0, r + 0.012 * rmax, ring_label(r), fontsize=7.0, ha="center", va="bottom",
                    color="#7A7A7A", zorder=9)
    ax.set_aspect("equal")
    ax.axis("off")


def radar_methods(ax, values, ang, rmax):
    """values[m] has the shape of ``ang``; one polyline per (method, sector), never joining two sectors
    (two sectors are two different models -- the same rule as c4)."""
    for row in ang:
        for t in row:
            ax.plot(*pol(np.array([0, rmax]), np.array([t, t])), color="#EFEFEF", linewidth=0.4, zorder=0)
    xs_all, ys_all = [], []
    for m in METHODS:
        if m not in values:
            continue
        ours = m == OURS
        for s in range(ang.shape[0]):
            v, t = values[m][s], ang[s]
            ok = np.isfinite(v)
            if not ok.any():
                continue
            x, y = pol(v[ok], t[ok])
            xs_all += list(x)
            ys_all += list(y)
            if ours:
                ax.fill(np.r_[0, x, 0], np.r_[0, y, 0], color=OURS_FILL, alpha=0.55, linewidth=0, zorder=1)
            ax.plot(x, y, color=COLOR[m], linewidth=1.5 if ours else 0.8, zorder=7 if ours else 3,
                    marker="o", markersize=2.0 if ours else 1.2, markeredgewidth=0)
    return xs_all, ys_all


def r2b_radar(sp, tau):
    """Two radars, greedy | official sampling. 30 spokes each: 5 sectors (models x tasks) x 6 benchmarks.
    The claim reads as: the blue ring is the outermost ring on every one of the 60 spokes."""
    ang = spoke_angles(5, 6, margin_deg=12.0)
    fig = plt.figure(figsize=(FULL_W_IN, 3.30))
    for i, mode in enumerate(MODES):
        ax = fig.add_axes([i * 0.5, 0.115, 0.5, 0.885])
        rmax = YMAX[mode]
        radar_frame(ax, rmax, YTICKS[mode][1:], ring_label=lambda r: f"{r:g}×", ref=1.0)
        vals = {m: np.vstack([sp[(blk, mode)][m] for blk in BLOCKS]) for m in METHODS}
        xs, ys = radar_methods(ax, vals, ang, rmax)
        lim = rmax * 1.64
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim * 0.92, lim * 0.92)
        fs.assert_inside(ax, xs, ys)
        for s, blk in enumerate(BLOCKS):
            mid = ang[s].mean()
            x, y = pol(rmax * 1.07, mid)
            ha = "left" if np.sin(mid) > 0.25 else ("right" if np.sin(mid) < -0.25 else "center")
            va = "bottom" if np.cos(mid) > 0.5 else ("top" if np.cos(mid) < -0.5 else "center")
            ax.text(x, y, BLOCK_LABEL[blk] + f"\n×{lead(sp, (blk, mode)):.2f}", fontsize=7.0,
                    ha=ha, va=va, color=fs.INK, linespacing=1.05)
        ax.text(0.02, 0.985, MODE_LABEL[mode], transform=ax.transAxes, fontsize=8.0, ha="left",
                va="top", color="#3A3A3A")
    legend_methods(fig, [0.03, 0.055, 0.95, 0.045])
    fig.text(0.034, 0.024, "Spokes in every sector, clockwise: MMSVG icon, illus.; MMSVG-2M icon, illus.; "
             "SArena icon, illus.   ×1.27: our lead over the best baseline.", fontsize=7.0, ha="left",
             va="center", color="#6A6A6A")
    align(fig, "r2b_radar")
    fs.save(fig, str(OUT / "r2b_radar"), width=FULL_W_IN)


# --------------------------------------------------------------------------------------------------
def r2c_radar_rel(sp, tau):
    """One radar, 60 spokes: 5 sectors (models x tasks), each split into a greedy half (white) and an official-
    sampling half. Every value is divided by ours in the same cell, so ours is the unit circle and the
    claim reads as: no ring ever touches the circle. Our absolute speed-up is printed under each sector."""
    ang = spoke_angles(5, 12, margin_deg=4.0, sub=(6, 5.0)).reshape(10, 6)
    fig = plt.figure(figsize=(COL_W_IN, 3.95))
    ax = fig.add_axes([0.0, 0.19, 1.0, 0.81])
    radar_frame(ax, 1.0, [0.25, 0.5, 0.75])
    vals = {}
    for m in METHODS:
        rows = []
        for blk in BLOCKS:
            for mode in MODES:
                rows.append(sp[(blk, mode)][m] / sp[(blk, mode)][OURS])
        vals[m] = np.vstack(rows)
    assert np.nanmax(np.vstack([vals[m] for m in METHODS if m != OURS])) < 1.0
    xs, ys = radar_methods(ax, vals, ang, 1.0)
    lim = 1.74
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.93, lim * 0.80)
    fs.assert_inside(ax, xs, ys)
    for b, blk in enumerate(BLOCKS):
        mid = np.r_[ang[2 * b], ang[2 * b + 1]].mean()
        x, y = pol(1.07, mid)
        ha = "left" if np.sin(mid) > 0.25 else ("right" if np.sin(mid) < -0.25 else "center")
        va = "bottom" if np.cos(mid) > 0.5 else ("top" if np.cos(mid) < -0.5 else "center")
        g, s_ = np.nanmean(sp[(blk, "greedy")][OURS]), np.nanmean(sp[(blk, "sample")][OURS])
        ax.text(x, y, BLOCK_LABEL[blk] + f"\nours {g:.1f}× | {s_:.1f}×", fontsize=7.0, ha=ha, va=va,
                color=fs.INK, linespacing=1.05)
    legend_methods(fig, [0.03, 0.055, 0.95, 0.12], ncol=3)
    fig.text(0.03, 0.030, "Radius: speed-up relative to ours in the same cell (rings at 0.25, 0.5,\n"
             "0.75). In every sector, clockwise: greedy half, then official sampling.", fontsize=7.0,
             ha="left", va="center", color="#6A6A6A", linespacing=1.1)
    fs.save(fig, str(OUT / "r2c_radar_rel"), width=COL_W_IN)


# --------------------------------------------------------------------------------------------------
def r2d_ticks(sp, tau):
    """The 30 cells side by side, as in c4, but each method is a short tick instead of a line through the
    cells, ours is the thick blue tick, and the band between the best baseline and ours is shaded. Tight
    limits per row and a two-row label (subset, then benchmark) instead of 30 rotated long names."""
    gap = 1.1

    def cx(b, j):
        return b * (6 + gap) + j

    fig = plt.figure(figsize=(FULL_W_IN, 3.20))
    left, right, top, bot, vgap = 0.066, 0.004, 0.135, 0.150, 0.045
    ph = (1 - top - bot - vgap) / 2
    pw = 1 - left - right
    for r, mode in enumerate(MODES):
        ax = fig.add_axes([left, bot + (1 - r) * (ph + vgap), pw, ph])
        X, Y = [], []
        for b, blk in enumerate(BLOCKS):
            key = (blk, mode)
            best = best_baseline(sp, key)
            for j in range(6):
                x, o = cx(b, j), sp[key][OURS][j]
                ax.add_patch(Rectangle((x - 0.40, best[j]), 0.80, o - best[j], facecolor=OURS_FILL,
                                       edgecolor="none", zorder=1))
                for m in METHODS[:-1]:
                    v = sp[key][m][j]
                    if np.isfinite(v):
                        ax.hlines(v, x - 0.31, x + 0.31, color=COLOR[m],
                                  linewidth=1.15 if m == "DDTree" else 0.95, zorder=3)
                        X.append(x)
                        Y.append(v)
                ax.hlines(o, x - 0.40, x + 0.40, color=COLOR[OURS], linewidth=1.9, zorder=5)
                X.append(x)
                Y.append(o)
            if b:
                ax.axvline(cx(b, 0) - (gap + 1) / 2, color="#E2E2E2", linewidth=0.5, zorder=0)
        ax.axhline(1.0, color=fs.REF, linewidth=0.55, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xlim(-0.65, cx(4, 5) + 0.65)
        ax.set_ylim(0, YMAX[mode])
        fs.assert_inside(ax, X, Y)
        ax.set_yticks(YTICKS[mode])
        ax.tick_params(axis="y", pad=1.5)
        ax.set_ylabel(f"speed-up, {MODE_LABEL[mode]}", fontsize=fs.SIZE["label"], labelpad=2)
        ax.set_xticks([])
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        if r == 0:
            for b, blk in enumerate(BLOCKS):
                g, s_ = lead(sp, (blk, "greedy")), lead(sp, (blk, "sample"))
                ax.text(cx(b, 2.5), 1.13, BLOCK_LABEL[blk], transform=trans, ha="center", va="bottom",
                        fontsize=7.0, color=fs.INK, linespacing=1.05)
                ax.text(cx(b, 2.5), 1.02, f"×{g:.2f} | ×{s_:.2f}", transform=trans,
                        ha="center", va="bottom", fontsize=7.0, color=COLOR[OURS])
        else:
            for b in range(5):
                for j in range(6):
                    ax.scatter(cx(b, j), -0.075, s=9, linewidths=0.4, transform=trans, clip_on=False,
                               **glyph_kw(j))
    legend_methods(fig, [left, 0.050, 1 - left - right, 0.05])
    ax = fig.add_axes([left, 0.004, 1 - left - right, 0.045])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x = 0.004
    for d, name in enumerate(DATASETS):
        ax.scatter(x + 0.006, 0.5, s=9, linewidths=0.4, **glyph_kw(2 * d))
        ax.scatter(x + 0.020, 0.5, s=9, linewidths=0.4, **glyph_kw(2 * d + 1))
        ax.text(x + 0.031, 0.5, name, fontsize=7.0, va="center", ha="left", color="#3A3A3A")
        x += 0.125
    ax.text(x + 0.01, 0.5, "filled = icon, hollow = illustration.   Blue: our lead over the best baseline, "
            "greedy | sampling", fontsize=7.0, va="center", ha="left", color="#6A6A6A")
    align(fig, "r2d_ticks")
    fs.save(fig, str(OUT / "r2d_ticks"), width=FULL_W_IN)


# --------------------------------------------------------------------------------------------------
def darken(hexcol, k=0.35):
    c = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = c * (1 - k)
    return "#" + "".join(f"{int(round(x)):02X}" for x in c)


def beeswarm(y_pt, d_pt):
    """x offsets (pt) for points at heights y_pt so that no two are closer than d_pt."""
    order = np.argsort(y_pt)
    placed, out = [], np.zeros(len(y_pt))
    for i in order:
        for k in range(200):
            off = ((k + 1) // 2) * d_pt * (1 if k % 2 else -1) if k else 0.0
            if all((off - px) ** 2 + (y_pt[i] - py) ** 2 >= d_pt ** 2 for px, py in placed):
                break
        placed.append((off, y_pt[i]))
        out[i] = off
    return out


def r2e_swarm(sp, tau):
    """The 396 comparisons behind 'ours is first in every cell', drawn one by one: each dot is one
    (block, mode, benchmark) cell, its height is that baseline's speed-up divided by ours in the same cell.
    The claim reads as: all 396 dots are below the line at 1. The number on top is the closest approach."""
    base = [m for m in METHODS if m != OURS]
    ratios = {}
    for m in base:
        v, md = [], []
        for key in sp:
            if applies(sp, key, m):
                r = sp[key][m] / sp[key][OURS]
                ok = np.isfinite(r)
                v += list(r[ok])
                md += [key[1]] * int(ok.sum())
        ratios[m] = (np.array(v), np.array(md))
    assert sum(len(ratios[m][0]) for m in base) == 396
    order = sorted(base, key=lambda m: np.median(ratios[m][0]))
    fig = plt.figure(figsize=(FULL_W_IN, 2.35))
    rect = [0.068, 0.105, 0.926, 0.875]
    ax = fig.add_axes(rect)
    ylim = (0.0, 1.11)
    h_pt = rect[3] * 2.35 * 72
    unit_pt = rect[2] * FULL_W_IN * 72 / len(order)
    d_pt, s = 3.15, 7.5
    X, Y = [], []
    for i, m in enumerate(order):
        v, md = ratios[m]
        offs = beeswarm((v - ylim[0]) / (ylim[1] - ylim[0]) * h_pt, d_pt) / unit_pt
        for mode in MODES:
            sel = md == mode
            if mode == "greedy":
                kw = dict(facecolor=COLOR[m], edgecolor=darken(COLOR[m], 0.45), linewidths=0.35)
            else:
                kw = dict(facecolor="#FFFFFF", edgecolor=darken(COLOR[m], 0.25), linewidths=0.75)
            ax.scatter(i + offs[sel], v[sel], s=s, zorder=4, **kw)
        X += list(i + offs)
        Y += list(v)
        ax.text(i, 1.022, f"{v.max():.2f}", ha="center", va="bottom", fontsize=7.0, color="#4A4A4A")
    ax.axhline(1.0, color=COLOR[OURS], linewidth=1.6, zorder=5)
    ax.text(-0.55, 0.985, "Ours = 1", ha="left", va="top", fontsize=7.5, color=COLOR[OURS],
            fontweight="bold")
    ax.text(-0.55, 1.022, "closest", ha="left", va="bottom", fontsize=7.0, color="#7A7A7A")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(*ylim)
    fs.assert_inside(ax, X, Y)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[m] for m in order], fontsize=7.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("speed-up relative to ours", fontsize=fs.SIZE["label"], labelpad=2)
    h = [Line2D([], [], marker="o", linestyle="", markersize=3.3, markerfacecolor="#9A9A9A",
                markeredgecolor="#4A4A4A", markeredgewidth=0.35),
         Line2D([], [], marker="o", linestyle="", markersize=3.3, markerfacecolor="#FFFFFF",
                markeredgecolor="#6A6A6A", markeredgewidth=0.75)]
    ax.legend(h, ["greedy", "official sampling"], loc="lower right", fontsize=7.0, frameon=False,
              handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    fs.save(fig, str(OUT / "r2e_swarm"), width=FULL_W_IN)


# --------------------------------------------------------------------------------------------------
def r2f_tau_speed(sp, tau):
    """Both numbers of every table-1 cell at once: tokens per round on x, speed-up on y, one dot per method x benchmark,
    one panel per block and mode. Ours sits alone in the top-right corner of all ten panels. It also shows
    why tokens per round alone does not rank the methods (EAGLE-3: tokens per round ~10 at ~3x; DARTree: longer tokens per round than DDTree,
    lower speed-up) -- the per-round cost, which is unit P1's subject."""
    tmax = {md: float(np.ceil(max(np.nanmax(tau[(b, md)][m]) for b in BLOCKS for m in METHODS) + 0.8))
            for md in MODES}
    fig = plt.figure(figsize=(FULL_W_IN, 3.25))
    left, right, top, bot, hgap, vgap = 0.066, 0.006, 0.082, 0.205, 0.011, 0.105
    pw = (1 - left - right - 4 * hgap) / 5
    ph = (1 - top - bot - vgap) / 2
    for r, mode in enumerate(MODES):
        for c, blk in enumerate(BLOCKS):
            ax = fig.add_axes([left + c * (pw + hgap), bot + (1 - r) * (ph + vgap), pw, ph])
            key = (blk, mode)
            X, Y = [], []
            for m in METHODS:
                if not applies(sp, key, m):
                    continue
                ours = m == OURS
                ax.scatter(tau[key][m], sp[key][m], s=9 if ours else 5.5, color=COLOR[m],
                           edgecolor="#FFFFFF" if ours else "none", linewidths=0.3, zorder=6 if ours else 3)
                X += list(tau[key][m])
                Y += list(sp[key][m])
            ax.axhline(1.0, color=fs.REF, linewidth=0.55, linestyle=(0, (3, 2)), zorder=1)
            ax.set_xlim(0, tmax[mode])
            ax.set_ylim(0, YMAX[mode])
            fs.assert_inside(ax, X, Y)
            ax.set_yticks(YTICKS[mode])
            ax.set_xticks([0, 5, 10, 15, 20] if tmax[mode] > 20 else [0, 5, 10, 15])
            ax.tick_params(axis="both", pad=1.5)
            if c == 0:
                ax.set_ylabel(f"speed-up, {MODE_LABEL[mode]}", fontsize=fs.SIZE["label"], labelpad=2)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("acceptance length τ", fontsize=fs.SIZE["label"], labelpad=1.5)
            if r == 0:
                ax.set_title(BLOCK_LABEL[blk], fontsize=fs.SIZE["tick"], pad=2.5, color=fs.INK,
                             linespacing=1.1)
    legend_methods(fig, [left, 0.006, 1 - left - right, 0.05])
    align(fig, "r2f_tau_speed")
    fs.save(fig, str(OUT / "r2f_tau_speed"), width=FULL_W_IN)


def main():
    sp, tau = load()
    check_claims(sp, tau)
    for fn in (r2a_bars_tight, r2b_radar, r2c_radar_rel, r2d_ticks, r2e_swarm, r2f_tau_speed):
        fn(sp, tau)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
