"""Round 4 of table 1 as a figure --  evening. Round 3 stays as it is (guide SS1-14); this round only scales up
the type and the glyphs and adds the bars + tokens per round combination the project decision to see.

User on round 3:
  1. In r3a / r3d / r3_combo the six benchmark glyphs (filled / hollow circle, square, triangle) are hard to see, and
     the method legend is small too. Make them bigger -- and not only the legends: every annotation in the figure,
     the x-axis ticks (the glyph row) and the labels on the bars.
  2. Also stack r3a over r3f as one figure, to compare with r3_combo (r3d over r3f).

Sizes (pt): ticks 8, axis titles 9, legends / headers 8.5, leads 9-9.5 (round 3: 7 / 8 / 7 / 7.5); legend glyphs 4x
the area, the glyph row under the cells ~2.7x, glyphs inside bars ~1.5x.
Files: r4a_bars, r4d_ticks, r4f_tau_speed, r4_combo_ticks (= r4d over r4f), r4_combo_bars (= r4a over r4f).
Data and checks: round 2's (table1.json, all tokens; the 396 / 396 pairwise check runs first).
Run:  python3 paper/claims/T1-main-table/round4/make.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("t1r2", HERE.parent / "round2/make.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
r2.OUT = HERE
fs = r2.fs
OUT = HERE
METHODS, OURS, COLOR, BLOCKS, BLOCK_LABEL = r2.METHODS, r2.OURS, r2.COLOR, r2.BLOCKS, r2.BLOCK_LABEL
MODES, YMAX, YTICKS, BENCH, SHORT = r2.MODES, r2.YMAX, r2.YTICKS, r2.BENCH, r2.SHORT
W = r2.FULL_W_IN
SZ = dict(tick=8.0, label=9.0, legend=8.5, header=8.5, lead=9.0, bar_lead=9.5, note=8.0, letter=12.0,
          g_legend=30, g_axis=24, g_bar=6.5, dot=15, dot_ours=26)
YLAB = {"greedy": "speed-up,\ngreedy", "sample": "speed-up,\nofficial sampling"}
plt.rcParams.update({"xtick.labelsize": SZ["tick"], "ytick.labelsize": SZ["tick"]})


# --------------------------------------------------------------------------------------------------
def legend_methods(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # measured text widths, leftover space shared evenly (as in legend_glyphs)
    texts = [ax.text(0, 0.5, SHORT[m], fontsize=SZ["legend"], va="center", ha="left",
                     color=COLOR[m] if m == OURS else "#2E2E2E", fontweight="bold" if m == OURS else "normal")
             for m in METHODS]
    rend = fig.canvas.get_renderer()
    ax_w = ax.get_window_extent(rend).width
    widths = [t.get_window_extent(rend).width / ax_w for t in texts]
    lead_in = 0.029
    gap = (0.995 - sum(widths) - len(METHODS) * lead_in) / (len(METHODS) - 1)
    assert gap > 0.008, ("method legend does not fit", gap)
    x = 0.004
    for m, t, w in zip(METHODS, texts, widths):
        ax.add_patch(Rectangle((x, 0.22), 0.021, 0.56, color=COLOR[m], linewidth=0))
        t.set_x(x + lead_in)
        x += lead_in + w + gap


def legend_glyphs(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # entry widths come from the measured text widths (capital Ms make "MMSVG-2M-..." much wider than its letter
    # count suggests), and the space that is left is shared out evenly between the entries
    texts = [ax.text(0, 0.5, name, fontsize=SZ["legend"], va="center", ha="left", color="#2E2E2E")
             for name in BENCH]
    rend = fig.canvas.get_renderer()
    ax_w = ax.get_window_extent(rend).width
    widths = [t.get_window_extent(rend).width / ax_w for t in texts]
    lead_in = 0.026                                     # glyph + its gap, in axes fraction
    gap = (0.995 - sum(widths) - len(BENCH) * lead_in) / (len(BENCH) - 1)
    assert gap > 0.008, ("glyph legend does not fit", gap)
    x = 0.004
    for j, (t, w) in enumerate(zip(texts, widths)):
        ax.scatter(x + 0.011, 0.5, s=SZ["g_legend"], linewidths=0.6, **r2.glyph_kw(j))
        t.set_x(x + lead_in)
        x += lead_in + w + gap


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=SZ["letter"], fontweight="bold", ha="left", va="top", color=fs.INK)


# --------------------------------------------------------------------------------------------------
def bars_grid(fig, rects, sp, titles=True):
    """r3a's 2 x 5 bar panels, bigger type; rects[mode] = five rectangles."""
    for r, mode in enumerate(MODES):
        for c, blk in enumerate(BLOCKS):
            ax = fig.add_axes(rects[mode][c])
            key = (blk, mode)
            ms = [m for m in METHODS if r2.applies(sp, key, m)]
            X, Y = [], []
            for i, m in enumerate(ms):
                v = sp[key][m]
                ax.bar(i, np.nanmean(v), width=0.86, color=COLOR[m], linewidth=0, zorder=2)
                for j in range(6):
                    x = i + (j - 2.5) * 0.124
                    ax.scatter(x, v[j], s=SZ["g_bar"], linewidths=0.38, zorder=5, **r2.glyph_kw(j))
                    X.append(x)
                    Y.append(v[j])
            ax.axhline(1.0, color=fs.REF, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
            ax.set_xlim(-0.62, len(ms) - 0.38)
            ax.set_ylim(0, YMAX[mode])
            fs.assert_inside(ax, X, Y)
            ax.set_xticks([])
            ax.set_yticks(YTICKS[mode])
            ax.tick_params(axis="y", pad=1.5)
            if c == 0:
                ax.set_ylabel(YLAB[mode], fontsize=SZ["label"], labelpad=2.5, linespacing=1.05)
            else:
                ax.set_yticklabels([])
            ax.text(0.04, 0.975, f"×{r2.lead(sp, key):.2f}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=SZ["bar_lead"], color=COLOR[OURS])
            ax.text(0.04, 0.735, "over best\nbaseline", transform=ax.transAxes, ha="left", va="top",
                    fontsize=SZ["note"], color="#5E5E5E", linespacing=1.12)
            if r == 0 and titles:
                ax.set_title(BLOCK_LABEL[blk], fontsize=SZ["header"], pad=3, color=fs.INK, linespacing=1.08)


def grid_rects(left, right, top_in, H, pw_gap, row_h, row_gap):
    """five equal columns, two rows; top_in = inches from the top of the figure to the top of row 1."""
    pw = (1 - left - right - 4 * pw_gap) / 5
    out = {}
    for r, mode in enumerate(MODES):
        y_top = top_in + r * (row_h + row_gap)
        out[mode] = [[left + c * (pw + pw_gap), (H - y_top - row_h) / H, pw, row_h / H] for c in range(5)]
    return out


def r4a_bars(sp, tau):
    H = 3.55
    fig = plt.figure(figsize=(W, H))
    rects = grid_rects(0.088, 0.004, 0.40, H, 0.011, 1.12, 0.20)
    bars_grid(fig, rects, sp)
    legend_methods(fig, [0.088, 0.265 / H, 0.908, 0.21 / H])
    legend_glyphs(fig, [0.088, 0.035 / H, 0.908, 0.21 / H])
    fs.save(fig, str(OUT / "r4a_bars"), width=W)


# --------------------------------------------------------------------------------------------------
GAP = 1.1


def cx(b, j):
    return b * (6 + GAP) + j


XLIM = (-0.65, cx(4, 5) + 0.65)


def ticks_rows(fig, rects, sp, header=True):
    axes = []
    for r, (mode, rect) in enumerate(zip(MODES, rects)):
        ax = fig.add_axes(rect)
        axes.append(ax)
        X, Y = [], []
        for b, blk in enumerate(BLOCKS):
            key = (blk, mode)
            best = r2.best_baseline(sp, key)
            for j in range(6):
                x, o = cx(b, j), sp[key][OURS][j]
                ax.add_patch(Rectangle((x - 0.40, best[j]), 0.80, o - best[j], facecolor=r2.OURS_FILL,
                                       edgecolor="none", zorder=1))
                for m in METHODS[:-1]:
                    v = sp[key][m][j]
                    if np.isfinite(v):
                        ax.hlines(v, x - 0.31, x + 0.31, color=COLOR[m],
                                  linewidth=1.25 if m == "DDTree" else 1.05, zorder=3)
                        X.append(x)
                        Y.append(v)
                ax.hlines(o, x - 0.40, x + 0.40, color=COLOR[OURS], linewidth=2.1, zorder=5)
                X.append(x)
                Y.append(o)
            if b:
                ax.axvline(cx(b, 0) - (GAP + 1) / 2, color="#E2E2E2", linewidth=0.55, zorder=0)
        ax.axhline(1.0, color=fs.REF, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, YMAX[mode])
        fs.assert_inside(ax, X, Y)
        ax.set_yticks(YTICKS[mode])
        ax.tick_params(axis="y", pad=1.5)
        ax.set_ylabel(YLAB[mode], fontsize=SZ["label"], labelpad=2.5, linespacing=1.05)
        ax.set_xticks([])
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for b, blk in enumerate(BLOCKS):
            ax.annotate(f"×{r2.lead(sp, (blk, mode)):.2f}", xy=(cx(b, 2.5), 1.0), xycoords=trans,
                        xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=SZ["lead"], color=COLOR[OURS], annotation_clip=False)
        if r == 0 and header:
            for b, blk in enumerate(BLOCKS):
                ax.annotate(BLOCK_LABEL[blk], xy=(cx(b, 2.5), 1.0), xycoords=trans, xytext=(0, 14.5),
                            textcoords="offset points", ha="center", va="bottom", fontsize=SZ["header"],
                            color=fs.INK, linespacing=1.05, annotation_clip=False)
        if r == 1:
            for b in range(5):
                for j in range(6):
                    ax.scatter(cx(b, j), 0, s=SZ["g_axis"], linewidths=0.55, clip_on=False, zorder=6,
                               transform=trans + matplotlib.transforms.ScaledTranslation(0, -8 / 72,
                                                                                         fig.dpi_scale_trans),
                               **r2.glyph_kw(j))
    # the shaded band's legend entry sits inside the OmniSVG-8B image block (most room above its cells), two
    # short lines so it never crosses a block separator; "x: block mean" goes to the caption
    h = [Rectangle((0, 0), 1, 1, facecolor=r2.OURS_FILL, edgecolor="none")]
    x0 = (cx(3, -0.45) - XLIM[0]) / (XLIM[1] - XLIM[0])
    axes[0].legend(h, ["our lead over\nthe best baseline"], loc="upper left", bbox_to_anchor=(x0, 1.0),
                   fontsize=SZ["legend"], frameon=False, handlelength=1.3, handleheight=1.7, borderaxespad=0.15,
                   handletextpad=0.4)
    return axes


def ticks_rects(left, right, top_in, H, row_h, row_gap):
    pw = 1 - left - right
    return [[left, (H - top_in - row_h) / H, pw, row_h / H],
            [left, (H - top_in - 2 * row_h - row_gap) / H, pw, row_h / H]]


def r4d_ticks(sp, tau):
    H = 3.80
    fig = plt.figure(figsize=(W, H))
    left = 0.088
    ticks_rows(fig, ticks_rects(left, 0.004, 0.52, H, 1.12, 0.24), sp)
    legend_methods(fig, [left, 0.265 / H, 0.908, 0.21 / H])
    legend_glyphs(fig, [left, 0.035 / H, 0.908, 0.21 / H])
    fs.save(fig, str(OUT / "r4d_ticks"), width=W)


# --------------------------------------------------------------------------------------------------
def scatter_grid(fig, rects, sp, tau, titles=True):
    tmax = {md: float(np.ceil(np.nanmax([np.nanmax(tau[(b, md)][m]) for b in BLOCKS for m in METHODS
                                         if r2.applies(sp, (b, md), m)]) + 0.8)) for md in MODES}
    for r, mode in enumerate(MODES):
        for c, blk in enumerate(BLOCKS):
            ax = fig.add_axes(rects[mode][c])
            key = (blk, mode)
            X, Y = [], []
            for m in METHODS:
                if not r2.applies(sp, key, m):
                    continue
                ours = m == OURS
                ax.scatter(tau[key][m], sp[key][m], s=SZ["dot_ours"] if ours else SZ["dot"], color=COLOR[m],
                           edgecolor="#FFFFFF", linewidths=0.5, zorder=6 if ours else 3)
                X += list(tau[key][m])
                Y += list(sp[key][m])
            ax.axhline(1.0, color=fs.REF, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
            ax.set_xlim(0, tmax[mode])
            ax.set_ylim(0, YMAX[mode])
            fs.assert_inside(ax, X, Y)
            ax.set_yticks(YTICKS[mode])
            ax.set_xticks([0, 10, 20] if tmax[mode] > 20 else [0, 5, 10, 15])
            ax.tick_params(axis="both", pad=1.5)
            if c == 0:
                ax.set_ylabel(YLAB[mode], fontsize=SZ["label"], labelpad=2.5, linespacing=1.05)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("acceptance length τ", fontsize=SZ["label"], labelpad=1.5)
            if r == 0 and titles:
                ax.set_title(BLOCK_LABEL[blk], fontsize=SZ["header"], pad=3, color=fs.INK, linespacing=1.08)


def r4f_tau_speed(sp, tau):
    H = 3.45
    fig = plt.figure(figsize=(W, H))
    rects = grid_rects(0.088, 0.006, 0.40, H, 0.011, 1.05, 0.30)
    scatter_grid(fig, rects, sp, tau)
    legend_methods(fig, [0.088, 0.03 / H, 0.908, 0.21 / H])
    fs.save(fig, str(OUT / "r4f_tau_speed"), width=W)


# --------------------------------------------------------------------------------------------------
def r4_combo_ticks(sp, tau):
    """a = r4d (speed-up per benchmark), b = r4f (tokens per round vs speed-up), b's columns under a's block segments."""
    H = 6.85
    fig = plt.figure(figsize=(W, H))
    left = 0.088
    pw = 1 - left - 0.004
    top_a, row_a, gap_a = 0.52, 1.12, 0.24
    ticks_rows(fig, ticks_rects(left, 0.004, top_a, H, row_a, gap_a), sp)
    span = XLIM[1] - XLIM[0]
    cols = [(left + pw * (cx(b, -0.45) - XLIM[0]) / span, pw * 5.9 / span) for b in range(5)]  # -0.45.. 5.45
    top_b = top_a + 2 * row_a + gap_a + 0.62          # glyph row + room for the panel letter
    row_b, gap_b = 1.02, 0.30
    rects = {mode: [[x0, (H - top_b - r * (row_b + gap_b) - row_b) / H, w, row_b / H] for x0, w in cols]
             for r, mode in enumerate(MODES)}
    scatter_grid(fig, rects, sp, tau, titles=False)
    letter(fig, 0.004, 1 - (top_a - 0.02) / H, "a")
    letter(fig, 0.004, 1 - (top_b - 0.20) / H, "b")
    legend_methods(fig, [left, 0.265 / H, 0.908, 0.21 / H])
    legend_glyphs(fig, [left, 0.035 / H, 0.908, 0.21 / H])
    fs.save(fig, str(OUT / "r4_combo_ticks"), width=W)


def r4_combo_bars(sp, tau):
    """a = r4a (bars, glyphs inside), b = r4f (tokens per round vs speed-up); both are 5-column grids on the same columns."""
    H = 6.65
    fig = plt.figure(figsize=(W, H))
    left, right, gap = 0.088, 0.006, 0.011
    top_a, row_a, gap_a = 0.40, 1.12, 0.20
    bars_grid(fig, grid_rects(left, right, top_a, H, gap, row_a, gap_a), sp)
    top_b = top_a + 2 * row_a + gap_a + 0.42
    scatter_grid(fig, grid_rects(left, right, top_b, H, gap, 1.02, 0.30), sp, tau, titles=False)
    letter(fig, 0.004, 1 - (top_a - 0.05) / H, "a")
    letter(fig, 0.004, 1 - (top_b - 0.12) / H, "b")
    legend_methods(fig, [left, 0.265 / H, 0.906, 0.21 / H])
    legend_glyphs(fig, [left, 0.035 / H, 0.906, 0.21 / H])
    fs.save(fig, str(OUT / "r4_combo_bars"), width=W)


def main():
    sp, tau = r2.load()
    r2.check_claims(sp, tau)
    for fn in (r4a_bars, r4d_ticks, r4f_tau_speed, r4_combo_ticks, r4_combo_bars):
        fn(sp, tau)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
