"""Table 1 as a figure, round 6 (main line session 5): compact layouts to choose from.

User on Figure 4 (round 5's r5_combo_ticks, 6.85 in tall): it takes most of a page, panel b (tokens per round against
speed-up, 2 x 5 scatters) is mostly white, and the form of a is undecided (round 5's ticks or round 4's bars).
"Draw as many forms as you can, I will look at them and then decide" -- so this round draws the candidates at print size.

Panel b repeats a's y (speed-up); what it adds is tokens per round. Every candidate keeps both quantities, both decodings, all
nine methods and all 60 settings. They differ in layout and in the mark:
  r6a_side_ticks    a = round 5's ticks (two rows), narrowed; b = tokens per round against speed-up with the five pairs pooled,
                    one panel per row, on that row's y range
  r6b_side_bars     as r6a, our value drawn as a bar, the baselines as ticks inside it
  r6c_side_dots     as r6a, baselines as dots, a line from the best baseline up to ours
  r6d_split_ticks   two rows, speed-up and tokens per round; inside every pair, greedy and sampling side by side
  r6e_four_ticks    four rows (speed-up and tokens per round, each under greedy and sampling), round 5's ticks
  r6f_four_bars     four rows, round 4's bars (mean of the six benchmarks, one glyph per benchmark)
Rounds 1-5 are untouched; round 2's data layer is imported read-only.
Data and checks: runs/2026-11-02/figdata/table1.json (all tokens); the 396 / 396 pairwise check runs first.
Run:  python3 paper/claims/T1-main-table/round6/make.py [name...]   (STRICT=0 to draft past the geometry check)
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory, ScaledTranslation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402
from sightline.paths import RUNS
sys.path.insert(0, str(RUNS / "2026-11-06/chores/c2"))
import c2lib  # noqa: E402
spec = importlib.util.spec_from_file_location("t1r2", HERE.parent / "round2/make.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402   # stand-in in paper/_house

H.apply()
STRICT = os.environ.get("STRICT", "1") != "0"
SZ = H.SZ
METHODS, OURS, COLOR, BLOCKS, BENCH, MODES = r2.METHODS, r2.OURS, r2.COLOR, r2.BLOCKS, r2.BENCH, r2.MODES
BASE = [m for m in METHODS if m != OURS]
SHORT = {m: m for m in METHODS} | {OURS: "Ours"}
FILL = H.C["ours_light"]          # our lead over the best baseline
BAR = "#E6EDF6"                   # r6b: our bar below the best baseline
SEP = "#E4E4E4"                   # thin line between two pairs
FULL = 495.0
AX_L, AX_R = 40.0, 3.0
HEAD = {b: b.replace(", ", "\n") for b in BLOCKS}
LIM = {("sp", "greedy"): (14.0, [0, 4, 8, 12]), ("sp", "sample"): (7.5, [0, 2, 4, 6]),
       ("tau", "greedy"): (26.0, [0, 10, 20]), ("tau", "sample"): (16.5, [0, 5, 10, 15])}
YLAB = {("sp", "greedy"): "speed-up,\ngreedy", ("sp", "sample"): "speed-up,\nofficial sampling",
        ("tau", "greedy"): "tokens per round,\ngreedy", ("tau", "sample"): "tokens per round,\nofficial sampling"}

SP, TAU = r2.load()
r2.check_claims(SP, TAU)
DATA = {"sp": SP, "tau": TAU}


# ------------------------------------------------------------------------------------------------ columns
class Cols:
    """x positions of the 6 benchmark columns of every pair, for one decoding per row or both side by side."""

    def __init__(self, modes):
        self.modes = list(modes)
        self.gb, self.gm = (1.1, 0.0) if len(self.modes) == 1 else (1.2, 0.6)
        self.lo = -0.65
        self.hi = self.x(4, len(self.modes) - 1, 5) + 0.65

    def x(self, b, g, j):
        per = 6 * len(self.modes) + self.gm * (len(self.modes) - 1)
        return b * (per + self.gb) + g * (6 + self.gm) + j

    def center(self, b, g=None):
        if g is None:
            return (self.x(b, 0, 0) + self.x(b, len(self.modes) - 1, 5)) / 2
        return (self.x(b, g, 0) + self.x(b, g, 5)) / 2

    def seps(self):
        last = len(self.modes) - 1
        return [(self.x(b - 1, last, 5) + self.x(b, 0, 0)) / 2 for b in range(1, 5)]


def marks(ax, q, cols, style, k=1.0):
    """One column per (pair, decoding, benchmark): the baselines and ours in the chosen mark."""
    arr = DATA[q]
    X, Y = [], []
    for b, blk in enumerate(BLOCKS):
        for g, md in enumerate(cols.modes):
            key = (blk, md)
            best = r2.best_baseline(arr, key)
            for j in range(6):
                x, o, bb = cols.x(b, g, j), arr[key][OURS][j], best[j]
                if style in ("ticks", "bars"):
                    if style == "bars":
                        ax.add_patch(Rectangle((x - 0.40, 0), 0.80, bb, facecolor=BAR, edgecolor="none", zorder=1))
                    ax.add_patch(Rectangle((x - 0.40, bb), 0.80, o - bb, facecolor=FILL, edgecolor="none", zorder=1))
                else:
                    ax.vlines(x, bb, o, color=COLOR[OURS], lw=1.5 * k, zorder=4)
                for m in BASE:
                    v = arr[key][m][j]
                    if not np.isfinite(v):
                        continue
                    if style == "dots":
                        ax.scatter([x], [v], s=8.5 * k * k, color=COLOR[m], edgecolor="white", linewidths=0.3,
                                   zorder=3)
                    else:
                        ax.hlines(v, x - 0.31, x + 0.31, color=COLOR[m], lw=(1.2 if m == "DDTree" else 1.0) * k,
                                  zorder=3)
                    X.append(x)
                    Y.append(v)
                if style == "dots":
                    ax.scatter([x], [o], s=17 * k * k, color=COLOR[OURS], edgecolor="white", linewidths=0.4,
                               zorder=5)
                else:
                    ax.hlines(o, x - 0.40, x + 0.40, color=COLOR[OURS], lw=1.9 * k, zorder=5)
                X.append(x)
                Y.append(o)
    return X, Y


def strip(fig, rect_pt, q, cols, style, lim_mode, ylab, k=1.0, lead=True):
    """One full row of columns (all five pairs) for quantity q."""
    ax = H.axes_pt(fig, *rect_pt)
    X, Y = marks(ax, q, cols, style, k)
    H.ref_line(ax, y=1.0, lw=0.6)
    for s in cols.seps():
        ax.axvline(s, color=SEP, lw=0.55, zorder=0)
    ymax, yt = LIM[(q, lim_mode)]
    ax.set_xlim(cols.lo, cols.hi)
    ax.set_ylim(0, ymax)
    r2.fs.assert_inside(ax, X, Y)
    ax.set_yticks(yt)
    ax.set_xticks([])
    ax.tick_params(axis="y", pad=1.5)
    ax.set_ylabel(ylab, fontsize=SZ["label"], labelpad=2.5, linespacing=1.05)
    if lead:
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        for b, blk in enumerate(BLOCKS):
            for g, md in enumerate(cols.modes):
                ax.annotate(f"×{r2.lead(DATA[q], (blk, md)):.2f}", xy=(cols.center(b, g), 1.0), xycoords=tr,
                            xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                            fontsize=SZ["lead"], color=COLOR[OURS], annotation_clip=False)
    return ax


def headers(ax, cols, dy_pt):
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for b, blk in enumerate(BLOCKS):
        ax.annotate(HEAD[blk], xy=(cols.center(b), 1.0), xycoords=tr, xytext=(0, dy_pt), textcoords="offset points",
                    ha="center", va="bottom", fontsize=SZ["header"], linespacing=1.05, annotation_clip=False)


def glyph_row(fig, ax, cols, s, dy_pt=7.0):
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    off = tr + ScaledTranslation(0, -dy_pt / 72, fig.dpi_scale_trans)
    for b in range(5):
        for g in range(len(cols.modes)):
            for j in range(6):
                ax.scatter(cols.x(b, g, j), 0, s=s, linewidths=0.5, clip_on=False, zorder=6, transform=off,
                           **r2.glyph_kw(j))


def group_labels(fig, ax, cols, dy_pt, names):
    """Under the glyph row: a thin line over each decoding's six columns and its name."""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    line = tr + ScaledTranslation(0, -dy_pt / 72, fig.dpi_scale_trans)
    for b in range(5):
        for g, md in enumerate(cols.modes):
            ax.plot([cols.x(b, g, 0) - 0.35, cols.x(b, g, 5) + 0.35], [0, 0], color=H.GROUP_LINE, lw=0.8,
                    clip_on=False, transform=line, zorder=1)
            ax.annotate(names[md], xy=(cols.center(b, g), 0), xycoords=tr, xytext=(0, -dy_pt - 1.5),
                        textcoords="offset points", ha="center", va="top", fontsize=SZ["note"] - 0.5, color=H.MUTED,
                        annotation_clip=False)


def legends(fig, top_pt, x_pt=AX_L, width_pt=None, band="lead over the best baseline", glyphs=True, lead_kind="band"):
    """Row 1: the nine methods and the shaded lead. Row 2: the six benchmark glyphs. Entries spread over the width."""
    meth = [H.item(SHORT[m], COLOR[m]) for m in METHODS]
    lead = [H.item(band, FILL, kind="band") if lead_kind == "band" else H.item(band, COLOR[OURS], kind="line", lw=1.5)]
    gl = [H.item(name, "#262626", kind="marker", marker=r2.glyph_kw(j)["marker"],
                 hollow=r2.glyph_kw(j)["facecolor"] == "#FFFFFF", ms=5.0, mew=0.8) for j, name in enumerate(BENCH)]
    gl = gl if glyphs else []

    def gap(items, w):
        return (w - 2.0 - c2lib._natural_width(fig, items, SZ["legend"])) / (len(items) - 1)

    # the shaded lead goes with the methods if it fits, else with the glyphs, else both rows start at the far left
    for x0, rows in ((x_pt, [meth + lead, gl]), (x_pt, [meth, gl + lead]), (4.0, [meth + lead, gl])):
        w = width_pt or FULL - x0 - AX_R
        rows = [r for r in rows if r]
        if all(gap(r, w) >= 6.0 for r in rows):
            break
    else:
        raise AssertionError("legend rows do not fit")
    y = top_pt
    for items in rows:
        y = H.legend(fig, items, x_pt=x0, top_pt=y, width_pt=w, gap_pt=gap(items, w))
    return y


# point areas (pt^2) in the tokens-per-round scatters. User on r6a: "Can the points in panel b be larger?" -- the first
# draft's 7 / 13 needed a zoom to tell the colours apart
S_BASE, S_OURS = 16.0, 25.0


def side_scatter(fig, rect_pt, mode, xlabel, blocks=BLOCKS, s_base=S_BASE, s_ours=S_OURS, xt=None):
    """Tokens per round against speed-up, the given pairs pooled; same y range as the row of ticks beside it."""
    ax = H.axes_pt(fig, *rect_pt)
    X, Y = [], []
    for m in METHODS:
        xs = np.concatenate([TAU[(b, mode)][m] for b in blocks])
        ys = np.concatenate([SP[(b, mode)][m] for b in blocks])
        ok = np.isfinite(xs) & np.isfinite(ys)
        if not ok.any():
            continue
        ours = m == OURS
        ax.scatter(xs[ok], ys[ok], s=s_ours if ours else s_base, color=COLOR[m], edgecolor="white", linewidths=0.45,
                   zorder=6 if ours else 3)
        X += list(xs[ok])
        Y += list(ys[ok])
    H.ref_line(ax, y=1.0, lw=0.6)
    ymax, yt = LIM[("sp", mode)]
    tmax, tt = LIM[("tau", mode)]
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, ymax)
    r2.fs.assert_inside(ax, X, Y)
    ax.set_yticks(yt)
    ax.set_xticks(xt or tt)
    ax.tick_params(axis="both", pad=1.5)
    if xlabel:
        ax.set_xlabel("tokens per round", fontsize=SZ["label"], labelpad=1.5)
    return ax


def align(fig, axes, ids, rows, cols, name, exempt=None):
    rows = [r for r in (rows or []) if len(r) > 1] or None      # stacked full-width rows: columns only
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=ids, row_groups=rows, column_groups=cols,
        exemptions=exempt or [], json_out=str(HERE / f"{name}.alignment.json"), tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True)


def save(fig, name):
    H.save(fig, str(HERE / name), width="full", strict=STRICT)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------ candidates
def side(name, style):
    """a: two rows of columns (greedy, sampling) on the left; b: tokens per round against speed-up on the right."""
    ROW, B_W, GAP_B = 84.0, 100.0, 24.0
    A_W = FULL - AX_L - AX_R - GAP_B - B_W
    top_row1 = 2 + 20 + 12
    gap = 18.0
    below = 25.0
    leg_top = top_row1 + 2 * ROW + gap + below + 3
    Ht = leg_top + 2 * SZ["legend"] * 1.45 + 2
    fig = plt.figure(figsize=(FULL / 72, Ht / 72))
    cols = Cols(["greedy"])
    k = 0.92
    a1 = strip(fig, (AX_L, Ht - top_row1 - ROW, A_W, ROW), "sp", cols, style, "greedy", YLAB[("sp", "greedy")], k)
    cols_s = Cols(["sample"])
    a2 = strip(fig, (AX_L, Ht - top_row1 - 2 * ROW - gap, A_W, ROW), "sp", cols_s, style, "sample",
               YLAB[("sp", "sample")], k)
    headers(a1, cols, 13.0)
    glyph_row(fig, a2, cols_s, s=17)
    bx = AX_L + A_W + GAP_B
    b1 = side_scatter(fig, (bx, Ht - top_row1 - ROW, B_W, ROW), "greedy", False)
    b2 = side_scatter(fig, (bx, Ht - top_row1 - 2 * ROW - gap, B_W, ROW), "sample", True)
    tr = blended_transform_factory(b1.transAxes, b1.transAxes)
    b1.annotate("all five pairs", xy=(0.5, 1.0), xycoords=tr, xytext=(0, 13.0), textcoords="offset points",
                ha="center", va="bottom", fontsize=SZ["header"], annotation_clip=False)
    legends(fig, leg_top, lead_kind="line" if style == "dots" else "band")
    H.letter(fig, 1.0, 2.0, "a")
    H.letter(fig, bx - 22.0, 2.0, "b")
    align(fig, [a1, a2, b1, b2], ["a1", "a2", "b1", "b2"], [["a1", "b1"], ["a2", "b2"]],
          [["a1", "a2"], ["b1", "b2"]], name,
          [{"panels": ["a1", "a2", "b1", "b2"], "checks": ["panel-width"],
            "reason": "a holds 30 columns per row and b one scatter; unequal widths by design"}])
    save(fig, name)


def r6a_side_ticks():
    side("r6a_side_ticks", "ticks")


def r6b_side_bars():
    side("r6b_side_bars", "bars")


def r6c_side_dots():
    side("r6c_side_dots", "dots")


def r6d_split_ticks():
    """Two rows (speed-up, tokens per round); every pair shows greedy and sampling side by side."""
    ROW = 92.0
    A_W = FULL - AX_L - AX_R
    top_row1 = 2 + 20 + 12
    gap = 18.0
    below = 12 + 13
    leg_top = top_row1 + 2 * ROW + gap + below + 4
    Ht = leg_top + 2 * SZ["legend"] * 1.45 + 2
    fig = plt.figure(figsize=(FULL / 72, Ht / 72))
    cols = Cols(["greedy", "sample"])
    k = 0.85
    a1 = strip(fig, (AX_L, Ht - top_row1 - ROW, A_W, ROW), "sp", cols, "ticks", "greedy", "speed-up", k)
    a2 = strip(fig, (AX_L, Ht - top_row1 - 2 * ROW - gap, A_W, ROW), "tau", cols, "ticks", "greedy",
               "tokens per round", k)
    headers(a1, cols, 13.0)
    glyph_row(fig, a2, cols, s=11, dy_pt=6.0)
    group_labels(fig, a2, cols, 12.0, {"greedy": "greedy", "sample": "sampling"})
    legends(fig, leg_top)
    align(fig, [a1, a2], ["sp", "tau"], [["sp"], ["tau"]], [["sp", "tau"]], "r6d_split_ticks")
    save(fig, "r6d_split_ticks")


FOUR = [("sp", "greedy"), ("sp", "sample"), ("tau", "greedy"), ("tau", "sample")]


def r6e_four_ticks():
    ROW, gap = 64.0, 16.0
    A_W = FULL - AX_L - AX_R
    top_row1 = 2 + 20 + 12
    below = 13.0
    leg_top = top_row1 + 4 * ROW + 3 * gap + below + 4
    Ht = leg_top + 2 * SZ["legend"] * 1.45 + 2
    fig = plt.figure(figsize=(FULL / 72, Ht / 72))
    axes = []
    for r, (q, md) in enumerate(FOUR):
        cols = Cols([md])
        axes.append(strip(fig, (AX_L, Ht - top_row1 - (r + 1) * ROW - r * gap, A_W, ROW), q, cols, "ticks", md,
                          YLAB[(q, md)], 0.95))
    headers(axes[0], Cols(["greedy"]), 13.0)
    glyph_row(fig, axes[-1], Cols(["sample"]), s=20)
    legends(fig, leg_top)
    ids = [f"{q}_{md}" for q, md in FOUR]
    align(fig, axes, ids, [[i] for i in ids], [ids], "r6e_four_ticks")
    save(fig, "r6e_four_ticks")


def r6f_four_bars():
    """Round 4's bars: one bar per method = mean of the six benchmarks, a glyph per benchmark on it."""
    ROW, gap, PG = 64.0, 16.0, 6.0
    A_W = FULL - AX_L - AX_R
    PW = (A_W - 4 * PG) / 5
    top_row1 = 2 + 20 + 12
    leg_top = top_row1 + 4 * ROW + 3 * gap + 6
    Ht = leg_top + 2 * SZ["legend"] * 1.45 + 2
    fig = plt.figure(figsize=(FULL / 72, Ht / 72))
    grid, ids = [], []
    for r, (q, md) in enumerate(FOUR):
        arr = DATA[q]
        ymax, yt = LIM[(q, md)]
        for c, blk in enumerate(BLOCKS):
            ax = H.axes_pt(fig, AX_L + c * (PW + PG), Ht - top_row1 - (r + 1) * ROW - r * gap, PW, ROW)
            key = (blk, md)
            ms = [m for m in METHODS if r2.applies(SP, key, m)]
            X, Y = [], []
            for i, m in enumerate(ms):
                v = arr[key][m]
                ax.bar(i, np.nanmean(v), width=0.84, color=COLOR[m], linewidth=0, zorder=2)
                for j in range(6):
                    x = i + (j - 2.5) * 0.122
                    ax.scatter(x, v[j], s=5.5, linewidths=0.35, zorder=5, **r2.glyph_kw(j))
                    X.append(x)
                    Y.append(v[j])
            H.ref_line(ax, y=1.0, lw=0.6)
            ax.set_xlim(-0.62, len(ms) - 0.38)
            ax.set_ylim(0, ymax)
            r2.fs.assert_inside(ax, X, Y)
            ax.set_xticks([])
            ax.set_yticks(yt)
            ax.tick_params(axis="y", pad=1.5)
            if c == 0:
                ax.set_ylabel(YLAB[(q, md)], fontsize=SZ["label"], labelpad=2.5, linespacing=1.05)
            else:
                ax.set_yticklabels([])
            ax.annotate(f"×{r2.lead(arr, key):.2f}", xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 1.5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=SZ["lead"],
                        color=COLOR[OURS], annotation_clip=False)
            if r == 0:
                ax.annotate(HEAD[blk], xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 13.0),
                            textcoords="offset points", ha="center", va="bottom", fontsize=SZ["header"],
                            linespacing=1.05, annotation_clip=False)
            grid.append(ax)
            ids.append(f"r{r}c{c}")
    rows = [[H.item(SHORT[m], COLOR[m]) for m in METHODS]]       # no shaded lead in this form
    rows.append([H.item(name, "#262626", kind="marker", marker=r2.glyph_kw(j)["marker"],
                        hollow=r2.glyph_kw(j)["facecolor"] == "#FFFFFF", ms=5.0, mew=0.8)
                 for j, name in enumerate(BENCH)])
    y = leg_top
    for items in rows:
        nat = c2lib._natural_width(fig, items, SZ["legend"])
        gp = (FULL - AX_L - AX_R - 2.0 - nat) / (len(items) - 1)
        y = H.legend(fig, items, x_pt=AX_L, top_pt=y, width_pt=FULL - AX_L - AX_R, gap_pt=gp)
    align(fig, grid, ids, [[f"r{r}c{c}" for c in range(5)] for r in range(4)],
          [[f"r{r}c{c}" for r in range(4)] for c in range(5)], "r6f_four_bars")
    save(fig, "r6f_four_bars")


def r6g_appx_pairs():
    """Appendix: round 5's b (tokens per round against speed-up) pair by pair, 2 x 5, at the enlarged point size.
    User: "panel b here already pools the five cases, so I suggest the full version of each case goes to the appendix"."""
    ROW, PG, top = 80.0, 8.0, 2 + 20
    PW = (FULL - AX_L - AX_R - 4 * PG) / 5
    gap = 22.0
    leg_top = top + 2 * ROW + gap + 25 + 3
    Ht = leg_top + SZ["legend"] * 1.45 + 2
    fig = plt.figure(figsize=(FULL / 72, Ht / 72))
    axes, ids = [], []
    for r, md in enumerate(MODES):
        for c, blk in enumerate(BLOCKS):
            ax = side_scatter(fig, (AX_L + c * (PW + PG), Ht - top - (r + 1) * ROW - r * gap, PW, ROW), md, r == 1,
                              blocks=[blk], s_base=14.0, s_ours=22.0)
            if c == 0:
                ax.set_ylabel(YLAB[("sp", md)], fontsize=SZ["label"], labelpad=2.5, linespacing=1.05)
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.annotate(HEAD[blk], xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 3.0),
                            textcoords="offset points", ha="center", va="bottom", fontsize=SZ["header"],
                            linespacing=1.05, annotation_clip=False)
            axes.append(ax)
            ids.append(f"r{r}c{c}")
    items = [H.item(SHORT[m], COLOR[m], kind="marker", ms=5.5, mew=0.0) for m in METHODS]
    nat = c2lib._natural_width(fig, items, SZ["legend"])
    H.legend(fig, items, x_pt=AX_L, top_pt=leg_top, width_pt=FULL - AX_L - AX_R,
             gap_pt=(FULL - AX_L - AX_R - 2.0 - nat) / (len(items) - 1))
    align(fig, axes, ids, [[f"r{r}c{c}" for c in range(5)] for r in range(2)],
          [[f"r{r}c{c}" for r in range(2)] for c in range(5)], "r6g_appx_pairs")
    save(fig, "r6g_appx_pairs")


def pages(entries):
    """Each candidate at the top of a paper page (8.5 x 11 in, text block 6.875 x 8.875 in), a 3-line caption and
    grey two-column text under it; drawn at 64 px per inch, side by side, so the share of the page reads at a glance."""
    from PIL import Image, ImageDraw, ImageFont
    dpi = 64
    pw, ph, lm, tm = int(8.5 * dpi), int(11 * dpi), int(0.8125 * dpi), int(1.0 * dpi)
    tw, th = int(6.875 * dpi), int(8.875 * dpi)
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    ncol = 4
    nrow = (len(entries) + ncol - 1) // ncol
    out = Image.new("RGB", (ncol * (pw + 16) + 16, nrow * (ph + 46)), "white")
    d0 = ImageDraw.Draw(out)
    for i, (n, png) in enumerate(entries):
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        im = Image.open(png).convert("RGB")
        h_in = im.height / im.width * 6.875
        im = im.resize((tw, int(round(h_in * dpi))), Image.LANCZOS)
        page.paste(im, (lm, tm))
        y = tm + im.height + int(0.12 * dpi)
        for k in range(3):                                   # caption, three lines
            d.rectangle([lm, y, lm + (tw if k < 2 else int(tw * 0.6)), y + 4], fill="#9A9A9A")
            y += int(0.16 * dpi)
        y += int(0.18 * dpi)
        cw, gap = int(3.25 * dpi), int(0.375 * dpi)
        while y < tm + th - 4:                               # two columns of body text
            for c in range(2):
                x0 = lm + c * (cw + gap)
                d.rectangle([x0, y, x0 + cw, y + 3], fill="#D0D0D0")
            y += int(0.152 * dpi)
        d.rectangle([0, 0, pw - 1, ph - 1], outline="#BBBBBB")
        x, y0 = 16 + (i % ncol) * (pw + 16), (i // ncol) * (ph + 46)
        out.paste(page, (x, y0 + 40))
        d0.text((x, y0 + 10), f"{n}  {h_in:.2f} in", fill="black", font=font)
    out.save(HERE / "_pages.png")


ALL = [r6a_side_ticks, r6b_side_bars, r6c_side_dots, r6d_split_ticks, r6e_four_ticks, r6f_four_bars]


APPX = [r6g_appx_pairs]


def main(names):
    for fn in APPX:
        if not names or fn.__name__ in names:
            fn()
    for fn in ALL:
        if not names or fn.__name__ in names:
            fn()
    pngs = [str(HERE / f"{fn.__name__}.png") for fn in ALL if (HERE / f"{fn.__name__}.png").exists()]
    H.sheet(pngs, str(HERE / "_sheet.png"), cols=2, title="T1 round 6 (11-09): compact layouts, print scale")
    r5 = HERE.parent / "round5/r5_combo_ticks.png"           # the current Figure 4, for comparison
    pages([("r5_combo_ticks (now)", r5)] + [(Path(p).stem, p) for p in pngs])


if __name__ == "__main__":
    main(sys.argv[1:])
