"""Proposition 1 figure, round 3 (main line): polish of the chosen form.

User, p2d is the main panel and p2a its companion; in p2d the dots and the trend lines run together and the
black arrow is hard to see. Changes (data unchanged -- round 2's th2data.py, block-32 head, 277 prompts, loop-free):
  * the two per-design median lines are replaced by ONE trend band shared by all 554 points (% of each log-x bin,
    median as a thin haloed line), drawn under the dots -- "both designs ride one curve" becomes the picture
    (PACT fig. 2a's shared trend band);
  * chain dots hollow grey, tree dots filled blue, so the overlap stays legible;
  * the arrow is gone: marginal densities show the shift instead (top: how long a prompt's runs are; right: the gain),
    with the two medians marked;
  * p3c stacks p2a over the scatter on one shared run-length axis, the window lines running through both.
Run:  python3 paper/claims/TH1-run-length/round3/make.py [names...]   (STRICT=0 to draft)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE.parent / "round2"))
import house as H  # noqa: E402
import th2data as T  # noqa: E402

T.check()
H.apply()
SZ, C, INK, MUTED = H.SZ, H.C, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W_PT = H.COL_W * 72
LEG_FS = 7.8
BLUE, GREY = C["ours"], "#8C8C8C"
BLUE_FILL, GREY_FILL = "#BFD0E6", "#DADADA"
BAND = "#E6E6E6"
PP = T.per_prompt()
X = {k: PP[k]["unlimited"] for k in (1, 16)}
Y = {k: PP[k]["gain"][(16, 32)] for k in (1, 16)}
MED = {k: (float(np.median(X[k])), float(np.median(Y[k]))) for k in (1, 16)}
XLIM, YLIM = (1, 2000), (0.95, 2.1)


def _fig(h_pt):
    return plt.figure(figsize=(W_PT / 72, h_pt / 72))


def _logx(ax, show=True):
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256, 1024]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}" if show else ""))


def _gain_axis(ax):
    ax.set_ylim(*YLIM)
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: {1.0: "×1", 2.0: "×2"}.get(v, f"×{v:g}")))


def _shared_band(ax):
    """One trend for all prompts of both designs: per log-x bin (>= 20 prompts) the 10th-90th percentile band and the
    median. Drawn under the dots."""
    x = np.concatenate([X[1], X[16]])
    y = np.concatenate([Y[1], Y[16]])
    edges = np.geomspace(XLIM[0], XLIM[1], 15)
    bx, lo, md, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 20:
            bx.append(np.sqrt(a * b))
            lo.append(np.percentile(y[m], 10))
            md.append(np.median(y[m]))
            hi.append(np.percentile(y[m], 90))
    band = ax.fill_between(bx, lo, hi, color=BAND, lw=0, zorder=1)
    band.set_gid("bg")
    ln, = ax.plot(bx, md, color="#5A5A5A", lw=1.2, zorder=2.5,
                  path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ln.set_gid("bg")       # the trend may pass under labels like a light band (guide SS4.3)
    return bx, md


def _dots(ax, s=8):
    sc1 = ax.scatter(X[1], Y[1], s=s, facecolors="none", edgecolors=GREY, linewidths=0.6, alpha=0.75, zorder=3)
    sc2 = ax.scatter(X[16], Y[16], s=s, color=BLUE, alpha=0.55, lw=0, zorder=4)
    for sc in (sc1, sc2):
        sc.set_rasterized(True)
    for k in (1, 16):
        ax.plot([MED[k][0]], [MED[k][1]], marker="D", ms=6.2, mfc="white", mec=BLUE if k == 16 else "#6E6E6E",
                mew=1.6, zorder=6)


def _refs(ax, left_x=1.15, short=False, below=False):
    """x2 named at the top left (low-run prompts gain little, so that corner is empty); x1 at the bottom right.
    below=True puts the x2 note in two lines under the line, left of x = 16 (p3c: the window lines cross the top)."""
    ln = ax.axhline(2.0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref")
    if below:
        ax.text(left_x, 1.975, "the most a\ndoubling gives", ha="left", va="top", fontsize=SZ["tick"] - 0.9,
                color=MUTED, linespacing=1.15)
    else:
        ax.text(left_x, 2.022, "the most a doubling gives" if short else "the most a doubling can give", ha="left",
                va="bottom", fontsize=SZ["tick"] - 0.9, color=MUTED)
    ln = ax.axhline(1.0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref")
    ax.text(XLIM[1] * 0.9, 1.022, "no gain", ha="right", va="bottom", fontsize=SZ["tick"] - 0.9, color=MUTED)


def _kde_log(v, grid):
    return gaussian_kde(np.log10(v), bw_method=0.28)(grid)


def _top_marginal(ax, labels=True):
    g = np.linspace(np.log10(XLIM[0]), np.log10(XLIM[1]), 400)
    peak = {}
    for k, col, fill in ((1, GREY, GREY_FILL), (16, BLUE, BLUE_FILL)):
        d = _kde_log(X[k], g)
        d = d / d.max()
        f = ax.fill_between(10 ** g, 0, d, color=fill, lw=0, alpha=0.9 if k == 1 else 0.85, zorder=1 if k == 1 else 2)
        f.set_gid("bg")
        ax.plot(10 ** g, d, color=col, lw=1.0, zorder=3)
        ax.plot([MED[k][0]] * 2, [0, np.interp(np.log10(MED[k][0]), g, d)], color=col, lw=1.0, ls=(0, (2, 1.5)),
                zorder=3)
        peak[k] = (10 ** g[np.argmax(d)], 1.0)
    if labels:   # above each hump: the grey name ends at its peak, the blue one starts at its peak
        ax.text(peak[1][0], 1.06, "1 candidate", ha="right", va="bottom", fontsize=SZ["tick"] - 0.6,
                color="#6E6E6E")
        ax.text(peak[16][0], 1.06, "16 candidates", ha="left", va="bottom", fontsize=SZ["tick"] - 0.6,
                color=BLUE, fontweight="bold")
    _logx(ax, show=False)
    ax.set_ylim(0, 1.42)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)


def _right_marginal(ax):
    g = np.linspace(YLIM[0], YLIM[1], 300)
    for k, col, fill in ((1, GREY, GREY_FILL), (16, BLUE, BLUE_FILL)):
        d = gaussian_kde(Y[k], bw_method=0.25)(g)
        d = d / d.max()
        f = ax.fill_betweenx(g, 0, d, color=fill, lw=0, alpha=0.9 if k == 1 else 0.85, zorder=1 if k == 1 else 2)
        f.set_gid("bg")
        ax.plot(d, g, color=col, lw=1.0, zorder=3)
        ym = MED[k][1]
        ax.plot([0, np.interp(ym, g, d)], [ym, ym], color=col, lw=1.0, ls=(0, (2, 1.5)), zorder=3)
        t = ax.text(1.04, ym, f"×{ym:.2f}", ha="left", va="center", fontsize=SZ["tick"] - 0.6,
                    color=BLUE if k == 16 else "#6E6E6E", fontweight="bold" if k == 16 else "normal")
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, 1.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)


# =============================================================================================== candidates
def p3a_joint():
    """Scatter with one shared trend band, top and right marginal densities (medians dashed, values on the right)."""
    H_PT = 198
    fig = _fig(H_PT)
    L, B, MW, MH, RW, TH = 40, 30, 150, 128, 40, 30
    ax = H.axes_pt(fig, L, B, MW, MH)
    axt = H.axes_pt(fig, L, B + MH + 3, MW, TH)
    axr = H.axes_pt(fig, L + MW + 3, B, RW, MH)
    _shared_band(ax)
    _dots(ax)
    _refs(ax, short=True)
    _logx(ax)
    _gain_axis(ax)
    ax.set_xlabel("average run length in the prompt (tokens)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("gain, window 16 → 32", fontsize=SZ["label"], labelpad=2)
    _top_marginal(axt)
    _right_marginal(axr)
    return fig


def p3b_top():
    """Scatter with the shared trend band and the top marginal only; the two median gains written next to their
    diamonds' row at the right edge of the plot."""
    H_PT = 196
    fig = _fig(H_PT)
    L, B, MW, MH, TH = 40, 30, 186, 128, 30
    ax = H.axes_pt(fig, L, B, MW, MH)
    axt = H.axes_pt(fig, L, B + MH + 3, MW, TH)
    _shared_band(ax)
    _dots(ax)
    _refs(ax)
    for k in (1, 16):
        ym = MED[k][1]
        ln, = ax.plot([MED[k][0] * 1.25, 480], [ym, ym], color=BLUE if k == 16 else "#8C8C8C", lw=0.7,
                      ls=(0, (1, 1.5)), zorder=2)
        ln.set_gid("ref")
        t = ax.text(1900, ym, f"×{ym:.2f}", ha="right", va="bottom", fontsize=SZ["tick"] - 0.5,
                    color=BLUE if k == 16 else "#6E6E6E", fontweight="bold" if k == 16 else "normal")
        t.set_gid("free")
    _logx(ax)
    _gain_axis(ax)
    ax.set_xlabel("average run length in the prompt (tokens)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("gain, window 16 → 32", fontsize=SZ["label"], labelpad=2)
    _top_marginal(axt)
    return fig


def p3c_pair():
    """Two panels on one run-length axis. a: where the tokens sit (share of tokens in runs at least x long, p2a).
    b: per prompt, the gain from doubling the window against the prompt's average run length, one shared trend band.
    The window lines (16, 32) run through both panels."""
    H_PT = 292
    fig = _fig(H_PT)
    items = [H.item("16 candidates (tree)", BLUE, kind="line", lw=2.0, bold=True, text_color=BLUE),
             H.item("1 candidate (chain)", GREY, kind="line", lw=1.5)]
    bot = H.legend(fig, items, x_pt=46, top_pt=2, width_pt=W_PT - 48, fontsize=LEG_FS, gap_pt=10)
    L, B, MW = 46, 30, W_PT - 46 - 6
    HB, HA, GAP = 132, 96, 12
    axb = H.axes_pt(fig, L, B, MW, HB)
    axa = H.axes_pt(fig, L, B + HB + GAP, MW, HA)
    assert B + HB + GAP + HA <= H_PT - bot - 2, (B + HB + GAP + HA, H_PT - bot)
    # a
    for k, col, lw in ((1, GREY, 1.5), (16, BLUE, 2.0)):
        xs, ys, _n = T.token_ccdf(k)
        axa.step(xs, 100 * ys, where="post", color=col, lw=lw, zorder=3 if k == 16 else 2)
        y32 = 100 * T.share_tokens_at_least(k, 32)
        axa.plot([32], [y32], "o", ms=4.0, color=col, zorder=5)
        if k == 16:
            axa.text(38, y32 + 3, f"{y32:.0f}%", ha="left", va="bottom", fontsize=SZ["tick"], color=col,
                     fontweight="bold")
        else:
            axa.text(22.6, y32 - 7, f"{y32:.0f}%", ha="center", va="top", fontsize=SZ["tick"], color=col)
    axa.set_ylim(0, 104)
    axa.set_yticks([0, 50, 100])
    _logx(axa, show=False)
    axa.set_ylabel("tokens in runs\nat least this long (%)", fontsize=SZ["label"] - 0.5, labelpad=2, linespacing=1.18)
    # b
    _shared_band(axb)
    _dots(axb, s=7)
    _refs(axb, below=True)
    for k in (1, 16):          # the two medians, named at the right edge (as in p3b)
        ym = MED[k][1]
        ln, = axb.plot([MED[k][0] * 1.25, 480], [ym, ym], color=BLUE if k == 16 else "#8C8C8C", lw=0.7,
                       ls=(0, (1, 1.5)), zorder=2)
        ln.set_gid("ref")
        t = axb.text(1900, ym, f"×{ym:.2f}", ha="right", va="bottom", fontsize=SZ["tick"] - 0.5,
                     color=BLUE if k == 16 else "#6E6E6E", fontweight="bold" if k == 16 else "normal")
        t.set_gid("free")
    _logx(axb)
    _gain_axis(axb)
    axb.set_xlabel("run length (tokens)", fontsize=SZ["label"] - 0.5, labelpad=1.5)
    axb.text(XLIM[1] * 0.9, 1.10, "one dot per prompt,\nat its average run length", ha="right", va="bottom",
             fontsize=SZ["tick"] - 0.9, color=MUTED, linespacing=1.15)
    axb.set_ylabel("gain, window 16 → 32", fontsize=SZ["label"] - 0.5, labelpad=2)
    # window lines through both panels, named once at the top of a
    for x in (16, 32):
        for a in (axa, axb):
            ln = a.axvline(x, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ln.set_gid("ref")
        axa.text(x * 1.07, 6, f"window {x}", rotation=90, ha="left", va="bottom", fontsize=SZ["tick"] - 1.0,
                 color=MUTED)
    # panel letters level with each axes top, left edge of the figure (guide SS5.2)
    H_top = H_PT
    H.letter(fig, 1.5, H_top - (B + HB + GAP + HA), "a")
    H.letter(fig, 1.5, H_top - (B + HB), "b")
    fig._th1_panels = (axa, axb)
    return fig


CANDIDATES = {"p3a_joint": p3a_joint, "p3b_top": p3b_top, "p3c_pair": p3c_pair}


def main(names):
    outs = []
    for n in names or list(CANDIDATES):
        fig = CANDIDATES[n]()
        if n == "p3c_pair":
            from audit_panel_alignment import require_matplotlib_panel_alignment
            fig.canvas.draw()
            axa, axb = fig._th1_panels      # only the two data panels; the legend overlay spans the whole figure
            require_matplotlib_panel_alignment(fig, json_out=str(HERE / f"{n}.alignment.json"),
                                               tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
                                               axes=[axa, axb], panel_ids=["a", "b"], column_groups=[["a", "b"]])
            print("panel alignment: PASS (a above b, shared left and right edges)")
        H.save(fig, str(HERE / n), width="col", strict=STRICT)
        plt.close(fig)
        outs.append(str(HERE / f"{n}.png"))
    if not names:
        H.sheet(outs, str(HERE / "_sheet.png"), cols=3, title="TH1 round 3: polish of p2d (same data as round 2)")


if __name__ == "__main__":
    main(sys.argv[1:])
