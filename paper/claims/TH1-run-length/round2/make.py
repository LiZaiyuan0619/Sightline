"""Proposition 1 (run-length law) evidence figure, round 2 (main line). Single-column candidates.

Claim (one sentence): with 16 candidates per position, the runs of correctly drafted tokens on SVG grow far longer
than the window, so the window decides how much of a round's potential is collected; doubling it pays far more for a
tree than for a chain.

Every candidate reads the SAME data (th2data.py): our block-32 head, 277 prompts with a loop-free part, greedy,
offset-1 labels, 1 candidate per position (chain) vs 16 (tree), windows 16 / 32 / 64 and unlimited.
Run:  python3 paper/claims/TH1-run-length/round2/make.py [names...]   (no names = all; STRICT=0 to draft)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import th2data as T  # noqa: E402

T.check()
H.apply()
SZ, C, INK, MUTED = H.SZ, H.C, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W_PT = H.COL_W * 72                  # 234 pt
LEG_FS = 7.8
COL = {1: "#8C8C8C", 16: C["ours"]}
LIGHT = {1: "#D9D9D9", 16: "#C9D7EA"}
NAME = {1: "1 candidate (chain)", 16: "16 candidates (tree)"}
PP = T.per_prompt()
PO = T.pooled()


def _fig(h_pt):
    return plt.figure(figsize=(W_PT / 72, h_pt / 72))


def _log(axis, ticks, fmt=lambda v: f"{v:g}"):
    getattr(axis.axes, f"set_{axis.axis_name}scale")("log")
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _p: fmt(v)))


def _two_row_axis(ax, xs, top_labels, groups, y_line=-0.115, y_text=-0.135, fs=None):
    """Window values on the first row, the group name under a thin line on the second (guide SS3.1)."""
    ax.set_xticks(xs)
    ax.set_xticklabels(top_labels, fontsize=fs or SZ["tick"])
    for name, (a, b), col in groups:
        ax.plot([a - 0.34, b + 0.34], [y_line, y_line], transform=ax.get_xaxis_transform(), clip_on=False,
                color=col, lw=0.9)
        ax.text((a + b) / 2, y_text, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=(fs or SZ["tick"]) - 0.2, color=col)


# =============================================================================================== candidates
def p2a_runs():
    """Where the tokens sit: share of all tokens in runs at least x tokens long (pooled over the 277 prompts).
    The window lines show how much of each design's text lies in runs the window has to cut."""
    fig = _fig(176)
    items = [H.item(NAME[16], COL[16], kind="line", lw=2.0, bold=True, text_color=COL[16]),
             H.item(NAME[1], COL[1], kind="line", lw=1.5)]
    bot = H.legend(fig, items, x_pt=46, top_pt=2, width_pt=W_PT - 48, fontsize=LEG_FS, gap_pt=10)
    ax = H.axes_pt(fig, 46, 30, W_PT - 46 - 8, 176 - 30 - bot - 4)
    for x in (16, 32):
        ln = ax.axvline(x, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ln.set_gid("ref")
        ax.text(x * 1.06, 4, f"window {x}", rotation=90, ha="left", va="bottom", fontsize=SZ["tick"] - 0.8,
                color=MUTED)
    for k in (1, 16):
        xs, ys, n = T.token_ccdf(k)
        ax.step(xs, 100 * ys, where="post", color=COL[k], lw=2.0 if k == 16 else 1.5, zorder=3 if k == 16 else 2)
        y32 = 100 * T.share_tokens_at_least(k, 32)
        ax.plot([32], [y32], "o", ms=4.2, color=COL[k], zorder=5)
        if k == 16:
            ax.text(38, y32 + 2.5, f"{y32:.0f}%", ha="left", va="bottom", fontsize=SZ["legend"], color=COL[k],
                    fontweight="bold")
        else:   # between the two window lines and under the curve, which is higher there
            ax.text(22.6, y32 - 6.0, f"{y32:.0f}%", ha="center", va="top", fontsize=SZ["legend"], color=COL[k])
    _log(ax.xaxis, [1, 4, 16, 64, 256, 1024])
    ax.set_xlim(1, 1700)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("run length (tokens between two misses)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens in runs at least\nthis long (%)", fontsize=SZ["label"], labelpad=2, linespacing=1.18)
    return fig


def p2b_prompts():
    """Per prompt, the share of the unlimited-window value that a window collects: violin + one dot per prompt,
    median bar on top; chain and tree, windows 16 / 32 / 64."""
    fig = _fig(182)
    ax = H.axes_pt(fig, 50, 40, W_PT - 50 - 4, 182 - 40 - 6)
    rng = np.random.default_rng(1106)
    xs, labs, pos = [], [], 0.0
    spans = {}
    for k in (1, 16):
        start = pos
        for W in T.WINDOWS:
            v = 100 * PP[k]["share"][W]
            vp = ax.violinplot([v], positions=[pos], widths=0.78, showextrema=False, showmedians=False)
            for b in vp["bodies"]:
                b.set_facecolor(LIGHT[k])
                b.set_edgecolor("none")
                b.set_alpha(1.0)
                b.set_gid("bg")
            jit = rng.uniform(-0.2, 0.2, len(v))
            sc = ax.scatter(pos + jit, v, s=2.2, color=COL[k], alpha=0.45, lw=0, zorder=3)
            sc.set_rasterized(True)
            med = float(np.median(v))
            ax.plot([pos - 0.3, pos + 0.3], [med, med], color=INK if k == 1 else C["ours"], lw=1.8, zorder=4,
                    solid_capstyle="butt")
            ax.text(pos, 104, f"{med:.0f}%", ha="center", va="bottom", fontsize=SZ["tick"] - 0.4,
                    color="#5E5E5E" if k == 1 else C["ours"], fontweight="bold" if k == 16 else "normal")
            xs.append(pos)
            labs.append(str(W))
            pos += 1.0
        spans[k] = (start, pos - 1.0)
        pos += 0.6
    ax.set_xlim(-0.6, pos - 1.0)
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    _two_row_axis(ax, xs, labs, [("1 candidate", spans[1], "#6E6E6E"), ("16 candidates", spans[16], C["ours"])])
    ax.text(-0.04, -0.085, "window", transform=ax.transAxes, ha="right", va="center", fontsize=SZ["tick"] - 0.8,
            color=MUTED)
    ax.set_ylabel("collected by the window\n(% of unlimited)", fontsize=SZ["label"], labelpad=4, linespacing=1.18)
    return fig


def p2c_paired():
    """Per prompt, tokens per round at windows 16, 32, 64 and unlimited (one faint line per prompt and design),
    median in bold with the median gain of each doubling."""
    fig = _fig(182)
    items = [H.item(NAME[16], COL[16], kind="line", lw=2.0, bold=True, text_color=COL[16]),
             H.item(NAME[1], COL[1], kind="line", lw=1.6)]
    bot = H.legend(fig, items, x_pt=36, top_pt=2, width_pt=W_PT - 38, fontsize=LEG_FS, gap_pt=10)
    ax = H.axes_pt(fig, 36, 30, W_PT - 36 - 6, 182 - 30 - bot - 4)
    X = [0, 1, 2, 3.25]
    for k in (1, 16):
        M = np.column_stack([PP[k]["tau"][W] for W in T.WINDOWS] + [PP[k]["unlimited"]])
        for row in M:
            ln, = ax.plot(X, row, color=COL[k], lw=0.45, alpha=0.07 if k == 16 else 0.09, zorder=1)
            ln.set_rasterized(True)
            ln.set_gid("bg")
        med = np.median(M, axis=0)
        ax.plot(X, med, color=COL[k], lw=2.0 if k == 16 else 1.6, marker="o", ms=4.0, zorder=4)
        for i, (a, b) in enumerate([(16, 32), (32, 64)]):
            g = float(np.median(PP[k]["gain"][(a, b)]))
            xm = (X[i] + X[i + 1]) / 2
            ym = np.sqrt(med[i] * med[i + 1])
            t = ax.text(xm, ym * (1.30 if k == 16 else 0.74), f"×{g:.2f}", ha="center",
                        va="bottom" if k == 16 else "top", fontsize=SZ["tick"] - 0.4, color=COL[k] if k == 16
                        else "#5E5E5E", fontweight="bold" if k == 16 else "normal")
            t.set_gid("free")
    _log(ax.yaxis, [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_ylim(1.6, 1800)
    ax.set_xlim(-0.3, 3.55)
    ax.set_xticks(X)
    ax.set_xticklabels(["16", "32", "64", "no limit"])
    ax.set_xlabel("window (positions verified per round)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens per round", fontsize=SZ["label"], labelpad=2)
    return fig


def p2d_scatter():
    """Per prompt: x = the prompt's average run length (= what an unlimited window would write per round), y = gain
    from doubling the window 16 -> 32. Chain and tree prompts fall on one curve -- the window's return depends only on
    how long the runs are -- and 16 candidates move the typical prompt up that curve (arrow between the medians)."""
    fig = _fig(178)
    items = [H.item(NAME[16], COL[16], kind="marker", ms=4.0), H.item(NAME[1], COL[1], kind="marker", ms=4.0)]
    bot = H.legend(fig, items, x_pt=40, top_pt=2, width_pt=W_PT - 42, fontsize=LEG_FS, gap_pt=10)
    ax = H.axes_pt(fig, 40, 30, W_PT - 40 - 6, 178 - 30 - bot - 4)
    for y, lab in ((2.0, "the most a doubling can give"), (1.0, "no gain")):
        ln = ax.axhline(y, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ln.set_gid("ref")
        ax.text(1.15, y + (0.02 if y == 2.0 else 0.02), lab, ha="left", va="bottom", fontsize=SZ["tick"] - 0.8,
                color=MUTED)
    for k in (1, 16):
        x = PP[k]["unlimited"]
        y = PP[k]["gain"][(16, 32)]
        sc = ax.scatter(x, y, s=5, color=COL[k], alpha=0.5 if k == 16 else 0.55, lw=0, zorder=3)
        sc.set_rasterized(True)
        # binned median trend on log-x (bins with at least 20 prompts)
        edges = np.geomspace(1, 2000, 14)
        bx, by = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x >= a) & (x < b)
            if m.sum() >= 20:
                bx.append(np.sqrt(a * b))
                by.append(np.median(y[m]))
        ax.plot(bx, by, color=COL[k], lw=2.0 if k == 16 else 1.6, zorder=4)
    # the same prompts, before and after widening: from the median chain prompt to the median tree prompt
    from matplotlib.patches import FancyArrowPatch
    p0 = (float(np.median(PP[1]["unlimited"])), float(np.median(PP[1]["gain"][(16, 32)])))
    p1 = (float(np.median(PP[16]["unlimited"])), float(np.median(PP[16]["gain"][(16, 32)])))
    for (x, y), k in ((p0, 1), (p1, 16)):
        ax.plot([x], [y], marker="D", ms=6.0, mfc="white", mec=COL[k], mew=1.6, zorder=7)
    arr = FancyArrowPatch((p0[0] * 1.18, p0[1] + 0.035), (p1[0] / 1.18, p1[1] - 0.03), arrowstyle="-|>",
                          mutation_scale=9, lw=1.3, color=INK, zorder=6, connectionstyle="arc3,rad=-0.25")
    ax.add_patch(arr)
    ax.text(1.3, 1.66, "median prompt,\n1 → 16 candidates", ha="left", va="center", fontsize=SZ["tick"] - 0.8,
            color=INK, linespacing=1.15)
    _log(ax.xaxis, [1, 4, 16, 64, 256, 1024])
    ax.set_xlim(1, 2000)
    ax.set_ylim(0.95, 2.12)
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"×{v:.2f}".rstrip("0").rstrip(".") if v != 2.0
                                               else "×2"))
    ax.set_xlabel("average run length in the prompt (tokens)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("gain, window 16 → 32", fontsize=SZ["label"], labelpad=2)
    return fig


def p2e_bars():
    """Pooled tokens per round at each window, chain vs tree; the unlimited-window value as a hollow bar (it is the
    reference a finite window cannot reach). Log axis so the chain's short bars stay readable."""
    fig = _fig(178)
    ax = H.axes_pt(fig, 36, 50, W_PT - 36 - 4, 178 - 50 - 6)
    xs, labs, pos = [], [], 0.0
    spans = {}
    for k in (1, 16):
        start = pos
        for W in list(T.WINDOWS) + ["unlimited"]:
            v = PO[k][W]
            hollow = W == "unlimited"
            ax.bar(pos, v, width=0.7, color="white" if hollow else COL[k], ec=COL[k], lw=1.0 if hollow else 0,
                   zorder=2)
            t = ax.text(pos, v * 1.08, f"{v:.1f}" if v < 20 else f"{v:.0f}", ha="center", va="bottom",
                        fontsize=SZ["tick"] - 0.6, color=COL[k] if k == 16 else "#5E5E5E",
                        fontweight="bold" if (k == 16 and W == 32) else "normal")
            t.set_gid("free")
            xs.append(pos)
            labs.append("no\nlimit" if hollow else str(W))
            pos += 1.0
        spans[k] = (start, pos - 1.0)
        pos += 0.6
    ax.set_yscale("log")
    ax.set_ylim(1, 160)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16, 32, 64, 128]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlim(-0.6, pos - 1.0)
    _two_row_axis(ax, xs, labs, [("1 candidate", spans[1], "#6E6E6E"), ("16 candidates", spans[16], C["ours"])],
                  y_line=-0.235, y_text=-0.255, fs=SZ["tick"] - 0.4)
    ax.text(-0.04, -0.075, "window", transform=ax.transAxes, ha="right", va="center", fontsize=SZ["tick"] - 0.8,
            color=MUTED)
    ax.set_ylabel("tokens per round", fontsize=SZ["label"], labelpad=2)
    return fig


CANDIDATES = {"p2a_runs": p2a_runs, "p2b_prompts": p2b_prompts, "p2c_paired": p2c_paired,
              "p2d_scatter": p2d_scatter, "p2e_bars": p2e_bars}


def main(names):
    outs = []
    for n in names or list(CANDIDATES):
        fig = CANDIDATES[n]()
        H.save(fig, str(HERE / n), width="col", strict=STRICT)
        plt.close(fig)
        outs.append(str(HERE / f"{n}.png"))
    if not names:
        H.sheet(outs, str(HERE / "_sheet.png"), cols=2,
                title="TH1 run-length law, round 2: one dataset (block-32 head, 277 prompts, healthy, greedy)")


if __name__ == "__main__":
    main(sys.argv[1:])
