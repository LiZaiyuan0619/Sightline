"""Q1 fair frontier, round 3: the three frontier cells of Figure 7 (main.pdf, d-f) folded into ONE cell.

User, Figure 7 becomes a 2 x 3 (cell a, the loop-control rose, leaves; its numbers are in the merged table and
in Figure 3f), and d-f carry less per square inch than the other cells, so they merge into one. Claim of the cell:
under the same loop rule and the same render-and-pick selection, our best of 8 reaches the original model's best-of-8
quality in 2.6-3.9x less time, and in the time of ONE sample of the original model we can pick from 8-16 candidates
(it can pick from 1-2).
Data: round 1's pack (runs/2026-11-04/figdata_b3/q1_frontier.json, 300 benchmark prompts per block, seconds include
rendering and scoring), through round 2's module; round 2's check_numbers runs first. Nothing is recomputed.
Candidates at the print size of one cell of the 2 x 3 (165 x 160 pt), one chart family each:
  f3a_overlay   the frontier, three blocks on one axis: seconds and quality both relative to the original model
  f3b_dumbbell seconds at the same selection (best of 8), per block, with our best of 16 and one original sample
  f3c_budget    staircase: how many candidates a time budget buys, per block
  f3d_lead_by_n bars: our lead at every N of the same selection
  f3e_strips    the three frontiers kept in their own units, stacked as strips in one cell
Run:  python3 paper/claims/Q1-quality-frontier/round3/make.py [name...]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("q1r2", HERE.parent / "round2/make.py")
Q1 = importlib.util.module_from_spec(spec)
sys.modules["q1r2"] = Q1
spec.loader.exec_module(Q1)
Q1.check_numbers()                       # x3.87 / x3.83 / x2.57; affordable 2/2/1 vs 16/16/8; passes 18/17/28%
H = Q1.H
H.apply()
C, INK, MUTED = H.C, H.INK, H.MUTED
AR, OU = Q1.AR, Q1.OU
BL, t, q = Q1.BLOCKS, Q1.t, Q1.q
ARL, SDL, NOF = Q1.AR_LADDER, Q1.SD_LADDER, Q1.NOF
W_PT, H_PT, LEG = 165.0, 160.0, 7.4
SHORT = dict(zip(BL, ["OmniSVG, text", "OmniSVG, image", "InternSVG, text"]))
MK = dict(zip(BL, ["o", "s", "^"]))
BCOL = dict(zip(BL, ["#C97B95", "#8E9E3C", "#5E7286"]))       # Figure 7e's block colours (Q3 round 2)


def xn(b, a):
    return t(b, a) / t(b, "ar1")


def yn(b, a):
    return q(b, a) / q(b, "ar8s")


def gmean(v):
    return float(np.exp(np.mean(np.log(v))))


def new_fig():
    return plt.figure(figsize=(W_PT / 72, H_PT / 72))


def leg(fig, items, top=2.0, gap=6.0):
    return H.legend(fig, items, x_pt=4.0, top_pt=top, width_pt=W_PT - 6.0, fontsize=LEG, gap_pt=gap)


def save(fig, name):
    H.save(fig, str(HERE / name), width=W_PT / 72)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- a overlay
def overlay_items(full_names=False):
    """full_names: the model names as the rest of Figure 7 prints them (OmniSVG-8B, text)."""
    nm = (lambda b: b.replace("-to-SVG", "")) if full_names else (lambda b: SHORT[b])
    return [H.item("original model + loop control", AR, kind="line", lw=1.1),
            H.item("Ours", OU, kind="line", lw=1.7)] + \
           [H.item(nm(b), INK, kind="marker", marker=MK[b], ms=3.6, hollow=True, mew=0.8) for b in BL]


def draw_overlay(ax):
    """The chosen form (user, ): both axes relative to the original model's best of 8 under the same loop rule
    and scorer, so the three blocks' grey curves end at the same point (1, 1) and our curves reach the same quality
    at 0.26-0.39 of its time; hollow markers at 0 = one sample of the original model (official decoding), where the
    gains are measured from. Used by f3a_overlay and by Figure 7 (EX3-grid round 3)."""
    X = lambda b, a: t(b, a) / t(b, "ar8s")
    Y = lambda b, a: q(b, a) / q(b, "ar8s")
    for b in BL:
        ax.plot([X(b, a) for a in ARL], [Y(b, a) for a in ARL], "-", color=AR, lw=1.0, marker=MK[b], ms=3.0,
                mfc=AR, mec="white", mew=0.4, zorder=3)
        ax.plot([X(b, a) for a in SDL], [Y(b, a) for a in SDL], "-", color=OU, lw=1.5, marker=MK[b], ms=3.4,
                mfc=OU, mec="white", mew=0.4, zorder=4)
        ax.plot([X(b, "ar1")], [0], marker=MK[b], ms=3.6, mfc="white", mec=AR, mew=0.8, ls="none", zorder=5)
    H.ref_line(ax, y=1, lw=0.7)
    xs8 = [X(b, "sd8") for b in BL]
    ax.annotate("", xy=(max(xs8) * 1.10, 1.0), xytext=(0.93, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=OU, lw=1.2, shrinkA=0, shrinkB=0), zorder=6)
    lo, hi = min(t(b, "ar8s") / t(b, "sd8") for b in BL), max(t(b, "ar8s") / t(b, "sd8") for b in BL)
    ax.text(1.9, 1.07, f"×{lo:.1f}–{hi:.1f}", ha="right", va="bottom", fontsize=8.0, color=OU, fontweight="bold",
            zorder=6)
    ax.text(1.07, 0.93, "its best\nof 8", ha="left", va="top", fontsize=7.0, color=AR, linespacing=1.0)
    ax.text(gmean([X(b, "sd16") for b in BL]) * 1.02, 1.30, "16", ha="center", va="bottom", fontsize=7.0, color=OU)
    ax.text(gmean([X(b, "sd1s") for b in BL]), -0.13, "1", ha="center", va="top", fontsize=7.0, color=OU)
    ax.text(gmean([X(b, "ar1") for b in BL]), -0.11, "1 sample", ha="center", va="top", fontsize=7.0, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.055, 2.0)
    ax.set_xticks([0.1, 0.2, 0.5, 1], ["0.1", "0.2", "0.5", "1"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.32, 1.45)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("seconds, × original's best of 8", fontsize=8.0, labelpad=1.5)
    ax.set_ylabel("quality gain,\n× original's best of 8", fontsize=7.5, labelpad=2, linespacing=1.05)
    return ax


def f3a_overlay():
    fig = new_fig()
    bot = leg(fig, overlay_items())
    draw_overlay(H.axes_pt(fig, 40.0, 26.0, W_PT - 46.0, H_PT - bot - 26.0 - 6.0))
    save(fig, "f3a_overlay")


# ----------------------------------------------------------------------------------------------- b dumbbell
def f3b_dumbbell():
    """Seconds (over one original sample) for the same selection, best of 8: the original model with the same loop
    rule (grey) vs ours (blue), the arrow is our lead; our best of 16 (hollow) still fits two blocks into one sample."""
    fig = new_fig()
    items = [H.item("original model + loop control, best of 8", AR, kind="marker", ms=4.4),
             H.item("Ours, best of 8", OU, kind="marker", ms=4.4, bold=True, text_color=OU),
             H.item("Ours, best of 16", OU, kind="marker", ms=4.4, hollow=True)]
    bot = leg(fig, items)
    ax = H.axes_pt(fig, 40.0, 30.0, W_PT - 46.0, H_PT - bot - 30.0 - 6.0)
    for i, b in enumerate(BL):
        ya, ys, y16 = xn(b, "ar8s"), xn(b, "sd8"), xn(b, "sd16")
        ax.annotate("", xy=(i, ys * 1.06), xytext=(i, ya / 1.06),
                    arrowprops=dict(arrowstyle="-|>", color=OU, lw=1.2, shrinkA=0, shrinkB=0), zorder=3)
        ax.plot([i], [ya], "o", ms=5.0, mfc=AR, mec="white", mew=0.5, zorder=4)
        ax.plot([i], [ys], "o", ms=5.0, mfc=OU, mec="white", mew=0.5, zorder=4)
        ax.plot([i + 0.24], [y16], "o", ms=4.6, mfc="white", mec=OU, mew=1.0, zorder=4)
        ax.text(i - 0.07, np.sqrt(ya * ys), f"×{ya / ys:.1f}", ha="right", va="center", fontsize=8.0, color=OU,
                fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=0.1), zorder=5)
    H.ref_line(ax, y=1, lw=0.7)
    ax.text(0.5, 1.04, "1 sample", ha="center", va="bottom", fontsize=7.0, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.28, 2.9)
    ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2]))
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:g}")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.6, 2.45)
    ax.set_xticks(range(3), [SHORT[b].replace(", ", "\n") for b in BL], fontsize=7.0, linespacing=1.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("seconds, × one original sample", fontsize=8.0, labelpad=2)
    save(fig, "f3b_dumbbell")


# ----------------------------------------------------------------------------------------------- c budget
def f3c_budget():
    """How many candidates a time budget (over one original sample) buys: steps up where each best-of-N run ends."""
    fig = new_fig()
    styles = dict(zip(BL, ["-", (0, (3.5, 1.5)), (0, (1, 1.2))]))
    items = [H.item("original model + loop control", AR, kind="line", lw=1.2),
             H.item("Ours", OU, kind="line", lw=1.7)] + \
            [H.item(SHORT[b], INK, kind="line", lw=1.0, ls=styles[b]) for b in BL]
    bot = leg(fig, items)
    ax = H.axes_pt(fig, 34.0, 26.0, W_PT - 40.0, H_PT - bot - 26.0 - 6.0)
    xmax = 2.6
    for b in BL:
        for fam, col, lw in ((ARL, AR, 1.1), (SDL, OU, 1.6)):
            xs = [xn(b, a) for a in fam]
            ns = [NOF[a] for a in fam]
            X = [xs[0]] + xs[1:] + [xmax]
            Y = ns + [ns[-1]]
            ax.step(X, Y, where="post", color=col, lw=lw, ls=styles[b], zorder=3 if col == OU else 2)
    H.ref_line(ax, x=1, lw=0.7)
    ours1 = [Q1.r1.affordable(b, "SD", t(b, "ar1")) for b in BL]
    ar1 = [Q1.r1.affordable(b, "AR", t(b, "ar1")) for b in BL]
    ax.text(1.06, 19.0, "/".join(map(str, ours1)), ha="left", va="bottom", fontsize=7.5, color=OU,
            fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=0.1), zorder=6)
    ax.text(1.06, 0.86, "/".join(map(str, ar1)), ha="left", va="top", fontsize=7.5, color=AR,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.1), zorder=6)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xscale("log")
    ax.set_xlim(0.09, xmax)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.62, 30)
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16]))
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:g}")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("time budget, × one original sample", fontsize=8.0, labelpad=1.5)
    ax.set_ylabel("candidates to pick from", fontsize=8.0, labelpad=2)
    save(fig, "f3c_budget")


# ----------------------------------------------------------------------------------------------- d lead by N
def f3d_lead_by_n():
    """At every N of the same selection (same loop rule, same scorer), the original model's seconds over ours."""
    fig = new_fig()
    items = [H.item(SHORT[b], BCOL[b]) for b in BL]
    bot = leg(fig, items)
    ax = H.axes_pt(fig, 34.0, 26.0, W_PT - 40.0, H_PT - bot - 26.0 - 6.0)
    pairs = [("ar1s", "sd1s"), ("ar2s", "sd2"), ("ar4s", "sd4"), ("ar8s", "sd8")]
    w = 0.26
    for j, b in enumerate(BL):
        v = [t(b, a) / t(b, s) for a, s in pairs]
        xs = np.arange(len(pairs)) + (j - 1) * w
        ax.bar(xs, v, w * 0.9, color=BCOL[b], lw=0, zorder=2)
        for x, y in zip(xs, v):
            if (j == 0 and x < 0.5) or (j == 2 and x > 2.5):
                ax.text(x, y + 0.08, f"{y:.1f}", ha="center", va="bottom", fontsize=7.0, color=INK)
    H.ref_line(ax, y=1, lw=0.7)
    ax.set_ylim(0, 6.0)
    ax.set_yticks([1, 2, 3, 4, 5], ["1×", "2×", "3×", "4×", "5×"])
    ax.set_xticks(range(len(pairs)), [str(NOF[a]) for a, _ in pairs])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlabel("best of N, same rule and scorer", fontsize=8.0, labelpad=1.5)
    ax.set_ylabel("less time than the original", fontsize=8.0, labelpad=2)
    save(fig, "f3d_lead_by_n")


# ----------------------------------------------------------------------------------------------- e strips
def f3e_strips():
    """d-f kept in their own units (CLIP gain, LPIPS drop) and real seconds, stacked as three strips in one cell."""
    fig = new_fig()
    items = [H.item("original model + loop control", AR, kind="line", lw=1.1),
             H.item("Ours", OU, kind="line", lw=1.7),
             H.item("1 sample", AR, kind="marker", marker="s", hollow=True, ms=4.0)]
    bot = leg(fig, items)
    left, right, bottom, gap = 36.0, 40.0, 24.0, 5.0
    sh = (H_PT - bot - bottom - 4.0 - 2 * gap) / 3
    axes = []
    for k, b in enumerate(BL):
        yb = bottom + (2 - k) * (sh + gap)
        ax = H.axes_pt(fig, left, yb, W_PT - left - right, sh)
        ax.plot([t(b, a) for a in ARL], [q(b, a) for a in ARL], "-", color=AR, lw=1.0, marker="o", ms=2.6,
                mfc=AR, mec="white", mew=0.3, zorder=3)
        ax.plot([t(b, a) for a in SDL], [q(b, a) for a in SDL], "-", color=OU, lw=1.5, marker="o", ms=3.0,
                mfc=OU, mec="white", mew=0.3, zorder=4)
        ax.plot([t(b, "ar1")], [0], "s", ms=3.8, mfc="white", mec=AR, mew=0.8, zorder=5)
        yq = q(b, "ar8s")
        ax.annotate("", xy=(t(b, "sd8") * 1.1, yq), xytext=(t(b, "ar8s") * 0.92, yq),
                    arrowprops=dict(arrowstyle="-|>", color=OU, lw=1.0, shrinkA=0, shrinkB=0), zorder=6)

        ax.set_xscale("log")
        ax.set_xlim(1.0, 70)
        hi = max(q(b, a) for a in SDL + ARL)
        ax.set_ylim(-0.25 * hi, 1.25 * hi)
        ax.set_yticks([0, round(hi * 0.8, 2 if hi < 0.1 else 1)])
        ax.yaxis.set_major_formatter(lambda v, _p: f"{v:g}")
        fam, task = SHORT[b].split(", ")
        fig.text((W_PT - right + 3.0) / W_PT, (yb + sh - 1.0) / H_PT,
                 fam + "\n" + task,
                 ha="left", va="top", fontsize=7.0, color=INK, linespacing=1.0)
        fig.text((W_PT - right + 3.0) / W_PT, (yb + 1.0) / H_PT, f"×{t(b, 'ar8s') / t(b, 'sd8'):.1f}",
                 ha="left", va="bottom", fontsize=7.5, color=OU, fontweight="bold")
        if k < 2:
            ax.set_xticks([1, 2, 5, 10, 20, 50], [])
        else:
            ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
            ax.set_xlabel("seconds per prompt", fontsize=8.0, labelpad=1.5)
        ax.xaxis.set_minor_formatter(NullFormatter())
        axes.append(ax)
    fig.text(4.0 / W_PT, (bottom + 1.5 * (sh + gap)) / H_PT, "CLIP / LPIPS gain over 1 sample", rotation=90,
             ha="left", va="center", fontsize=8.0, color=INK)
    save(fig, "f3e_strips")


ALL = [f3a_overlay, f3b_dumbbell, f3c_budget, f3d_lead_by_n, f3e_strips]

if __name__ == "__main__":
    names = sys.argv[1:]
    for fn in ALL:
        if not names or fn.__name__ in names:
            try:
                fn()
            except AssertionError as e:
                print("GEOMETRY", fn.__name__, e)
    pngs = [str(HERE / f"{fn.__name__}.png") for fn in ALL if (HERE / f"{fn.__name__}.png").exists()]
    H.sheet(pngs, str(HERE / "_sheet.png"), cols=3, title="Q1 round 3 (11-10): the frontier in one cell, 165 x 160 pt")
