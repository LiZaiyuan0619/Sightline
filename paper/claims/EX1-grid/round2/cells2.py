"""The first experiments composite, round 2: the user's picks from round 1, drawn into their composite
boxes with the margins fixed per row, so the plot areas line up.

User, night, on round 1:
  a  Figure 5   c5a_tall (both panels labelled)
  b  Figure 7   keep the radar; give it room by narrowing c
  c  Figure 8   not the arrows (Figure 5 already has arrows and points): c8a or c8d -> c8d, the draft's wrong rate at
                structure vs numbers (the claim of the paragraph; the user's pick of between the same two)
  d  Figure 9   c9a for now; the per-block full version in the appendix is main.pdf's Figure 27 (survival curves).
                Then (same night): "for figure 9 use c9b after all, the ridges" -> d_ridges
  e  Figure 10  keep the stacked bars (c10a). Round 1's cell put greedy and official sampling as an unlabelled upper and
                lower bar; here they are two small panels side by side with their names on top.
Layout (points, 495 wide): a is the left column (185 x 300); row 1 = b (180) + c (130); row 2 = d (150) + e (160).
Data: round 1's data.py through round 1's cells.py (signed-off loaders, asserted).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round1"))
import cells as R1  # noqa: E402

D, H = R1.D, R1.H
C, M, INK, MUTED = R1.C, R1.M, R1.INK, R1.MUTED
R1.LEG_DX = 12.0
LEG_DX = 12.0

FULL, TOP_PAD, ROW_H = 495.0, 2.0, 150.0
FIG_H = 2 * ROW_H + TOP_PAD
A_W, B_W, C_W, D_W, E_W = 185.0, 180.0, 130.0, 156.0, 154.0
T1, B1 = 36.0, 36.0          # row 1: the legend band above the plots; c's two-level block axis below
T2, B2 = 30.0, 25.0          # row 2: legend + e's panel titles above; the x tick labels and title below
BOX = {"a": (0.0, 0.0, A_W, 2 * ROW_H), "b": (A_W, ROW_H, B_W, ROW_H), "c": (A_W + B_W, ROW_H, C_W, ROW_H),
       "d": (A_W, 0.0, D_W, ROW_H), "e": (A_W + D_W, 0.0, E_W, ROW_H)}
KEY = R1.KEY


def legend_over_plot(fig, box, t, items, x_pad=4.0):
    """One legend row sitting right above the plot area (its band is t points tall)."""
    H.legend(fig, items, x_pt=box[0] + x_pad, top_pt=R1.top_of(fig, box) + t - 13.0, width_pt=box[2] - x_pad - 2,
             fontsize=KEY)


# ---------------------------------------------------------------------------------------------------------- a
def a_landscape(fig, box):
    x0, y0, w, h = box
    R1.LEG_DX = 20.0
    bot = R1._land_legend(fig, box)
    R1.LEG_DX = LEG_DX
    assert bot - R1.top_of(fig, box) <= T1 - 2, bot - R1.top_of(fig, box)
    l, r, gap = 31.0, 6.0, 8.0
    top = y0 + h - T1
    ph = top - (y0 + B2) - gap
    h1, h2 = 0.56 * ph, 0.44 * ph
    ax1 = H.axes_pt(fig, x0 + l, y0 + B2 + h2 + gap, w - l - r, h1)
    ax2 = H.axes_pt(fig, x0 + l, y0 + B2, w - l - r, h2)
    R1._land_panel(ax1, "fid", R1.LAB_FID, True)
    ax1.set_ylim(37, 6.5)
    ax1.yaxis.set_major_locator(FixedLocator([10, 20, 30]))
    ax1.set_ylabel("FID", labelpad=2)
    ax1.tick_params(axis="x", labelbottom=False)
    R1.free(ax1.text(0.98, 0.97, "lower is better", transform=ax1.transAxes, ha="right", va="top", fontsize=7.0,
                     color=MUTED))
    R1._land_panel(ax2, "clip", R1.LAB_CLIP, False)
    ax2.set_ylim(18.8, 26.0)
    ax2.yaxis.set_major_locator(FixedLocator([20, 22, 24]))
    ax2.set_ylabel("CLIP-T2I", labelpad=2)
    ax2.set_xlabel("seconds per SVG (log scale)", labelpad=1.5)
    R1.free(ax2.text(0.98, 0.97, "higher is better", transform=ax2.transAxes, ha="right", va="top", fontsize=7.0,
                     color=MUTED))
    return [ax1, ax2]


# ---------------------------------------------------------------------------------------------------------- b
def b_radar(fig, box):
    """Round 1's radar (radius = speed-up as a share of ours) in a wider cell, with our own speed-up range per model
    back under each name as in the signed-off single-column radar."""
    T = R1.T
    bot = R1._t3_legend(fig, box, with_ours=True)
    x0, y0, w, h = box
    leg_h = bot - R1.top_of(fig, box)
    Dd = 98.0
    lo_y, hi_y = y0 + 11.0, y0 + h - leg_h - 12.0          # room for the one-line labels at 0 and 180 degrees
    cy = (lo_y + hi_y) / 2
    W_, Hf = fig.get_figwidth() * 72, fig.get_figheight() * 72
    ax = fig.add_axes([(x0 + (w - Dd) / 2) / W_, (cy - Dd / 2) / Hf, Dd / W_, Dd / Hf], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    sector, step = 60.0, 12.0
    th, meta = [], []
    for i, f in enumerate(T.FAMILIES):
        for j, (mo, k) in enumerate(R1.SPOKES):
            th.append(np.deg2rad(i * sector + (j - 1.5) * step))
            meta.append((f, mo, k))
    th = np.array(th)
    Rr = {m: np.array([R1._ratio(f, m, mo, k) for f, mo, k in meta]) for m in R1.BASES}
    for i in range(len(T.FAMILIES)):
        idx = np.arange(4 * i, 4 * i + 4)
        tt = np.linspace(th[idx[0]] - np.deg2rad(step / 2), th[idx[-1]] + np.deg2rad(step / 2), 80)
        base = np.interp(tt, th[idx], Rr["DDTree"][idx])
        ax.fill_between(tt, np.minimum(base, 1.0), 1.0, color=C["ours_light"], lw=0, zorder=1).set_gid("bg")
        ax.plot(tt, base, color=M["DDTree"], lw=0.9, zorder=3)
    for rr in (0.5, 0.75):
        ax.plot(np.linspace(0, 2 * np.pi, 361), [rr] * 361, color="#E4E4E4", lw=0.5, zorder=0)[0].set_gid("ref")
    ax.plot(np.linspace(0, 2 * np.pi, 361), [1.0] * 361, color=C["ours"], lw=1.5, zorder=5)
    for m in R1.BASES:
        if m != "DDTree":
            ax.scatter(th, Rr[m], s=6.0, color=M[m], edgecolor="none", zorder=4)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    for i, f in enumerate(T.FAMILIES):
        ours = [T.spd(f, "Ours", mo, T.SUBSETS[f][k]) for mo, k in R1.SPOKES]
        rng = f"{min(ours):.1f}–{max(ours):.1f}×"
        ang = i * sector
        if ang in (0, 180):
            txt, ha, va, rr = f"{f}, {rng}", "center", "bottom" if ang == 0 else "top", 1.1
        else:
            name = f.replace("StarVector-", "StarVector\n")
            txt, ha, va, rr = f"{name}\n{rng}", "left" if ang < 180 else "right", "center", 1.12
        R1.free(ax.text(np.deg2rad(ang), rr, txt, ha=ha, va=va, fontsize=7.0, color=INK, linespacing=1.0))
    return ax


# ---------------------------------------------------------------------------------------------------------- c
def c_wrong_bars(fig, box):
    ax = R1.ax_in(fig, box, l=31, b=B1, r=3, t=T1)
    w = 0.36
    for i, (b, *_r) in enumerate(R1.ORDER):
        r = D.wrong_rate(b)
        for dx, key, col in ((-w / 2 - 0.01, "structure", R1.STRUCT_C), (w / 2 + 0.01, "numbers", R1.NUM_C)):
            v = 100 * r[key]
            ax.bar(i + dx, v, width=w, color=col, edgecolor="none", zorder=3)
            s = f"{v:.1f}" if v < 2 else f"{v:.0f}"
            lean = -0.07 if key == "structure" else 0.0
            R1.free(ax.text(i + dx + lean, v + 0.3, s, ha="center", va="bottom", fontsize=7.0))
    ax.set_ylim(0, 12.5)
    ax.yaxis.set_major_locator(FixedLocator([0, 5, 10]))
    ax.set_ylabel("draft wrong (%)", labelpad=2)
    ax.set_xlim(-0.6, len(R1.ORDER) - 0.4)
    R1.block_axis2(ax)
    legend_over_plot(fig, box, T1, [H.item("structure", R1.STRUCT_C), H.item("numbers", R1.NUM_C)])
    return ax


# ---------------------------------------------------------------------------------------------------------- d
def d_dots(fig, box):
    ax = R1.ax_in(fig, box, l=37, b=B2, r=3, t=T2)
    for j, m in enumerate(R1.WM):
        col = M[m]
        for i, md, v in R1._fills(m):
            dx = -0.28 + 0.14 * i
            ax.scatter([j + dx], [v], s=11 if m != "Ours" else 13, marker="o",
                       facecolor=col if md == "greedy" else "white", edgecolor=col,
                       linewidths=0.8, zorder=4 if m == "Ours" else 3)
    ax.set_xlim(-0.6, len(R1.WM) - 0.4)
    ax.set_ylim(0, 58)
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40]))
    ax.set_ylabel("rounds that fill\nthe window (%)", labelpad=2.5, linespacing=1.2)
    R1.method_axis(ax, R1.WM)
    legend_over_plot(fig, box, T2, [H.item("greedy", INK, kind="marker", ms=4.6),
                                    H.item("official sampling", INK, kind="marker", hollow=True, ms=4.6)])
    return ax


def d_ridges(fig, box):
    """Round 1's c9b (user the ridges after all): tokens per round, greedy, pooled over the five blocks,
    each ridge scaled to its own peak; a short dark tick under each ridge marks that decoder's window."""
    ax = R1.ax_in(fig, box, l=33, b=B2, r=4, t=T2)
    n = len(R1.WM)
    xs = np.arange(1, 33)
    for j, m in enumerate(R1.WM):
        base = n - 1 - j
        hgt = np.zeros(32)
        for b, *_r in R1.ORDER:
            hist = D.window(b, "greedy", m)["hist"]
            hgt += np.array([hist.get(int(k), 0.0) for k in xs])
        hgt /= len(R1.ORDER)
        hh = 0.86 * hgt / hgt.max()
        ax.bar(xs, hh, bottom=base, width=1.0, color=M[m], edgecolor="none", alpha=0.95 if m == "Ours" else 0.85,
               zorder=3)
        ax.plot([0.5, 32.5], [base, base], color="#D7D7D7", lw=0.5, zorder=2)[0].set_gid("ref")
        cap = D.CAP[m]
        ax.plot([cap, cap], [base - 0.16, base], color=INK, lw=1.0, zorder=4, clip_on=False)
    ax.set_yticks(np.arange(n) + 0.3)
    ax.set_yticklabels(R1.WM[::-1], fontsize=7.0)
    for tl in ax.get_yticklabels():
        if tl.get_text() == "Ours":
            tl.set_color(M["Ours"])
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.2, n)
    ax.set_xlim(0.5, 32.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 16, 32]))
    ax.set_xlabel("tokens per round", labelpad=1.5)
    legend_over_plot(fig, box, T2, [H.item("the decoder's window", INK, kind="marker", marker="|", ms=6.0)])
    return ax


# ---------------------------------------------------------------------------------------------------------- e
SEG = (("finished", "#C9D3DD"), ("cap", "#E9E1D2"), ("loop", R1.NUM_C))


def e_stacked(fig, box):
    x0, y0, w, h = box
    l, r, gap = 47.0, 7.0, 10.0
    pw = (w - l - r - gap) / 2
    axes = []
    for k, (md, title) in enumerate((("greedy", "greedy"), ("sample", "official sampling"))):
        ax = H.axes_pt(fig, x0 + l + k * (pw + gap), y0 + B2, pw, h - B2 - T2)
        n = len(R1.TGT)
        for j, t in enumerate(R1.TGT):
            y = n - 1 - j
            s = D.loops(t, md)
            left = 0.0
            for key, col in SEG:
                ax.barh(y, 100 * s[key], left=left, height=0.62, color=col, edgecolor="none", zorder=3)
                left += 100 * s[key]
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.55, n - 0.45)
        ax.xaxis.set_major_locator(FixedLocator([0, 50, 100]))
        ax.tick_params(axis="x", labelsize=7.0)
        ax.spines["left"].set_visible(False)
        if k == 0:
            ax.set_yticks(range(n))
            ax.set_yticklabels([R1.TGT_NAME[t] for t in R1.TGT][::-1], fontsize=7.0, linespacing=1.0)
            ax.tick_params(axis="y", length=0, pad=2)
        else:
            ax.set_yticks([])
        R1.free(ax.text(0.5, 1.0, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=KEY))
        axes.append(ax)
    W_ = fig.get_figwidth() * 72
    R1.free(fig.text((x0 + l + pw + gap / 2) / W_, (y0 + 2.0) / (fig.get_figheight() * 72), "outputs (%)",
                     ha="center", va="bottom", fontsize=R1.LAB))
    H.legend(fig, [H.item("finished", SEG[0][1]), H.item("length cap", SEG[1][1]), H.item("in a loop", SEG[2][1])],
             x_pt=x0 + 4 + LEG_DX, top_pt=R1.top_of(fig, box) + 1.0, width_pt=w - 6 - LEG_DX, fontsize=KEY)
    return axes


CELLS = {"a": a_landscape, "b": b_radar, "c": c_wrong_bars, "d": d_ridges, "e": e_stacked}
