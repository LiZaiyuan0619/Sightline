"""The third experiments composite (main.pdf Figures 7-13 as of night), round 1: one 3 x 3 figure.

User, compose these as a 3 x 3 the way the main line proposed. Each cell is redrawn at its print size from the
signed-off figure's own data layer (imported, its checks run on import); nothing is recomputed.
  row 1  a  Figure 7   loop control for every decoder (L round 3's rose, its own helpers, a 104 pt disc)
         b  Figure 8   why escape then stop (L round 3's two strips)
         c  Figure 10  target forward passes within one original sample (Q1 round 2)
  row 2  d-f Figure 9  the quality frontier (Q1 round 2's row, same geometry and labels)
  row 3  g  Figure 11  three VLM judges (Q2 round 1): short row names, the seconds column dropped (Figures 9 and 10 show time)
         h  Figure 12  selection must use the rendering (Q3 round 2)
         i  Figure 13  window against concurrency (P5 round 2's heatmap, tighter rows)
Run:  python3 paper/claims/EX3-grid/round1/make.py   (STRICT=0 to draft past the geometry check)
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
CLAIMS = HERE.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402   # stand-in in paper/_house


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load("l3", CLAIMS / "L-loop-control/round3/make.py")
Q1 = _load("q1r2", CLAIMS / "Q1-quality-frontier/round2/make.py")
Q2 = _load("q2r1", CLAIMS / "Q2-vlm-judge/round1/make.py")
Q3 = _load("q3r2", CLAIMS / "Q3-render-selection/round2/make.py")
P5 = _load("p5r2", CLAIMS / "P5-serving/round2/make.py")
H = L.H
H.apply()
SZ, C, M, INK, MUTED = H.SZ, H.C, H.METHOD, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
NAME = "ex3_grid"
FULL, CW = 495.0, 165.0
R1, R2, R3 = 150.0, 172.0, 175.0             # row heights, top to bottom
TOP = 2.0
FIG_H = TOP + R1 + R2 + R3
LEG = 7.4                                     # legend size inside a cell
# the frontier row's plot columns (Q1 round 2's geometry); rows 1 and 3 line up with them where they can
FL, FG, FR = 58.0, 58.0, 6.0
PW = (FULL - FL - FR - 2 * FG) / 3


def y_row(k):
    return {1: R3 + R2, 2: R3, 3: 0.0}[k]


def top_of_row(k):
    return FIG_H - (y_row(k) + {1: R1, 2: R2, 3: R3}[k])


def leg(fig, items, x_pt, top_pt, width_pt, gap=6.0):
    return H.legend(fig, items, x_pt=x_pt, top_pt=top_pt, width_pt=width_pt, fontsize=LEG, gap_pt=gap)


def free(t):
    t.set_gid("free")
    return t


# ------------------------------------------------------------------------------------------------------- a rose
def a_rose(fig):
    F = L.fair()
    x0, y0, w, h = 0.0, y_row(1), CW, R1
    items = [H.item("DDTree", L.DD), H.item("+ loop control", L.DD_T), H.item("Ours", L.OU),
             H.item("+ loop control", L.OU_T, text_color=INK, bold=False),
             H.item("icon", INK, kind="marker", marker="^", ms=4.4),
             H.item("illus.", INK, kind="marker", marker="^", hollow=True, ms=4.4)]
    bot = leg(fig, items, x0 + 14, top_of_row(1), w - 16, gap=5.0)
    D = 90.0
    used = bot - top_of_row(1)
    cy = y0 + (h - used) / 2 - 1
    W_, Hf = fig.get_figwidth() * 72, fig.get_figheight() * 72
    ax = fig.add_axes([(x0 + (w - D) / 2) / W_, (cy - D / 2) / Hf, D / W_, D / Hf], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    r0 = 2.6
    to_r = lambda v: r0 + v
    tt = np.linspace(0, 2 * np.pi, 361)
    for g in (4, 8, 12):
        ax.plot(tt, np.full_like(tt, to_r(g)), color="#E3E3E3", lw=0.5, zorder=1)[0].set_gid("ref")
    ax.text(0, to_r(8), "8×", ha="center", va="center", fontsize=7.0, color=MUTED, zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
    ax.set_ylim(0, to_r(13.4))
    th = L.polar_angles()
    bw = np.deg2rad(15.5)
    for c in L.CELLS:
        v, t = F[c], th[c]
        for dx, base_v, top_v, dark, tint in ((-8.6, v["dd"], v["dd_lc"], L.DD, L.DD_T),
                                              (8.6, v["ou"], v["ou_lc"], L.OU, L.OU_T)):
            a = t + np.deg2rad(dx)
            ax.bar(a, base_v, width=bw, bottom=r0, color=dark, linewidth=0, zorder=3)
            ax.bar(a, top_v - base_v, width=bw, bottom=r0 + base_v, color=tint, linewidth=0, zorder=3)
        tip = to_r(max(v["dd_lc"], v["ou_lc"]))
        ax.plot([t], [tip + 0.9], ls="none", marker="^", ms=3.6, mec=INK, mfc=L.GLYPH[c[1]], mew=0.7, zorder=6)
        free(L.polar_text(ax, t, tip + 1.9, f"×{v['ou_lc'] / v['dd_lc']:.2f}", fontsize=7.0, color=L.OU,
                          fontweight="bold", zorder=6))
    for k, b in enumerate(L.BLOCKS):
        mdl, task = L.SHORT[b]
        free(L.polar_text(ax, np.deg2rad(k * 90 + 45), to_r(13.4) + 2.6, f"{mdl}\n{task}", fontsize=7.0, color=INK,
                          linespacing=1.0))
    return ax


# ------------------------------------------------------------------------------------------------------- b strips
def b_strips(fig):
    Q, V = L.ladder()
    x0, y0 = CW, y_row(1)
    items = [H.item("one setting", L.THIN, kind="line", lw=1.0), H.item("median", L.OU, kind="line", lw=2.0)]
    leg(fig, items, x0 + 14, top_of_row(1), CW - 16)
    left = FL + FG + PW - x0 - PW + (CW - PW - 4) * 0 + 0
    lx = x0 + 56.0
    bot_h, top_h, gap, yb = 60.0, 34.0, 5.0, 26.0
    ax2 = H.axes_pt(fig, lx, y0 + yb, PW, bot_h)
    ax1 = H.axes_pt(fig, lx, y0 + yb + bot_h + gap, PW, top_h)
    xs, keys = [0, 1, 2, 3], ["lossless"] + L.RULES
    for ax in (ax1, ax2):
        ax.axvspan(1.62, 2.38, color=C["ours_light"], alpha=0.55, lw=0, zorder=0).set_gid("bg")
        ax.set_xlim(-0.35, 3.35)
    for s in L.SETTINGS:
        ls = "-" if s[1] == "greedy" else (0, (2.4, 1.4))
        ax1.plot(xs, [Q[s][k]["saved"] for k in keys], color=L.THIN, lw=0.7, ls=ls, zorder=2)
        ax2.plot(xs, [Q[s][k]["d"] for k in keys], color=L.THIN, lw=0.7, ls=ls, zorder=2)
    ax1.plot(xs, [float(np.median([Q[s][k]["saved"] for s in L.SETTINGS])) for k in keys], color=L.OU, lw=1.8,
             marker="o", ms=3.6, zorder=4)
    ax2.plot(xs, [float(np.median([Q[s][k]["d"] for s in L.SETTINGS])) for k in keys], color=L.OU, lw=1.8,
             marker="o", ms=3.6, zorder=4)
    H.ref_line(ax2, y=0, lw=0.7)
    H.ref_line(ax1, y=1, lw=0.7)
    ax1.set_ylim(0.97, 1.40)
    ax1.set_yticks([1.0, 1.2], ["1.0", "1.2×"])
    ax2.set_ylim(-0.98, 0.68)
    ax2.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "+0.5"])
    free(ax1.text(-0.3, 1.38, "forward passes saved", ha="left", va="top", fontsize=7.0, color=MUTED))
    ax2.set_ylabel("quality change,\n% of lossless", fontsize=8.0, labelpad=2, linespacing=1.1)
    ax1.set_xticks([])
    ax1.spines["bottom"].set_visible(False)
    ax2.set_xticks(xs, ["lossless", "rollback", "escape\nthen stop", "stop"], fontsize=7.0, linespacing=1.0)
    ax2.tick_params(axis="x", length=0, pad=2)
    for lab_ in ax2.get_xticklabels():
        if lab_.get_text().startswith("escape"):
            lab_.set_color(L.OU)
            lab_.set_fontweight("bold")
    for i, r in enumerate(L.RULES, start=1):
        nw = V[r].count("worse")
        free(ax2.text(i, -0.93, f"{nw} of 8\nworse", ha="center", va="bottom", fontsize=7.0, linespacing=1.0,
                      color=C["loss"] if nw > 4 else MUTED))
    return ax2


# ------------------------------------------------------------------------------------------------------- c compute
def c_compute(fig):
    x0, y0 = 2 * CW, y_row(1)
    items = [H.item("original model + loop control", Q1.AR), H.item("Ours", Q1.OU),
             H.item("1 sample of the original model", H.REF, kind="line", lw=0.9, ls=(0, (3, 2)))]
    leg(fig, items, x0 + 14, top_of_row(1), CW - 16, gap=5.0)
    ax = H.axes_pt(fig, x0 + 54.0, y0 + 34.0, PW, R1 - 34.0 - 40.0)
    arms = [("ar8s", Q1.AR, 8), ("sd1s", Q1.OU, 1), ("sd8", Q1.OU, 8), ("sd16", Q1.OU, 16)]
    w, xs0 = 0.205, np.arange(3)
    ticks, tl = [], []
    for k, (arm, col, n) in enumerate(arms):
        v = [100 * Q1.G[(b, arm)]["target_forward"] / Q1.G[(b, "ar1")]["target_forward"] for b in Q1.BLOCKS]
        xs = xs0 + (k - 1.5) * w
        ax.bar(xs, v, w * 0.9, color=col, linewidth=0, zorder=2)
        for x, y in zip(xs, v):
            free(ax.text(x, y + (14 if arm == "sd16" else 3), f"{y:.0f}", ha="center", va="bottom", fontsize=7.0,
                         color=INK))
            ticks.append(x)
            tl.append(str(n))
    H.ref_line(ax, y=100, lw=0.7)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 50, 100, 150], ["0", "50", "100", "150%"])
    ax.set_ylabel("target forward passes", fontsize=8.0, labelpad=2)
    ax.set_xticks(ticks, tl)
    ax.tick_params(axis="x", length=0, pad=2.0, labelsize=7.0)
    ax.set_xlim(-0.5, 2.5)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x, b in zip(xs0, Q1.BLOCKS):
        mdl, task = b.split(", ")
        fam, size = mdl.split("-")
        ax.text(x, -0.15, f"{fam}\n{size}, {task.replace('-to-SVG', '')}", transform=tr, ha="center", va="top",
                fontsize=7.0, linespacing=1.0, color=INK)
    return ax


# ------------------------------------------------------------------------------------------------------- d-f frontier
def def_frontier(fig):
    y0 = y_row(2)
    items = [H.item("original model + loop control, best of N", Q1.AR, kind="line", lw=1.2),
             H.item("Ours, best of N", Q1.OU, kind="line", lw=1.8, bold=True, text_color=Q1.OU),
             H.item("original model, 1 sample", Q1.AR, kind="marker", marker="s", hollow=True, ms=4.6),
             H.item("authors' pipeline, first valid of 8", Q1.AR, kind="marker", marker="D", hollow=True, ms=4.2)]
    leg(fig, items, FL, top_of_row(2) + 1, FULL - FL - 4, gap=10.0)
    bottom, ph = 28.0, 104.0
    axes = [H.axes_pt(fig, FL + i * (PW + FG), y0 + bottom, PW, ph) for i in range(3)]
    for ax, b in zip(axes, Q1.BLOCKS):
        t, q = Q1.t, Q1.q
        H.ref_line(ax, x=t(b, "ar1"), lw=0.7)
        yq = q(b, "ar8s")
        H.ref_line(ax, y=yq, lw=0.7)
        for arms, c, lw, ms in ((Q1.AR_LADDER, Q1.AR, 1.2, 3.8), (Q1.SD_LADDER, Q1.OU, 1.8, 4.4)):
            ax.plot([t(b, a) for a in arms], [q(b, a) for a in arms], "-", color=c, lw=lw, zorder=3, marker="o", ms=ms,
                    mfc=c, mec="white", mew=0.6)
        ax.plot(t(b, "ar1"), q(b, "ar1"), "s", ms=4.6, mfc="white", mec=Q1.AR, mew=1.0, zorder=4)
        ax.plot(t(b, "ar8-fv"), q(b, "ar8-fv"), "D", ms=4.2, mfc="white", mec=Q1.AR, mew=1.0, zorder=4)
        ax.annotate("", xy=(t(b, "sd8"), yq), xytext=(t(b, "ar8s"), yq),
                    arrowprops=dict(arrowstyle="-|>", color=Q1.OU, lw=1.2, shrinkA=3, shrinkB=3), zorder=5)
        ax.set_xscale("log")
        allx = [t(b, a) for a in Q1.AR_LADDER + Q1.SD_LADDER + ["ar1", "ar8-fv"]]
        ally = [q(b, a) for a in Q1.AR_LADDER + Q1.SD_LADDER + ["ar1", "ar8-fv"]]
        ax.set_xlim(0.78 * min(allx), 1.5 * max(allx))
        pad = 0.16 * (max(ally) - min(ally))
        ax.set_ylim(min(ally) - 1.3 * pad, max(ally) + 1.6 * pad)
        H.fs.assert_inside(ax, allx, ally)
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("seconds per prompt", fontsize=8.0, labelpad=1.5)
        ax.set_ylabel(Q1.ylabel(b), fontsize=8.0, labelpad=2)
        mdl, task = b.split(", ")
        ax.set_title(f"{mdl}, {task}", fontsize=8.0, pad=3, color=INK)
        ax.yaxis.set_label_coords(-0.27, 0.5)
    fig.canvas.draw()
    off = {"OmniSVG-8B, text-to-SVG": dict(lead=(0, -4), at=None), "OmniSVG-8B, image-to-SVG": dict(lead=(0, -4), at=None),
           "InternSVG-8B, text-to-SVG": dict(lead=(5, 4.5), at=1.0)}
    for ax, b in zip(axes, Q1.BLOCKS):
        t, q = Q1.t, Q1.q
        for arms, c, xy, ha in ((Q1.AR_LADDER, Q1.AR, (5.0, -3.0), "left"), (Q1.SD_LADDER, Q1.OU, (-5.0, 3.5), "right")):
            for a in arms:
                xy_, ha_ = ((5.0, -3.0), "left") if a == "sd1s" else (xy, ha)
                ax.annotate(str(Q1.NOF[a]), (t(b, a), q(b, a)), textcoords="offset points", xytext=xy_, ha=ha_,
                            va="center", fontsize=7.0, color=c, zorder=2.5,
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.15))
        yq = q(b, "ar8s")
        dx, dy = off[b]["lead"]
        xl = np.sqrt(t(b, "sd8") * t(b, "ar8s")) if off[b]["at"] is None else t(b, "ar8s") * off[b]["at"]
        ax.annotate(f"×{t(b, 'ar8s') / t(b, 'sd8'):.1f}", (xl, yq), textcoords="offset points", xytext=(dx, dy),
                    ha="center", va="bottom" if dy > 0 else "top", fontsize=8.5, color=Q1.OU, fontweight="bold",
                    zorder=2.5, bbox=dict(facecolor="white", edgecolor="none", pad=0.15))
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(t(b, "ar1"), 0.985, "1 sample ", transform=tr, ha="right", va="top", fontsize=7.0, color=MUTED)
    return axes


# ------------------------------------------------------------------------------------------------------- g judges
def g_judges(fig):
    x0, y0 = 0.0, 0.0
    items = [H.item("Ours preferred", C["ours"]), H.item("tie", Q2.TIE_C), H.item("original model preferred", Q2.AR_LIGHT)]
    leg(fig, items, x0 + 14, top_of_row(3), CW - 16, gap=5.0)
    R = Q2.rows()
    n_of = lambda a: re.findall(r"\d+", Q2.NAME[a])[0] if re.findall(r"\d+", Q2.NAME[a]) else "1"
    left = 46.0
    ax = H.axes_pt(fig, x0 + left, y0 + 26.0, CW - left - 6, R3 - 26.0 - 40.0)
    y, prev, ys = 0.0, None, []
    for b, a, c in R:
        if prev is not None and b != prev:
            y += 0.8
        prev = b
        win, tie, loss = Q2.Q.counts(b, a, c)
        nn = win + tie + loss
        wv, tv, lv = win / nn, tie / nn, loss / nn
        ax.barh(y, wv, left=tv / 2, height=0.66, color=C["ours"], lw=0, zorder=2)
        ax.barh(y, tv, left=-tv / 2, height=0.66, color=Q2.TIE_C, lw=0, zorder=2)
        ax.barh(y, lv, left=-tv / 2 - lv, height=0.66, color=Q2.AR_LIGHT, lw=0, zorder=2)
        p = Q2.Q.pref(b, a, c)
        free(ax.text(tv / 2 + wv + 0.03, y, f"{p:.2f}", ha="left", va="center", fontsize=7.0, color=C["ours"],
                     fontweight="bold" if p > 0.6 else "normal"))
        ys.append((y, b, f"{n_of(a)} vs {n_of(c)}"))
        y += 1.0
    for bb in dict.fromkeys(b for _v, b, _s in ys):
        vv = [v for v, b, _s in ys if b == bb]
        ax.plot([0, 0], [min(vv) - 0.36, max(vv) + 0.36], color=H.REF, lw=0.7, ls=(0, (3, 2)), zorder=3)[0].set_gid("ref")
    ax.set_xlim(-0.75, 1.0)
    ax.set_ylim(y - 0.35, -1.5)
    ax.set_yticks([v for v, *_ in ys], [s for *_, s in ys], fontsize=7.0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(FixedLocator([-0.5, 0, 0.5]))
    ax.xaxis.set_major_formatter(lambda v, _p: f"{abs(v) * 100:.0f}%")
    ax.set_xlabel("share of judgements", fontsize=8.0, labelpad=1.5)
    firsts = {}
    for v, b, _s in ys:
        firsts.setdefault(b, v)
    for b, v in firsts.items():
        free(ax.text(-0.75, v - 0.62, b.replace(", text-to-SVG", ", text"), ha="left", va="bottom", fontsize=7.0,
                     fontweight="bold", color=INK))
    free(ax.text(-0.02, -1.45, "best of", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.0,
                 color=MUTED))
    return ax


# ------------------------------------------------------------------------------------------------------- h rules
def h_rules(fig):
    x0, y0 = CW, 0.0
    items = [H.item(Q3.BLAB[b], Q3.BCOL[b], kind="marker", ms=4.0) for b in Q3.BLOCKS] + [H.item("median", "#D6D6D6")]
    leg(fig, items, x0 + 14, top_of_row(3), CW - 16, gap=5.0)
    left = 70.0
    ax = H.axes_pt(fig, x0 + left, y0 + 26.0, CW - left - 6, R3 - 26.0 - 40.0)
    ax.axvline(0, color=H.REF, lw=0.7, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    for i, r in enumerate(Q3.ORDER):
        vals = [(b, Q3.eff(b, r, m)) for b, m in Q3.cells() if Q3.eff(b, r, m) is not None]
        med = float(np.median([v for _, v in vals]))
        ours = r == "render score"
        ax.barh(i, med, height=0.56, color=C["ours_light"] if ours else "#E2E2E2", lw=0, zorder=1)
        for (b, v), dy in zip(vals, np.linspace(-0.17, 0.17, len(vals))):
            ax.plot([v], [i + dy], "o", ms=3.6, mfc=Q3.BCOL[b], mec="white", mew=0.4, zorder=3)
        free(ax.text(max(v for _, v in vals) + 0.05, i, Q3.fmt(med), ha="left", va="center", fontsize=7.0,
                     color=C["ours"] if ours else INK, fontweight="bold" if ours else "normal"))
    ax.set_ylim(len(Q3.ORDER) - 0.5, -0.6)
    ax.set_yticks(range(len(Q3.ORDER)), [Q3.RLAB[r] for r in Q3.ORDER], fontsize=7.0)
    for tl in ax.get_yticklabels():
        if tl.get_text().startswith("render score"):
            tl.set_color(C["ours"])
            tl.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.52, 1.1)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.xaxis.set_major_formatter(lambda v, _p: Q3.fmt(v))
    ax.set_xlabel("gain over a random pick (s.d.)", fontsize=8.0, labelpad=1.5)
    return ax


# ------------------------------------------------------------------------------------------------------- i serving
def i_serving(fig):
    pd_ = P5.pd_
    x0, y0 = 2 * CW, 0.0
    wins = [32, 24, 16, 8]
    cmap = LinearSegmentedColormap.from_list("lead", ["#5E9A9A", "#A9CFCF", "#F4F4F4", "#9DB7D9", "#2A5C97"])
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=0.30)
    items = [H.item("ahead of DFlash's best", "#6E9AC4"), H.item("level", "#E9E9E9"), H.item("behind", "#7FB3B3"),
             H.item("the window we pick", INK, kind="marker", marker="s", hollow=True, ms=5.0, mew=1.1)]
    leg(fig, items, x0 + 14, top_of_row(3), CW - 16, gap=5.0)
    left, rh, tl_h, gap, bottom = 62.0, 9.6, 9.0, 6.0, 24.0
    ah = rh * (len(wins) + 1)
    axes = []
    for k, mode in enumerate(("greedy", "sample")):
        yb = y0 + bottom + (0 if k == 1 else ah + tl_h + gap)
        ax = H.axes_pt(fig, x0 + left, yb, CW - left - 3, ah)
        Z = np.array([[np.log(pd_.spd(mode, pd_.OURS[w], c) / pd_.best(mode, "base", c)[0]) for c in pd_.CONC]
                      for w in wins])
        ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto", extent=(-0.5, len(pd_.CONC) - 0.5, len(wins) - 0.5, -0.5),
                  interpolation="nearest", zorder=1)
        for i, w in enumerate(wins):
            for j, c in enumerate(pd_.CONC):
                free(ax.text(j, i, f"{pd_.spd(mode, pd_.OURS[w], c):.1f}", ha="center", va="center", fontsize=7.0,
                             color=INK, zorder=3))
        for j, c in enumerate(pd_.CONC):
            v, _w = pd_.best(mode, "base", c)
            ax.add_patch(Rectangle((j - 0.5, len(wins) - 0.5), 1, 1, facecolor="white", edgecolor="#D0D0D0", lw=0.4,
                                   zorder=1))
            free(ax.text(j, len(wins), f"{v:.1f}", ha="center", va="center", fontsize=7.0, color=P5.TEAL_TXT,
                         zorder=3))
            i = wins.index(pd_.best(mode, "ours", c)[1])
            r = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=INK, lw=1.1, zorder=4)
            r.set_gid("bg")
            ax.add_patch(r)
        ax.set_xlim(-0.5, len(pd_.CONC) - 0.5)
        ax.set_ylim(len(wins) + 0.5, -0.5)
        ax.set_yticks(range(len(wins) + 1), [f"window {w}" for w in wins] + ["DFlash, best"], fontsize=7.0)
        ax.get_yticklabels()[-1].set_color(P5.TEAL_TXT)
        ax.tick_params(axis="y", length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if k == 1:
            ax.set_xticks(range(len(pd_.CONC)), [str(c) for c in pd_.CONC], fontsize=7.0)
            ax.tick_params(axis="x", length=0, pad=2)
            ax.set_xlabel("concurrent requests", fontsize=8.0, labelpad=1.5)
        else:
            ax.set_xticks([])
        free(ax.text(-0.5, -0.62, "greedy" if mode == "greedy" else "official sampling", ha="left", va="bottom",
                     fontsize=7.0, color=INK))
        axes.append(ax)
    return axes


def grid():
    fig = plt.figure(figsize=(FULL / 72, FIG_H / 72))
    made = dict(a=a_rose(fig), b=b_strips(fig), c=c_compute(fig))
    d, e, f = def_frontier(fig)
    made.update(d=d, e=e, f=f, g=g_judges(fig), h=h_rules(fig))
    i1, i2 = i_serving(fig)
    made["i"] = i1
    for ax in fig.axes:
        if ax.get_label() == "_house_overlay":
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    fig.canvas.draw()
    for pid, (xc, k) in dict(a=(0, 1), b=(1, 1), c=(2, 1), d=(0, 2), e=(1, 2), f=(2, 2), g=(0, 3), h=(1, 3), i=(2, 3)).items():
        H.letter(fig, xc * CW + 1.0, top_of_row(k) + (20.0 if k == 2 else 0.0), pid)
    require_matplotlib_panel_alignment(
        fig, axes=[made["b"], made["c"], made["d"], made["e"], made["f"]], panel_ids=["b", "c", "d", "e", "f"],
        row_groups=[["d", "e", "f"]], column_groups=[["b", "e"], ["c", "f"]],
        exemptions=[{"panels": ["b", "e"], "checks": ["column"], "reason": "b's strips sit 2 pt right of e's plot"},
                    {"panels": ["c", "f"], "checks": ["column"], "reason": "c's bars sit level with f to 2 pt"}],
        json_out=str(HERE / f"{NAME}.alignment.json"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    H.save(fig, str(HERE / NAME), width="full", strict=STRICT)
    plt.close(fig)


if __name__ == "__main__":
    grid()
    print("wrote", HERE / f"{NAME}.pdf")
