"""L loop control, round 3 --  evening. Rounds 1 and 2 stay as they are (guide SS1-14).

User on round 2: "Apart from f3_radar and f4_lead, which are passable, the others are poor, and even these two are flat" -- and on round 1:
the fairness figure (l3) is right but too wide, it gets ONE single-column panel and should be denser; the
"loop control is free" figure (l1) is worth having but too ugly and messy to read.
Round 3 is the first round on the house style (paper/_house): point-based legends, the paper-wide encoding,
and the geometry check that would have stopped round 2's collisions.

Two claims, one figure each (single column):
  F  fairness. The same loop control given to the strongest baseline (DDTree) and to us: it speeds both up
     (DDTree x1.15-1.59, ours x1.06-1.46) and with it on both sides we are still first in all eight SArena
     cells (lead x1.13-1.41 lossless -> x1.05-1.19). Like for like only: lossless vs lossless, with vs with
     (with loop control, DDTree beats our LOSSLESS run in 6 of 8 cells -- a different tier, never compared).
  Q  loop control is free. On the rule we use (escape then stop) quality moves by -0.34%.. +0.26% of the
     lossless run, 7 of 8 settings inside their 95% CI (the eighth is better); the harsher 'stop' saves more
     forward passes but is worse in 7 of 8; 'rollback' is safe but saves the least.

Candidates (each a different chart family, guide SS3.2 / SS7.3):
  fa_rose      polar stacked bars: 8 cells x {DDTree, Ours}, dark = lossless, tint = what loop control adds
  fb_radar     radar from 4x outwards: four rings, our lead (both with loop control) shaded
  fc_hbars     horizontal stacked bars, the round-1 l3 turned on its side so it fits one column
  fd_latency   seconds per SVG on a log axis, one arrow per decoder from lossless to loop control, the
               original model as the reference dot
  qa_scissors two strips on one x axis (the rule ladder): forward passes saved rises, quality stays flat until
               'stop'; thin line = one setting, thick = the median (PACT fig. 6c's "scissors")
  qb_boxes     box + points per rule, points coloured by verdict (better / unchanged / worse)
  qc_verdict   one stacked bar per rule: how many of the 8 settings got better / stayed / got worse, with the
               forward passes saved at the end of the bar
  qd_plane     the median path through (forward passes saved, quality change), settings as faint dots

Data: runs/2026-11-04/figdata_b3/l_rules.json (built by runs/2026-11-04/figdata_b3.py; each block asserts a
published number). F uses its 'sarena' block (full official benchmark, measured seconds; the (fast) rows are
escape then stop on the lossless row above them). Q uses its 'ladder' block (900 bench rows, 300 for InternSVG
image-to-SVG, paired against the same head's lossless run, 95% bootstrap CI).
Run:  python3 paper/claims/L-loop-control/round3/make.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402
from sightline.paths import RUNS

H.apply()
SZ, C, M = H.SZ, H.C, H.METHOD
DATA = json.loads((RUNS / "2026-11-04/figdata_b3/l_rules.json").read_text())

BLOCKS = ["OmniSVG-8B, text-to-SVG", "InternSVG-8B, text-to-SVG", "OmniSVG-8B, image-to-SVG",
          "InternSVG-8B, image-to-SVG"]
SHORT = {"OmniSVG-8B, text-to-SVG": ("OmniSVG-8B", "text"), "InternSVG-8B, text-to-SVG": ("InternSVG-8B", "text"),
         "OmniSVG-8B, image-to-SVG": ("OmniSVG-8B", "image"), "InternSVG-8B, image-to-SVG": ("InternSVG-8B", "image")}
SUBSETS = ["icon", "illustration"]
SUB_LABEL = {"icon": "icon", "illustration": "illus."}
CELLS = [(b, s) for b in BLOCKS for s in SUBSETS]
# tints for "what loop control adds" -- a lighter step of each method's own colour (ENCODING.md SS1)
DD, DD_T = M["DDTree"], "#8FB9B3"
OU, OU_T = M["Ours"], "#86A9D4"


# --------------------------------------------------------------------------------------------------------
def fair():
    S = {(r["block"], r["subset"], r["method"]): r["seconds"] for r in DATA["sarena"]}
    out = {}
    for c in CELLS:
        ar = S[c + ("AR",)]
        out[c] = dict(ar_s=ar, **{k: ar / S[c + (m,)] for k, m in
                                  (("dd", "DDTree"), ("dd_lc", "DDTree (fast)"), ("ou", "Ours"), ("ou_lc", "Ours (fast)"))},
                      **{k + "_s": S[c + (m,)] for k, m in
                         (("dd", "DDTree"), ("dd_lc", "DDTree (fast)"), ("ou", "Ours"), ("ou_lc", "Ours (fast)"))})
    lead0 = [v["ou"] / v["dd"] for v in out.values()]
    lead1 = [v["ou_lc"] / v["dd_lc"] for v in out.values()]
    gdd = [v["dd_lc"] / v["dd"] for v in out.values()]
    gou = [v["ou_lc"] / v["ou"] for v in out.values()]
    # the published ranges (round 1 README, inventory SS0.10); fail here, not on the page
    assert f"{min(lead0):.2f}-{max(lead0):.2f}" == "1.13-1.41", (min(lead0), max(lead0))
    assert f"{min(lead1):.2f}-{max(lead1):.2f}" == "1.05-1.19", (min(lead1), max(lead1))
    assert f"{min(gdd):.2f}-{max(gdd):.2f}" == "1.15-1.59" and f"{min(gou):.2f}-{max(gou):.2f}" == "1.06-1.46"
    assert all(x > 1 for x in lead1)
    return out


SETTINGS = [(b, m) for b in BLOCKS for m in ("greedy", "sample")]
RULES = ["rollback", "escape-then-stop", "stop"]
RULE_LABEL = {"lossless": "lossless", "rollback": "rollback", "escape-then-stop": "escape\nthen stop", "stop": "stop"}


def ladder():
    """-> Q[(block, mode)][rule] = dict(saved, d, lo, hi) with d / lo / hi in % of the lossless run."""
    rows = {(r["target"], r["mode"], r["rule"]): r for r in DATA["ladder"]}
    Q = {}
    for b, m in SETTINGS:
        base = rows[(b, m, "lossless")]
        ref = base["quality"][base["primary"]][0]
        Q[(b, m)] = {"lossless": dict(saved=1.0, d=0.0, lo=0.0, hi=0.0, metric=base["primary"])}
        for r in RULES:
            q = rows[(b, m, r)]["quality"][base["primary"]]
            Q[(b, m)][r] = dict(saved=rows[(b, m, r)]["nfe_vs_lossless"], d=100 * q[0] / ref,
                                lo=100 * q[1] / ref, hi=100 * q[2] / ref, metric=base["primary"])
    # verdicts, as published: rollback 5 / 8 unchanged (3 better); escape then stop 7 / 8 (1 better); stop 1 / 8
    verdict = {r: [("better" if Q[s][r]["lo"] > 0 else "worse" if Q[s][r]["hi"] < 0 else "same") for s in SETTINGS]
               for r in RULES}
    assert [verdict["rollback"].count(k) for k in ("same", "better", "worse")] == [5, 3, 0], verdict["rollback"]
    assert [verdict["escape-then-stop"].count(k) for k in ("same", "better", "worse")] == [7, 1, 0]
    assert [verdict["stop"].count(k) for k in ("same", "better", "worse")] == [1, 0, 7]
    d_ets = [Q[s]["escape-then-stop"]["d"] for s in SETTINGS]
    assert f"{min(d_ets):+.2f}..{max(d_ets):+.2f}" == "-0.34..+0.26", (min(d_ets), max(d_ets))
    return Q, verdict


def cell_name(c, sep="\n"):
    b, s = c
    mdl, task = SHORT[b]
    return f"{mdl} {task}{sep}{SUB_LABEL[s]}"


def fair_legend(fig, top_pt, x_pt=4, width_pt=None):
    return H.legend(fig, [H.item("DDTree", DD), H.item("+ loop control", DD_T),
                          H.item("Ours", OU), H.item("+ loop control", OU_T, text_color=H.INK, bold=False)],
                    x_pt=x_pt, top_pt=top_pt, width_pt=width_pt, gap_pt=8)


# --------------------------------------------------------------------------------------------------------
# polar layout shared by fa_rose and fb_radar: one quadrant per model x task, icon then illustration clockwise
def polar_angles():
    th = {}
    for k, b in enumerate(BLOCKS):
        th[(b, "icon")] = np.deg2rad(k * 90 + 25)
        th[(b, "illustration")] = np.deg2rad(k * 90 + 65)
    return th


def polar_text(ax, theta, r, s, **kw):
    """Text at (theta, r) on a north-up clockwise polar axes, anchored away from the centre."""
    x, y = np.sin(theta), np.cos(theta)
    ha = "center" if abs(x) < 0.3 else ("left" if x > 0 else "right")
    va = "center" if abs(y) < 0.3 else ("bottom" if y > 0 else "top")
    return ax.text(theta, r, s, ha=kw.pop("ha", ha), va=kw.pop("va", va), **kw)


GLYPH = {"icon": H.INK, "illustration": "white"}
GLYPH_ITEMS = [H.item("SArena-Icon", H.INK, kind="marker", marker="^", ms=5),
               H.item("SArena-Illus.", H.INK, kind="marker", marker="^", hollow=True, ms=5)]


def polar_axes(fig, Ht_pt, top_pt, D):
    W_pt = fig.get_figwidth() * 72
    ax = fig.add_axes([(W_pt - D) / 2 / W_pt, (Ht_pt - top_pt - D) / Ht_pt, D / W_pt, D / Ht_pt], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    return ax


def polar_rings(ax, to_r, top):
    tt = np.linspace(0, 2 * np.pi, 361)
    for g in (4, 8, 12):
        ln, = ax.plot(tt, np.full_like(tt, to_r(g)), color="#E3E3E3", lw=0.6, zorder=1)
        ln.set_gid("ref")
        ax.text(0, to_r(g), f"{g}×", ha="center", va="center", fontsize=7.2, color=H.MUTED, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.set_ylim(0, top)


def polar_blocks(ax, r):
    for k, b in enumerate(BLOCKS):
        mdl, task = SHORT[b]
        polar_text(ax, np.deg2rad(k * 90 + 45), r, f"{mdl}\n{task}-to-SVG", fontsize=SZ["header"], color=H.INK,
                   linespacing=1.05)


def polar_tips(ax, F, th, to_r):
    """Subset glyph and our lead (both with loop control) just outside each cell's longer bar / ring point."""
    for c in CELLS:
        v = F[c]
        tip = to_r(max(v["dd_lc"], v["ou_lc"]))
        ax.plot([th[c]], [tip + 0.75], ls="none", marker="^", ms=4.6, mec=H.INK, mfc=GLYPH[c[1]], mew=0.8, zorder=6)
        polar_text(ax, th[c], tip + 1.55, f"×{v['ou_lc'] / v['dd_lc']:.2f}", fontsize=SZ["legend"], color=OU,
                   fontweight="bold", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))


def fa_rose(F):
    """Polar stacked bars; radius = r0 + speed-up, so a bar's length is its speed-up (nothing is truncated)."""
    W = H.COL_W
    items = [H.item("DDTree", DD), H.item("+ loop control", DD_T), H.item("Ours", OU),
             H.item("+ loop control", OU_T, text_color=H.INK, bold=False)] + GLYPH_ITEMS
    leg = H.legend_height(items, W, x_pt=6, width_pt=W * 72 - 12, gap_pt=8)
    D = 150.0
    Ht_pt = leg + 10 + D + 8
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=6, top_pt=0, width_pt=W * 72 - 12, gap_pt=8)
    ax = polar_axes(fig, Ht_pt, leg + 10, D)
    r0 = 2.6
    to_r = lambda v: r0 + v
    polar_rings(ax, to_r, to_r(13.4))
    th = polar_angles()
    bw = np.deg2rad(15.5)
    for c in CELLS:
        v, t = F[c], th[c]
        for dx, base_v, top_v, dark, tint in ((-8.6, v["dd"], v["dd_lc"], DD, DD_T), (8.6, v["ou"], v["ou_lc"], OU, OU_T)):
            a = t + np.deg2rad(dx)
            ax.bar(a, base_v, width=bw, bottom=r0, color=dark, linewidth=0, zorder=3)
            ax.bar(a, top_v - base_v, width=bw, bottom=r0 + base_v, color=tint, linewidth=0, zorder=3)
    polar_tips(ax, F, th, to_r)
    polar_blocks(ax, to_r(13.4) + 0.2)
    H.save(fig, str(HERE / "fa_rose"), width="col")
    plt.close(fig)


def fb_radar(F):
    """Radar from 3x outwards (round 2's f3 started at 0 and its rings crowded at 5-11x). Our lead with loop
    control on both sides is the shaded band; dashed = lossless, solid = with loop control."""
    W = H.COL_W
    items = [H.item("DDTree", DD, kind="line", ls=(0, (3, 1.6)), lw=1.1), H.item("+ loop control", DD, kind="line", lw=1.8),
             H.item("Ours", OU, kind="line", ls=(0, (3, 1.6)), lw=1.1),
             H.item("+ loop control", OU, kind="line", lw=2.2, text_color=H.INK, bold=False),
             H.item("our lead, both with loop control", C["ours_light"], kind="band")] + GLYPH_ITEMS
    leg = H.legend_height(items, W, x_pt=6, width_pt=W * 72 - 12, gap_pt=8)
    D = 150.0
    Ht_pt = leg + 10 + D + 8
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=6, top_pt=0, width_pt=W * 72 - 12, gap_pt=8)
    ax = polar_axes(fig, Ht_pt, leg + 10, D)
    lo = 3.0
    to_r = lambda v: v - lo
    polar_rings(ax, to_r, to_r(13.4))
    th = polar_angles()
    order = sorted(CELLS, key=lambda c: th[c])
    ang = np.array([th[c] for c in order] + [th[order[0]] + 2 * np.pi])
    for c in order:
        sp, = ax.plot([th[c], th[c]], [0, to_r(12.4)], color="#ECECEC", lw=0.6, zorder=1)
        sp.set_gid("ref")
    ring = lambda key: np.array([to_r(F[c][key]) for c in order] + [to_r(F[order[0]][key])])
    # the band between two polygons, as one polygon (straight chords, like the rings)
    band = ax.fill(np.r_[ang, ang[::-1]], np.r_[ring("ou_lc"), ring("dd_lc")[::-1]], color=C["ours_light"], lw=0,
                   zorder=2)
    for b_ in band:
        b_.set_gid("bg")
    for key, col, lw, ls in (("dd", DD, 1.1, (0, (3, 1.6))), ("ou", OU, 1.1, (0, (3, 1.6))),
                             ("dd_lc", DD, 1.8, "-"), ("ou_lc", OU, 2.2, "-")):
        ax.plot(ang, ring(key), color=col, lw=lw, ls=ls, zorder=4 if "ou" in key else 3, solid_joinstyle="round")
    polar_tips(ax, F, th, to_r)
    polar_blocks(ax, to_r(13.4) + 0.2)
    H.save(fig, str(HERE / "fb_radar"), width="col")
    plt.close(fig)


def rows_layout(top_pt, bar_h=4.6, head_h=11.0, pair_gap=1.0, gap=4.0, group_gap=1.5):
    """y layout (points from the figure top): 4 groups, a header line each, then 2 rows x 2 bars."""
    y, pos, heads = top_pt, {}, {}
    for b in BLOCKS:
        heads[b] = y + head_h * 0.5
        y += head_h
        for s in SUBSETS:
            pos[(b, s)] = (y + bar_h / 2, y + bar_h + pair_gap + bar_h / 2)
            y += 2 * bar_h + pair_gap + gap
        y += group_gap
    return pos, heads, y - gap - group_gap


def rows_figure(items, bar_h, bottom=30.0, left=34.0, right=30.0, gap_pt=8):
    W = H.COL_W
    leg = H.legend_height(items, W, x_pt=6, width_pt=W * 72 - 12, gap_pt=gap_pt)
    pos, heads, y_end = rows_layout(leg + 4, bar_h=bar_h)
    Ht_pt = y_end + 3 + bottom
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=6, top_pt=0, width_pt=W * 72 - 12, gap_pt=gap_pt)
    top = leg + 4
    ax = H.axes_pt(fig, left, bottom, W * 72 - left - right, Ht_pt - bottom - top)
    ax.set_ylim(0, Ht_pt - bottom - top)
    to_y = lambda yy: (Ht_pt - bottom) - yy                   # points from the figure top -> axes data y
    for b in BLOCKS:
        mdl, task = SHORT[b]
        ax.text(0.0, to_y(heads[b]), f"{mdl}, {task}-to-SVG", ha="left", va="center", fontsize=SZ["header"],
                color=H.INK, transform=ax.get_yaxis_transform(), clip_on=False, zorder=2, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    for c in CELLS:
        ax.text(-0.02, to_y((pos[c][0] + pos[c][1]) / 2), SUB_LABEL[c[1]], ha="right", va="center",
                fontsize=SZ["tick"], color=H.INK, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    return fig, ax, pos, to_y


def fc_hbars(F):
    """Round 1's l3 on its side: 8 rows x two stacked bars; dark = lossless, tint = what loop control adds."""
    items = [H.item("DDTree", DD), H.item("+ loop control", DD_T), H.item("Ours", OU),
             H.item("+ loop control", OU_T, text_color=H.INK, bold=False)]
    bh = 4.6
    fig, ax, pos, to_y = rows_figure(items, bh)
    ax.set_xlim(0, 12.2)
    for c in CELLS:
        v = F[c]
        for yc, base_v, top_v, dark, tint in ((pos[c][0], v["dd"], v["dd_lc"], DD, DD_T), (pos[c][1], v["ou"], v["ou_lc"], OU, OU_T)):
            yy = to_y(yc)
            ax.barh(yy, base_v, height=bh, color=dark, lw=0, zorder=3)
            ax.barh(yy, top_v - base_v, left=base_v, height=bh, color=tint, lw=0, zorder=3)
        ax.text(v["ou_lc"] + 0.18, to_y(pos[c][1]), f"×{v['ou_lc'] / v['dd_lc']:.2f}", ha="left", va="center",
                fontsize=SZ["legend"], color=OU, fontweight="bold", zorder=5, clip_on=False, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12], ["0", "2", "4", "6", "8", "10", "12×"])
    ax.set_xlabel("speed-up over the original model", fontsize=SZ["label"], labelpad=2)
    for xg in (2, 4, 6, 8, 10, 12):
        ax.axvline(xg, color="#EDEDED", lw=0.6, zorder=0).set_gid("ref")
    H.save(fig, str(HERE / "fc_hbars"), width="col")
    plt.close(fig)


def fd_latency(F):
    """Seconds per SVG on a log axis. Grey dot: the original model. Per decoder: a tick at its lossless time, a
    dot at its time with loop control, joined by a line. Our dot is left of DDTree's in every row."""
    items = [H.item("original model", C["ar"], kind="marker", ms=5), H.item("DDTree", DD, kind="line", lw=1.5),
             H.item("Ours", OU, kind="line", lw=1.8),
             H.item("lossless", H.INK, kind="marker", marker="|", ms=6, mew=1.2),
             H.item("with loop control", H.INK, kind="marker", ms=4.6)]
    bh = 5.2
    fig, ax, pos, to_y = rows_figure(items, bh, right=8.0)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 60)
    for c in CELLS:
        v = F[c]
        ymid = to_y((pos[c][0] + pos[c][1]) / 2)
        ax.plot([v["ar_s"]], [ymid], ls="none", marker="o", ms=5, mfc=C["ar"], mec="white", mew=0.5, zorder=4)
        for yc, a, b_, col, lw in ((pos[c][0], v["dd_s"], v["dd_lc_s"], DD, 1.5), (pos[c][1], v["ou_s"], v["ou_lc_s"], OU, 1.8)):
            yy = to_y(yc)
            ax.plot([a, b_], [yy, yy], color=col, lw=lw, solid_capstyle="butt", zorder=4)
            ax.plot([a], [yy], ls="none", marker="|", ms=6, mew=1.2, color=col, zorder=5)
            ax.plot([b_], [yy], ls="none", marker="o", ms=4.6, mfc=col, mec="white", mew=0.4, zorder=5)
    ax.set_xticks([1, 3, 10, 30], ["1", "3", "10", "30 s"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("seconds per SVG", fontsize=SZ["label"], labelpad=2)
    for xg in (1, 3, 10, 30):
        ax.axvline(xg, color="#EDEDED", lw=0.6, zorder=0).set_gid("ref")
    H.save(fig, str(HERE / "fd_latency"), width="col")
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------
THIN = "#B7C5D8"


def qa_scissors(Q, V):
    """Two strips on the rule ladder. Thin = one setting (model x task x decoding), thick = the median."""
    W = H.COL_W
    items = [H.item("one setting, greedy", THIN, kind="line", lw=1.0),
             H.item("official sampling", THIN, kind="line", lw=1.0, ls=(0, (2.4, 1.4))),
             H.item("median", OU, kind="line", lw=2.0)]
    leg = H.legend_height(items, W, x_pt=4, width_pt=W * 72 - 8, gap_pt=8)
    left, right = 48.0, 6.0
    top_h, bot_h, gap, y_bot = 62.0, 104.0, 6.0, 30.0
    Ht_pt = y_bot + bot_h + gap + top_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=4, top_pt=0, width_pt=W * 72 - 8, gap_pt=8)
    ax2 = H.axes_pt(fig, left, y_bot, W * 72 - left - right, bot_h)
    ax1 = H.axes_pt(fig, left, y_bot + bot_h + gap, W * 72 - left - right, top_h)
    xs, keys = [0, 1, 2, 3], ["lossless"] + RULES
    for ax in (ax1, ax2):
        ax.axvspan(1.62, 2.38, color=C["ours_light"], alpha=0.55, lw=0, zorder=0).set_gid("bg")
        ax.set_xlim(-0.3, 3.3)
    for s in SETTINGS:
        ls = "-" if s[1] == "greedy" else (0, (2.4, 1.4))
        ax1.plot(xs, [Q[s][k]["saved"] for k in keys], color=THIN, lw=0.8, ls=ls, zorder=2)
        ax2.plot(xs, [Q[s][k]["d"] for k in keys], color=THIN, lw=0.8, ls=ls, zorder=2)
    med_s = [float(np.median([Q[s][k]["saved"] for s in SETTINGS])) for k in keys]
    med_d = [float(np.median([Q[s][k]["d"] for s in SETTINGS])) for k in keys]
    ax1.plot(xs, med_s, color=OU, lw=2.0, marker="o", ms=4.2, zorder=4)
    ax2.plot(xs, med_d, color=OU, lw=2.0, marker="o", ms=4.2, zorder=4)
    H.ref_line(ax2, y=0)
    H.ref_line(ax1, y=1)
    ax1.set_ylim(0.97, 1.40)
    ax1.set_yticks([1.0, 1.1, 1.2, 1.3], ["1.0", "1.1", "1.2", "1.3×"])
    ax2.set_ylim(-0.98, 0.68)
    ax2.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "+0.5"])
    H.fs.assert_inside(ax1, [0] * 32, [Q[s][k]["saved"] for s in SETTINGS for k in keys])
    H.fs.assert_inside(ax2, [0] * 32, [Q[s][k]["d"] for s in SETTINGS for k in keys])
    ax1.set_ylabel("forward passes\nsaved", fontsize=SZ["label"], labelpad=3)
    ax2.set_ylabel("quality change,\n% of lossless", fontsize=SZ["label"], labelpad=3)
    ax1.set_xticks([])
    ax1.spines["bottom"].set_visible(False)
    ax2.set_xticks(xs, [RULE_LABEL[k] for k in keys])
    ax2.tick_params(axis="x", length=0, pad=3)
    for lab_ in ax2.get_xticklabels():
        if lab_.get_text().startswith("escape"):
            lab_.set_color(OU)
            lab_.set_fontweight("bold")
    for i, r in enumerate(RULES, start=1):
        nw = V[r].count("worse")
        ax2.text(i, -0.9, f"{nw} of 8 worse", ha="center", va="bottom", fontsize=7.6,
                 color=C["loss"] if nw else H.MUTED)
    H.save(fig, str(HERE / "qa_scissors"), width="col")
    plt.close(fig)


def qb_boxes(Q, V):
    """Box + points per rule; a point is one setting, coloured by its paired 95% CI."""
    W = H.COL_W
    items = [H.item("better", C["gain"], kind="marker", ms=5), H.item("unchanged", C["same"], kind="marker", ms=5),
             H.item("worse", C["loss"], kind="marker", ms=5)]
    left, right, bottom = 48.0, 6.0, 33.0
    leg = H.legend_height(items, W, x_pt=left - 2, width_pt=W * 72 - left - right, gap_pt=10)
    Ht_pt = leg + 4 + 128 + bottom
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=left - 2, top_pt=0, width_pt=W * 72 - left - right, gap_pt=10)
    ax = H.axes_pt(fig, left, bottom, W * 72 - left - right, 128)
    col = {"better": C["gain"], "same": C["same"], "worse": C["loss"]}
    for i, r in enumerate(RULES):
        d = np.array([Q[s][r]["d"] for s in SETTINGS])
        q1, q2, q3 = np.percentile(d, [25, 50, 75])
        edge = OU if r == "escape-then-stop" else "#8A8A8A"
        ax.add_patch(Rectangle((i - 0.23, q1), 0.46, q3 - q1, facecolor="none", edgecolor=edge, lw=1.1, zorder=2))
        ax.plot([i - 0.23, i + 0.23], [q2, q2], color=edge, lw=1.8, zorder=3)
        ax.plot([i, i], [d.min(), q1], color=edge, lw=0.9, zorder=2)
        ax.plot([i, i], [q3, d.max()], color=edge, lw=0.9, zorder=2)
        order = np.argsort(d)
        jit = np.empty(len(d))
        jit[order] = np.tile([-0.11, 0.11], len(d) // 2)          # alternate left / right up the box
        for x_, y_, vv in zip(i + jit, d, V[r]):
            ax.plot([x_], [y_], ls="none", marker="o", ms=4.6, mfc=col[vv], mec="white", mew=0.5, zorder=4)
    H.ref_line(ax, y=0)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.75, 0.68)
    ax.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "+0.5"])
    ax.set_ylabel("quality change,\n% of lossless", fontsize=SZ["label"], labelpad=3)
    med = [float(np.median([Q[s][r]["saved"] for s in SETTINGS])) for r in RULES]
    ax.set_xticks(range(3), [RULE_LABEL[r].replace("\n", " ") for r in RULES])
    ax.tick_params(axis="x", length=0, pad=3)
    for lab_ in ax.get_xticklabels():
        if lab_.get_text().startswith("escape"):
            lab_.set_color(OU)
            lab_.set_fontweight("bold")
    for i, m in enumerate(med):
        ax.text(i, -0.105, f"{100 * (1 - 1 / m):.0f}% fewer passes", ha="center", va="top", fontsize=7.5,
                color=H.MUTED, transform=ax.get_xaxis_transform())
    H.save(fig, str(HERE / "qb_boxes"), width="col")
    plt.close(fig)


def qc_verdict(Q, V):
    """One stacked bar per rule (8 settings: better / unchanged / worse); a column on the right gives the
    forward passes saved (median of the 8)."""
    W = H.COL_W
    items = [H.item("better", C["gain"]), H.item("unchanged", C["same"]), H.item("worse", C["loss"])]
    left, bottom, right = 68.0, 26.0, 44.0
    leg = H.legend_height(items, W, x_pt=4, width_pt=W * 72 - right - 8, gap_pt=10)
    Ht_pt = leg + 6 + 66 + bottom
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=4, top_pt=0, width_pt=W * 72 - right - 8, gap_pt=10)
    ax = H.axes_pt(fig, left, bottom, W * 72 - left - right, 66)
    col = {"better": C["gain"], "same": C["same"], "worse": C["loss"]}
    for i, r in enumerate(RULES):
        y = 2 - i
        x = 0
        for k in ("better", "same", "worse"):
            n = V[r].count(k)
            if n:
                ax.barh(y, n, left=x, height=0.62, color=col[k], lw=0, zorder=3)
                ax.text(x + n / 2, y, str(n), ha="center", va="center", fontsize=SZ["legend"], color="white", gid="free")
                x += n
        m = float(np.median([Q[s][r]["saved"] for s in SETTINGS]))
        ax.text(8.35, y, f"×{m:.2f}", ha="left", va="center", fontsize=SZ["legend"], clip_on=False,
                color=OU if r == "escape-then-stop" else H.INK, fontweight="bold" if r == "escape-then-stop" else "normal")
    ax.text(8.35, 2.62, "passes\nsaved", ha="left", va="bottom", fontsize=7.5, color=H.MUTED, clip_on=False,
            linespacing=1.12)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([2, 1, 0], [RULE_LABEL[r].replace("\n", " ") for r in RULES])
    ax.tick_params(axis="y", length=0)
    for lab_ in ax.get_yticklabels():
        if lab_.get_text().startswith("escape"):
            lab_.set_color(OU)
            lab_.set_fontweight("bold")
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("settings, of 8", fontsize=SZ["label"], labelpad=1)
    ax.spines["left"].set_visible(False)
    H.save(fig, str(HERE / "qc_verdict"), width="col")
    plt.close(fig)


def qd_plane(Q, V):
    """The median path through (forward passes saved, quality change); faint dots are the 8 settings per rule."""
    W = H.COL_W
    left, right, bottom = 48.0, 8.0, 30.0
    Ht_pt = 168.0
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    ax = H.axes_pt(fig, left, bottom, W * 72 - left - right, Ht_pt - bottom - 6)
    keys = ["lossless"] + RULES
    mx = [float(np.median([Q[s][k]["saved"] for s in SETTINGS])) for k in keys]
    my = [float(np.median([Q[s][k]["d"] for s in SETTINGS])) for k in keys]
    H.ref_line(ax, y=0)
    col = {"better": C["gain"], "same": "#B9C3CF", "worse": C["loss"]}
    for k in RULES:
        for s, vv in zip(SETTINGS, V[k]):
            ax.plot([Q[s][k]["saved"]], [Q[s][k]["d"]], ls="none", marker="o", ms=3.4, mfc=col[vv], mec="none",
                    alpha=0.75 if vv != "same" else 1.0, zorder=2)
    ax.plot(mx, my, color=OU, lw=1.8, zorder=4)
    for k, x_, y_ in zip(keys, mx, my):
        ax.plot([x_], [y_], ls="none", marker="o", ms=6.2 if k == "escape-then-stop" else 5,
                mfc=OU if k != "lossless" else "white", mec=OU, mew=1.2, zorder=5)
    offs = {"lossless": (-6, 0, "right"), "rollback": (0, 9, "center"), "escape-then-stop": (6, 7, "left"),
            "stop": (7, 0, "left")}
    for k, x_, y_ in zip(keys, mx, my):
        dx, dy, ha = offs[k]
        ax.annotate(RULE_LABEL[k].replace("\n", " "), (x_, y_), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=SZ["legend"], color=OU if k == "escape-then-stop" else H.INK,
                    fontweight="bold" if k == "escape-then-stop" else "normal", zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))
    ax.set_xlim(0.9, 1.37)
    ax.set_ylim(-0.75, 0.68)
    H.fs.assert_inside(ax, [Q[s][k]["saved"] for s in SETTINGS for k in RULES], [Q[s][k]["d"] for s in SETTINGS for k in RULES])
    ax.set_xticks([1.0, 1.1, 1.2, 1.3], ["1.0", "1.1", "1.2", "1.3×"])
    ax.set_yticks([-0.5, 0, 0.5], ["−0.5", "0", "+0.5"])
    ax.set_xlabel("forward passes saved", fontsize=SZ["label"], labelpad=2)
    ax.set_ylabel("quality change,\n% of lossless", fontsize=SZ["label"], labelpad=3)
    H.save(fig, str(HERE / "qd_plane"), width="col")
    plt.close(fig)


def main():
    F = fair()
    Q, V = ladder()
    print("fairness and ladder numbers reproduce the published ranges")
    for fn, arg in ((fa_rose, (F,)), (fb_radar, (F,)), (fc_hbars, (F,)), (fd_latency, (F,)),
                    (qa_scissors, (Q, V)), (qb_boxes, (Q, V)), (qc_verdict, (Q, V)), (qd_plane, (Q, V))):
        try:
            fn(*arg)
            print("wrote", fn.__name__)
        except AssertionError as e:
            print("CHECK", fn.__name__, e)


if __name__ == "__main__":
    main()
