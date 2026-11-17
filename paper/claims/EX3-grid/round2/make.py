"""Figure 7 (main.pdf, ), round 2: the 3 x 3 without its first cell, two re-layouts.

Why (session 7): the merged quality table now has a loop-control group with all six decoders (AR, EAGLE-3,
DFlash, DDTree, DARTree, Ours, each with the same rule; runs/2026-11-09/table-fill). Round 1's cell a (the rose: DDTree
and Ours with and without loop control, eight settings) draws four of that table's rows again, and any "gain per
decoder" cell would only divide two of its columns (figures guide 2.3: a quantity computable from numbers already shown
is not new data). So cell a goes; the Method sentence and 4.4 cite the table instead.

Two layouts, every cell redrawn at its new print size from round 1's code and data layers (nothing recomputed):
  L2  row 1: loop actions | forward passes (each half the width); row 2: the frontier; row 3: judges | rules | serving
  L3  row 1: loop actions | forward passes | judges; row 2: the frontier; row 3: rules | serving (each half the width,
      the serving heatmap gets the room it lacked in round 1)
Letters run a, b, c... in reading order in each layout.
Run:  python3 paper/claims/EX3-grid/round2/make.py [L2|L3]   (STRICT=0 to draft past the geometry check)
"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ex3r1", HERE.parent / "round1/make.py")
R = importlib.util.module_from_spec(spec)
sys.modules["ex3r1"] = R
spec.loader.exec_module(R)                      # round 1: data layers (their checks run on import) and the frontier row
L, Q1, Q2, Q3, P5, H = R.L, R.Q1, R.Q2, R.Q3, R.P5, R.H
C, INK, MUTED = H.C, H.INK, H.MUTED
leg, free = R.leg, R.free
STRICT = os.environ.get("STRICT", "1") != "0"
FULL, CW, HALF = R.FULL, R.CW, R.FULL / 2
FL, FG, PW = R.FL, R.FG, R.PW


class Rows:
    """Row heights (top to bottom) of one layout; round 1's y_row / top_of_row read R.R1-R3 and R.FIG_H, so the frontier
    row (drawn by round 1's own function) follows the same numbers."""

    def __init__(self, r1, r2, r3):
        R.R1, R.R2, R.R3 = r1, r2, r3
        R.FIG_H = R.TOP + r1 + r2 + r3
        self.h = {1: r1, 2: r2, 3: r3}

    @staticmethod
    def y(k):
        return R.y_row(k)

    @staticmethod
    def top(k):
        return R.top_of_row(k)


# ----------------------------------------------------------------------------------------------- loop actions (strips)
def strips(fig, rows, k, x0, w, plot_x=56.0):
    Q, V = L.ladder()
    y0 = rows.y(k)
    items = [H.item("one setting", L.THIN, kind="line", lw=1.0), H.item("median", L.OU, kind="line", lw=2.0)]
    leg(fig, items, x0 + 14, rows.top(k), w - 16)
    lx, pw = x0 + plot_x, w - plot_x - 8.0
    bot_h, top_h, gap, yb = 60.0, 34.0, 5.0, 26.0
    ax2 = H.axes_pt(fig, lx, y0 + yb, pw, bot_h)
    ax1 = H.axes_pt(fig, lx, y0 + yb + bot_h + gap, pw, top_h)
    xs, keys = [0, 1, 2, 3], ["lossless"] + L.RULES
    for ax in (ax1, ax2):
        ax.axvspan(1.62, 2.38, color=C["ours_light"], alpha=0.55, lw=0, zorder=0).set_gid("bg")
        ax.set_xlim(-0.35, 3.35)
    for s in L.SETTINGS:
        ls = "-" if s[1] == "greedy" else (0, (2.4, 1.4))
        ax1.plot(xs, [Q[s][kk]["saved"] for kk in keys], color=L.THIN, lw=0.7, ls=ls, zorder=2)
        ax2.plot(xs, [Q[s][kk]["d"] for kk in keys], color=L.THIN, lw=0.7, ls=ls, zorder=2)
    ax1.plot(xs, [float(np.median([Q[s][kk]["saved"] for s in L.SETTINGS])) for kk in keys], color=L.OU, lw=1.8,
             marker="o", ms=3.6, zorder=4)
    ax2.plot(xs, [float(np.median([Q[s][kk]["d"] for s in L.SETTINGS])) for kk in keys], color=L.OU, lw=1.8,
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


# ----------------------------------------------------------------------------------------------- forward passes (bars)
def compute(fig, rows, k, x0, w, plot_x=54.0):
    y0 = rows.y(k)
    items = [H.item("original model + loop control", Q1.AR), H.item("Ours", Q1.OU),
             H.item("1 sample of the original model", H.REF, kind="line", lw=0.9, ls=(0, (3, 2)))]
    leg(fig, items, x0 + 14, rows.top(k), w - 16, gap=5.0)
    ax = H.axes_pt(fig, x0 + plot_x, y0 + 34.0, w - plot_x - 8.0, rows.h[k] - 34.0 - 40.0)
    arms = [("ar8s", Q1.AR, 8), ("sd1s", Q1.OU, 1), ("sd8", Q1.OU, 8), ("sd16", Q1.OU, 16)]
    bw, xs0 = 0.205, np.arange(3)
    ticks, tl = [], []
    for j, (arm, col, n) in enumerate(arms):
        v = [100 * Q1.G[(b, arm)]["target_forward"] / Q1.G[(b, "ar1")]["target_forward"] for b in Q1.BLOCKS]
        xs = xs0 + (j - 1.5) * bw
        ax.bar(xs, v, bw * 0.9, color=col, linewidth=0, zorder=2)
        for x, y in zip(xs, v):
            free(ax.text(x, y + (14 if arm == "sd16" and w < 200 else 3), f"{y:.0f}", ha="center", va="bottom",
                         fontsize=7.0, color=INK))
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
        lab = f"{fam}\n{size}, {task.replace('-to-SVG', '')}" if w < 200 else f"{mdl}\n{task}"
        ax.text(x, -0.15, lab, transform=tr, ha="center", va="top", fontsize=7.0, linespacing=1.0, color=INK)
    return ax


# ----------------------------------------------------------------------------------------------- VLM judges
def judges(fig, rows, k, x0, w, left=46.0):
    y0 = rows.y(k)
    items = [H.item("Ours preferred", C["ours"]), H.item("tie", Q2.TIE_C), H.item("original model preferred", Q2.AR_LIGHT)]
    leg(fig, items, x0 + 14, rows.top(k), w - 16, gap=5.0)
    RR = Q2.rows()
    n_of = lambda a: re.findall(r"\d+", Q2.NAME[a])[0] if re.findall(r"\d+", Q2.NAME[a]) else "1"
    ax = H.axes_pt(fig, x0 + left, y0 + 26.0, w - left - 6, rows.h[k] - 26.0 - 40.0)
    y, prev, ys = 0.0, None, []
    for b, a, c in RR:
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


# ----------------------------------------------------------------------------------------------- selection rules
def rules(fig, rows, k, x0, w, left=70.0):
    y0 = rows.y(k)
    items = [H.item(Q3.BLAB[b], Q3.BCOL[b], kind="marker", ms=4.0) for b in Q3.BLOCKS] + [H.item("median", "#D6D6D6")]
    leg(fig, items, x0 + 14, rows.top(k), w - 16, gap=5.0)
    ax = H.axes_pt(fig, x0 + left, y0 + 26.0, w - left - 6, rows.h[k] - 26.0 - 40.0)
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


# ----------------------------------------------------------------------------------------------- serving heatmap
def serving(fig, rows, k, x0, w, left=62.0, rh=9.6, label_x=False):
    pd_ = P5.pd_
    y0 = rows.y(k)
    wins = [32, 24, 16, 8]
    cmap = LinearSegmentedColormap.from_list("lead", ["#5E9A9A", "#A9CFCF", "#F4F4F4", "#9DB7D9", "#2A5C97"])
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=0.30)
    items = [H.item("ahead of DFlash's best", "#6E9AC4"), H.item("level", "#E9E9E9"), H.item("behind", "#7FB3B3"),
             H.item("the window we pick", INK, kind="marker", marker="s", hollow=True, ms=5.0, mew=1.1)]
    leg(fig, items, x0 + 14, rows.top(k), w - 16, gap=5.0)
    tl_h, gap, bottom = 9.0, 6.0, 24.0
    ah = rh * (len(wins) + 1)
    axes = []
    for j0, mode in enumerate(("greedy", "sample")):
        yb = y0 + bottom + (0 if j0 == 1 else ah + tl_h + gap)
        ax = H.axes_pt(fig, x0 + left, yb, w - left - 3, ah)
        Z = np.array([[np.log(pd_.spd(mode, pd_.OURS[wn], c) / pd_.best(mode, "base", c)[0]) for c in pd_.CONC]
                      for wn in wins])
        ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto", extent=(-0.5, len(pd_.CONC) - 0.5, len(wins) - 0.5, -0.5),
                  interpolation="nearest", zorder=1)
        for i, wn in enumerate(wins):
            for j, c in enumerate(pd_.CONC):
                s = f"{pd_.spd(mode, pd_.OURS[wn], c):.1f}" + ("×" if label_x else "")
                free(ax.text(j, i, s, ha="center", va="center", fontsize=7.0, color=INK, zorder=3))
        for j, c in enumerate(pd_.CONC):
            v, _w = pd_.best(mode, "base", c)
            ax.add_patch(Rectangle((j - 0.5, len(wins) - 0.5), 1, 1, facecolor="white", edgecolor="#D0D0D0", lw=0.4,
                                   zorder=1))
            free(ax.text(j, len(wins), f"{v:.1f}" + ("×" if label_x else ""), ha="center", va="center", fontsize=7.0,
                         color=P5.TEAL_TXT, zorder=3))
            i = wins.index(pd_.best(mode, "ours", c)[1])
            r = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=INK, lw=1.1, zorder=4)
            r.set_gid("bg")
            ax.add_patch(r)
        ax.set_xlim(-0.5, len(pd_.CONC) - 0.5)
        ax.set_ylim(len(wins) + 0.5, -0.5)
        ax.set_yticks(range(len(wins) + 1), [f"window {wn}" for wn in wins] + ["DFlash, best"], fontsize=7.0)
        ax.get_yticklabels()[-1].set_color(P5.TEAL_TXT)
        ax.tick_params(axis="y", length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if j0 == 1:
            ax.set_xticks(range(len(pd_.CONC)), [str(c) for c in pd_.CONC], fontsize=7.0)
            ax.tick_params(axis="x", length=0, pad=2)
            ax.set_xlabel("concurrent requests", fontsize=8.0, labelpad=1.5)
        else:
            ax.set_xticks([])
        free(ax.text(-0.5, -0.62, "greedy" if mode == "greedy" else "official sampling", ha="left", va="bottom",
                     fontsize=7.0, color=INK))
        axes.append(ax)
    return axes


# ----------------------------------------------------------------------------------------------- layouts
def finish(fig, name, letters, made, row_groups, column_groups, exemptions):
    for ax in fig.axes:
        if ax.get_label() == "_house_overlay":
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    fig.canvas.draw()
    for pid, (x, top) in letters.items():
        H.letter(fig, x, top, pid)
    R.require_matplotlib_panel_alignment(
        fig, axes=[made[p] for p in made], panel_ids=list(made), row_groups=row_groups, column_groups=column_groups,
        exemptions=exemptions, json_out=str(HERE / f"{name}.alignment.json"), tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        strict=False)
    H.save(fig, str(HERE / name), width="full", strict=STRICT)
    plt.close(fig)
    print("wrote", HERE / f"{name}.pdf")


def L2():
    """Row 1: loop actions | forward passes, half the width each; rows 2-3 as round 1 (judges | rules | serving)."""
    rows = Rows(150.0, 172.0, 175.0)
    fig = plt.figure(figsize=(FULL / 72, R.FIG_H / 72))
    made = {"a": strips(fig, rows, 1, 0.0, HALF, plot_x=FL - 2.0), "b": compute(fig, rows, 1, HALF, HALF, plot_x=60.0)}
    c, d, e = R.def_frontier(fig)
    made.update(c=c, d=d, e=e)
    made["f"] = judges(fig, rows, 3, 0.0, CW)
    made["g"] = rules(fig, rows, 3, CW, CW)
    made["h"] = serving(fig, rows, 3, 2 * CW, CW)[0]
    letters = {"a": (1.0, rows.top(1)), "b": (HALF + 1.0, rows.top(1)), "c": (1.0, rows.top(2) + 20.0),
               "d": (CW + 1.0, rows.top(2) + 20.0), "e": (2 * CW + 1.0, rows.top(2) + 20.0),
               "f": (1.0, rows.top(3)), "g": (CW + 1.0, rows.top(3)), "h": (2 * CW + 1.0, rows.top(3))}
    finish(fig, "ex3_L2", letters, made, row_groups=[["c", "d", "e"]], column_groups=[["a", "c"]],
           exemptions=[{"panels": ["a", "c"], "checks": ["column"], "reason": "the strips start 2 pt left of c"}])


def L3():
    """Row 1: loop actions | forward passes | judges; row 2: the frontier; row 3: rules | serving, half the width each."""
    rows = Rows(160.0, 172.0, 165.0)
    fig = plt.figure(figsize=(FULL / 72, R.FIG_H / 72))
    made = {"a": strips(fig, rows, 1, 0.0, CW, plot_x=FL - 2.0), "b": compute(fig, rows, 1, CW, CW, plot_x=56.0),
            "c": judges(fig, rows, 1, 2 * CW, CW)}
    d, e, f = R.def_frontier(fig)
    made.update(d=d, e=e, f=f)
    made["g"] = rules(fig, rows, 3, 0.0, HALF, left=74.0)
    made["h"] = serving(fig, rows, 3, HALF, HALF, left=64.0, rh=9.6)[0]
    letters = {"a": (1.0, rows.top(1)), "b": (CW + 1.0, rows.top(1)), "c": (2 * CW + 1.0, rows.top(1)),
               "d": (1.0, rows.top(2) + 20.0), "e": (CW + 1.0, rows.top(2) + 20.0),
               "f": (2 * CW + 1.0, rows.top(2) + 20.0), "g": (1.0, rows.top(3)), "h": (HALF + 1.0, rows.top(3))}
    finish(fig, "ex3_L3", letters, made, row_groups=[["d", "e", "f"]], column_groups=[["a", "d"]],
           exemptions=[{"panels": ["a", "d"], "checks": ["column"], "reason": "the strips start 2 pt left of d"}])


def pages(entries):
    """Round 6's page mock (T1-main-table/round6/make.py): each candidate at the top of a paper page with a three-line
    caption and grey two-column text under it."""
    from PIL import Image, ImageDraw, ImageFont
    dpi = 64
    pw, ph, lm, tm = int(8.5 * dpi), int(11 * dpi), int(0.8125 * dpi), int(1.0 * dpi)
    tw, th = int(6.875 * dpi), int(8.875 * dpi)
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    ncol = len(entries)
    out = Image.new("RGB", (ncol * (pw + 16) + 16, ph + 46), "white")
    d0 = ImageDraw.Draw(out)
    for i, (n, png) in enumerate(entries):
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        im = Image.open(png).convert("RGB")
        h_in = im.height / im.width * 6.875
        im = im.resize((tw, int(round(h_in * dpi))), Image.LANCZOS)
        page.paste(im, (lm, tm))
        y = tm + im.height + int(0.12 * dpi)
        for k in range(3):
            d.rectangle([lm, y, lm + (tw if k < 2 else int(tw * 0.6)), y + 4], fill="#9A9A9A")
            y += int(0.16 * dpi)
        y += int(0.18 * dpi)
        cw, gap = int(3.25 * dpi), int(0.375 * dpi)
        while y < tm + th - 4:
            for c in range(2):
                x0 = lm + c * (cw + gap)
                d.rectangle([x0, y, x0 + cw, y + 3], fill="#D0D0D0")
            y += int(0.152 * dpi)
        d.rectangle([0, 0, pw - 1, ph - 1], outline="#BBBBBB")
        x = 16 + i * (pw + 16)
        out.paste(page, (x, 40))
        d0.text((x, 10), f"{n}  {h_in:.2f} in", fill="black", font=font)
    out.save(HERE / "_pages.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["L2", "L3"]
    for nm in which:
        {"L2": L2, "L3": L3}[nm]()
    pngs = [str(HERE / f"ex3_{nm}.png") for nm in ("L2", "L3") if (HERE / f"ex3_{nm}.png").exists()]
    pages([("round 1 (now)", str(HERE.parent / "round1/ex3_grid.png"))] + [(Path(p).stem, p) for p in pngs])
