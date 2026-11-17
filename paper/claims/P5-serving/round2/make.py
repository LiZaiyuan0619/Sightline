"""P5 round 2 -- one head, one knob, the whole load range (SGLang).  night, house style, single column.

Claim: in a production engine our block-32 head serves every load with one knob, the verification window. At low
load a long window beats the block-16 head's best window by %; as load rises both heads turn the window
down (ours 32 -> 8, DFlash's 16 -> 8) and at concurrent requests they are within 3.1% in 5 of 6 settings.
The crossover around 4-8 concurrent costs up to 9% (greedy, 8 concurrent) -- drawn, not hidden.

Data: pdata.py (runs/2026-11-04/b2w/b2w.json; asserts the published cells). One column (3.25 in), greedy and the
official sampling folded into one panel where the chart allows it.

Run:  python3 paper/claims/P5-serving/round2/make.py [names]     (STRICT=0 for drafts)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import pdata as pd_  # noqa: E402

H.apply()
SZ, C, M, INK, MUTED = H.SZ, H.C, H.METHOD, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W = H.COL_W * 72                     # 234 pt
TEAL_TXT = "#4F8C8C"                 # DFlash's light teal, darkened for text
BASE_C = "#5FA3A3"                   # DFlash's block-16 head: the DFlash family colour, a step darker to read as a line
X = {c: i for i, c in enumerate(pd_.CONC)}
LEG_FS = 7.8


def finish(fig):
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if "\n" in t.get_text() and getattr(t, "_linespacing", 1.2) < 1.12:
            t.set_linespacing(1.12)


def ref(a):
    a.set_gid("ref")
    return a


def conc_axis(ax, label=True):
    ax.set_xticks(range(len(pd_.CONC)), [str(c) for c in pd_.CONC])
    ax.set_xlim(-0.35, len(pd_.CONC) - 0.65)
    if label:
        ax.set_xlabel("concurrent requests", fontsize=SZ["label"], labelpad=1.5)


def times_log(ax, lo, hi, ticks):
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:g}×")


def save(fig, name):
    finish(fig)
    H.save(fig, str(HERE / name), width="col", strict=STRICT)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ a envelope
def p5a_envelope():
    """Each head at its best window, per load. Light blue = where ours leads, light grey = where DFlash's does.
    The two rows above the plot are the window each head picked at that load (the knob)."""
    items = [H.item("Ours, block 32", C["ours"], kind="line", lw=1.8, marker="o", with_marker=True, ms=4.2,
                    bold=True, text_color=C["ours"]),
             H.item("DFlash, block 16", BASE_C, kind="line", lw=1.5, marker="o", with_marker=True, ms=4.0),
             H.item("greedy", INK, kind="marker", ms=4.0),
             H.item("official sampling", INK, kind="marker", hollow=True, ms=4.0, mew=0.9)]
    leg = H.legend_height(items, H.COL_W, x_pt=4, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    head, bottom, ah = 36.0, 30.0, 118.0
    Ht = leg + 2 + head + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=4, top_pt=0, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, bottom, W - 44 - 6, ah)
    times_log(ax, 1.0, 11.5, [1, 2, 4, 8])
    xs = np.arange(len(pd_.CONC))
    allv = []
    for mode, ls, hollow in (("sample", (0, (3.2, 1.6)), True), ("greedy", "-", False)):
        o = np.array([pd_.best(mode, "ours", c)[0] for c in pd_.CONC])
        b = np.array([pd_.best(mode, "base", c)[0] for c in pd_.CONC])
        allv += list(o) + list(b)
        xf = np.linspace(0, len(xs) - 1, 400)
        of, bf = np.exp(np.interp(xf, xs, np.log(o))), np.exp(np.interp(xf, xs, np.log(b)))
        f1 = ax.fill_between(xf, bf, of, where=of >= bf, color=C["ours_light"], lw=0, zorder=1, interpolate=True)
        f2 = ax.fill_between(xf, of, bf, where=bf > of, color="#E3E3E3", lw=0, zorder=1, interpolate=True)
        f1.set_gid("bg"), f2.set_gid("bg")
        ax.plot(xs, b, color=BASE_C, lw=1.5 if not hollow else 1.1, ls=ls, zorder=3, solid_capstyle="round")
        ax.plot(xs, o, color=C["ours"], lw=1.9 if not hollow else 1.3, ls=ls, zorder=4, solid_capstyle="round")
        for ys, col in ((b, BASE_C), (o, C["ours"])):
            ax.plot(xs, ys, "o", ms=4.0 if hollow else 4.4, mfc="white" if hollow else col, mec=col,
                    mew=0.9 if hollow else 0.5, zorder=5)
    for c, txt, dy in ((1, "+29%", 1.09), (2, "+24%", 1.09)):
        pass
    g = [pd_.lead("greedy", c) for c in pd_.CONC]
    ax.text(0.55, np.sqrt(pd_.best("greedy", "ours", 1)[0] * pd_.best("greedy", "ours", 2)[0]) * 1.08,
            f"+{100 * min(g[:3]):.0f} to +{100 * max(g[:3]):.0f}%", ha="center", va="bottom",
            fontsize=SZ["legend"] - 0.6, color=C["ours"], fontweight="bold")
    ax.text(5.0, 1.13, "level, within 3%", ha="center", va="center", fontsize=SZ["legend"] - 0.8, color=MUTED)
    conc_axis(ax)
    ax.set_ylabel("speed-up over the\noriginal model", fontsize=SZ["label"], labelpad=2)
    # the knob: the window each head picked (greedy; under sampling ours picks the same, DFlash's switches at 16)
    for k, (head_, col, name) in enumerate((("ours", C["ours"], "Ours"), ("base", TEAL_TXT, "DFlash"))):
        y = 1.0 + (head - 2) / ah * (0.55 if k == 0 else 0.15) + 0.0
        for c in pd_.CONC:
            w = pd_.best("greedy", head_, c)[1]
            ax.text(X[c], 1.0, str(w), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=SZ["tick"] - 0.6, color=col, fontweight="bold" if head_ == "ours" else "normal")
    # place the two window rows in axes-fraction space above the plot
    for t in ax.texts[-14:]:
        k = 0 if t.get_color() == C["ours"] else 1
        t.set_position((t.get_position()[0], 1.0 + (head - 3 - k * 10.5) / ah - 0.02))
    for k, (name, col) in enumerate((("Ours", C["ours"]), ("DFlash", TEAL_TXT))):
        ax.text(-0.42, 1.0 + (head - 3 - k * 10.5) / ah - 0.02, name, transform=ax.get_xaxis_transform(), ha="right",
                va="bottom", fontsize=SZ["tick"] - 0.6, color=col, fontweight="bold" if k == 0 else "normal")
    ax.text(3.0, 1.0 + (head - 3 + 10.5) / ah - 0.02, "verification window each head uses", ha="center",
            va="bottom", transform=ax.get_xaxis_transform(), fontsize=SZ["tick"] - 0.6, color=MUTED)
    H.fs.assert_inside(ax, xs, allv)
    save(fig, "p5a_envelope")


# ------------------------------------------------------------------------------------------------ b heat
def p5b_knob():
    """Our head at every window x every load; colour = against the block-16 head's best window at that load
    (blue = ours ahead, teal = behind, white = level); the outlined cell in each column is the window we pick.
    The outlines walk from window 32 at one request to window 8 at 64: blue at low load, light grey at high load,
    light teal in 4 of the 14 columns around the crossover (4-9% behind), drawn as measured."""
    wins = [32, 24, 16, 8]
    rows_lab = [f"window {w}" for w in wins]
    cmap = LinearSegmentedColormap.from_list("lead", ["#5E9A9A", "#A9CFCF", "#F4F4F4", "#9DB7D9", "#2A5C97"])
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=0.30)
    leg_items = [H.item("ahead of DFlash's best", "#6E9AC4"), H.item("level", "#E9E9E9"),
                 H.item("behind", "#7FB3B3"),
                 H.item("the window we pick", INK, kind="marker", marker="s", hollow=True, ms=6.0, mew=1.2)]
    leg = H.legend_height(leg_items, H.COL_W, x_pt=4, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    rh, gap, bottom, top_lab = 12.5, 16.0, 30.0, 11.0
    ah = rh * (len(wins) + 1)
    Ht = leg + 2 + 2 * (ah + top_lab) + gap + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, leg_items, x_pt=4, top_pt=0, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    for k, mode in enumerate(("greedy", "sample")):
        y0 = bottom + (0 if k == 1 else ah + top_lab + gap)
        ax = H.axes_pt(fig, 70, y0, W - 70 - 3, ah)
        Z = np.array([[np.log(pd_.spd(mode, pd_.OURS[w], c) / pd_.best(mode, "base", c)[0]) for c in pd_.CONC]
                      for w in wins])
        ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto", extent=(-0.5, len(pd_.CONC) - 0.5, len(wins) - 0.5, -0.5),
                  interpolation="nearest", zorder=1)
        for i, w in enumerate(wins):
            for j, c in enumerate(pd_.CONC):
                v = pd_.spd(mode, pd_.OURS[w], c)
                t = ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=SZ["tick"] - 0.9, color=INK,
                            zorder=3)
                t.set_gid("free")
        # the DFlash head's best, as a last row
        for j, c in enumerate(pd_.CONC):
            v, w = pd_.best(mode, "base", c)
            ax.add_patch(Rectangle((j - 0.5, len(wins) - 0.5), 1, 1, facecolor="white", edgecolor="#D0D0D0",
                                   lw=0.4, zorder=1))
            t = ax.text(j, len(wins), f"{v:.1f}×", ha="center", va="center", fontsize=SZ["tick"] - 0.9,
                        color=TEAL_TXT, zorder=3)
            t.set_gid("free")
        for j, c in enumerate(pd_.CONC):
            wsel = pd_.best(mode, "ours", c)[1]
            i = wins.index(wsel)
            r = Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=INK, lw=1.3, zorder=4)
            r.set_gid("bg")
            ax.add_patch(r)
        ax.set_xlim(-0.5, len(pd_.CONC) - 0.5)
        ax.set_ylim(len(wins) + 0.5, -0.5)
        ax.set_yticks(range(len(wins) + 1), [f"Ours, window {w}" for w in wins] + ["DFlash, best"],
                      fontsize=SZ["tick"] - 0.8)
        for t in ax.get_yticklabels():
            if t.get_text().startswith("DFlash"):
                t.set_color(TEAL_TXT)
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.5, -0.62, "greedy" if mode == "greedy" else "official sampling", ha="left", va="bottom",
                fontsize=SZ["legend"] - 0.4, color=INK, transform=ax.transData)
        if k == 1:
            ax.set_xticks(range(len(pd_.CONC)), [str(c) for c in pd_.CONC])
            ax.set_xlabel("concurrent requests", fontsize=SZ["label"], labelpad=1.5)
        else:
            ax.set_xticks([])
    save(fig, "p5b_knob")


# ------------------------------------------------------------------------------------------------ c frontier
def p5c_frontier():
    """Serving frontier: whole-engine throughput against the wait per request, each head at its best window at
    each load; the original model for reference. Lower right is better."""
    items = [H.item("Ours, block 32", C["ours"], kind="line", lw=1.8, marker="o", with_marker=True, ms=4.0,
                    bold=True, text_color=C["ours"]),
             H.item("DFlash, block 16", BASE_C, kind="line", lw=1.5, marker="o", with_marker=True, ms=3.8),
             H.item("official sampling", INK, kind="line", lw=1.0, ls=(0, (3.2, 1.6)), marker="o", with_marker=True,
                    hollow=True, ms=3.6)]
    leg = H.legend_height(items, H.COL_W, x_pt=4, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 30.0, 128.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=4, top_pt=0, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 40, bottom, W - 40 - 6, ah)
    ax.set_xscale("log")
    ax.set_yscale("log")
    allx, ally = [], []
    for mode, ls, hollow in (("sample", (0, (3.2, 1.6)), True), ("greedy", "-", False)):
        for head_, col, lw in (("base", BASE_C, 1.4), ("ours", C["ours"], 1.8)):
            arms = pd_.OURS if head_ == "ours" else pd_.BASE
            pts = []
            for c in pd_.CONC:
                _, w = pd_.best(mode, head_, c)
                ce = pd_.cell(mode, arms[w], c)
                pts.append((ce["tok_per_s"], ce["latency_s"]))
            xs_, ys_ = zip(*pts)
            allx += xs_
            ally += ys_
            ax.plot(xs_, ys_, color=col, lw=lw if not hollow else lw * 0.7, ls=ls, zorder=4 if head_ == "ours" else 3)
            ax.plot(xs_, ys_, "o", ms=3.6 if hollow else 4.0, mfc="white" if hollow else col, mec=col,
                    mew=0.8 if hollow else 0.5, zorder=5)
            if head_ == "ours" and mode == "greedy":
                for c, (x, y) in zip(pd_.CONC, pts):
                    off = {1: (1.0, 0.86, "center", "top"), 8: (0.93, 1.08, "right", "bottom"),
                           64: (0.93, 1.05, "right", "bottom")}
                    if c in off:
                        fx, fy, ha, va = off[c]
                        ax.text(x * fx, y * fy, f"{c} req.", ha=ha, va=va, fontsize=SZ["tick"] - 1.0, color=C["ours"])
    ax.set_xlim(600, 24000)
    ax.set_ylim(0.9, 30)
    ax.xaxis.set_major_locator(FixedLocator([1000, 3000, 10000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v / 1000:g}k")
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:g}")
    ax.set_xlabel("tokens per second, whole engine", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("wait per request (s)", fontsize=SZ["label"], labelpad=2)
    H.fs.assert_inside(ax, allx, ally)
    save(fig, "p5c_frontier")


# ------------------------------------------------------------------------------------------------ d lead
def p5d_lead():
    """Our best window against the block-16 head's best window, per load: ahead by % at low load, level at
    high load, a dip at the crossover. The grey band is +-3%."""
    items = [H.item("greedy", C["ours"], kind="line", lw=1.8, marker="o", with_marker=True, ms=4.4),
             H.item("official sampling", C["ours"], kind="line", lw=1.2, ls=(0, (3.2, 1.6)), marker="o",
                    with_marker=True, hollow=True, ms=4.0),
             H.item("within 3%", "#E6E6E6", kind="band")]
    leg = H.legend_height(items, H.COL_W, x_pt=4, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 30.0, 112.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=4, top_pt=0, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, bottom, W - 44 - 6, ah)
    ax.axhspan(-3, 3, color="#E6E6E6", lw=0, zorder=0).set_gid("bg")
    ref(ax.axhline(0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1))
    xs = np.arange(len(pd_.CONC))
    allv = []
    for mode, ls, hollow in (("sample", (0, (3.2, 1.6)), True), ("greedy", "-", False)):
        v = np.array([100 * pd_.lead(mode, c) for c in pd_.CONC])
        allv += list(v)
        ax.plot(xs, v, color=C["ours"], lw=1.8 if not hollow else 1.2, ls=ls, zorder=3)
        ax.plot(xs, v, "o", ms=4.0 if hollow else 4.4, mfc="white" if hollow else C["ours"], mec=C["ours"],
                mew=0.9 if hollow else 0.5, zorder=4)
    for c in (1, 2, 4):
        v = 100 * pd_.lead("greedy", c)
        ax.text(X[c] + 0.12, v + 1.2, f"+{v:.0f}%", ha="left", va="bottom", fontsize=SZ["tick"] - 0.8,
                color=C["ours"])
    v8 = 100 * pd_.lead("greedy", 8)
    ax.text(X[8] + 0.18, v8 - 1.2, f"{v8:.0f}%".replace("-", "−"), ha="left", va="top", fontsize=SZ["tick"] - 0.8,
            color=INK)
    conc_axis(ax)
    ax.set_ylim(-14, 36)
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.yaxis.set_major_formatter(lambda v, _p: (f"{v:+.0f}".replace("-", "−") if v else "0"))
    ax.set_ylabel("our best window vs\nDFlash's best (%)", fontsize=SZ["label"], labelpad=2)
    H.fs.assert_inside(ax, xs, allv)
    save(fig, "p5d_lead")


# ------------------------------------------------------------------------------------------------ e same window
def p5e_same_window():
    """Defence: at the same window the two heads are level (window 8: 4.65 vs 4.67 at one request), so the
    lead at low load is the longer window our head can use, not a better head at equal footing."""
    items = [H.item("Ours, block 32", C["ours"], kind="line", lw=1.6), H.item("DFlash, block 16", BASE_C, kind="line",
                                                                           lw=1.4),
             H.item("window 16", INK, kind="marker", ms=4.2), H.item("window 8", INK, kind="marker", marker="s",
                                                                    ms=3.8)]
    leg = H.legend_height(items, H.COL_W, x_pt=4, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 30.0, 112.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=4, top_pt=0, width_pt=W - 6, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, bottom, W - 44 - 6, ah)
    times_log(ax, 1.5, 8.5, [2, 4, 8])
    xs = np.arange(len(pd_.CONC))
    allv = []
    for w, mk in ((16, "o"), (8, "s")):
        for head_, arm, col, lw in (("base", pd_.BASE[w], BASE_C, 1.3), ("ours", pd_.OURS[w], C["ours"], 1.5)):
            v = [pd_.spd("greedy", arm, c) for c in pd_.CONC]
            allv += v
            ax.plot(xs, v, color=col, lw=lw, zorder=3)
            ax.plot(xs, v, mk, ms=4.0 if mk == "o" else 3.6, mfc=col, mec="white", mew=0.5, zorder=4)
    conc_axis(ax)
    ax.set_ylabel("speed-up over the\noriginal model (greedy)", fontsize=SZ["label"], labelpad=2)
    H.fs.assert_inside(ax, xs, allv)
    save(fig, "p5e_same_window")


ALL = {"p5a_envelope": p5a_envelope, "p5b_knob": p5b_knob, "p5c_frontier": p5c_frontier, "p5d_lead": p5d_lead,
       "p5e_same_window": p5e_same_window}

# ------------------------------------------------------------------------------------------------ f knob + same window
def p5f_knob_full():
    """p5b with DFlash's two windows as rows of their own, so the same-window comparison (p5e) is read off the same
    panel: 'Ours, window 8' and 'DFlash, window 8' sit in one column. Every cell is coloured against DFlash's best at
    that load; black frame = the window we pick, teal frame = the one DFlash picks."""
    rows = [("ours", 32), ("ours", 24), ("ours", 16), ("ours", 8), ("base", 16), ("base", 8)]
    cmap = LinearSegmentedColormap.from_list("lead", ["#5E9A9A", "#A9CFCF", "#F4F4F4", "#9DB7D9", "#2A5C97"])
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=0.30)
    leg_items = [H.item("ahead of DFlash's best", "#6E9AC4"), H.item("level", "#E9E9E9"), H.item("behind", "#7FB3B3"),
                 H.item("our pick", INK, kind="marker", marker="s", hollow=True, ms=6.0, mew=1.2),
                 H.item("DFlash's pick", BASE_C, kind="marker", marker="s", hollow=True, ms=6.0, mew=1.2)]
    leg = H.legend_height(leg_items, H.COL_W, x_pt=4, width_pt=W - 6, fontsize=LEG_FS, gap_pt=6)
    rh, gap, bottom, top_lab, sep = 11.5, 15.0, 30.0, 11.0, 3.0
    ah = rh * len(rows) + sep
    Ht = leg + 2 + 2 * (ah + top_lab) + gap + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, leg_items, x_pt=4, top_pt=0, width_pt=W - 6, fontsize=LEG_FS, gap_pt=6)
    for k, mode in enumerate(("greedy", "sample")):
        y0 = bottom + (0 if k == 1 else ah + top_lab + gap)
        ax = H.axes_pt(fig, 70, y0, W - 70 - 3, ah)
        yk = [i + (sep / rh if head_ == "base" else 0) for i, (head_, _) in enumerate(rows)]
        for (head_, w), y in zip(rows, yk):
            arm = (pd_.OURS if head_ == "ours" else pd_.BASE)[w]
            for j, c in enumerate(pd_.CONC):
                v = pd_.spd(mode, arm, c)
                z = np.log(v / pd_.best(mode, "base", c)[0])
                ax.add_patch(Rectangle((j - 0.5, y - 0.5), 1, 1, facecolor=cmap(norm(z)), edgecolor="white", lw=0.5,
                                       zorder=1))
                t = ax.text(j, y, f"{v:.1f}×", ha="center", va="center", fontsize=SZ["tick"] - 0.9,
                            color=INK if head_ == "ours" else "#2F5F5F", zorder=3)
                t.set_gid("free")
        for j, c in enumerate(pd_.CONC):
            for head_, col in (("ours", INK), ("base", BASE_C)):
                wsel = pd_.best(mode, head_, c)[1]
                i = rows.index((head_, wsel))
                r = Rectangle((j - 0.5 + 0.03, yk[i] - 0.5 + 0.05), 0.94, 0.9, facecolor="none", edgecolor=col, lw=1.3,
                              zorder=4)
                r.set_gid("bg")
                ax.add_patch(r)
        ax.set_xlim(-0.5, len(pd_.CONC) - 0.5)
        ax.set_ylim(yk[-1] + 0.5, -0.5)
        ax.set_yticks(yk, [("Ours" if h == "ours" else "DFlash") + f", window {w}" for h, w in rows],
                      fontsize=SZ["tick"] - 0.8)
        for t in ax.get_yticklabels():
            if t.get_text().startswith("DFlash"):
                t.set_color(TEAL_TXT)
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(-0.5, -0.62, "greedy" if mode == "greedy" else "official sampling", ha="left", va="bottom",
                fontsize=SZ["legend"] - 0.4, color=INK)
        if k == 1:
            ax.set_xticks(range(len(pd_.CONC)), [str(c) for c in pd_.CONC])
            ax.set_xlabel("concurrent requests", fontsize=SZ["label"], labelpad=1.5)
        else:
            ax.set_xticks([])
    save(fig, "p5f_knob_full")


ALL["p5f_knob_full"] = p5f_knob_full


if __name__ == "__main__":
    for n in sys.argv[1:] or list(ALL):
        ALL[n]()
