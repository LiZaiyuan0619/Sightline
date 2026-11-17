"""Q2 -- three VLM judges, round 1 (night, house style, single column).

Claim: three judges from three vendors agree with the render-and-score frontier. Our best of 8 is preferred over
one sample of the original model about 2: 1 (0.69 / 0.67) while taking less time than that one sample; against the
original model's own best of 8 it is a tie (0.49 / 0.51) at a quarter of the time; and our single sample ties the
original model's single sample (0.48 / 0.52), so the judges carry no bias for or against our outputs.

Data: qdata.py (judge reports + seconds from the Q1 data pack). Two blocks: OmniSVG-8B and InternSVG-8B text-to-SVG.
Run:  python3 paper/claims/Q2-vlm-judge/round1/make.py   (STRICT=0 for drafts)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import qdata as Q  # noqa: E402

H.apply()
SZ, C, M, INK, MUTED = H.SZ, H.C, H.METHOD, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W = H.COL_W * 72
LEG_FS = 7.6
BLK = list(Q.RUNS)
BLK_LAB = {"OmniSVG-8B, text-to-SVG": "OmniSVG-8B, text-to-SVG", "InternSVG-8B, text-to-SVG": "InternSVG-8B, text-to-SVG"}
AR_C, AR_LIGHT, TIE_C = C["ar"], "#BDBDBD", "#E4E4E4"
NAME = {"sd1s": "Ours, 1 sample", "sd8": "Ours, best of 8", "sd16": "Ours, best of 16", "ar1s": "1 sample",
        "ar8s": "best of 8"}
COMPS = [("sd1s", "ar1s"), ("sd8", "ar1s"), ("sd8", "ar8s"), ("sd16", "ar8s")]


def finish(fig):
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if "\n" in t.get_text() and getattr(t, "_linespacing", 1.2) < 1.12:
            t.set_linespacing(1.12)


def save(fig, name):
    finish(fig)
    H.save(fig, str(HERE / name), width="col", strict=STRICT)
    plt.close(fig)


def rows():
    out = []
    for b in BLK:
        for a, c in COMPS:
            if (a, c) in Q.P[b]["claude-opus-5-5"]:
                out.append((b, a, c))
    return out


def sec(b, arm):
    return Q.SECONDS[(b, arm)]


# ------------------------------------------------------------------------------------------------ a win / tie / loss
def q2a_winbars():
    """User-study bars: judgements pooled over the three judges (1,800 per row: 300 prompts x 2 orders x 3 judges).
    Right of the centre = ours preferred, left = the original model preferred, the grey middle = ties.
    The right column is seconds per prompt, ours vs the original model."""
    items = [H.item("Ours preferred", C["ours"]), H.item("tie", TIE_C), H.item("original model preferred", AR_LIGHT)]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    R = rows()
    rh, head, bottom = 15.0, 12.0, 26.0
    ah = rh * len(R) + 10
    Ht = leg + 2 + head + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    left, right = 104.0, 40.0
    ax = H.axes_pt(fig, left, bottom, W - left - right, ah)
    ys = []
    y = 0.0
    prev = None
    for b, a, c in R:
        if prev is not None and b != prev:
            y += 0.7
        prev = b
        win, tie, loss = Q.counts(b, a, c)
        n = win + tie + loss
        w, t, l = win / n, tie / n, loss / n
        ax.barh(y, w, left=t / 2, height=0.66, color=C["ours"], lw=0, zorder=2)
        ax.barh(y, t, left=-t / 2, height=0.66, color=TIE_C, lw=0, zorder=2)
        ax.barh(y, l, left=-t / 2 - l, height=0.66, color=AR_LIGHT, lw=0, zorder=2)
        p = Q.pref(b, a, c)
        ax.text(t / 2 + w + 0.02, y, f"{p:.2f}", ha="left", va="center", fontsize=SZ["tick"] - 0.6,
                color=C["ours"], fontweight="bold" if p > 0.6 else "normal")
        ax.text(1.0 + (right / 2) / (W - left - right), y, f"{sec(b, a):.1f} / {sec(b, c):.1f}",
                ha="center", va="center", fontsize=SZ["tick"] - 1.0, color=INK, transform=ax.get_yaxis_transform())
        ys.append((y, b, a, c))
        y += 1.0
    ax.axvline(0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=3).set_gid("ref")
    ax.set_xlim(-0.75, 0.95)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_yticks([v for v, *_ in ys],
                  [f"ours {NAME[a].replace('Ours, ', '')} vs {NAME[c]}" for _, _, a, c in ys],
                  fontsize=SZ["tick"] - 0.9)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.xaxis.set_major_locator(FixedLocator([-0.5, 0, 0.5]))
    ax.xaxis.set_major_formatter(lambda v, _p: f"{abs(v) * 100:.0f}%")
    ax.set_xlabel("share of judgements", fontsize=SZ["label"], labelpad=1.5)
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    # block headers and the seconds header
    blocks_y = {}
    for v, b, *_ in ys:
        blocks_y.setdefault(b, []).append(v)
    for b, vs in blocks_y.items():
        from matplotlib.transforms import blended_transform_factory
        ax.text(-0.012, min(vs) - 0.62, BLK_LAB[b], ha="right", va="bottom", fontsize=SZ["legend"] - 0.6, color=INK,
                transform=blended_transform_factory(ax.transAxes, ax.transData), clip_on=False, fontweight="bold")
    ax.text(1.0 + (right / 2) / (W - left - right), -0.62, "seconds,\nours / theirs", ha="center",
            va="bottom", fontsize=7.0, color=MUTED, transform=ax.get_yaxis_transform())
    save(fig, "q2a_winbars")


# ------------------------------------------------------------------------------------------------ b forest
def q2b_forest():
    """Forest plot: each judge's preference with its 95% CI, the mean as a diamond; the dashed line is a tie."""
    jm = {"claude-opus-5-5": ("o", "Claude Opus 5.5"), "gpt-6.1-sol": ("s", "GPT-6.1"), "gemini-3.8-flash": ("^", "Gemini 3.8 Flash")}
    items = [H.item(lab, INK, kind="marker", marker=mk, ms=4.0) for _, (mk, lab) in jm.items()] + \
            [H.item("mean of three", C["ours"], kind="marker", marker="D", ms=4.6)]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    R = rows()
    rh, bottom = 17.0, 26.0
    ah = rh * len(R) + 14
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    left = 104.0
    ax = H.axes_pt(fig, left, bottom, W - left - 6, ah)
    ax.axvline(0.5, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    y, prev, ticks, labs = 0.0, None, [], []
    allx = []
    for b, a, c in R:
        if prev is not None and b != prev:
            y += 0.8
        prev = b
        for k, j in enumerate(Q.JUDGES):
            d = Q.P[b][j][(a, c)]
            yy = y + (k - 1) * 0.22
            ax.plot([d["lo"], d["hi"]], [yy, yy], color="#8A8A8A", lw=0.9, zorder=2, solid_capstyle="butt")
            ax.plot([d["pref"]], [yy], jm[j][0], ms=3.6, mfc=INK, mec="white", mew=0.4, zorder=3)
            allx += [d["lo"], d["hi"]]
        p = Q.pref(b, a, c)
        ax.plot([p], [y], "D", ms=4.6, mfc=C["ours"], mec="white", mew=0.5, zorder=4)
        ticks.append(y)
        labs.append(f"{NAME[a].replace('Ours, ', '')} vs {NAME[c]}")
        y += 1.0
    ax.set_ylim(y - 0.35, -0.75)
    ax.set_yticks(ticks, labs, fontsize=SZ["tick"] - 0.8)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.4, 0.78)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel("judges' preference for ours (0.5 = tie)", fontsize=SZ["label"], labelpad=1.5)
    by = {}
    for t, (b, *_r) in zip(ticks, R):
        by.setdefault(b, []).append(t)
    for b, vs in by.items():
        from matplotlib.transforms import blended_transform_factory
        ax.text(-0.012, min(vs) - 0.55, BLK_LAB[b], ha="right", va="bottom", fontsize=SZ["legend"] - 0.6, color=INK,
                transform=blended_transform_factory(ax.transAxes, ax.transData), clip_on=False, fontweight="bold")
    H.fs.assert_inside(ax, allx, [0])
    save(fig, "q2b_forest")


# ------------------------------------------------------------------------------------------------ c curve over N
def q2c_curve():
    """Preference over one sample of the original model, as the number of candidates grows: the two best-of-N lines lie
    on top of each other (same quality at the same N); the numbers are seconds per prompt -- ours is the cheap one."""
    items = [H.item("Ours, best of N", C["ours"], kind="line", lw=1.8, marker="o", with_marker=True, ms=4.2,
                    bold=True, text_color=C["ours"]),
             H.item("original model, best of N", AR_C, kind="line", lw=1.4, marker="o", with_marker=True, ms=3.8),
             H.item("OmniSVG-8B", INK, kind="line", lw=1.2), H.item("InternSVG-8B", INK, kind="line", lw=1.0,
                                                                    ls=(0, (3.2, 1.6)))]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 30.0, 124.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, bottom, W - 44 - 8, ah)
    Ns = [1, 2, 4, 8, 16]
    X = {n: i for i, n in enumerate(Ns)}
    ax.axhline(0.5, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    ally = []
    for b, ls in zip(BLK, ("-", (0, (3.2, 1.6)))):
        ar = [(1, 0.5, sec(b, "ar1s"))] + [(n, Q.pref(b, f"ar{n}s", "ar1s"), sec(b, f"ar{n}s")) for n in (2, 4, 8)]
        sd = [(1, Q.pref(b, "sd1s", "ar1s"), sec(b, "sd1s"))] + \
             [(n, Q.pref(b, f"sd{n}", "ar1s"), sec(b, f"sd{n}")) for n in (2, 4, 8, 16)]
        for pts, col, lw, z in ((ar, AR_C, 1.4, 3), (sd, C["ours"], 1.8, 4)):
            xs = [X[n] for n, _, _ in pts]
            ys = [p for _, p, _ in pts]
            ally += ys
            ax.plot(xs, ys, color=col, lw=lw if ls == "-" else lw * 0.8, ls=ls, zorder=z)
            ax.plot(xs, ys, "o", ms=4.0, mfc=col, mec="white", mew=0.5, zorder=z + 1)
        if ls == "-":
            for n, p, s in sd:
                if n in (8, 16):
                    ax.text(X[n], p + 0.012, f"{s:.1f} s", ha="center", va="bottom", fontsize=7.0,
                            color=C["ours"])
            for n, p, s in ar:
                if n in (8,):
                    ax.text(X[n] + 0.08, p - 0.012, f"{s:.1f} s", ha="left", va="top", fontsize=7.0,
                            color=AR_C)
    ax.set_xticks(range(len(Ns)), [str(n) for n in Ns])
    ax.set_xlim(-0.3, len(Ns) - 0.6)
    ax.set_ylim(0.44, 0.74)
    ax.set_yticks([0.5, 0.6, 0.7])
    ax.set_xlabel("candidates N (best of N, rendered and scored)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("judges prefer it over one\nsample of the original model", fontsize=SZ["label"] - 0.5, labelpad=2)
    H.fs.assert_inside(ax, [0, 4], ally)
    save(fig, "q2c_curve")


# ------------------------------------------------------------------------------------------------ d frontier (judges)
def q2d_frontier():
    """The Q1 frontier with the judges as the quality axis: preference over one sample of the original model against
    seconds per prompt; same N labels as Q1."""
    items = [H.item("Ours, best of N", C["ours"], kind="line", lw=1.8, marker="o", with_marker=True, ms=4.2,
                    bold=True, text_color=C["ours"]),
             H.item("original model, best of N", AR_C, kind="line", lw=1.4, marker="o", with_marker=True, ms=3.8),
             H.item("OmniSVG-8B", INK, kind="line", lw=1.2), H.item("InternSVG-8B", INK, kind="line", lw=1.0,
                                                                    ls=(0, (3.2, 1.6)))]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 30.0, 124.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, bottom, W - 44 - 8, ah)
    ax.set_xscale("log")
    ax.axhline(0.5, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    allx, ally = [], []
    for b, ls in zip(BLK, ("-", (0, (3.2, 1.6)))):
        ar = [(1, 0.5, sec(b, "ar1s"))] + [(n, Q.pref(b, f"ar{n}s", "ar1s"), sec(b, f"ar{n}s")) for n in (2, 4, 8)]
        sd = [(1, Q.pref(b, "sd1s", "ar1s"), sec(b, "sd1s"))] + \
             [(n, Q.pref(b, f"sd{n}", "ar1s"), sec(b, f"sd{n}")) for n in (2, 4, 8, 16)]
        for pts, col, lw, z in ((ar, AR_C, 1.4, 3), (sd, C["ours"], 1.8, 4)):
            xs = [s for _, _, s in pts]
            ys = [p for _, p, _ in pts]
            allx += xs
            ally += ys
            ax.plot(xs, ys, color=col, lw=lw if ls == "-" else lw * 0.8, ls=ls, zorder=z)
            ax.plot(xs, ys, "o", ms=4.0, mfc=col, mec="white", mew=0.5, zorder=z + 1)
            if ls == "-":
                for n, p, s in pts:
                    if n == 8:
                        ax.text(s / 1.07, p + 0.006, "best of 8", ha="right", va="bottom", fontsize=7.0, color=col)
    ax.set_xlim(1.0, 70)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:g}")
    ax.set_ylim(0.44, 0.74)
    ax.set_yticks([0.5, 0.6, 0.7])
    ax.set_xlabel("seconds per prompt", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("judges prefer it over one\nsample of the original model", fontsize=SZ["label"] - 0.5, labelpad=2)
    H.fs.assert_inside(ax, allx, ally)
    save(fig, "q2d_frontier")


ALL = {"q2a_winbars": q2a_winbars, "q2b_forest": q2b_forest, "q2c_curve": q2c_curve, "q2d_frontier": q2d_frontier}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(ALL):
        ALL[n]()
