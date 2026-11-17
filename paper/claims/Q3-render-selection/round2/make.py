"""Q3 round 2: the same claim and data as round 1 (../round1/q3data.py, standardized gains), drawn
cleaner. User on round 1's q3a_rows: "Enough information, but the form is messy. Is there another kind of chart that shows it more clearly and nicely?"

  q3d_rulebars   one row per rule: the median over the seven model x metric cells as a bar from zero, the seven cells
                 as dots coloured by model; 5 rows instead of 35 marks with 35 intervals
  q3e_bump       one column per model x metric, the five rules ranked 1-5 in each; one line per rule -- the render score
                 sits on top in 5 of 7 columns, the highest likelihood at the bottom in all 7; effect sizes on the
                 render score's points
  q3f_dumbbell   one row per model x metric, a line from the best text-only rule to the render score, the highest
                 likelihood as a dark mark left of zero; three marks per row instead of five

Run: python3 paper/claims/Q3-render-selection/round2/make.py   (STRICT=0 for drafts)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R1 = HERE.parent / "round1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(R1))
import house as H  # noqa: E402
import q3data as Q  # noqa: E402

H.apply()
SZ, C, INK, MUTED = H.SZ, H.C, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W = H.COL_W * 72
LEG_FS = 7.6
BLOCKS = ["OmniSVG-8B, text-to-SVG", "OmniSVG-8B, image-to-SVG", "InternSVG-8B, text-to-SVG, T=0.6"]
BLAB = {"OmniSVG-8B, text-to-SVG": "OmniSVG-8B, text", "OmniSVG-8B, image-to-SVG": "OmniSVG-8B, image",
        "InternSVG-8B, text-to-SVG, T=0.6": "InternSVG-8B, text"}
BCOL = {"OmniSVG-8B, text-to-SVG": "#C4859B", "OmniSVG-8B, image-to-SVG": "#8E9A4F",   # not method colours
        "InternSVG-8B, text-to-SVG, T=0.6": "#6E7F8F"}
MLAB = {"aesthetic": "aesthetic", "hps": "HPS", "ssim": "SSIM", "lpips": "LPIPS", "mse": "MSE"}
MSHORT = {"aesthetic": "aesth.", "hps": "HPS", "ssim": "SSIM", "lpips": "LPIPS", "mse": "MSE"}
NPROMPT = {"OmniSVG-8B, text-to-SVG": 900, "OmniSVG-8B, image-to-SVG": 900, "InternSVG-8B, text-to-SVG, T=0.6": 300}
RLAB = {"max likelihood": "highest likelihood", "min likelihood": "lowest likelihood", "text consensus": "text consensus",
        "render consensus": "pixel consensus", "render score": "render score (ours)"}
ORDER = ["render score", "text consensus", "min likelihood", "render consensus", "max likelihood"]


def finish(fig):
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if "\n" in t.get_text() and getattr(t, "_linespacing", 1.2) < 1.12:
            t.set_linespacing(1.12)


def save(fig, name):
    finish(fig)
    H.save(fig, str(HERE / name), width="col", strict=STRICT)
    plt.close(fig)


def cells():
    return [(b, m) for b in BLOCKS for m in Q.independent(b)]


def eff(b, rule, m):
    d = Q.delta(b, rule, m)
    if d is None:
        return None
    sd = (d[2] - d[1]) / (2 * 1.96) * NPROMPT[b] ** 0.5
    return d[0] / sd


def fmt(v):
    return (f"{v:+.1f}".replace("-", "−")) if v else "0"


# ------------------------------------------------------------------------------------------------ d rule bars
def q3d_rulebars():
    items = [H.item(BLAB[b], BCOL[b], kind="marker", ms=4.2) for b in BLOCKS] + \
            [H.item("median", "#D6D6D6")]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    rh, bottom = 21.0, 28.0
    ah = rh * len(ORDER) + 4
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 84, bottom, W - 84 - 8, ah)
    ax.axvline(0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    for i, r in enumerate(ORDER):
        vals = [(b, eff(b, r, m)) for b, m in cells() if eff(b, r, m) is not None]
        med = float(np.median([v for _, v in vals]))
        ours = r == "render score"
        ax.barh(i, med, height=0.56, color=C["ours_light"] if ours else "#E2E2E2", lw=0, zorder=1)
        jit = np.linspace(-0.17, 0.17, len(vals))
        for (b, v), dy in zip(vals, jit):
            ax.plot([v], [i + dy], "o", ms=4.2, mfc=BCOL[b], mec="white", mew=0.5, zorder=3)
        ax.text(max(v for _, v in vals) + 0.05, i, fmt(med), ha="left", va="center", fontsize=7.4,
                color=C["ours"] if ours else INK, fontweight="bold" if ours else "normal")
    ax.set_ylim(len(ORDER) - 0.5, -0.6)
    ax.set_yticks(range(len(ORDER)), [RLAB[r] for r in ORDER], fontsize=SZ["tick"] - 0.4)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("render score"):
            t.set_color(C["ours"]), t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.52, 1.02)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.xaxis.set_major_formatter(lambda v, _p: fmt(v))
    ax.set_xlabel("gain over a random pick (s.d.)", fontsize=SZ["label"], labelpad=1.5)
    save(fig, "q3d_rulebars")


# ------------------------------------------------------------------------------------------------ e bump (ranks)
def q3e_bump():
    rules = ["render score", "text consensus", "min likelihood", "render consensus", "max likelihood"]
    rcol = {"render score": C["ours"], "text consensus": "#5A5A5A", "min likelihood": "#9A9A9A",
            "render consensus": "#C4C4C4", "max likelihood": "#2E2E2E"}
    rls = {"render score": "-", "text consensus": "-", "min likelihood": (0, (2.5, 1.5)), "render consensus": (0, (1, 1.2)),
           "max likelihood": "-"}
    X = cells()
    items = [H.item(RLAB[r], rcol[r], kind="line", lw=2.0 if r == "render score" else 1.3, ls=rls[r], marker="o",
                    with_marker=True, ms=4.4 if r == "render score" else 3.6, bold=r == "render score",
                    text_color=C["ours"] if r == "render score" else None) for r in rules]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 36.0, 104.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    left, right = 30.0, 8.0
    ax = H.axes_pt(fig, left, bottom, W - left - right, ah)
    xs = np.arange(len(X), dtype=float)
    # gaps between models
    pos, k, prev = [], 0.0, None
    for b, m in X:
        if prev is not None and b != prev:
            k += 0.45
        pos.append(k)
        k += 1.0
        prev = b
    pos = np.array(pos)
    ranks = {r: [] for r in rules}
    for j, (b, m) in enumerate(X):
        avail = [r for r in rules if eff(b, r, m) is not None]
        order = sorted(avail, key=lambda r: -eff(b, r, m))
        for r in rules:
            ranks[r].append(order.index(r) + 1 if r in avail else np.nan)
    for r in rules[::-1]:
        y = np.array(ranks[r], float)
        ok = ~np.isnan(y)
        ax.plot(pos[ok], y[ok], color=rcol[r], lw=2.0 if r == "render score" else 1.3, ls=rls[r],
                zorder=4 if r == "render score" else 3, solid_capstyle="round")
        ax.plot(pos[ok], y[ok], "o", ms=5.4 if r == "render score" else 4.2, mfc=rcol[r], mec="white", mew=0.6,
                zorder=5 if r == "render score" else 3)
    for j, (b, m) in enumerate(X):
        v = eff(b, "render score", m)
        rr = ranks["render score"][j]
        if rr == 1:
            ax.text(pos[j], rr - 0.3, fmt(v), ha="center", va="bottom", fontsize=7.0, color=C["ours"])
        else:
            ax.text(pos[j] - 0.12, rr + 0.22, fmt(v), ha="right", va="top", fontsize=7.0, color=C["ours"])
    ax.set_ylim(5.55, 0.35)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylabel("rank (1 = best)", fontsize=SZ["label"] - 0.5, labelpad=2)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(pos[0] - 0.4, pos[-1] + 0.4)
    groups = []
    for b in BLOCKS:
        g = [p for p, (bb, _m) in zip(pos, X) if bb == b]
        groups.append((BLAB[b].replace(", ", "\n"), g))
    H.fs.two_row_xaxis(ax, list(pos), [MSHORT[m] for _b, m in X], groups, line_down_pt=11.0, name_down_pt=12.0,
                       pad=0.4, fs_group=SZ["tick"] - 0.8)
    for t in ax.get_xticklabels():
        t.set_fontsize(SZ["tick"] - 1.0)
    save(fig, "q3e_bump")


# ------------------------------------------------------------------------------------------------ f dumbbell
def q3f_dumbbell():
    items = [H.item("render score (ours)", C["ours"], kind="marker", ms=5.0, bold=True, text_color=C["ours"]),
             H.item("best rule without a score", "#7A7A7A", kind="marker", marker="D", ms=4.0),
             H.item("highest likelihood", "#2E2E2E", kind="marker", marker="v", ms=4.6)]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    X = cells()
    rh, bottom = 14.0, 28.0
    ah = rh * len(X) + 2 * 8
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 96, bottom, W - 96 - 8, ah)
    ax.axvline(0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    y, prev, ticks, labs, heads = 0.0, None, [], [], []
    for b, m in X:
        if b != prev:
            if prev is not None:
                y += 0.6
            heads.append((b, y))
            prev = b
        rnd = eff(b, "render score", m)
        others = {r: eff(b, r, m) for r in ("text consensus", "min likelihood", "render consensus")
                  if eff(b, r, m) is not None}
        best_r = max(others, key=others.get)
        bt = others[best_r]
        lik = eff(b, "max likelihood", m)
        ax.plot([min(bt, rnd), max(bt, rnd)], [y, y], color="#C9C9C9", lw=2.2, zorder=2, solid_capstyle="round")
        ax.plot([lik], [y], "v", ms=4.6, mfc="#2E2E2E", mec="white", mew=0.4, zorder=3)
        ax.plot([bt], [y], "D", ms=4.0, mfc="#7A7A7A", mec="white", mew=0.4, zorder=3)
        ax.plot([rnd], [y], "o", ms=5.2, mfc=C["ours"], mec="white", mew=0.5, zorder=4)
        ticks.append(y)
        labs.append(MLAB[m])
        y += 1.0
    ax.set_ylim(y - 0.4, -0.95)
    ax.set_yticks(ticks, labs, fontsize=SZ["tick"] - 0.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for b, yh in heads:
        ax.text(-0.012, yh - 0.55, BLAB[b], ha="right", va="bottom", fontsize=SZ["legend"] - 0.8, color=INK, transform=tr,
                fontweight="bold")
    ax.set_xlim(-0.52, 1.0)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.xaxis.set_major_formatter(lambda v, _p: fmt(v))
    ax.set_xlabel("gain over a random pick (s.d.)", fontsize=SZ["label"], labelpad=1.5)
    save(fig, "q3f_dumbbell")


ALL = {"q3d_rulebars": q3d_rulebars, "q3e_bump": q3e_bump, "q3f_dumbbell": q3f_dumbbell}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(ALL):
        ALL[n]()
