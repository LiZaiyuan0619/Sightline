"""Q3 -- picking needs the render. Round 1, night, house style, single column.

Claim: given eight candidates, a pick that never renders does not recover the render score's gain. The model's own
confidence (highest likelihood) picks WORSE than random on every independent metric; text consensus (self-consistency,
MBR over token n-grams) recovers part of the gain; pixel consensus of the renders recovers nothing; only scoring the
render helps across the board. (Independent metrics only: the render score picks by CLIP / DINO itself.)

Scope: OmniSVG-8B text / image, InternSVG-8B text at its main temperature T=0.6 (T=0.3 and T=1.0 in the appendix).
x = the rule's gain over a random pick in standard deviations of the paired per-prompt difference (the SD is recovered
from each row's 95% bootstrap CI and its number of prompts), so every metric shares one scale. (A first draft divided by
the render score's own gain; that ratio blows up where the render gain is small and noisy -- aesthetic -- so it was
replaced.)
Data: q3data.py. Run: python3 paper/claims/Q3-render-selection/round1/make.py   (STRICT=0 for drafts)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
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
MLAB = {"aesthetic": "aesthetic", "hps": "HPS", "ssim": "SSIM", "lpips": "LPIPS", "mse": "MSE"}
STYLE = {  # the non-render rules in greys with distinct marks; the render score is ours (blue)
    "max likelihood": dict(color="#3E3E3E", marker="v", ms=4.8, label="highest likelihood"),
    "min likelihood": dict(color="#9A9A9A", marker="^", ms=4.4, label="lowest likelihood"),
    "text consensus": dict(color="#6F6F6F", marker="D", ms=4.0, label="text consensus"),
    "render consensus": dict(color="#B8B8B8", marker="o", ms=4.0, label="pixel consensus"),
    "render score": dict(color=C["ours"], marker="o", ms=5.2, label="render score (ours)"),
}
LO, HI = -0.55, 1.0
NPROMPT = {"OmniSVG-8B, text-to-SVG": 900, "OmniSVG-8B, image-to-SVG": 900, "InternSVG-8B, text-to-SVG, T=0.6": 300}


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
    return [(b, m) for b in BLOCKS for m in Q.independent(b)]


def norm(b, rule, m):
    """(effect, lo, hi) in SD units of the paired per-prompt difference."""
    d = Q.delta(b, rule, m)
    if d is None:
        return None
    sd = (d[2] - d[1]) / (2 * 1.96) * NPROMPT[b] ** 0.5
    return tuple(v / sd for v in d)


def q3a_rows():
    items = [H.item(STYLE[r]["label"], STYLE[r]["color"], kind="marker", marker=STYLE[r]["marker"], ms=STYLE[r]["ms"],
                    bold=r == "render score", text_color=C["ours"] if r == "render score" else None) for r in Q.RULES]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    R = rows()
    rh, bottom = 13.0, 28.0
    ah = rh * len(R) + 3 * 7
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 96, bottom, W - 96 - 8, ah)
    ax.axvline(0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    y, prev, ticks, labs, heads = 0.0, None, [], [], []
    for b, m in R:
        if b != prev:
            if prev is not None:
                y += 0.55
            heads.append((b, y))
            prev = b
        for k, rule in enumerate(Q.RULES):
            v = norm(b, rule, m)
            if v is None:
                continue
            st = STYLE[rule]
            yy = y + (k - 2) * 0.13
            x, lo, hi = v
            xc = min(max(x, LO + 0.04), HI - 0.04)
            if x != xc:      # off the scale: an edge mark with the value
                ax.annotate(f"{x:.1f}", (xc, yy), xytext=(4 if x > 0 else -4, 0), textcoords="offset points",
                            ha="left" if x > 0 else "right", va="center", fontsize=7.0, color=st["color"])
            else:
                ax.plot([max(lo, LO), min(hi, HI)], [yy, yy], color=st["color"], lw=0.8, zorder=2, alpha=0.9)
            ax.plot([xc], [yy], st["marker"], ms=st["ms"] - 0.8, mfc=st["color"], mec="white", mew=0.4, zorder=3)
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
        ax.text(-0.012, yh - 0.5, BLAB[b], ha="right", va="bottom", fontsize=SZ["legend"] - 0.8, color=INK,
                transform=tr, fontweight="bold")
    ax.set_xlim(LO, HI)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:+.1f}".replace("-", "\u2212") if v else "0")
    ax.set_xlabel("gain over a random pick (s.d.)", fontsize=SZ["label"], labelpad=1.5)
    save(fig, "q3a_rows")


def q3b_ladder():
    """Rules ordered from worst to best on the x axis, one line per model x independent metric; every line starts below
    zero (highest likelihood is worse than random) and ends at 1 (the render score)."""
    order = ["max likelihood", "render consensus", "min likelihood", "text consensus", "render score"]
    cols = {"OmniSVG-8B, text-to-SVG": "#2F6D6D", "OmniSVG-8B, image-to-SVG": "#B4A8D4",
            "InternSVG-8B, text-to-SVG, T=0.6": "#C3A77C"}
    items = [H.item(BLAB[b], cols[b], kind="line", lw=1.3, marker="o", with_marker=True, ms=3.4) for b in BLOCKS]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 34.0, 120.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 46, bottom, W - 46 - 8, ah)
    ax.axhline(0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    ally = []
    for b, m in rows():
        xs, ys = [], []
        for i, r in enumerate(order):
            v = norm(b, r, m)
            if v is None:
                continue
            xs.append(i)
            ys.append(min(max(v[0], LO + 0.05), HI - 0.05))
        ally += ys
        ax.plot(xs, ys, color=cols[b], lw=1.1, zorder=2, alpha=0.95)
        ax.plot(xs, ys, "o", ms=3.4, mfc=cols[b], mec="white", mew=0.4, zorder=3)
    ax.set_xticks(range(len(order)), ["highest\nlikelihood", "pixel\nconsensus", "lowest\nlikelihood", "text\nconsensus",
                                      "render\nscore"], fontsize=SZ["tick"] - 1.0)
    for t in ax.get_xticklabels():
        if "render\nscore" in t.get_text():
            t.set_color(C["ours"]), t.set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(LO, HI)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:+.1f}".replace("-", "\u2212") if v else "0")
    ax.set_ylabel("gain over a random\npick (s.d.)", fontsize=SZ["label"], labelpad=2)
    save(fig, "q3b_ladder")


def q3c_strip():
    """Rules on the x axis (worst to best), the seven model x metric gains as dots, the median as a bar."""
    order = ["max likelihood", "render consensus", "min likelihood", "text consensus", "render score"]
    items = [H.item("median", "#D9D9D9"), H.item("one model x metric", INK, kind="marker", ms=3.6),
             H.item("render score (ours)", C["ours"], kind="marker", ms=4.4, bold=True, text_color=C["ours"])]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    bottom, ah = 34.0, 112.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=W - 5, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 46, bottom, W - 46 - 8, ah)
    ax.axhline(0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1).set_gid("ref")
    rng = np.random.default_rng(0)
    for i, r in enumerate(order):
        vals = [norm(b, r, m)[0] for b, m in rows() if norm(b, r, m) is not None]
        med = float(np.median(vals))
        col = C["ours"] if r == "render score" else "#D9D9D9"
        ax.bar(i, med, width=0.62, color=C["ours_light"] if r == "render score" else col, lw=0, zorder=1)
        jit = np.linspace(-0.2, 0.2, len(vals))
        ax.plot(i + jit, vals, "o", ms=3.4, mfc=C["ours"] if r == "render score" else "#5A5A5A", mec="white", mew=0.4,
                zorder=3, ls="none")
        ax.text(i, max(vals) + 0.05, f"{med:+.2f}".replace("-", "\u2212"), ha="center", va="bottom", fontsize=7.0,
                color=C["ours"] if r == "render score" else INK)
    ax.set_xticks(range(len(order)), ["highest\nlikelihood", "pixel\nconsensus", "lowest\nlikelihood",
                                      "text\nconsensus", "render\nscore"], fontsize=SZ["tick"] - 1.0)
    for t in ax.get_xticklabels():
        if "render\nscore" in t.get_text():
            t.set_color(C["ours"]), t.set_fontweight("bold")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.55, 1.0)
    ax.set_yticks([-0.5, 0, 0.5])
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:+.1f}".replace("-", "\u2212") if v else "0")
    ax.set_ylabel("gain over a random\npick (s.d.)", fontsize=SZ["label"], labelpad=2)
    save(fig, "q3c_strip")


ALL = {"q3a_rows": q3a_rows, "q3b_ladder": q3b_ladder, "q3c_strip": q3c_strip}

if __name__ == "__main__":
    for n in sys.argv[1:] or list(ALL):
        ALL[n]()
