"""Speed-quality landscape, round 2 --  evening. Round 1 stays as it is.

User on round 1:
  - FID and CLIP can share one x axis: one figure, FID on top, CLIP below (saves space).
  - The dots are still too small; the text is hard to read -- make both bigger.
  - The legend is confusing: shapes (circle, square) should be labelled directly on the plot, not mixed into a legend.
  - Drop the words "reported" / "measured"; those details go to the appendix.

What changed: two stacked panels on a shared log-seconds axis; markers ~2x the area, labels 8.5 pt; our four arrows
(each model's single sample and best of 8, AR -> ours) are named by leader lines in a band above the data, so the
circle / square never needs a legend; the legend keeps colours only (as patches). The loop-control points (Ours (fast))
are left out here -- at this marker size they sit on top of Ours, and table 2 carries them.
Data: as round 1 (runs/2026-11-03/table2_refrows.json + tex/tables/table2a.tex; SArena-Icon only).
Run: python3 paper/claims/T2-landscape/round2/make.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("land1", HERE.parent / "round1/make.py")
l1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l1)
fs = l1.fs
W = l1.FULL_W_IN
OURS_C, AR_C = l1.OURS_C, l1.AR_C
OPT_C, LLM_C = "#C9A27E", "#9C8CC0"
CAT2 = {"optimisation": OPT_C, "LLM, reported": LLM_C, "LLM, measured": LLM_C}   # one colour per kind
SZ = dict(tick=8.0, label=9.0, text=8.5, other=55, ar=55, ours=70)
plt.rcParams.update({"xtick.labelsize": SZ["tick"], "ytick.labelsize": SZ["tick"]})

# where each other system's name goes, per panel: "r" right of the point, "l" left, "ur" above right
SIDE = {"fid": {"VectorFusion": "r", "SVGDreamer": "l", "LLM4SVG-7B": "l", "OmniSVG-3B": "l", "SVGen-3B": "r",
                "SVGen-7B": "r", "HiVG-3B": "r"},
        "clip": {"VectorFusion": "r", "SVGDreamer": "l", "LLM4SVG-7B": "r", "OmniSVG-3B": "l", "SVGen-3B": "dr",
                 "SVGen-7B": "r", "HiVG-3B": "ur"}}
XLIM = (0.5, 450.0)


def fx(x):
    """data x -> axes fraction on the log axis"""
    return np.log10(x / XLIM[0]) / np.log10(XLIM[1] / XLIM[0])


def panel(ax, others, ours, key, better_low, ylim, yticks, ylab, band_y):
    X, Y = [], []
    for o in others:
        if o["name"] == "InternSVG-8B":
            continue                                  # drawn below with the band labels
        ax.scatter(o["s"], o[key], s=SZ["other"], color=CAT2[o["cat"]], edgecolor="white", linewidths=0.6, zorder=4)
        X.append(o["s"])
        Y.append(o[key])
        side = SIDE[key][o["name"]]
        dx, dy, ha, va = {"r": (5, 0, "left", "center"), "l": (-5, 0, "right", "center"),
                          "ur": (4, 6, "left", "bottom"), "dr": (4, -6, "left", "top")}[side]
        ax.annotate(o["name"], (o["s"], o[key]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=SZ["text"], color="#3E3E3E")
    isvg = next(o for o in others if o["name"] == "InternSVG-8B")
    ax.scatter(isvg["s"], isvg[key], s=SZ["other"], color=CAT2[isvg["cat"]], edgecolor="white", linewidths=0.6,
               zorder=4)
    X.append(isvg["s"])
    Y.append(isvg[key])
    anchors = [(isvg["s"], isvg[key], "InternSVG-8B", "#6E5FA0")]
    for model, mk in (("OmniSVG-8B", "o"), ("InternSVG-8B", "s")):
        for a_lab, b_lab, name in (("AR", "Ours", model), ("AR x8", "Ours x8", f"{model}, best of 8")):
            a, b = ours[(model, a_lab)], ours[(model, b_lab)]
            ax.annotate("", xy=(b["s"] * 1.16, b[key]), xytext=(a["s"] / 1.16, a[key]),
                        arrowprops=dict(arrowstyle="-|>", color=OURS_C, lw=1.0, shrinkA=0, shrinkB=0,
                                        mutation_scale=8), zorder=3)
            ax.scatter(a["s"], a[key], s=SZ["ar"], marker=mk, facecolor="white", edgecolor=AR_C, linewidths=1.1,
                       zorder=5)
            ax.scatter(b["s"], b[key], s=SZ["ours"], marker=mk, color=OURS_C, edgecolor="white", linewidths=0.6,
                       zorder=6)
            X += [a["s"], b["s"]]
            Y += [a[key], b[key]]
            anchors.append((np.sqrt(a["s"] * b["s"]), (a[key] + b[key]) / 2, name, OURS_C))
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    if better_low:
        ax.invert_yaxis()                             # better is up
    fs.assert_inside(ax, X, Y)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylab, fontsize=SZ["label"], labelpad=2)
    ax.tick_params(axis="both", pad=1.5)
    # band labels: ONE row, in the order of the anchors' x, so no leader line crosses another line or another
    # label (both ends are monotonic); widths measured on the text alone, then packed and shifted to fit
    anchors.sort(key=lambda t: t[0])
    rend = ax.figure.canvas.get_renderer()
    ax_w = ax.get_window_extent(rend).width
    widths = []
    for _, _, text, _ in anchors:
        probe = ax.text(0, 0, text, fontsize=SZ["text"], transform=ax.transAxes)
        widths.append(probe.get_window_extent(rend).width / ax_w)
        probe.remove()
    lefts, edge = [], 0.005
    for (x, _, _, _), w in zip(anchors, widths):
        left = max(fx(x) - 0.5 * w, edge)
        lefts.append(left)
        edge = left + w + 0.014
    over = lefts[-1] + widths[-1] - 0.995
    if over > 0:
        lefts = [l - over for l in lefts]
    assert lefts[0] >= 0.0, ("band labels do not fit in one row", lefts[0])
    for (x, y, text, col), left in zip(anchors, lefts):
        ax.annotate(text, (x, y), xytext=(left, band_y), textcoords="axes fraction", ha="left", va="center",
                    fontsize=SZ["text"], color=col, zorder=7,
                    arrowprops=dict(arrowstyle="-", color="#ABABAB", lw=0.55, shrinkA=1, shrinkB=4,
                                    relpos=(0.5, 0.0)))


def main():
    others, ours = l1.load()
    H = 3.95
    fig = plt.figure(figsize=(W, H))
    left, pw = 0.075, 0.918
    top = fig.add_axes([left, 0.585, pw, 0.405])
    bot = fig.add_axes([left, 0.155, pw, 0.405], sharex=top)
    # the "better" end of each axis gets an empty band for the names of the crowded arrows
    panel(top, others, ours, "fid", True, (-5.0, 36.5), [10, 20, 30], "FID", band_y=0.925)
    panel(bot, others, ours, "clip", False, (19.0, 27.7), [20, 22, 24], "CLIP-T2I", band_y=0.925)
    plt.setp(top.get_xticklabels(), visible=False)
    top.tick_params(axis="x", length=2.5)
    bot.set_xticks([1, 3, 10, 30, 100, 300])
    bot.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    bot.set_xlabel("seconds per SVG on SArena-Icon (log scale)", fontsize=SZ["label"], labelpad=1.5)
    for ax, txt in ((top, "lower is better"), (bot, "higher is better")):
        ax.text(0.995, 0.03 if ax is top else 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=SZ["tick"], color="#7A7A7A")
    # colours only, as patches: what the shapes mean is written on the plot
    lg = fig.add_axes([left, 0.0, pw, 0.075])
    lg.axis("off")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    items = [(OPT_C, None, "optimisation-based generators"), (LLM_C, None, "other SVG LLMs"),
             ("white", AR_C, "original decoding (AR)"), (OURS_C, None, "with our decoding")]
    x = 0.0
    for face, edge, text in items:
        lg.add_patch(Rectangle((x, 0.28), 0.018, 0.44, facecolor=face, edgecolor=edge or face, linewidth=1.0))
        t = lg.text(x + 0.026, 0.5, text, fontsize=SZ["text"], va="center", ha="left", color="#2E2E2E")
        fig.canvas.draw()
        x += 0.026 + t.get_window_extent().width / lg.get_window_extent().width + 0.045
    assert x < 1.02, x
    fs.save(fig, str(HERE / "l4_stacked"), width=W)
    print("wrote l4_stacked")


if __name__ == "__main__":
    main()
