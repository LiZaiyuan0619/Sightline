"""Q1 -- the fair speed/quality frontier (component 3), round 1,.

Claim: under a protocol where the original model gets the same loop rule and the same render-and-pick selection,
our decoding reaches the quality of the original model's 8-sample pipeline 2.6-3.9x faster, and inside the wall
clock of ONE official AR sample it can afford 16 candidates (8 on InternSVG) -- better quality than AR's 8.
A second, hardware-free reading: eight of our candidates cost 3.6-6.0x fewer target forward passes than one AR
sample.

Caliber: 300 benchmark prompts present in every arm, one H200 per shard, HF eager, bf16; seconds include rendering
and scoring; quality is paired per prompt against one AR sample with a 95% bootstrap CI.
Data: runs/2026-11-04/figdata_b3/q1_frontier.json (runs/2026-11-04/figdata_b3.py; reproduces logs/STATUS.md 1.3).

Candidates, one chart family each (guide 3.2, 7.3):
  q1a_frontier    the frontier itself: seconds (log) vs quality, one line per family, N on every point
  q1b_matched     grouped bars: seconds at matched quality, and candidates affordable at matched latency
  q1c_budget      staircase: how many candidates a wall-clock budget buys, AR vs ours
  q1d_trajectory the three components as arrows in the (seconds, quality) plane, from the official AR sample
  q1e_compute     target forward passes as a share of one AR sample -- no hardware in the comparison
Run: python3 paper/claims/Q1-quality-frontier/round1/make.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import figstyle as fs  # noqa: E402
from sightline.paths import RUNS

fs.apply_style()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = json.loads((RUNS / "2026-11-04/figdata_b3/q1_frontier.json").read_text())["rows"]

COL_W, FULL_W = 3.25, 6.875                       # two-column page
SZ = dict(tick=8.0, label=9.0, legend=8.5, note=8.5, lead=9.0, letter=12.0)
plt.rcParams.update({"xtick.labelsize": SZ["tick"], "ytick.labelsize": SZ["tick"]})

BLOCKS = ["OmniSVG-8B, text-to-SVG", "OmniSVG-8B, image-to-SVG", "InternSVG-8B, text-to-SVG"]
TITLE = {b: b.replace(", ", "\n") for b in BLOCKS}
C_AR, C_OURS, C_FILL = "#8C7B63", "#1D4E89", "#BFD0E6"
C_LOSSLESS, C_LOOP, C_PICK = "#1D4E89", "#C98A1E", "#5B8C5A"      # the three components
METRIC_NAME = {"clip": "CLIP score", "lpips": "LPIPS"}

G = {(r["block"], r["arm"]): r for r in DATA}
PRIMARY = {b: G[(b, "ar1")]["primary"] for b in BLOCKS}
SIGN = {b: G[(b, "ar1")]["better"] for b in BLOCKS}                # +1 higher is better, -1 lower is better


def q(block, arm):
    """quality of an arm, oriented so that larger is better, in the block's primary metric."""
    r = G[(block, arm)]
    return SIGN[block] * r["quality"][PRIMARY[block]][0]


def t(block, arm):
    return G[(block, arm)]["seconds"]


AR_LADDER = ["ar1s", "ar2s", "ar4s", "ar8s"]
SD_LADDER = ["sd1s", "sd2", "sd4", "sd8", "sd16"]
NOF = {"ar1s": 1, "ar2s": 2, "ar4s": 4, "ar8s": 8, "sd1s": 1, "sd2": 2, "sd4": 4, "sd8": 8, "sd16": 16}


def ylabel(block):
    m = METRIC_NAME[PRIMARY[block]]
    return f"{m} gain vs\none AR sample" if SIGN[block] > 0 else f"{m} drop vs\none AR sample"


def affordable(block, family, budget):
    """largest N this family can render and pick from inside the budget."""
    arms = AR_LADDER if family == "AR" else SD_LADDER
    return max(NOF[a] for a in arms if t(block, a) <= budget)


def three_panels(height, left=0.094, right=0.996, bottom=0.175, top=0.80, gap=0.105):
    fig = plt.figure(figsize=(FULL_W, height))
    pw = (right - left - 2 * gap) / 3
    return fig, [fig.add_axes([left + i * (pw + gap), bottom, pw, top - bottom]) for i in range(3)]


def pin_ylabels(axes, x=-0.200):
    """Same label offset in every panel, so a wide tick label cannot push one off the page."""
    for ax in axes:
        ax.yaxis.set_label_coords(x, 0.5)


def side_note(ax, x, y, text, color, va="top"):
    """Write next to a vertical guide, on whichever side of it has room."""
    lo, hi = ax.get_xlim()
    right_room = np.log10(hi / x) > np.log10(x / lo)
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(4 if right_room else -4, 0),
                ha="left" if right_room else "right", va=va, fontsize=SZ["note"], color=color,
                linespacing=1.1)


# --------------------------------------------------------------------------------------------- q1a
def q1a_frontier():
    """The frontier: seconds per prompt (log) against paired quality, the AR ladder and ours, N on every point."""
    fig, axes = three_panels(2.55)
    for i, (ax, b) in enumerate(zip(axes, BLOCKS)):
        for arms, c, lab in ((AR_LADDER, C_AR, "original model"), (SD_LADDER, C_OURS, "Ours")):
            xs, ys = [t(b, a) for a in arms], [q(b, a) for a in arms]
            ax.plot(xs, ys, "-", color=c, lw=1.5 if c == C_OURS else 1.1, zorder=3,
                    marker="o", ms=5.0 if c == C_OURS else 4.2, mfc=c, mec="white", mew=0.6)
            for a, x, y in zip(arms, xs, ys):
                dy = 9 if c == C_OURS else -12
                ax.annotate(str(NOF[a]), (x, y), textcoords="offset points", xytext=(0, dy),
                            ha="center", fontsize=SZ["note"], color=c)
        # the model authors' own pipeline, and a plain AR sample: two reference points, named on the spot
        for arm, mk, txt in (("ar1", "s", "1 sample, official"), ("ar8-fv", "D", "official pipeline, 8")):
            ax.plot(t(b, arm), q(b, arm), mk, ms=4.6, mfc="white", mec=C_AR, mew=1.1, zorder=4)
        # the two readings
        yq, xa = q(b, "ar8s"), t(b, "ar1")
        ax.axhline(yq, color=fs.REF, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.axvline(xa, color=fs.REF, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.annotate("", xy=(t(b, "sd8"), yq), xytext=(t(b, "ar8s"), yq),
                    arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=1.2, shrinkA=2, shrinkB=2))
        ax.annotate(f"×{t(b, 'ar8s') / t(b, 'sd8'):.1f}", (np.sqrt(t(b, "sd8") * t(b, "ar8s")), yq),
                    textcoords="offset points", xytext=(0, 5), ha="center", va="bottom",
                    fontsize=SZ["lead"], color=C_OURS, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("wall clock per prompt (s)", fontsize=SZ["label"], labelpad=1.5)
        ax.set_ylabel(ylabel(b), fontsize=SZ["label"], labelpad=1.5)
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.set_xlim(0.8 * min(t(b, a) for a in SD_LADDER), 1.45 * max(t(b, a) for a in ["ar8-fv"]))
        ys = [q(b, a) for a in AR_LADDER + SD_LADDER + ["ar1", "ar8-fv"]]
        pad = 0.17 * (max(ys) - min(ys))
        ax.set_ylim(min(ys) - pad, max(ys) + 1.5 * pad)
        fs.assert_inside(ax, [t(b, a) for a in AR_LADDER + SD_LADDER + ["ar1", "ar8-fv"]], ys)
        ax.set_title(TITLE[b], fontsize=SZ["legend"], pad=3.5, color="#2E2E2E", linespacing=1.1)
    pin_ylabels(axes)
    for ax, b in zip(axes, BLOCKS):
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(t(b, "ar1"), 0.025, "one AR sample", transform=tr, rotation=90, ha="right", va="bottom",
                fontsize=SZ["note"], color=fs.REF)
    legend(fig, [0.094, 0.935, 0.902, 0.055],
           [(C_AR, "o", "original model + the same loop rule and selection"),
            (C_AR, "s", "one sample, official decoding"), (C_AR, "D", "official pipeline, 8 samples"),
            (C_OURS, "o", "Ours")],
           open_marks=("one sample, official decoding", "official pipeline, 8 samples"))
    fs.save(fig, str(HERE / "q1a_frontier"), width=FULL_W)


def legend(fig, rect, items, open_marks=()):
    """One row, entries spaced by measured text width (guide 4.4: name every glyph, no coding rules)."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    texts = [ax.text(0, 0.5, s, fontsize=SZ["legend"], va="center", ha="left",
                     color=c if s == "Ours" else "#2E2E2E", fontweight="bold" if s == "Ours" else "normal")
             for c, _, s in items]
    rend = fig.canvas.get_renderer()
    aw = ax.get_window_extent(rend).width
    ws = [x.get_window_extent(rend).width / aw for x in texts]
    lead = 0.018
    gap = (0.995 - sum(ws) - len(items) * lead) / max(1, len(items) - 1)
    assert gap > 0.004, ("legend does not fit", gap)
    x = 0.004
    for (c, mk, s), txt, w in zip(items, texts, ws):
        filled = s not in open_marks
        ax.plot([x + 0.006], [0.5], mk, ms=5.0, mfc=c if filled else "white", mec=c,
                mew=0.6 if filled else 1.1, clip_on=False)
        txt.set_x(x + lead)
        x += lead + w + gap


# --------------------------------------------------------------------------------------------- q1b
def q1b_matched():
    """Two questions as two bar rows: how long to reach the same quality, how many candidates at the same latency."""
    H = 3.05
    fig = plt.figure(figsize=(COL_W, H))
    xs = np.arange(3)
    ax = fig.add_axes([0.200, 0.555, 0.785, 0.335])
    for k, (arm, c) in enumerate((("ar8s", C_AR), ("sd8", C_OURS))):
        v = [t(b, arm) for b in BLOCKS]
        ax.bar(xs + (k - 0.5) * 0.34, v, 0.32, color=c, linewidth=0, zorder=2)
        for x, y in zip(xs, v):
            ax.text(x + (k - 0.5) * 0.34, y * 1.07, f"{y:.0f}", ha="center", va="bottom",
                    fontsize=SZ["note"], color=c)
    for x, b in zip(xs, BLOCKS):
        ax.text(x, max(t(b, "ar8s"), t(b, "sd8")) * 2.3, f"×{t(b, 'ar8s') / t(b, 'sd8'):.1f}", ha="center",
                va="bottom", fontsize=SZ["lead"], color=C_OURS, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(2, 260)
    ax.set_yticks([5, 10, 50], ["5", "10", "50"])
    ax.set_ylabel("seconds at the\nsame quality", fontsize=SZ["label"], labelpad=1.5)
    ax.set_xticks(xs, ["" for _ in xs])
    ax.tick_params(axis="x", length=0)

    ax2 = fig.add_axes([0.200, 0.175, 0.785, 0.300])
    for k, (fam, c) in enumerate((("AR", C_AR), ("Ours", C_OURS))):
        vals = [affordable(b, fam, t(b, "ar1")) for b in BLOCKS]
        ax2.bar(xs + (k - 0.5) * 0.34, vals, 0.32, color=c, linewidth=0, zorder=2)
        for x, y in zip(xs, vals):
            ax2.text(x + (k - 0.5) * 0.34, y + 0.35, str(y), ha="center", va="bottom",
                     fontsize=SZ["note"], color=c)
    for x, b in zip(xs, BLOCKS):
        r = affordable(b, "Ours", t(b, "ar1")) / affordable(b, "AR", t(b, "ar1"))
        ax2.text(x, 17.4, f"×{r:.0f}", ha="center", va="bottom", fontsize=SZ["lead"], color=C_OURS,
                 fontweight="bold")
    ax2.set_ylim(0, 21)
    ax2.set_yticks([0, 8, 16])
    ax2.set_ylabel("candidates inside one\nofficial AR sample", fontsize=SZ["label"], labelpad=1.5)
    ax2.set_xticks(xs, [TITLE[b] for b in BLOCKS], fontsize=SZ["tick"], linespacing=1.1)
    ax2.tick_params(axis="x", length=0, pad=2)
    legend(fig, [0.035, 0.935, 0.95, 0.055], [(C_AR, "s", "original model"), (C_OURS, "s", "Ours")])
    fs.letter_at(fig, 2, (1 - 0.890) * H * 72 + 1, "a")
    fs.letter_at(fig, 2, (1 - 0.475) * H * 72 + 1, "b")
    fs.save(fig, str(HERE / "q1b_matched"), width=COL_W)


# --------------------------------------------------------------------------------------------- q1c
def q1c_budget():
    """Staircase: what a wall-clock budget buys. y = candidates rendered and picked from, x = the budget."""
    fig, axes = three_panels(2.45)
    for ax, b in zip(axes, BLOCKS):
        for arms, c, step_lab in ((AR_LADDER, C_AR, None), (SD_LADDER, C_OURS, None)):
            xs = [t(b, a) for a in arms] + [200.0]
            ys = [NOF[a] for a in arms]
            ax.step(xs, ys + [ys[-1]], where="post", color=c, lw=1.6 if c == C_OURS else 1.1, zorder=3)
            ax.plot([t(b, a) for a in arms], ys, "o", ms=4.4, mfc=c, mec="white", mew=0.6, zorder=4)
        ax.axvline(t(b, "ar1"), color=fs.REF, lw=0.6, ls=(0, (4, 3)), zorder=1)
        lo, hi = affordable(b, "AR", t(b, "ar1")), affordable(b, "Ours", t(b, "ar1"))
        ax.annotate("", xy=(t(b, "ar1") * 0.955, hi), xytext=(t(b, "ar1") * 0.955, lo),
                    arrowprops=dict(arrowstyle="-|>", color=C_OURS, lw=1.3, shrinkA=1, shrinkB=1))
        ax.text(t(b, "ar1") * 0.90, np.sqrt(lo * hi), f"×{hi // lo}", ha="right", va="center",
                fontsize=SZ["lead"], color=C_OURS, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_ylim(0.8, 26)
        ax.set_xlim(0.9 * t(b, "sd1s"), 90)
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.set_xlabel("wall-clock budget (s)", fontsize=SZ["label"], labelpad=1.5)
        ax.set_ylabel("candidates it buys", fontsize=SZ["label"], labelpad=1.5)
        ax.set_title(TITLE[b], fontsize=SZ["legend"], pad=3.5, color="#2E2E2E", linespacing=1.1)
    pin_ylabels(axes, -0.145)
    for ax, b in zip(axes, BLOCKS):
        side_note(ax, t(b, "ar1"), 25.0, "one official\nAR sample", fs.REF)
    legend(fig, [0.094, 0.935, 0.902, 0.055],
           [(C_AR, "o", "original model, same loop rule and selection"), (C_OURS, "o", "Ours")])
    fs.save(fig, str(HERE / "q1c_budget"), width=FULL_W)


# --------------------------------------------------------------------------------------------- q1d
def q1d_trajectory():
    """The three components as three arrows from the model's own official single sample."""
    fig, axes = three_panels(2.6)
    STEPS = [("ar1", "sd1", C_LOSSLESS, "lossless speculative decoding"),
             ("sd1", "sd1s", C_LOOP, "loop control"),
             ("sd1s", "sd16", C_PICK, "render and pick 16")]
    for ax, b in zip(axes, BLOCKS):
        end = "sd16" if t(b, "sd16") <= t(b, "ar1") else "sd8"
        steps = [(a, (end if c is C_PICK else z), c, lab.replace("16", str(NOF[end])))
                 for a, z, c, lab in STEPS]
        for a, z, c, lab in steps:
            ax.annotate("", xy=(t(b, z), q(b, z)), xytext=(t(b, a), q(b, a)),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.7, shrinkA=3.5, shrinkB=3.5,
                                        mutation_scale=11))
        pts = ["ar1", "sd1", "sd1s", end]
        ax.plot([t(b, p) for p in pts], [q(b, p) for p in pts], "o", ms=4.6, mfc="white", mec="#555555",
                mew=0.9, zorder=5, ls="none")
        ax.plot([t(b, end)], [q(b, end)], "o", ms=6.4, mfc=C_OURS, mec="white", mew=0.7, zorder=6)
        ax.plot([t(b, "ar8s")], [q(b, "ar8s")], "D", ms=5.0, mfc=C_AR, mec="white", mew=0.6, zorder=6)
        lo, hi = ax.get_xlim()
        rightish = np.log10(t(b, "ar8s") / lo) > 0.72 * np.log10(hi / lo)
        ax.annotate("the model's own\n8 of 1", (t(b, "ar8s"), q(b, "ar8s")), textcoords="offset points",
                    xytext=(4 if rightish else 0, -7), ha="right" if rightish else "center", va="top",
                    fontsize=SZ["note"], color=C_AR, linespacing=1.1)
        ax.annotate("one AR sample", (t(b, "ar1"), q(b, "ar1")), textcoords="offset points", xytext=(0, -9),
                    ha="center", va="top", fontsize=SZ["note"], color="#555555")
        ax.annotate("loop control", (np.sqrt(t(b, "sd1") * t(b, "sd1s")), q(b, "sd1s")),
                    textcoords="offset points", xytext=(0, 10), ha="center", va="bottom",
                    fontsize=SZ["note"], color=C_LOOP,
                    arrowprops=dict(arrowstyle="-", color=C_LOOP, lw=0.6, shrinkA=1, shrinkB=3))
        ax.set_xscale("log")
        ax.set_xlabel("wall clock per prompt (s)", fontsize=SZ["label"], labelpad=1.5)
        ax.set_ylabel(ylabel(b), fontsize=SZ["label"], labelpad=1.5)
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        xs = [t(b, p) for p in pts + ["ar8s"]]
        ys = [q(b, p) for p in pts + ["ar8s"]]
        ax.set_xlim(0.7 * min(xs), 1.5 * max(xs))
        pad = 0.20 * (max(ys) - min(ys))
        ax.set_ylim(min(ys) - 1.6 * pad, max(ys) + pad)
        fs.assert_inside(ax, xs, ys)
        ax.set_title(TITLE[b], fontsize=SZ["legend"], pad=3.5, color="#2E2E2E", linespacing=1.1)
    pin_ylabels(axes)
    legend(fig, [0.094, 0.935, 0.902, 0.055],
           [(C_LOSSLESS, "s", "lossless speculative decoding"), (C_LOOP, "s", "loop control"),
            (C_PICK, "s", "render and pick")])
    fs.save(fig, str(HERE / "q1d_trajectory"), width=FULL_W)


# --------------------------------------------------------------------------------------------- q1e
def q1e_compute():
    """No hardware in the comparison: target forward passes, as a share of one AR sample."""
    H = 2.15
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0.195, 0.235, 0.790, 0.645])
    arms = [("ar8s", C_AR, "model, 8 of 1"), ("sd1s", C_FILL, "Ours, 1"), ("sd8", "#5E87B5", "Ours, 8 of 1"),
            ("sd16", C_OURS, "Ours, 16 of 1")]
    xs = np.arange(3)
    w = 0.21
    for k, (arm, c, lab) in enumerate(arms):
        v = [100 * G[(b, arm)]["target_forward"] / G[(b, "ar1")]["target_forward"] for b in BLOCKS]
        ax.bar(xs + (k - 1.5) * w, v, w * 0.92, color=c, linewidth=0, zorder=2)
        for x, y in zip(xs, v):
            ax.text(x + (k - 1.5) * w, y + 3, f"{y:.0f}", ha="center", va="bottom", fontsize=SZ["note"],
                    color=c if c != C_FILL else "#6E7F96")
    ax.axhline(100, color=fs.REF, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(-0.42, 101, "one AR sample", ha="left", va="bottom", fontsize=SZ["note"], color=fs.REF)
    ax.set_ylim(0, 160)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("target forward passes\n(% of one AR sample)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_xticks(xs, [TITLE[b] for b in BLOCKS], fontsize=SZ["tick"], linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=2)
    legend(fig, [0.035, 0.925, 0.95, 0.065], [(c, "s", lab) for _, c, lab in arms])
    fs.save(fig, str(HERE / "q1e_compute"), width=COL_W)


if __name__ == "__main__":
    for b in BLOCKS:
        print(f"{b:28s} sd8/ar8s ×{t(b, 'ar8s') / t(b, 'sd8'):.2f}   "
              f"inside one AR sample: {max(NOF[a] for a in SD_LADDER if t(b, a) <= t(b, 'ar1'))} candidates   "
              f"sd8 forwards {G[(b, 'sd8')]['target_forward'] / G[(b, 'ar1')]['target_forward']:.0%} of ar1")
    q1a_frontier()
    q1b_matched()
    q1c_budget()
    q1d_trajectory()
    q1e_compute()
