"""Q1 fair frontier, round 2 -- polish of the two forms the user picked (q1a + q1e). Round 1 stays.

User: main text = q1a_frontier + q1e_compute; q1d_trajectory goes to the appendix.
What round 2 changes, on the house style (paper/_house):
  - the collisions of round 1: the rotated 'one AR sample' lay across the grey curve and its '2', the '×2.6' sat
    on its arrow and on the '16' point, q1e's 'one AR sample' was printed over a bar, q1e's legend swatches
    touched their words;
  - the original model is grey (it is the reference; round 1's taupe was close to SpS's brown);
  - words: '8 of 1' -> 'best of 8'; 'official' only for the authors' own pipeline, the plain sample is
    'original model, 1 sample' (table 2 already confused 'official sampling' with 'Official 8');
  - q1e: one blue for ours, the number of candidates written under each bar instead of three blues.
Numbers are round 1's (same data pack, same functions, imported).
Data: runs/2026-11-04/figdata_b3/q1_frontier.json (300 benchmark prompts in every arm, one H200 per shard,
HF eager, bf16; seconds include rendering and scoring; quality paired per prompt against one sample of the
original model, 95% bootstrap CI).
Run:  python3 paper/claims/Q1-quality-frontier/round2/make.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("q1r1", HERE.parent / "round1/make.py")
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402

H.apply()
SZ, C = H.SZ, H.C
BLOCKS, G, q, t, NOF = r1.BLOCKS, r1.G, r1.q, r1.t, r1.NOF
AR_LADDER, SD_LADDER = r1.AR_LADDER, r1.SD_LADDER
AR, OU = C["ar"], C["ours"]
METRIC = {"clip": "CLIP score", "lpips": "LPIPS"}


def check_numbers():
    """The published readings (round 1 README): x3.87 / x3.83 / x2.57 at matched quality; inside one sample of
    the original model AR can pick 2 / 2 / 1 and we 16 / 16 / 8; 8 of ours = 18% / 17% / 28% of its passes."""
    sp = [t(b, "ar8s") / t(b, "sd8") for b in BLOCKS]
    assert [f"{x:.2f}" for x in sp] == ["3.87", "3.83", "2.57"], sp
    assert [r1.affordable(b, "AR", t(b, "ar1")) for b in BLOCKS] == [2, 2, 1]
    assert [r1.affordable(b, "SD", t(b, "ar1")) for b in BLOCKS] == [16, 16, 8]
    fw = [round(100 * G[(b, "sd8")]["target_forward"] / G[(b, "ar1")]["target_forward"]) for b in BLOCKS]
    assert fw == [18, 17, 28], fw


def ylabel(b):
    m = METRIC[r1.PRIMARY[b]]
    return f"{m} {'gain' if r1.SIGN[b] > 0 else 'drop'} over\n1 sample of the original model"


def q1a_frontier():
    W = H.FULL_W
    items = [H.item("original model + loop control, best of N", AR, kind="line", lw=1.2),
             H.item("Ours, best of N", OU, kind="line", lw=1.8, bold=True, text_color=OU),
             H.item("original model, 1 sample", AR, kind="marker", marker="s", hollow=True, ms=4.8),
             H.item("authors' pipeline, first valid of 8", AR, kind="marker", marker="D", hollow=True, ms=4.4)]
    leg = H.legend_height(items, W, x_pt=58, width_pt=W * 72 - 60, gap_pt=12)
    left, gap_x, right = 58.0, 58.0, 6.0
    pw = (W * 72 - left - right - 2 * gap_x) / 3
    bottom, ph, title_h = 30.0, 118.0, 17.0
    Ht_pt = bottom + ph + title_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=58, top_pt=0, width_pt=W * 72 - 60, gap_pt=12)
    axes = [H.axes_pt(fig, left + i * (pw + gap_x), bottom, pw, ph) for i in range(3)]
    for ax, b in zip(axes, BLOCKS):
        # one sample of the original model: its time is the budget line
        xa = t(b, "ar1")
        H.ref_line(ax, x=xa)
        yq = q(b, "ar8s")
        H.ref_line(ax, y=yq)
        for arms, c, lw, ms in ((AR_LADDER, AR, 1.2, 4.2), (SD_LADDER, OU, 1.8, 5.0)):
            xs, ys = [t(b, a) for a in arms], [q(b, a) for a in arms]
            ax.plot(xs, ys, "-", color=c, lw=lw, zorder=3, marker="o", ms=ms, mfc=c, mec="white", mew=0.6)
        ax.plot(t(b, "ar1"), q(b, "ar1"), "s", ms=4.8, mfc="white", mec=AR, mew=1.1, zorder=4)
        ax.plot(t(b, "ar8-fv"), q(b, "ar8-fv"), "D", ms=4.4, mfc="white", mec=AR, mew=1.1, zorder=4)
        # the matched-quality reading: from the original model's best of 8 to ours
        ax.annotate("", xy=(t(b, "sd8"), yq), xytext=(t(b, "ar8s"), yq),
                    arrowprops=dict(arrowstyle="-|>", color=OU, lw=1.2, shrinkA=3, shrinkB=3), zorder=5)
        ax.set_xscale("log")
        allx = [t(b, a) for a in AR_LADDER + SD_LADDER + ["ar1", "ar8-fv"]]
        ally = [q(b, a) for a in AR_LADDER + SD_LADDER + ["ar1", "ar8-fv"]]
        ax.set_xlim(0.78 * min(allx), 1.5 * max(allx))
        pad = 0.16 * (max(ally) - min(ally))
        ax.set_ylim(min(ally) - 1.3 * pad, max(ally) + 1.6 * pad)
        H.fs.assert_inside(ax, allx, ally)
        ax.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("seconds per prompt", fontsize=SZ["label"], labelpad=1.5)
        ax.set_ylabel(ylabel(b), fontsize=SZ["label"], labelpad=2)
        mdl, task = b.split(", ")
        ax.set_title(f"{mdl}, {task}", fontsize=SZ["header"], pad=4, color=H.INK)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.255, 0.5)
    fig.canvas.draw()
    # labels: N beside every point, the speed-up over the arrow, '1 sample' on the budget line
    # where the speed-up sits: under the arrow's middle, except on InternSVG, where the arrow is short and the
    # original model's curve runs right under it -- there it goes above, right of our '16'
    off = {"OmniSVG-8B, text-to-SVG": dict(lead=(0, -4), at=None), "OmniSVG-8B, image-to-SVG": dict(lead=(0, -4), at=None),
           "InternSVG-8B, text-to-SVG": dict(lead=(5, 4.5), at=1.0)}
    for ax, b in zip(axes, BLOCKS):
        for arms, c, xy, ha in ((AR_LADDER, AR, (5.5, -3.0), "left"), (SD_LADDER, OU, (-5.5, 3.5), "right")):
            for a in arms:
                # our first point sits by the y axis: its label goes to the right instead
                xy_, ha_ = ((5.5, -3.0), "left") if a == "sd1s" else (xy, ha)
                ax.annotate(str(NOF[a]), (t(b, a), q(b, a)), textcoords="offset points", xytext=xy_,
                            ha=ha_, va="center", fontsize=SZ["note"], color=c, zorder=2.5,
                            bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
        yq = q(b, "ar8s")
        dx, dy = off[b]["lead"]
        xl = np.sqrt(t(b, "sd8") * t(b, "ar8s")) if off[b]["at"] is None else t(b, "ar8s") * off[b]["at"]
        ax.annotate(f"×{t(b, 'ar8s') / t(b, 'sd8'):.1f}", (xl, yq),
                    textcoords="offset points", xytext=(dx, dy), ha="center", va="bottom" if dy > 0 else "top",
                    fontsize=SZ["lead"], color=OU, fontweight="bold", zorder=2.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(t(b, "ar1"), 0.985, "1 sample ", transform=tr, ha="right", va="top", fontsize=7.5, color=H.MUTED)
    H.save(fig, str(HERE / "q1a_frontier"), width="full")
    plt.close(fig)


def q1e_compute():
    """No hardware in the comparison: target forward passes as a share of one sample of the original model."""
    W = H.COL_W
    items = [H.item("original model + loop control", AR), H.item("Ours", OU),
             H.item("1 sample of the original model", H.REF, kind="line", lw=0.9, ls=(0, (3, 2)))]
    leg = H.legend_height(items, W, x_pt=46, width_pt=W * 72 - 50, gap_pt=10)
    left, bottom, ph = 46.0, 40.0, 100.0
    Ht_pt = bottom + ph + 6 + leg
    fig = plt.figure(figsize=(W, Ht_pt / 72))
    H.legend(fig, items, x_pt=46, top_pt=0, width_pt=W * 72 - 50, gap_pt=10)
    ax = H.axes_pt(fig, left, bottom, W * 72 - left - 4, ph)
    arms = [("ar8s", AR, 8), ("sd1s", OU, 1), ("sd8", OU, 8), ("sd16", OU, 16)]
    w, x0 = 0.205, np.arange(3)
    ticks, tlabels = [], []
    for k, (arm, col, n) in enumerate(arms):
        v = [100 * G[(b, arm)]["target_forward"] / G[(b, "ar1")]["target_forward"] for b in BLOCKS]
        xs = x0 + (k - 1.5) * w
        ax.bar(xs, v, w * 0.9, color=col, linewidth=0, zorder=2)
        for x, y in zip(xs, v):
            ax.text(x, y + 2.5, f"{y:.0f}", ha="center", va="bottom", fontsize=SZ["note"], color=H.INK)
            ticks.append(x)
            tlabels.append(str(n))
    H.ref_line(ax, y=100)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 50, 100, 150], ["0", "50", "100", "150%"])
    ax.set_ylabel("target forward passes", fontsize=SZ["label"], labelpad=2)
    ax.set_xticks(ticks, tlabels)
    ax.tick_params(axis="x", length=0, pad=3.0, labelsize=7.5)
    ax.set_xlim(-0.5, 2.5)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for x, b in zip(x0, BLOCKS):
        mdl, task = b.split(", ")
        ax.text(x, -0.13, f"{mdl}\n{task}", transform=tr, ha="center", va="top", fontsize=SZ["tick"],
                linespacing=1.05, color=H.INK)
    ax.text(-0.5 - 0.06, -0.092, "best of", transform=tr, ha="right", va="center", fontsize=7.5, color=H.MUTED)
    H.save(fig, str(HERE / "q1e_compute"), width="col")
    plt.close(fig)


if __name__ == "__main__":
    check_numbers()
    print("Q1 numbers reproduce round 1's README")
    for fn in (q1a_frontier, q1e_compute):
        try:
            fn()
            print("wrote", fn.__name__)
        except AssertionError as e:
            print("CHECK", fn.__name__, e)
