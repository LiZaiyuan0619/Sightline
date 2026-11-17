"""D5 -- how much training the draft head needs, and where it starts. Round 1, night, house style, drawn at the
size of one cell of the design-choice row (paper/claims/D-group/round1, 2.29 x 2.08 in) so it can join that row.

Claim (loop-free content, the D1-D3 caliber): on OmniSVG-8B text-to-SVG the head's acceptance climbs 33% from epoch 1
to epoch 4 and stays within 1% after (12.20 / 12.46 / 12.37 at epochs 4 / 5 / 6); for image-to-SVG, training the head
from scratch on 100k images is no better than reusing the text-to-SVG head as is (8.65 vs 8.77), while warm-starting it
from that head gives +15% (10.06).

Data: d5data.py (result rows, split as runs/2026-11-04/figdata_b4.split_tau). Greedy, 300 MMSVG-Bench prompts.
Run: python3 paper/claims/D5-training/round1/make.py   (STRICT=0 for drafts)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "D-group/round1"))
import d5data as D  # noqa: E402
import dpanels as P  # noqa: E402  (read-only: the row's geometry and helpers)

H, SZ, C, M, INK, MUTED = P.H, P.SZ, P.C, P.M, P.INK, P.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
YL = "tokens accepted\nper round"
GREY = "#8C8C8C"


def tau(e):
    return D.EP[e]["tau_healthy"]


def st(k):
    return D.ST[k]["tau_healthy"]


def d5a_twopart(fig, box):
    """One axis, two parts: left, the epoch curve (text-to-SVG head); right, where the image-to-SVG head starts from --
    the text-to-SVG head used as is, a head trained from scratch, and a head warm-started from the text-to-SVG one."""
    P._legend(fig, box, [H.item("by epoch, text-to-SVG", C["ours"], kind="line", lw=1.8, marker="o",
                                with_marker=True, ms=4.0),
                         H.item("within 1% of the best epoch", "#EDEDED", kind="band")], gap=6.0)
    ax = P._ax(fig, box)
    xs_e = [0.8 * e for e in range(1, 7)]
    ys_e = [tau(e) for e in range(1, 7)]
    top = max(ys_e)
    ax.axhspan(0.99 * top, top, color="#E6E6E6", lw=0, zorder=0).set_gid("bg")
    ax.plot(xs_e, ys_e, color=C["ours"], lw=1.8, zorder=3, solid_capstyle="round")
    ax.plot(xs_e, ys_e, "o", ms=4.0, mfc=C["ours"], mec="white", mew=0.5, zorder=4)
    xs_s = {"zero-shot": 5.8, "from scratch": 6.7, "warm start": 7.6}
    for k, x in xs_s.items():
        c = C["ours"] if k == "warm start" else GREY
        ax.plot([x], [st(k)], "o", ms=4.8, mfc=c, mec="white", mew=0.5, zorder=4)
    ax.add_patch(FancyArrowPatch((xs_s["zero-shot"], st("zero-shot")), (xs_s["warm start"], st("warm start")),
                                 arrowstyle="-|>,head_length=3.6,head_width=2.0", mutation_scale=1, color=C["ours"],
                                 lw=1.0, shrinkA=3.5, shrinkB=4.0, zorder=3))
    fz = SZ["legend"] - 1.2
    ax.text(xs_s["zero-shot"] - 0.16, st("zero-shot"), "as is", ha="right", va="center", fontsize=fz, color=GREY)
    ax.text(xs_s["from scratch"] + 0.16, st("from scratch") - 0.05, "from\nscratch", ha="left", va="center",
            fontsize=fz, color=GREY)
    ax.text(xs_s["warm start"], st("warm start") + 0.22, f"warm\n+{100 * (st('warm start') / st('zero-shot') - 1):.0f}%",
            ha="center", va="bottom", fontsize=fz + 0.3, color=C["ours"], fontweight="bold")
    ax.text(0.8 * 4.0, tau(4) - 0.3, f"+{100 * (tau(4) / tau(1) - 1):.0f}% by epoch 4", ha="left", va="top",
            fontsize=fz + 0.2, color=C["ours"])
    ax.set_xlim(0.35, 8.45)
    ax.set_ylim(7.9, 13.1)
    H.fs.two_row_xaxis(ax, xs_e, [str(e) for e in range(1, 7)],
                       [("epoch", xs_e)], line_down_pt=11.0, name_down_pt=12.0, pad=0.3, fs_group=SZ["tick"] - 0.4)
    from matplotlib.transforms import blended_transform_factory, ScaledTranslation
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    lo, hi = min(xs_s.values()) - 0.3, max(xs_s.values()) + 0.3
    ax.plot([lo, hi], [0, 0], transform=tr + ScaledTranslation(0, -11.0 / 72, fig.dpi_scale_trans), color=H.GROUP_LINE,
            lw=0.65, clip_on=False)
    ax.annotate("image-to-SVG", xy=((lo + hi) / 2, 0), xycoords=tr, xytext=(0, -12.0), textcoords="offset points",
                ha="center", va="top", fontsize=SZ["tick"] - 0.4, color=INK, annotation_clip=False)
    ax.set_ylabel(YL, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    H.fs.assert_inside(ax, xs_e + list(xs_s.values()), ys_e + [st(k) for k in xs_s])
    return [ax]


def d5b_marginal(fig, box):
    """What each extra epoch adds (text-to-SVG), and what each starting point adds over reusing the text head
    (image-to-SVG): diminishing returns after epoch 3; scratch training adds nothing, warm start adds 15%."""
    P._legend(fig, box, [H.item("gain over the step before", C["ours"]),
                         H.item("no gain or a loss", GREY)], gap=7.0)
    ax = P._ax(fig, box)
    steps = [(f"{e - 1}→{e}", tau(e) / tau(e - 1) - 1) for e in range(2, 7)]
    starts = [("scratch", st("from scratch") / st("zero-shot") - 1), ("warm", st("warm start") / st("zero-shot") - 1)]
    xs = list(range(len(steps))) + [len(steps) + 0.7, len(steps) + 1.7]
    vals = [v for _, v in steps] + [v for _, v in starts]
    for x, v in zip(xs, vals):
        ax.bar(x, 100 * v, width=0.62, color=C["ours"] if v > 0.005 else GREY, lw=0, zorder=2)
        ax.text(x, 100 * v + (0.6 if v >= 0 else -0.9), f"{100 * v:+.0f}".replace("-", "−"), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.0, color=INK)
    H.ref_line(ax, y=0)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(-4, 24)
    H.fs.two_row_xaxis(ax, xs, [s for s, _ in steps] + [s for s, _ in starts],
                       [("epochs", list(range(len(steps)))), ("vs zero-shot", xs[-2:])], line_down_pt=11.0,
                       name_down_pt=12.0, pad=0.4, fs_group=SZ["tick"] - 0.4)
    for t in ax.get_xticklabels():
        t.set_fontsize(7.0)
    xl = ax.get_xticklabels()
    xl[-2].set_text("scr."), xl[-1].set_text("warm")
    ax.set_xticks(xs, [t.get_text() for t in xl])
    for t in ax.get_xticklabels():
        t.set_fontsize(7.0)
    ax.set_ylabel("gain in tokens\nper round (%)", fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    return [ax]


def d5c_two_calibers(fig, box):
    """The epoch curve in both calibers: all tokens (table 1's count) and loop-free content; the loop-free line
    flattens from epoch 4, the all-token line too -- training longer does not just learn the loops."""
    P._legend(fig, box, [H.item("all tokens", P.ALL_C, kind="line", lw=1.3, marker="o", with_marker=True, ms=3.6),
                         H.item("loop-free content", C["ours"], kind="line", lw=1.8, marker="o", with_marker=True,
                                ms=4.0, bold=True, text_color=C["ours"])], gap=7.0)
    ax = P._ax(fig, box)
    xs = list(range(1, 7))
    ya = [D.EP[e]["tau_all"] for e in xs]
    yh = [tau(e) for e in xs]
    band = ax.fill_between(xs, yh, ya, color=C["loop_light"], lw=0, zorder=1)
    band.set_gid("bg")
    ax.plot(xs, ya, color=P.ALL_C, lw=1.3, zorder=3)
    ax.plot(xs, ya, "o", ms=3.6, mfc=P.ALL_C, mec="white", mew=0.5, zorder=4)
    ax.plot(xs, yh, color=C["ours"], lw=1.8, zorder=3)
    ax.plot(xs, yh, "o", ms=4.0, mfc=C["ours"], mec="white", mew=0.5, zorder=4)
    ax.set_xticks(xs)
    ax.set_xlim(0.6, 6.4)
    ax.set_ylim(8.5, 17.2)
    ax.set_xlabel("training epochs", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel(YL, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    H.fs.assert_inside(ax, xs, ya + yh)
    return [ax]


def d5d_start(fig, box):
    """Where the image-to-SVG head starts from, against the training it gets: arrows from the text-to-SVG head used
    as is (zero-shot) to the head trained from scratch and to the warm-started one; same 100k x 3 epochs."""
    P._legend(fig, box, [H.item("zero-shot", GREY, kind="marker", ms=4.4), H.item("from scratch", INK, kind="marker",
                                                                                 hollow=True, ms=4.4, mew=0.9),
                         H.item("warm start (ours)", C["ours"], kind="marker", ms=4.8, bold=True,
                                text_color=C["ours"])], gap=6.0)
    ax = P._ax(fig, box)
    keys = ["zero-shot", "from scratch", "warm start"]
    vals = {k: (st(k), D.ST[k]["tau_all"]) for k in keys}
    for j, (cal, lab) in enumerate(((0, "loop-free"), (1, "all tokens"))):
        x = j
        z = vals["zero-shot"][cal]
        for k, c, hollow in (("from scratch", INK, True), ("warm start", C["ours"], False)):
            v = vals[k][cal]
            dx = -0.12 if k == "from scratch" else 0.12
            ax.add_patch(FancyArrowPatch((x + dx, z), (x + dx, v), arrowstyle="-|>,head_length=3.4,head_width=1.9",
                                         mutation_scale=1, color=c, lw=1.1, shrinkA=2.5, shrinkB=3.0, zorder=3))
            ax.plot([x + dx], [v], "o", ms=4.6, mfc="white" if hollow else c, mec=c, mew=0.9 if hollow else 0.5,
                    zorder=4)
        ax.plot([x - 0.12, x + 0.12], [z, z], "o", ms=4.4, mfc=GREY, mec="white", mew=0.5, zorder=4)
        ax.text(x + 0.26, vals["warm start"][cal], f"+{100 * (vals['warm start'][cal] / z - 1):.0f}%", ha="left",
                va="center", fontsize=SZ["legend"] - 1.0, color=C["ours"], fontweight="bold")
    ax.set_xticks([0, 1], ["loop-free\ncontent", "all\ntokens"])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.5, 1.65)
    ax.set_ylim(7.8, 14.2)
    ax.set_ylabel(YL, fontsize=SZ["label"], labelpad=2, linespacing=1.0)
    return [ax]


CANDIDATES = {"d5a_twopart": d5a_twopart, "d5b_marginal": d5b_marginal, "d5c_two_calibers": d5c_two_calibers,
              "d5d_start": d5d_start}


def solo(name):
    fig = plt.figure(figsize=(P.BOX_W / 72, P.BOX_H / 72))
    CANDIDATES[name](fig, (0, 0, P.BOX_W, P.BOX_H))
    P.finish(fig)
    H.save(fig, str(HERE / name), width=P.BOX_W / 72, strict=STRICT)
    plt.close(fig)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(CANDIDATES):
        solo(n)
