"""D2 -- why the draft block stops at 32. Round 1, evening (house style).

Claim (rewritten on the loop-free-segment caliber, ): at the same recipe and the same training budget, doubling
the draft block from 32 to 64 tokens does buy all-token acceptance (tokens per round 16.17 -> 17.01) and wall clock
(x8.77 -> x10.39). On loop-free content it buys nothing: loop-free tokens per round is 12.36 at block 32 and 12.24 at block 64.
Everything the longer block adds sits inside repetition loops (loop tokens per round 24.06 -> 30.92), and the shipped system
cuts those with loop control -- so the extra 240 GPU-hours of training buy content the system throws away.

The second panel of the unit is the draft head's depth (2 / 5 / 10 layers at the same one-epoch budget): deeper
heads do accept more per round, but they cost more per round, so the token rate does not improve.

Caliber: 300 MMSVG-Bench prompts; all arms in $SIGHTLINE_RUNS/r1027/b64-eval were trained on the same 300k x 3 epochs and
evaluated in the same job, so the wall clock IS comparable inside this unit (unlike D1's ladder).

Candidates, one chart family each (guide SS3.2 / SS7.3):
  b1_two_calibers grouped bars: all-token and loop-free acceptance side by side, block 16 / 32 / 64
  b2_stack         stacked bars: what is loop-free content and what is only inside loops
  b3_slope         slope chart: every arm's all-token value drops to its loop-free value, lines cross
  b4_depth         the draft head's depth: tokens per round against ms per round, equal-rate rays

Data: runs/2026-11-04/figdata_b4/d2_block.json (runs/2026-11-04/figdata_b4.py).
Run:  python3 paper/claims/D2-block-length/round1/make.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402
from sightline.paths import RUNS

H.apply()
SZ, C = H.SZ, H.C
RAW = json.loads((RUNS / "2026-11-04/figdata_b4/d2_block.json").read_text())
D, DEPTH = RAW["cells"], RAW["depth"]

# the equal-budget comparison: same recipe (top-16 tree at 127 nodes), same 300k x 3 epochs
LEN = ["16", "32", "64"]
KEY = {"16": "16|6ep|tree", "32": "32|3ep|tree", "64": "64|3ep|tree"}
LEN_C = {"16": H.METHOD["DFlash"], "32": C["ours"], "64": "#9AAEC8"}
LEN_LAB = {"16": "16", "32": "32", "64": "64"}


def cell(ln, mode="greedy"):
    return D[f"{KEY[ln]}|{mode}"]


def check():
    g32, g64 = cell("32"), cell("64")
    assert (round(g32["tau_healthy"], 2), round(g64["tau_healthy"], 2)) == (12.36, 12.24)
    print("D2: all tokens %.2f -> %.2f (+%.0f%%), loop-free %.2f -> %.2f (%+.0f%%), inside loops %.1f -> %.1f" % (
        g32["tau_all"], g64["tau_all"], 100 * (g64["tau_all"] / g32["tau_all"] - 1),
        g32["tau_healthy"], g64["tau_healthy"], 100 * (g64["tau_healthy"] / g32["tau_healthy"] - 1),
        g32["tau_loop"], g64["tau_loop"]))


def mode_panels(items, ph=104.0, bottom=32.0, left=46.0, gap_x=50.0, right=6.0, title_h=15.0, n=2, gap_leg=12):
    W = H.FULL_W
    leg = H.legend_height(items, W, x_pt=left, width_pt=W * 72 - left - 4, gap_pt=gap_leg)
    pw = (W * 72 - left - right - (n - 1) * gap_x) / n
    Ht = bottom + ph + title_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht / 72))
    H.legend(fig, items, x_pt=left, top_pt=0, width_pt=W * 72 - left - 4, gap_pt=gap_leg)
    axes = [H.axes_pt(fig, left + i * (pw + gap_x), bottom, pw, ph) for i in range(n)]
    return fig, axes


MODES = [("greedy", "greedy"), ("sample", "official sampling")]


# ------------------------------------------------------------------------------------------------------- b1
def b1_two_calibers():
    """The whole claim in one reading: the left bar of each pair rises with the block, the right one does not."""
    items = [H.item("all tokens, as table 1 counts them", "#C9C9C9"),
             H.item("loop-free content only", C["ours"], bold=True, text_color=C["ours"])]
    fig, axes = mode_panels(items)
    for ax, (mode, lab) in zip(axes, MODES):
        xs = np.arange(3)
        for k, (field, col) in enumerate((("tau_all", "#C9C9C9"), ("tau_healthy", None))):
            for i, ln in enumerate(LEN):
                v = cell(ln, mode)[field]
                c = col or LEN_C[ln]
                ax.bar(i + (k - 0.5) * 0.4, v, 0.36, color=c, lw=0, zorder=3)
                ax.text(i + (k - 0.5) * 0.4, v + 0.25, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=SZ["legend"], color=H.INK)
        ax.set_xticks(xs, [LEN_LAB[l] for l in LEN])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlabel("draft block length", fontsize=SZ["label"], labelpad=2)
        ax.set_ylabel("tokens accepted per round", fontsize=SZ["label"], labelpad=2)
        ax.set_ylim(0, max(cell(l, mode)["tau_all"] for l in LEN) * 1.22)
        ax.set_title(lab, fontsize=SZ["header"], pad=4, color=H.INK)
        # the two readings of the 32 -> 64 step
        a = cell("32", mode)
        b = cell("64", mode)
        ax.text(1.5, ax.get_ylim()[1] * 0.985, f"32 → 64: all tokens {100 * (b['tau_all'] / a['tau_all'] - 1):+.0f}%,"
                f" loop-free {100 * (b['tau_healthy'] / a['tau_healthy'] - 1):+.0f}%", ha="center", va="top",
                fontsize=7.6, color=H.MUTED)
    H.save(fig, str(HERE / "b1_two_calibers"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- b2
def b2_stack():
    """One bar per block length, split into what a round accepts on loop-free content and what only loops add."""
    items = [H.item("loop-free content", C["ours"]), H.item("only inside repetition loops", C["loop_light"])]
    fig, axes = mode_panels(items)
    for ax, (mode, lab) in zip(axes, MODES):
        for i, ln in enumerate(LEN):
            c = cell(ln, mode)
            ax.bar(i, c["tau_healthy"], 0.6, color=LEN_C[ln], lw=0, zorder=3)
            ax.bar(i, c["tau_all"] - c["tau_healthy"], 0.6, bottom=c["tau_healthy"], color=C["loop_light"], lw=0,
                   zorder=3)
            ax.text(i, c["tau_healthy"] / 2, f"{c['tau_healthy']:.1f}", ha="center", va="center",
                    fontsize=SZ["legend"], color="white", gid="free")
            ax.text(i, c["tau_all"] + 0.25, f"{c['tau_all']:.1f}", ha="center", va="bottom", fontsize=SZ["legend"],
                    color=H.INK)
        ln = H.ref_line(ax, y=cell("32", mode)["tau_healthy"])
        ax.set_xticks(range(3), [LEN_LAB[l] for l in LEN])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_xlabel("draft block length", fontsize=SZ["label"], labelpad=2)
        ax.set_ylabel("tokens accepted per round", fontsize=SZ["label"], labelpad=2)
        ax.set_ylim(0, max(cell(l, mode)["tau_all"] for l in LEN) * 1.22)
        ax.set_title(lab, fontsize=SZ["header"], pad=4, color=H.INK)
        ax.text(-0.42, ax.get_ylim()[1] * 0.99, "dashed line: what block 32 already reaches on loop-free content",
                ha="left", va="top", fontsize=7.4, color=H.MUTED)
    H.save(fig, str(HERE / "b2_stack"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- b3
def b3_slope():
    """Every arm drops from its all-token value to its loop-free value. Block 64's line is the steepest and ends
    below block 32's -- the ranking swaps once the loops are taken out."""
    items = [H.item(f"block {LEN_LAB[l]}", LEN_C[l], kind="line", lw=2.0 if l == "32" else 1.4, marker="o",
                    with_marker=True, ms=5.0) for l in LEN]
    fig, axes = mode_panels(items)
    for ax, (mode, lab) in zip(axes, MODES):
        for ln in LEN:
            c = cell(ln, mode)
            ours = ln == "32"
            ax.plot([0, 1], [c["tau_all"], c["tau_healthy"]], "-o", color=LEN_C[ln], lw=2.2 if ours else 1.4,
                    ms=5.8 if ours else 4.8, mfc=LEN_C[ln], mec="white", mew=0.6, zorder=4 if ours else 3)
            dy = {"16": 0, "32": 5.5, "64": -5.5}[ln]       # 32 and 64 end within a token of each other
            ax.annotate(f"block {LEN_LAB[ln]}", (1, c["tau_healthy"]), textcoords="offset points", xytext=(6, dy),
                        ha="left", va="center", fontsize=SZ["legend"], color=LEN_C[ln],
                        fontweight="bold" if ours else "normal")
        ax.set_xlim(-0.14, 1.52)
        ax.set_xticks([0, 1], ["all tokens", "loop-free"])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_ylabel("tokens accepted per round", fontsize=SZ["label"], labelpad=2)
        vals = [cell(l, mode)[f] for l in LEN for f in ("tau_all", "tau_healthy")]
        ax.set_ylim(min(vals) * 0.88, max(vals) * 1.08)
        H.fs.assert_inside(ax, [0, 1] * 3, vals)
        ax.set_title(lab, fontsize=SZ["header"], pad=4, color=H.INK)
    H.save(fig, str(HERE / "b3_slope"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- b4
def b4_depth():
    """The other knob: how deep the draft head is. Grey rays are equal token rate."""
    rows = [d for d in DEPTH if d["design"] == "D1-top16-N127"]
    label = {"depth 2": "2 layers", "depth 5 (ours)": "5 layers", "depth 10": "10 layers"}
    col = {"depth 2": "#C3C3C3", "depth 5 (ours)": C["ours"], "depth 10": "#9AAEC8"}
    items = [H.item(label[k], col[k], kind="marker", ms=5.8 if "ours" in k else 5.0) for k in label]
    fig, axes = mode_panels(items)
    for ax, (mode, lab) in zip(axes, MODES):
        rs = [r for r in rows if r["mode"] == mode]
        xs = [r["ms_round"] for r in rs]
        ys = [r["tau_healthy"] for r in rs]
        x0, x1 = min(xs) * 0.92, max(xs) * 1.10
        for rate in (400, 450, 500, 550):
            ln, = ax.plot([x0, x1], [rate * x0 / 1000, rate * x1 / 1000], color="#E6E6E6", lw=0.7, zorder=1)
            ln.set_gid("ref")
        for r in rs:
            ours = "ours" in r["head"]
            ax.plot([r["ms_round"]], [r["tau_healthy"]], "o", ms=6.6 if ours else 5.4, mfc=col[r["head"]],
                    mec="white", mew=0.7, zorder=4)
        ax.set_xlim(x0, x1)
        ax.set_ylim(min(ys) * 0.90, max(ys) * 1.10)
        H.fs.assert_inside(ax, xs, ys)
        ax.set_xlabel("ms per verification round", fontsize=SZ["label"], labelpad=2)
        ax.set_ylabel("tokens accepted per round,\nloop-free content", fontsize=SZ["label"], labelpad=2)
        ax.set_title(lab, fontsize=SZ["header"], pad=4, color=H.INK)
    H.save(fig, str(HERE / "b4_depth"), width="full")
    plt.close(fig)


if __name__ == "__main__":
    check()
    names = []
    for fn in (b1_two_calibers, b2_stack, b3_slope, b4_depth):
        try:
            fn()
            names.append(fn.__name__)
            print("wrote", fn.__name__)
        except AssertionError as e:
            print("CHECK", fn.__name__, e)
    H.sheet([str(HERE / f"{n}.png") for n in names], str(HERE / "_sheet.png"), cols=1,
            title="D2 round 1 - why the draft block stops at 32 (double column, print scale)")
