"""D3 -- the shape of the draft tree: how wide, how big, and what a node buys. Round 1,  (house style).

Claims the unit defends:
  width   at a fixed budget of 127 nodes, the curve flattens: from k=16 on, every block is within 1% of its own
          better of {16, 32} (OmniSVG text 15.95 vs 15.80, InternSVG 18.33 vs 18.47, image 13.17 vs 13.18), while
          k=2 costs %. So 16 is where the curve goes flat, NOT a peak -- on the image block k=8 is 2% higher.
          The claim list said "k=16 is the peak"; that held on the one block it was measured on. Say
          "flat from 16 on, so we take the narrowest width on the plateau".
  budget the budget curve flattens: 15 -> 255 nodes moves tokens per round 8.68 -> 14.09 but speed-up only 5.15x -> 7.69x,
          and the last doubling (127 -> 255) is worth +1% of wall clock.
  where   69% of the budget goes to the drafter's own top-1 continuation, and that is where 96% of the accepted
          tokens come from; the siblings are the insurance, not the engine.
  cost    with the longer block, a node is better spent: we pay 7.2-8.7 tree nodes per accepted token where
          DDTree pays 9.9-11.5 and EAGLE-3 11.6-12.9.

Caliber: 300 MMSVG-Bench prompts, greedy, same-job AR (runs/2026-10-29/width.md); the budget split is measured on
loop-free positions only (runs/2026-10-31/analysis/tree_budget). Chain drafters (DFlash, PLD) are not on the node
panel: a chain has no siblings, so "nodes per accepted token" is not the same quantity for them.

Candidates, one chart family each (guide SS3.2 / SS7.3):
  c1_width     tokens per round against tree width, one line per block, the peak marked
  c2_budget    the budget curve, tokens per round and speed-up on the same node count (log x)
  c3_spend     where the nodes go and what they bring back, by sibling rank and by depth
  c4_nodes     nodes spent per accepted token, every tree method, every block

Data: runs/2026-11-04/figdata_b4/d3_tree.json (runs/2026-11-04/figdata_b4.py).
Run:  python3 paper/claims/D3-tree-shape/round1/make.py
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
D = json.loads((RUNS / "2026-11-04/figdata_b4/d3_tree.json").read_text())
BLOCKS = ["OmniSVG-8B text-to-SVG", "InternSVG-8B text-to-SVG", "OmniSVG-8B image-to-SVG"]
BLK_C = {BLOCKS[0]: C["ours"], BLOCKS[1]: "#6E9AC4", BLOCKS[2]: "#9AAEC8"}
TREE_METHODS = ["EAGLE-3", "DDTree", "DARTree", "Ours"]


def check():
    w = {r["k"]: r["tau"] for r in D["width"] if r["block"] == BLOCKS[0]}
    assert (w[2], w[16], w[32]) == (14.25, 15.95, 15.80)
    # the plateau, stated as it actually is: from k=16 on, every block is within 1% of its better of {16, 32}
    for b in BLOCKS:
        ws = {r["k"]: r["tau"] for r in D["width"] if r["block"] == b}
        best = max(ws[16], ws[32])
        assert ws[16] > 0.99 * best and ws[32] > 0.99 * best, (b, ws)
    b = {r["nodes"]: r for r in D["budget"] if r["block"] == BLOCKS[2]}
    npt = D["nodes_per_token"]
    ours = [npt[k] for k in npt if k.endswith("|Ours") and "greedy" in k]
    dd = [npt[k] for k in npt if k.endswith("|DDTree") and "greedy" in k]
    print("D3: width flat from 16 on (%.2f at 16, %.2f at 32); 127 -> 255 nodes is +%.0f%% tau but +%.0f%% speed; "
          "ours %.1f-%.1f nodes per accepted token, DDTree %.1f-%.1f" % (
              w[16], w[32], 100 * (b[255]["tau"] / b[127]["tau"] - 1),
              100 * (b[255]["speedup"] / b[127]["speedup"] - 1), min(ours), max(ours), min(dd), max(dd)))


def panels(items, n=3, ph=104.0, bottom=32.0, left=46.0, gap_x=48.0, right=6.0, title_h=15.0, gap_leg=12):
    W = H.FULL_W
    leg = H.legend_height(items, W, x_pt=left, width_pt=W * 72 - left - 4, gap_pt=gap_leg) if items else 2.0
    pw = (W * 72 - left - right - (n - 1) * gap_x) / n
    Ht = bottom + ph + title_h + 4 + leg
    fig = plt.figure(figsize=(W, Ht / 72))
    if items:
        H.legend(fig, items, x_pt=left, top_pt=0, width_pt=W * 72 - left - 4, gap_pt=gap_leg)
    return fig, [H.axes_pt(fig, left + i * (pw + gap_x), bottom, pw, ph) for i in range(n)]


def short(b):
    mdl, task = b.rsplit(" ", 1)
    return f"{mdl}, {task}"


# ------------------------------------------------------------------------------------------------------- c1
def c1_width():
    """tokens per round against the number of candidates a parent may have, at the standard budget of 127 nodes."""
    items = [H.item("the width we ship, 16", C["ours"], kind="marker", ms=6.4)]
    fig, axes = panels(items)
    for ax, b in zip(axes, BLOCKS):
        rs = sorted([r for r in D["width"] if r["block"] == b], key=lambda r: r["k"])
        xs = [r["k"] for r in rs]
        ys = [r["tau"] for r in rs]
        ax.plot(xs, ys, "-o", color="#9AAEC8", lw=1.5, ms=4.6, mfc="#9AAEC8", mec="white", mew=0.6, zorder=3)
        k16 = next(r for r in rs if r["k"] == 16)
        ax.plot([16], [k16["tau"]], "o", ms=7.0, mfc=C["ours"], mec="white", mew=0.7, zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(k) for k in xs])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(1.6, 40)
        lo, hi = min(ys), max(ys)
        ax.set_ylim(lo - 0.28 * (hi - lo), hi + 0.30 * (hi - lo))
        H.fs.assert_inside(ax, xs, ys)
        ax.set_xlabel("candidates per parent", fontsize=SZ["label"], labelpad=2)
        ax.set_ylabel("tokens accepted\nper round", fontsize=SZ["label"], labelpad=2)
        ax.set_title(short(b), fontsize=SZ["header"], pad=4, color=H.INK)
        # the k=32 delta, on whichever side of the last point has room
        up = ys[-1] >= k16["tau"]
        ax.annotate(f"{100 * (ys[-1] / k16['tau'] - 1):+.1f}% at 32", (32, ys[-1]), textcoords="offset points",
                    xytext=(-2, 10 if up else -10), ha="right", va="bottom" if up else "top", fontsize=7.6,
                    color=H.MUTED)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.205, 0.5)
    H.save(fig, str(HERE / "c1_width"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- c2
def c2_budget():
    """How much a bigger tree buys. Left: what a round accepts. Right: what the wall clock does about it."""
    items = [H.item("the budget we ship, 127 nodes", C["ours"], kind="marker", ms=6.4)]
    fig, axes = panels(items, n=2, gap_x=56.0, left=54.0)
    rs = sorted([r for r in D["budget"] if r["block"] == BLOCKS[2]], key=lambda r: r["nodes"])
    xs = [r["nodes"] for r in rs]
    for ax, field, ylab in ((axes[0], "tau", "tokens accepted\nper round"),
                            (axes[1], "speedup", "speed-up over\nthe original model")):
        ys = [r[field] for r in rs]
        ax.plot(xs, ys, "-o", color="#9AAEC8", lw=1.5, ms=4.8, mfc="#9AAEC8", mec="white", mew=0.6, zorder=3)
        i = xs.index(127)
        ax.plot([127], [ys[i]], "o", ms=7.0, mfc=C["ours"], mec="white", mew=0.7, zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(n) for n in xs])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(12, 330)
        lo, hi = min(ys), max(ys)
        ax.set_ylim(lo - 0.18 * (hi - lo), hi + 0.26 * (hi - lo))
        H.fs.assert_inside(ax, xs, ys)
        ax.set_xlabel("nodes in the draft tree", fontsize=SZ["label"], labelpad=2)
        ax.set_ylabel(ylab, fontsize=SZ["label"], labelpad=2)
        ax.annotate(f"{100 * (ys[-1] / ys[i] - 1):+.0f}%", ((127 * 255) ** 0.5, (ys[i] + ys[-1]) / 2),
                    textcoords="offset points", xytext=(0, -10), ha="center", va="top",
                    fontsize=SZ["legend"], color=H.MUTED)
    axes[0].set_title("what a bigger tree accepts", fontsize=SZ["header"], pad=4, color=H.INK)
    axes[1].set_title("what it is worth in wall clock", fontsize=SZ["header"], pad=4, color=H.INK)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.115, 0.5)
    H.save(fig, str(HERE / "c2_budget"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- c3
def c3_spend():
    """Two margins of the same 127 nodes: by sibling rank (left) and by depth in the block (right)."""
    items = [H.item("share of the nodes", "#C3C9D2"), H.item("share of the accepted tokens", C["ours"])]
    fig, axes = panels(items, n=2, gap_x=56.0)
    sp = D["spend"]["ours"]
    ranks = [str(i) for i in range(9)]
    ax = axes[0]
    for k, (field, col) in enumerate((("spend", "#C3C9D2"), ("accepted", C["ours"]))):
        ys = [100 * sp[field]["rank"][r] for r in ranks]
        ax.bar(np.arange(9) + (k - 0.5) * 0.4, ys, 0.36, color=col, lw=0, zorder=3)
    ax.text(0, 100 * sp["accepted"]["rank"]["0"] + 2.5, f"{100 * sp['accepted']['rank']['0']:.0f}% of the accepted "
            f"tokens come from\n{100 * sp['spend']['rank']['0']:.0f}% of the nodes", ha="left", va="bottom",
            fontsize=7.6, color=H.INK, linespacing=1.1)
    ax.set_xticks(range(9), ["top-1", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th+"])
    ax.tick_params(axis="x", length=0, pad=2, labelsize=7.4)
    ax.set_xlabel("the drafter's rank for that parent", fontsize=SZ["label"], labelpad=2)
    ax.set_ylabel("share of the budget (%)", fontsize=SZ["label"], labelpad=2)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("by sibling rank", fontsize=SZ["header"], pad=4, color=H.INK)

    ax = axes[1]
    deps = sorted(int(d) for d in sp["spend"]["depth"])
    for field, col, lab in (("spend", "#C3C9D2", "nodes"), ("accepted", C["ours"], "accepted")):
        ys = [100 * sp[field]["depth"][str(d)] for d in deps]
        ax.fill_between(deps, 0, ys, color=col, alpha=0.9 if field == "spend" else 0.75, lw=0,
                        zorder=2 if field == "spend" else 3)
    ax.set_xlim(1, max(deps))
    ax.set_ylim(0, max(100 * sp["spend"]["depth"][str(d)] for d in deps) * 1.3)
    ax.set_xlabel("position inside the draft block", fontsize=SZ["label"], labelpad=2)
    ax.set_ylabel("share of the budget (%)", fontsize=SZ["label"], labelpad=2)
    ax.set_title("by position in the block", fontsize=SZ["header"], pad=4, color=H.INK)
    ax.text(max(deps) * 0.97, ax.get_ylim()[1] * 0.97, "the nodes crowd into the first few positions;\n"
            "what they accept is spread evenly across the block", ha="right", va="top", fontsize=7.6,
            color=H.MUTED, linespacing=1.1)
    for a in axes:
        a.yaxis.set_label_coords(-0.115, 0.5)
    H.save(fig, str(HERE / "c3_spend"), width="full")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- c4
def c4_nodes():
    """What a node buys, across every tree drafter and every block of table 1."""
    npt = D["nodes_per_token"]
    blocks, series = [], {m: [] for m in TREE_METHODS}
    for key in npt:
        blk, mode, meth = key.split("|")
        if mode != "greedy" or meth not in TREE_METHODS:
            continue
        if blk not in blocks:
            blocks.append(blk)
    blocks = [b for b in ["OmniSVG-8B, text-to-SVG", "OmniSVG-4B, text-to-SVG", "InternSVG-8B, text-to-SVG",
                          "OmniSVG-8B, image-to-SVG", "InternSVG-8B, image-to-SVG"] if b in blocks]
    for m in TREE_METHODS:
        series[m] = [npt.get(f"{b}|greedy|{m}") for b in blocks]
    items = [H.item(m, H.METHOD[m]) for m in TREE_METHODS]
    W = H.FULL_W
    leg = H.legend_height(items, W, x_pt=46, width_pt=W * 72 - 50, gap_pt=12)
    left, bottom, ph = 46.0, 44.0, 118.0
    Ht = bottom + ph + 6 + leg
    fig = plt.figure(figsize=(W, Ht / 72))
    H.legend(fig, items, x_pt=46, top_pt=0, width_pt=W * 72 - 50, gap_pt=12)
    ax = H.axes_pt(fig, left, bottom, W * 72 - left - 6, ph)
    w = 0.2
    for k, m in enumerate(TREE_METHODS):
        ys = series[m]
        xs = np.arange(len(blocks)) + (k - 1.5) * w
        ok = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.bar([x for x, _ in ok], [y for _, y in ok], w * 0.9, color=H.METHOD[m], lw=0, zorder=3)
        for x, y in ok:
            ax.text(x, y + 0.25, f"{y:.1f}", ha="center", va="bottom", fontsize=7.4,
                    color=C["ours"] if m == "Ours" else H.INK, fontweight="bold" if m == "Ours" else "normal")
    ax.set_xticks(range(len(blocks)), ["" for _ in blocks])
    ax.tick_params(axis="x", length=0)
    tr = ax.get_xaxis_transform()
    for i, b in enumerate(blocks):
        mdl, task = b.split(", ")
        ax.text(i, -0.03, f"{mdl}\n{task}", transform=tr, ha="center", va="top", fontsize=SZ["tick"],
                linespacing=1.05, color=H.INK)
    ax.set_ylim(0, max(v for s in series.values() for v in s if v) * 1.18)
    ax.set_ylabel("tree nodes spent per accepted token", fontsize=SZ["label"], labelpad=2)
    ax.set_xlim(-0.55, len(blocks) - 0.45)
    H.save(fig, str(HERE / "c4_nodes"), width="full")
    plt.close(fig)


if __name__ == "__main__":
    check()
    names = []
    for fn in (c1_width, c2_budget, c3_spend, c4_nodes):
        try:
            fn()
            names.append(fn.__name__)
            print("wrote", fn.__name__)
        except AssertionError as e:
            print("CHECK", fn.__name__, e)
    H.sheet([str(HERE / f"{n}.png") for n in names], str(HERE / "_sheet.png"), cols=1,
            title="D3 round 1 - the shape of the draft tree (double column, print scale)")
