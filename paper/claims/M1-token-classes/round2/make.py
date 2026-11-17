"""M1 round 2 --  night. One message, one panel, the five blocks complete, no method ranking.

User on round 1: m1a looks like a method comparison in which ours is not always first; m1b switches to shares
and nobody knows why it shows tokens / DDTree / ours only, or why rejections; m1c's data are visibly incomplete; m1d
is too hard to read.
What M1 is for: a property of SVG output, not of any method -- most of it is structure a draft almost always gets
right, and the misses sit on the numbers. So round 2 shows ONE draft (ours; every learned draft is within a few
points, appendix), all five blocks, and plain quantities.

Candidates (single column, 3.25 in):
  m1e_bars   how often the draft's token is right: structure tokens vs numbers, per block
  m1g_share numbers: their share of the output vs their share of the draft's misses, per block
  m1h_text   one real SVG, printed as text; where each round stopped is highlighted (always on a number)
Data: as round 1 (profile classes, loop-free positions; OmniSVG-4B from the 4B-vocabulary recompute).
Run: python3 paper/claims/M1-token-classes/round2/make.py
"""
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("m1r1", ROOT / "paper/claims/M1-token-classes/round1/make.py")
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
fs, SZ, COLOR, BLOCKS, PROF, GROUP, GCOL = r1.fs, r1.SZ, r1.COLOR, r1.BLOCKS, r1.PROF, r1.GROUP, r1.GCOL
sys.path.insert(0, str(ROOT / "src"))
from svgtok.types import CMD_BY_ID, coord_xy, color_hex, token_type, ARC_START  # noqa: E402
from sightline.paths import RUNS

CW = 3.25
SHORT = {BLOCKS[0]: "OmniSVG\n8B, text", BLOCKS[1]: "OmniSVG\n4B, text", BLOCKS[2]: "InternSVG\n8B, text",
         BLOCKS[3]: "OmniSVG\n8B, image", BLOCKS[4]: "InternSVG\n8B, image"}


def group_acc(blk, mode="greedy", m="Ours"):
    """share-weighted draft accuracy per group (eos excluded: it is a single position per row)."""
    d = PROF[blk][mode][m]["classes"]
    out = {}
    for g in ("structure", "numbers"):
        ks = [k for k, v in d.items() if GROUP.get(k) == g and k != "eos" and v["acc"] is not None]
        w = np.array([d[k]["share"] for k in ks])
        a = np.array([d[k]["acc"] for k in ks])
        out[g] = float((w * a).sum() / w.sum())
    return out


def label_axis(ax):
    ax.set_xticks(range(len(BLOCKS)))
    ax.set_xticklabels([SHORT[b] for b in BLOCKS], fontsize=7.5, linespacing=1.12)
    ax.tick_params(axis="x", length=0, pad=2)


# --------------------------------------------------------------------------------------------------
def m1e_bars():
    fig = plt.figure(figsize=(CW, 2.6))
    ax = fig.add_axes([0.17, 0.22, 0.81, 0.62])
    Y = []
    for i, b in enumerate(BLOCKS):
        g = group_acc(b)
        for j, (grp, col) in enumerate((("structure", GCOL["structure"]), ("numbers", GCOL["numbers"]))):
            x = i + (j - 0.5) * 0.40
            v = (1 - g[grp]) * 100                   # how often the draft is wrong at such a token
            ax.bar(x, v, width=0.38, color=col, linewidth=0, zorder=2)
            ax.text(x, v + 0.25, f"{v:.0f}" if v >= 1.5 else f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                    color="#2E2E2E")
            Y.append(v)
    ax.set_ylim(0, 13)
    fs.assert_inside(ax, [0, 4], Y)
    ax.set_yticks([0, 5, 10])
    ax.set_ylabel("draft wrong (%)", fontsize=SZ["label"], labelpad=2)
    label_axis(ax)
    r1.m2.legend_row(fig, [0.17, 0.88, 0.82, 0.08], [(GCOL["structure"], "structure (commands, markup)"),
                                                     (GCOL["numbers"], "numbers")])
    fs.save(fig, str(HERE / "m1e_bars"), width=CW)


def m1g_share():
    fig = plt.figure(figsize=(CW, 2.6))
    ax = fig.add_axes([0.17, 0.22, 0.81, 0.62])
    for i, b in enumerate(BLOCKS):
        tok = r1.group_shares(b, "greedy", "Ours", "share")["numbers"] * 100
        mis = r1.group_shares(b, "greedy", "Ours", "rej_share")["numbers"] * 100
        for j, (v, col) in enumerate(((tok, "#EBD3A4"), (mis, "#C98A1E"))):
            x = i + (j - 0.5) * 0.40
            ax.bar(x, v, width=0.38, color=col, linewidth=0, zorder=2)
            ax.text(x, v + 1.0, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#2E2E2E")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("numbers (%)", fontsize=SZ["label"], labelpad=2)
    label_axis(ax)
    r1.m2.legend_row(fig, [0.17, 0.88, 0.82, 0.08], [("#EBD3A4", "of the output"), ("#C98A1E", "of the draft's misses")])
    fs.save(fig, str(HERE / "m1g_share"), width=CW)


# --------------------------------------------------------------------------------------------------
def token_text(t):
    ty = token_type(t)
    if ty == "cmd":
        return CMD_BY_ID[t]
    if ty == "coord":
        x, y = coord_xy(t)
        return f"{x},{y}"
    if ty == "color":
        return color_hex(t)
    if ty == "arc":
        return str(t - ARC_START)
    return None                                   # bos / eos / text tokens: not printed


def m1h_text():
    """The first four rounds of one real InternSVG-8B text-to-SVG output (ours, greedy, no loop, no repeated span),
    printed as the SVG code it is. A grey bar starts a new round; the highlighted piece is where a round stopped
    because the draft was wrong there. Round 1 fills the whole 32-token window with the header; the others stop on
    numbers. Data: runs/2026-11-03/figdata_b1/m1_example_isvg.json (runs/2026-11-03/figdata_b1_m1_example.py)."""
    ex = json.loads((RUNS / "2026-11-03/figdata_b1/m1_example_isvg.json").read_text())
    toks, rounds, cap = ex["tokens"], ex["rounds"], ex["cap"]
    assert ex["gidx"] == 0 and rounds == [32, 8, 14, 10], rounds
    start_of, stop = {}, set()
    s = 1                                          # token 0 comes from the prefill
    for j, e in enumerate(rounds):
        start_of[s] = j
        if e < cap:
            stop.add(s + e - 1)
        s += e
    fig = plt.figure(figsize=(CW, 1.62))
    ax = fig.add_axes([0.015, 0.25, 0.97, 0.73])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rend = fig.canvas.get_renderer()
    aw = ax.get_window_extent(rend).width
    x, y, line_h, fsz = 0.0, 0.90, 0.25, 7.4
    n_num = 0
    for i, piece in enumerate(toks[:s]):
        if i in start_of and i > 1:
            x += 0.008
            ax.plot([x - 0.004, x - 0.004], [y - 0.09, y + 0.09], color="#9A9A9A", lw=1.0)
        t = ax.text(x, y, piece, fontsize=fsz, family="monospace", ha="left", va="center", color="#2E2E2E")
        w = t.get_window_extent(rend).width / aw
        if x + w > 0.998:
            t.remove()
            x, y = 0.0, y - line_h
            t = ax.text(x, y, piece.lstrip(), fontsize=fsz, family="monospace", ha="left", va="center",
                        color="#2E2E2E")
            w = t.get_window_extent(rend).width / aw
        if i in stop:
            n_num += bool(re.search(r"[0-9]", piece))
            ax.add_patch(FancyBboxPatch((x - 0.002, y - 0.085), w + 0.004, 0.17, boxstyle="round,pad=0.004",
                                        facecolor="#F1C27D", edgecolor="none", zorder=0))
        x += w
    assert y > 0.05, "text ran out of the panel"
    print(f"m1h: {len(rounds)} rounds, {len(stop)} stopped by a wrong draft token, {n_num} of them on a number")
    lg = fig.add_axes([0.015, 0.02, 0.97, 0.15])
    lg.axis("off")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    lg.add_patch(FancyBboxPatch((0.0, 0.22), 0.045, 0.56, boxstyle="round,pad=0.01", facecolor="#F1C27D",
                                edgecolor="none"))
    lg.text(0.06, 0.5, "round stopped here (draft wrong)", fontsize=SZ["note"], va="center")
    lg.plot([0.66, 0.66], [0.15, 0.85], color="#9A9A9A", lw=1.0)
    lg.text(0.68, 0.5, "next round", fontsize=SZ["note"], va="center")
    fs.save(fig, str(HERE / "m1h_text"), width=CW)


def main():
    for b in BLOCKS:
        g = group_acc(b)
        print(f"{b:28s} structure {g['structure']:.3f}  numbers {g['numbers']:.3f}")
    for fn in (m1e_bars, m1g_share, m1h_text):
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
