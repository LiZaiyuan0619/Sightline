"""Speed-quality landscape: where the accelerated SVG LLMs stand among other SVG generators -- round 1, evening.

Why: table 2 used to open with eight reference rows (other SVG generators). They answer a different question from the
rest of the table (where do these models stand in the field, not which decoding is faster), and mixing them in is what
made the table hard to read (user). User: move them out and try a speed-quality figure; worried the gaps might not
look large -- "draw it and we look".

Claim: on SArena-Icon (text-to-SVG), decoding OmniSVG-8B / InternSVG-8B with our method moves them 5-10x to the left
(1.0-1.1 s per SVG) at unchanged quality, which makes them the fastest SVG generators here by a wide margin; best of 8
with our decoding (2.8 / 5.3 s) is still faster than any other system while it has the best quality.

Data: the numbers printed in table 2 (v13), nothing re-measured:
  runs/2026-11-03/table2_refrows.json   reference rows (5 reported by InternSVG: quality from its tables 2-3, time from
                                        its table 19, Icon only; 3 measured by us on 300 SArena prompts)
  tex/tables/table2a.tex                our two models (AR, Ours, Ours (fast), AR x8, Ours x8), full SArena
Only SArena-Icon: the reported systems have no time for Illustration.

Candidates: l1_fid (single column), l2_clip (single column), l3_both (double column, FID | CLIP-T2I).
Run: python3 paper/claims/T2-landscape/round1/make.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import figstyle as fs  # noqa: E402
from sightline.paths import RUNS

fs.apply_style()
COL_W_IN, FULL_W_IN = 3.25, 6.875
ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REF = RUNS / "2026-11-03/table2_refrows.json"
T2A = RUNS.parent / "tables/table2a.tex"

OURS_C, AR_C = "#1D4E89", "#6E6E6E"
CAT_C = {"optimisation": "#C9A27E", "LLM, reported": "#9C8CC0", "LLM, measured": "#7FAFAF"}
CAT_OF = {"VectorFusion": "optimisation", "SVGDreamer": "optimisation",
          "LLM4SVG-7B": "LLM, reported", "OmniSVG-3B": "LLM, reported", "InternSVG-8B": "LLM, reported",
          "SVGen-7B": "LLM, measured", "SVGen-3B": "LLM, measured", "HiVG-3B": "LLM, measured"}
# Icon columns of a table-2 row: FID, FID-C, CLIP-T2I, CLIP-I2I, s
COL = {"FID": 0, "CLIP-T2I": 2, "s": 4}


def strip(c):
    for _ in range(3):
        c = re.sub(r"^\\(?:textbf|underline)\{(.*)\}$", r"\1", c.strip())
    return c


def load():
    ref = json.loads(REF.read_text())
    others = []
    for grp in ("ref-reported", "ref-measured"):
        for r in ref[grp]:
            v = r["cells"]
            others.append(dict(name=r["method"], cat=CAT_OF[r["method"]], fid=float(v[COL["FID"]]),
                               clip=float(v[COL["CLIP-T2I"]]), s=float(v[COL["s"]])))
    ours, model = {}, None
    for line in T2A.read_text().splitlines():
        m = re.search(r"\\textbf\{(OmniSVG-8B|InternSVG-8B)\}", line)
        if m and "multicolumn{11}" in line:
            model = m.group(1)
            continue
        if model and "&" in line and line.rstrip().endswith(r"\\") and not line.startswith(("Method", " &")):
            raw = [x.strip() for x in line.rstrip()[:-2].split("&")]
            c = [raw[0]] + [strip(x) for x in raw[1:]]          # the row name keeps its \textbf, the cells lose it
            lab = {"AR": "AR", r"\textbf{Ours}": "Ours", r"\textbf{Ours} (fast)": "Ours (fast)",
                   r"AR $\times$8": "AR x8", r"\textbf{Ours} $\times$8": "Ours x8"}.get(c[0])
            if lab:
                ours[(model, lab)] = dict(fid=float(c[1 + COL["FID"]]), clip=float(c[1 + COL["CLIP-T2I"]]),
                                          s=float(c[1 + COL["s"]]))
    assert len(others) == 8 and len(ours) == 10, (len(others), sorted(ours))
    # anchors from the printed table (guide SS8: reproduce a printed number before drawing)
    assert ours[("OmniSVG-8B", "Ours")]["s"] == 1.01 and ours[("InternSVG-8B", "AR x8")]["fid"] == 8.75
    return others, ours


def draw(ax, others, ours, key, ylab, better_low, label_size=7.0):
    xs, ys = [], []
    for o in others:
        ax.scatter(o["s"], o[key], s=26, color=CAT_C[o["cat"]], edgecolor="white", linewidths=0.5, zorder=4)
        xs.append(o["s"])
        ys.append(o[key])
    for model, mk in (("OmniSVG-8B", "o"), ("InternSVG-8B", "s")):
        ar, us = ours[(model, "AR")], ours[(model, "Ours")]
        ar8, us8 = ours[(model, "AR x8")], ours[(model, "Ours x8")]
        fast = ours[(model, "Ours (fast)")]
        for a, b in ((ar, us), (ar8, us8)):
            ax.annotate("", xy=(b["s"] * 1.12, b[key]), xytext=(a["s"] / 1.12, a[key]),
                        arrowprops=dict(arrowstyle="-|>", color=OURS_C, lw=0.8, shrinkA=0, shrinkB=0,
                                        mutation_scale=6), zorder=3)
        ax.scatter([ar["s"], ar8["s"]], [ar[key], ar8[key]], s=24, marker=mk, facecolor="white",
                   edgecolor=AR_C, linewidths=0.9, zorder=5)
        ax.scatter([us["s"], fast["s"], us8["s"]], [us[key], fast[key], us8[key]], s=30, marker=mk,
                   color=OURS_C, edgecolor="white", linewidths=0.5, zorder=6)
        xs += [ar["s"], ar8["s"], us["s"], fast["s"], us8["s"]]
        ys += [ar[key], ar8[key], us[key], fast[key], us8[key]]
    ax.set_xscale("log")
    ax.set_xlim(0.5, 450)
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.10
    if better_low:
        ax.set_ylim(max(0.0, lo - 3.6 * pad), hi + pad)    # room above the best points for the label column
    else:
        ax.set_ylim(lo - pad, hi + 4.0 * pad)
    fs.assert_inside(ax, xs, ys)
    ax.set_xticks([1, 3, 10, 30, 100, 300])
    ax.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    ax.set_xlabel("seconds per SVG (log scale)", fontsize=fs.SIZE["label"], labelpad=1.5)
    ax.set_ylabel(ylab, fontsize=fs.SIZE["label"], labelpad=2)
    if better_low:
        ax.invert_yaxis()            # better is up, as for every other quality axis in the paper
    ax.tick_params(axis="both", pad=1.5)
    # direct labels (guide SS4.4). Sparse points get a label beside them; the three that sit among the arrows
    # get a leader line to a column in the empty upper right, in the order of their heights
    side = {"VectorFusion": "right", "SVGDreamer": "left", "LLM4SVG-7B": "right", "OmniSVG-3B": "left",
            "SVGen-7B": "right", "SVGen-3B": "right", "HiVG-3B": "right"}
    for o in others:
        if o["name"] in side:
            ha = "right" if side[o["name"]] == "left" else "left"      # side = where the label goes
            ax.annotate(o["name"], (o["s"], o[key]), xytext=(-4 if ha == "right" else 4, 0),
                        textcoords="offset points", ha=ha, va="center", fontsize=label_size, color="#4A4A4A")
    rep = next(o for o in others if o["name"] == "InternSVG-8B")
    crowd = [(rep["s"], rep[key], "InternSVG-8B (reported)", "#4A4A4A"),
             (ours[("OmniSVG-8B", "AR")]["s"], ours[("OmniSVG-8B", "AR")][key], "OmniSVG-8B, AR", AR_C),
             (ours[("InternSVG-8B", "AR")]["s"], ours[("InternSVG-8B", "AR")][key], "InternSVG-8B, AR", AR_C)]
    lo_, hi_ = sorted(ax.get_ylim())                          # (the FID axis is inverted by now)
    good_end = lo_ if better_low else hi_                     # the "better" end of the axis
    span = hi_ - lo_
    crowd.sort(key=lambda t: t[1], reverse=not better_low)    # best first = top first
    for i, (x, y, text, col) in enumerate(crowd):
        ty = good_end + (i + 0.6) * span * 0.085 * (1 if better_low else -1)
        ax.annotate(text, (x, y), xytext=(40, ty), textcoords="data", ha="left", va="center",
                    fontsize=label_size, color=col,
                    arrowprops=dict(arrowstyle="-", color="#B0B0B0", lw=0.45, shrinkA=1.5, shrinkB=3.0))


def legend(ax, loc, ncol=1, anchor=None):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", ms=4.2, color=CAT_C["optimisation"]),
         Line2D([], [], marker="o", ls="", ms=4.2, color=CAT_C["LLM, reported"]),
         Line2D([], [], marker="o", ls="", ms=4.2, color=CAT_C["LLM, measured"]),
         Line2D([], [], marker="o", ls="", ms=4.2, mfc="white", mec=AR_C, mew=0.9),
         Line2D([], [], marker="o", ls="", ms=4.6, color=OURS_C)]
    ax.legend(h, ["optimisation-based", "SVG LLMs (reported)", "SVG LLMs (measured)",
                  "AR, AR ×8 (○ OmniSVG-8B, □ InternSVG-8B)", "Ours, Ours (fast), Ours ×8"],
              loc=loc, fontsize=7.0, frameon=False, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.35,
              ncol=ncol, bbox_to_anchor=anchor, columnspacing=1.0)


def l1_fid(others, ours):
    fig = plt.figure(figsize=(COL_W_IN, 3.6))
    ax = fig.add_axes([0.15, 0.34, 0.83, 0.645])
    draw(ax, others, ours, "fid", "FID on SArena-Icon (lower is better)", better_low=True)
    legend(ax, "upper left", anchor=(-0.16, -0.17))
    fs.save(fig, str(OUT / "l1_fid"), width=COL_W_IN)


def l2_clip(others, ours):
    fig = plt.figure(figsize=(COL_W_IN, 3.6))
    ax = fig.add_axes([0.15, 0.34, 0.83, 0.645])
    draw(ax, others, ours, "clip", "CLIP-T2I on SArena-Icon", better_low=False)
    legend(ax, "upper left", anchor=(-0.16, -0.17))
    fs.save(fig, str(OUT / "l2_clip"), width=COL_W_IN)


def l3_both(others, ours):
    fig = plt.figure(figsize=(FULL_W_IN, 2.85))
    a = fig.add_axes([0.07, 0.25, 0.42, 0.73])
    b = fig.add_axes([0.575, 0.25, 0.42, 0.73])
    draw(a, others, ours, "fid", "FID (lower is better)", better_low=True)
    draw(b, others, ours, "clip", "CLIP-T2I", better_low=False)
    legend(a, "upper left", ncol=3, anchor=(-0.1, -0.2))
    for ax, s in ((a, "a"), (b, "b")):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.065, bb.y1, s, fontsize=12, fontweight="bold", ha="left", va="top")
    fs.save(fig, str(OUT / "l3_both"), width=FULL_W_IN)


def main():
    others, ours = load()
    for k in ("OmniSVG-8B", "InternSVG-8B"):
        ar, us = ours[(k, "AR")], ours[(k, "Ours")]
        print(f"{k}: AR {ar['s']} s -> ours {us['s']} s (x{ar['s'] / us['s']:.1f}); "
              f"x8 {ours[(k, 'AR x8')]['s']} -> {ours[(k, 'Ours x8')]['s']} s")
    fastest_other = min(others, key=lambda o: o["s"])
    print(f"fastest other system: {fastest_other['name']} {fastest_other['s']} s")
    for fn in (l1_fid, l2_clip, l3_both):
        fn(others, ours)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
