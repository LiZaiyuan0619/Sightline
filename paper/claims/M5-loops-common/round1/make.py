"""M5: repetition loops are a property of SVG LLMs, not of structured generation in general -- round 1, night.

Claim: with their own decoding, the seven SVG targets spend % of their greedy output tokens inside a repetition
loop (one detector, one ruler, the target's own AR output); OmniLottie-3B, which writes Lottie JSON, only 6-12%.
Sampling lowers the share for some targets (StarVector-1B to 1-3%), which the figure shows as it is.
Data: runs/2026-10-30/analysis/loop_share.md (sightline/svgtok/loops.analyze on each target's AR rows).
Candidates: m5a_bars (tokens in loops per target, greedy / sampling), m5b_ends (how outputs end, greedy),
            m5c_strip (every subset as a dot), m5d_icon_illus (icon vs illustration, greedy).
Run: python3 paper/claims/M5-loops-common/round1/make.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("m2r2", ROOT / "paper/claims/M2-block-ceiling/round2/make.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)
fs, SZ, W = m2.fs, m2.SZ, m2.W
TARGETS = ["OmniSVG-8B, text-to-SVG", "OmniSVG-4B, text-to-SVG", "InternSVG-8B, text-to-SVG",
           "OmniSVG-8B, image-to-SVG", "InternSVG-8B, image-to-SVG", "StarVector-1B, image-to-SVG",
           "StarVector-8B, image-to-SVG", "OmniLottie-3B, text-to-Lottie"]
LABEL = {t: t.replace(", ", "\n") for t in TARGETS}
SVG_C, LOT_C = "#C98A1E", "#7FAFAF"          # SVG targets amber (the loop colour of this paper), Lottie teal


def load():
    rows = []
    for line in (RUNS / "2026-10-30/analysis/loop_share.md").read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| target") or line.startswith("|---"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        pct = lambda s: float(s.rstrip("%")) / 100
        rows.append(dict(target=c[0], mode=c[1], subset=c[2], n=int(c[3]), cap=int(c[4]), eos=pct(c[5]),
                         capped=pct(c[6]), periodic=pct(c[7]), loop_tok=pct(c[8])))
    assert {r["target"] for r in rows} == set(TARGETS), {r["target"] for r in rows}
    return rows


def agg(rows, target, mode, key, sub=None):
    """row-weighted mean over the subsets (each subset has 150 or 300 prompts)."""
    rs = [r for r in rows if r["target"] == target and r["mode"] == mode and (sub is None or sub in r["subset"])]
    w = np.array([r["n"] for r in rs], float)
    return float((w * np.array([r[key] for r in rs])).sum() / w.sum())


def xlabels(ax, xs):
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[t] for t in TARGETS], fontsize=7.5, linespacing=1.12)
    ax.tick_params(axis="x", length=0, pad=2)
    for t in ax.get_xticklabels():
        if "Lottie" in t.get_text():
            t.set_color("#3E7E7E")


# --------------------------------------------------------------------------------------------------
def m5a_bars(rows):
    fig = plt.figure(figsize=(W, 2.55))
    ax = fig.add_axes([0.075, 0.24, 0.92, 0.62])
    for i, t in enumerate(TARGETS):
        c = LOT_C if "Lottie" in t else SVG_C
        for j, (md, alpha) in enumerate((("greedy", 1.0), ("sample", 0.45))):
            v = agg(rows, t, md, "loop_tok") * 100
            x = i + (j - 0.5) * 0.38
            ax.bar(x, v, width=0.36, color=c, alpha=alpha, linewidth=0, zorder=2)
            ax.text(x, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#2E2E2E")
    ax.set_xlim(-0.6, len(TARGETS) - 0.4)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("output tokens inside\na repetition loop (%)", fontsize=SZ["label"], labelpad=2, linespacing=1.05)
    xlabels(ax, range(len(TARGETS)))
    m2.legend_row(fig, [0.075, 0.91, 0.5, 0.07], [(SVG_C, "greedy"), ("#E9CFA0", "official sampling")])
    fs.save(fig, str(HERE / "m5a_bars"), width=W)


def m5b_ends(rows):
    """How each target's outputs end (greedy): with EOS, cut at the length cap without a loop, cut inside a loop."""
    fig = plt.figure(figsize=(W, 2.55))
    ax = fig.add_axes([0.075, 0.24, 0.92, 0.62])
    cols = {"EOS": "#B9C6D3", "capped": "#E4D7C3", "loop": SVG_C}
    for i, t in enumerate(TARGETS):
        eos = agg(rows, t, "greedy", "eos") * 100
        per = agg(rows, t, "greedy", "periodic") * 100
        cap = agg(rows, t, "greedy", "capped") * 100 - per
        b = 0
        for k, v in (("EOS", eos), ("capped", cap), ("loop", per)):
            ax.bar(i, v, bottom=b, width=0.62, color=cols[k], linewidth=0, zorder=2)
            if k == "loop":
                ax.text(i, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white")
            b += v
    ax.set_xlim(-0.6, len(TARGETS) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("outputs (%), greedy", fontsize=SZ["label"], labelpad=2)
    xlabels(ax, range(len(TARGETS)))
    m2.legend_row(fig, [0.075, 0.91, 0.75, 0.07], [(cols["EOS"], "finished (EOS)"),
                                                  (cols["capped"], "hit the length cap"),
                                                  (cols["loop"], "hit the cap inside a loop")])
    fs.save(fig, str(HERE / "m5b_ends"), width=W)


def m5c_strip(rows):
    """Every benchmark subset of every target as one dot (filled greedy, hollow sampling); bar = greedy mean."""
    rng = np.random.default_rng(1)
    fig = plt.figure(figsize=(W, 2.55))
    ax = fig.add_axes([0.075, 0.24, 0.92, 0.62])
    X, Y = [], []
    for i, t in enumerate(TARGETS):
        c = LOT_C if "Lottie" in t else SVG_C
        ax.bar(i, agg(rows, t, "greedy", "loop_tok") * 100, width=0.62, color=c, alpha=0.25, linewidth=0, zorder=1)
        for r in (r for r in rows if r["target"] == t):
            x = i + rng.uniform(-0.2, 0.2)
            ax.scatter([x], [r["loop_tok"] * 100], s=14, marker="o",
                       facecolor=c if r["mode"] == "greedy" else "white", edgecolor=c, linewidths=0.9, zorder=3)
            X.append(x)
            Y.append(r["loop_tok"] * 100)
    ax.set_xlim(-0.6, len(TARGETS) - 0.4)
    ax.set_ylim(0, 68)
    fs.assert_inside(ax, X, Y)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("output tokens inside\na repetition loop (%)", fontsize=SZ["label"], labelpad=2, linespacing=1.05)
    xlabels(ax, range(len(TARGETS)))
    h = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=SVG_C, mec=SVG_C),
         Line2D([], [], marker="o", ls="", ms=4.5, mfc="white", mec=SVG_C, mew=0.9)]
    fig.legend(h, ["greedy, one benchmark subset", "official sampling"], loc="upper left", bbox_to_anchor=(0.07, 0.99),
               ncol=2, fontsize=SZ["legend"], frameon=False, handletextpad=0.2, columnspacing=1.2)
    fs.save(fig, str(HERE / "m5c_strip"), width=W)


def m5d_icon_illus(rows):
    """Greedy, icons vs illustrations: longer outputs loop more."""
    fig = plt.figure(figsize=(W, 2.55))
    ax = fig.add_axes([0.075, 0.24, 0.92, 0.62])
    for i, t in enumerate(TARGETS):
        c = LOT_C if "Lottie" in t else SVG_C
        subs = ("real", "synthetic") if "Lottie" in t else ("icon", "illustration")
        for j, sub in enumerate(subs):
            v = agg(rows, t, "greedy", "loop_tok", sub) * 100
            x = i + (j - 0.5) * 0.38
            ax.bar(x, v, width=0.36, color=c, alpha=0.55 if j == 0 else 1.0, linewidth=0, zorder=2)
            ax.text(x, v + 0.8, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color="#2E2E2E")
    ax.set_xlim(-0.6, len(TARGETS) - 0.4)
    ax.set_ylim(0, 62)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("tokens in loops (%),\ngreedy", fontsize=SZ["label"], labelpad=2, linespacing=1.05)
    xlabels(ax, range(len(TARGETS)))
    m2.legend_row(fig, [0.075, 0.91, 0.75, 0.07], [("#E9CFA0", "icons (Lottie: real)"),
                                                  (SVG_C, "illustrations (Lottie: synthetic)")])
    fs.save(fig, str(HERE / "m5d_icon_illus"), width=W)


def main():
    rows = load()
    for t in TARGETS:
        print(f"{t:32s} greedy {agg(rows, t, 'greedy', 'loop_tok'):.0%}  sampling {agg(rows, t, 'sample', 'loop_tok'):.0%}")
    for fn in (m5a_bars, m5b_ends, m5c_strip, m5d_icon_illus):
        fn(rows)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
