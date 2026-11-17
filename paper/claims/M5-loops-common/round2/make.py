"""M5 round 2 --  night. Three changes asked by the user on round 1, nothing else:
  1. OmniLottie-3B out: it writes Lottie JSON, not SVG, and only served as a contrast; the claim is about SVG LLMs
     (the Lottie contrast goes to the appendix / one sentence).
  2. Horizontal: models x decoding modes on the y axis, the share of outputs on the x axis (fits one sub-figure).
  3. Both decodings: greedy AND each model's official sampling (no new experiment -- loop_share.md has both).
Claim: with their own decoding, % of the seven SVG LLMs' greedy outputs end inside a repetition loop; under
official sampling % still do, the exception being StarVector-1B (3%) -- shown as it is.
Data: runs/2026-10-30/analysis/loop_share.md (row-weighted over the benchmark subsets, as round 1).
Run: python3 paper/claims/M5-loops-common/round2/make.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m5r1", HERE.parent / "round1/make.py")
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
fs, SZ, m2 = r1.fs, r1.SZ, r1.m2
TARGETS = [t for t in r1.TARGETS if "Lottie" not in t]
SHORT = {t: t.replace(", text-to-SVG", "\ntext").replace(", image-to-SVG", "\nimage") for t in TARGETS}
COLS = {"EOS": "#B9C6D3", "capped": "#E4D7C3", "loop": r1.SVG_C}
CW = 3.25


def main():
    rows = r1.load()
    H = 3.35
    fig = plt.figure(figsize=(CW, H))
    ax = fig.add_axes([0.42, 0.125, 0.51, 0.77])
    y, ticks, groups = 0.0, [], []
    for t in TARGETS:
        ys = []
        for md, lab in (("greedy", "greedy"), ("sample", "sampling")):
            eos = r1.agg(rows, t, md, "eos") * 100
            per = r1.agg(rows, t, md, "periodic") * 100
            cap = r1.agg(rows, t, md, "capped") * 100 - per
            x = 0.0
            for k, v in (("EOS", eos), ("capped", cap), ("loop", per)):
                ax.barh(y, v, left=x, height=0.78, color=COLS[k], linewidth=0, zorder=2)
                x += v
            ax.text(101.5, y, f"{per:.0f}", ha="left", va="center", fontsize=7.5, color=r1.SVG_C)
            ticks.append((y, lab))
            ys.append(y)
            y -= 1.0
        groups.append((t, ys))
        y -= 0.45
    ax.set_ylim(y + 0.45 + 0.35, 0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("outputs (%)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_yticks([p for p, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=2)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for t, ys in groups:
        ax.text(-0.31, sum(ys) / 2, SHORT[t], transform=trans, ha="right", va="center", fontsize=SZ["tick"],
                color="#2E2E2E", linespacing=1.1)
    ax.text(101.5, 0.95, "loop", ha="left", va="bottom", fontsize=7.0, color=r1.SVG_C)
    m2.legend_row(fig, [0.02, 0.925, 0.97, 0.06], [(COLS["EOS"], "finished"), (COLS["capped"], "hit length cap"),
                                                  (COLS["loop"], "cap inside a loop")])
    fs.save(fig, str(HERE / "m5e_horizontal"), width=CW)
    for t in TARGETS:
        print(f"{t:30s} in a loop: greedy {r1.agg(rows, t, 'greedy', 'periodic'):.0%}  sampling "
              f"{r1.agg(rows, t, 'sample', 'periodic'):.0%}")
    print("wrote m5e_horizontal")


if __name__ == "__main__":
    main()
