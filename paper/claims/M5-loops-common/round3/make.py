"""M5 round 3 --  (B line, C2): round 2's m5e_horizontal moved onto the house style. Style only: same data
(runs/2026-10-30/analysis/loop_share.md via round 1's loader), same bars, same order, same numbers.
What changes:
  - the decoding row labels read "greedy" / "official sampling" (ENCODING.md SS3; round 2 wrote "sampling");
  - to fit the longer label, the left margin is measured from the widest model name and row label instead of a fixed
    fraction (house rule size from the content); the plot area is 110 pt wide instead of 119 (bars 8% shorter);
  - the legend row by house.legend; house.apply; house.save (blocks on a geometry FAIL).
Run: python paper/claims/M5-loops-common/round3/make.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(RUNS / "2026-11-06/chores/c2"))
import c2lib as C  # noqa: E402

H = C.H
spec = importlib.util.spec_from_file_location("m5r2", HERE.parent / "round2/make.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
r1 = r2.r1
H.apply()
MODE_LAB = (("greedy", "greedy"), ("sample", "official sampling"))
TICK_FS, GROUP_FS = 7.5, H.SZ["tick"]


def _width_pt(s, fs, fig):
    t = fig.text(0, 0, s, fontsize=fs, linespacing=1.1)
    w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    t.remove()
    return w


def main():
    rows = r1.load()
    W, Ht = r2.CW * 72, 3.35 * 72
    fig = plt.figure(figsize=(r2.CW, Ht / 72))
    tick_w = max(_width_pt(l, TICK_FS, fig) for _, l in MODE_LAB)
    group_w = max(_width_pt(r2.SHORT[t], GROUP_FS, fig) for t in r2.TARGETS)
    pad, gap_g = 2.0, 5.0                                   # tick pad (round 2), gap between model and row label
    left = 2.0 + group_w + gap_g + tick_w + pad
    right = 16.4                                            # round 2's: the loop-share numbers' column
    ax = H.axes_pt(fig, left, 0.125 * Ht, W - left - right, 0.77 * Ht)
    y, ticks, groups = 0.0, [], []
    for t in r2.TARGETS:
        ys = []
        for md, lab in MODE_LAB:
            eos = r1.agg(rows, t, md, "eos") * 100
            per = r1.agg(rows, t, md, "periodic") * 100
            cap = r1.agg(rows, t, md, "capped") * 100 - per
            x = 0.0
            for k, v in (("EOS", eos), ("capped", cap), ("loop", per)):
                ax.barh(y, v, left=x, height=0.78, color=r2.COLS[k], linewidth=0, zorder=2)
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
    ax.set_xlabel("outputs (%)", fontsize=H.SZ["label"], labelpad=1.5)
    ax.set_yticks([p for p, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=TICK_FS)
    ax.tick_params(axis="y", length=0, pad=pad)
    trans = blended_transform_factory(fig.dpi_scale_trans, ax.transData)
    for t, ys in groups:
        ax.text((2.0 + group_w) / 72, sum(ys) / 2, r2.SHORT[t], transform=trans, ha="right", va="center",
                fontsize=GROUP_FS, color="#2E2E2E", linespacing=1.1)
    ax.text(101.5, 0.95, "loop", ha="left", va="bottom", fontsize=7.0, color=r1.SVG_C)
    C.legend_rect(fig, [0.02, 0.925, 0.97, 0.06], C.patch_items(
        [(r2.COLS["EOS"], "finished"), (r2.COLS["capped"], "hit length cap"), (r2.COLS["loop"], "cap inside a loop")]),
        gap_cap_frac=0.05)
    old_w = 0.51 * W
    print(f"plot width {W - left - right:.1f} pt (round 2: {old_w:.1f} pt), left margin {left:.1f} pt")
    H.save(fig, str(HERE / "m5e_horizontal"), width="col")
    for t in r2.TARGETS:
        print(f"{t:30s} in a loop: greedy {r1.agg(rows, t, 'greedy', 'periodic'):.0%}  official sampling "
              f"{r1.agg(rows, t, 'sample', 'periodic'):.0%}")


if __name__ == "__main__":
    main()
