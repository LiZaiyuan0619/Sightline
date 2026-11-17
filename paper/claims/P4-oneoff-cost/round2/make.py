"""P4 round 2 --  (B line, C2): round 1's p4a_stack (appendix, user ) moved onto the house style.
Style only: same stage timings (runs/2026-10-30/analysis/timing_split_i2.md, anchors asserted by round 1), same
stacked bars, same labels on the bars. What changes: the first bar is named "original model" (ENCODING.md SS3; round 1
wrote "AR"); the legend row by house.legend; house.apply; house.save (blocks on a geometry FAIL).
Colour stays round 1's: here colour encodes the STAGE (preprocess / vision / prefill / decode), and only our decode
segment takes our blue, as finalized; the original model's grey of ENCODING.md SS1 is for figures that colour by
method, so it is not forced onto this one.
Run: python paper/claims/P4-oneoff-cost/round2/make.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import matplotlib.pyplot as plt
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(RUNS / "2026-11-06/chores/c2"))
import c2lib as C  # noqa: E402

H = C.H
spec = importlib.util.spec_from_file_location("p4r1", HERE.parent / "round1/make.py")
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
H.apply()
LABEL = {"AR": "original\nmodel"}


def p4a_stack():
    fig = plt.figure(figsize=(r1.CW, 2.7))
    ax = fig.add_axes([0.17, 0.15, 0.81, 0.63])
    for i, m in enumerate(r1.NAMES):
        b = 0.0
        for name, v in (("preprocess", r1.PRE), ("vision encoder", r1.VIS), ("prefill", r1.PREF),
                        ("decode", r1.DEC[m])):
            ax.bar(i, v / 1000, bottom=b, width=0.62, color=r1.STAGE_C[name] if m != "Ours" or name != "decode"
                   else r1.COLOR["Ours"], linewidth=0, zorder=2)
            b += v / 1000
        share = r1.ONE / (r1.ONE + r1.DEC[m]) * 100
        ax.text(i, b + 0.15, f"{b:.2f} s\none-off {share:.0f}%", ha="center", va="bottom", fontsize=7.0,
                linespacing=1.0, color=r1.COLOR["Ours"] if m == "Ours" else "#2E2E2E")
    ax.set_xlim(-0.6, len(r1.NAMES) - 0.4)
    ax.set_ylim(0, 11.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("seconds per image", fontsize=H.SZ["label"], labelpad=2)
    ax.set_xticks(range(len(r1.NAMES)))
    ax.set_xticklabels([LABEL.get(m, m) for m in r1.NAMES], fontsize=H.SZ["tick"], linespacing=1.05)
    ax.tick_params(axis="x", length=0, pad=2)
    C.legend_rect(fig, [0.17, 0.88, 0.82, 0.09], C.patch_items(
        [(r1.STAGE_C["preprocess"], "preprocess"), (r1.STAGE_C["vision encoder"], "vision"),
         (r1.STAGE_C["prefill"], "prefill"), (r1.STAGE_C["decode"], "decode")]), gap_cap_frac=0.05)
    H.save(fig, str(HERE / "p4a_stack"), width="col")


if __name__ == "__main__":
    p4a_stack()
    print("wrote p4a_stack")
