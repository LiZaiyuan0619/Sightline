"""P4: once decoding is fast, image-to-SVG's one-off cost is a visible share -- round 1, night.

Claim: every image-to-SVG request pays a fixed 196 ms before the first token (preprocess 73 + vision encoder 37 +
prefill 86). For the original model that is 2.1% of the 9.25 s; with our decoding it is 12.1% of 1.62 s, so the
end-to-end speed-up is 5.7x for a 6.4x faster decode -- and the fixed part, not decoding, is what bounds it next.
Data: runs/2026-10-30/analysis/timing_split_i2.md (OmniSVG-8B, MMSVG-Bench 8 icons + 8 illustrations, greedy, one
node). Methods timed: AR, DFlash, DDTree (strongest baseline), ours.
Candidates: p4a_stack (ms per image by stage), p4b_share (one-off share), p4c_amdahl (end-to-end vs decode speed-up),
            p4d_stages (each stage, AR vs ours, log scale).
Run: python3 paper/claims/P4-oneoff-cost/round1/make.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("m2r2", ROOT / "paper/claims/M2-block-ceiling/round2/make.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)
fs, SZ, COLOR = m2.fs, m2.SZ, m2.COLOR
CW = 3.25
PRE, VIS, PREF, AR_DEC = 73.0, 37.0, 86.0, 9055.0
DEC = {"AR": AR_DEC, "DFlash": 1893.0, "DDTree": 1519.0, "Ours": 1424.0}
NAMES = ["AR", "DFlash", "DDTree", "Ours"]
STAGE_C = {"preprocess": "#C9B6D6", "vision encoder": "#A891C2", "prefill": "#7E6AA6", "decode": "#9AAABB"}
ONE = PRE + VIS + PREF
# anchors from the printed table (guide SS8)
assert ONE == 196 and abs(ONE / (ONE + DEC["Ours"]) - 0.121) < 0.001 and abs(ONE / (ONE + AR_DEC) - 0.021) < 0.001


def p4a_stack():
    fig = plt.figure(figsize=(CW, 2.7))
    ax = fig.add_axes([0.17, 0.12, 0.81, 0.66])
    for i, m in enumerate(NAMES):
        b = 0.0
        for name, v in (("preprocess", PRE), ("vision encoder", VIS), ("prefill", PREF), ("decode", DEC[m])):
            ax.bar(i, v / 1000, bottom=b, width=0.62, color=STAGE_C[name] if m != "Ours" or name != "decode" else
                   COLOR["Ours"], linewidth=0, zorder=2)
            b += v / 1000
        share = ONE / (ONE + DEC[m]) * 100
        ax.text(i, b + 0.15, f"{b:.2f} s\none-off {share:.0f}%", ha="center", va="bottom", fontsize=7.0,
                linespacing=1.0, color=COLOR["Ours"] if m == "Ours" else "#2E2E2E")
    ax.set_xlim(-0.6, len(NAMES) - 0.4)
    ax.set_ylim(0, 11.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("seconds per image", fontsize=SZ["label"], labelpad=2)
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels(NAMES, fontsize=SZ["tick"])
    ax.tick_params(axis="x", length=0, pad=2)
    m2.legend_row(fig, [0.17, 0.88, 0.82, 0.09], [(STAGE_C["preprocess"], "preprocess"),
                                                 (STAGE_C["vision encoder"], "vision"), (STAGE_C["prefill"], "prefill"),
                                                 (STAGE_C["decode"], "decode")])
    fs.save(fig, str(HERE / "p4a_stack"), width=CW)


def p4b_share():
    fig = plt.figure(figsize=(CW, 2.4))
    ax = fig.add_axes([0.17, 0.14, 0.81, 0.70])
    for i, m in enumerate(NAMES):
        share = ONE / (ONE + DEC[m]) * 100
        ax.bar(i, share, width=0.6, color=COLOR["Ours"] if m == "Ours" else "#A891C2", linewidth=0, zorder=2)
        ax.text(i, share + 0.3, f"{share:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#2E2E2E")
    ax.set_xlim(-0.6, len(NAMES) - 0.4)
    ax.set_ylim(0, 14.5)
    ax.set_yticks([0, 5, 10])
    ax.set_ylabel("fixed one-off part\nof the latency (%)", fontsize=SZ["label"], labelpad=2, linespacing=1.05)
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels(NAMES, fontsize=SZ["tick"])
    ax.tick_params(axis="x", length=0, pad=2)
    fs.save(fig, str(HERE / "p4b_share"), width=CW)


def p4c_amdahl():
    """End-to-end speed-up against decode speed-up for a fixed 196 ms per request: the curve flattens toward 47x
    (= 1 / 2.1%); the measured methods sit on it."""
    f = ONE / (ONE + AR_DEC)
    fig = plt.figure(figsize=(CW, 2.6))
    ax = fig.add_axes([0.16, 0.16, 0.80, 0.80])
    s = np.logspace(0, 2, 200)
    ax.plot(s, 1 / (f + (1 - f) / s), color="#7E6AA6", lw=1.3, zorder=2)
    ax.plot(s, s, color="#D0D0D0", lw=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(1 / f, color="#B9A9D2", lw=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax.text(1.1, 1 / f * 1.08, f"bound: {1 / f:.0f}×", fontsize=SZ["note"], color="#7E6AA6", va="bottom")
    ax.text(40, 36, "no fixed cost", fontsize=SZ["note"], color="#9A9A9A", rotation=33, ha="center")
    for m in ("DFlash", "DDTree", "Ours"):
        sd = AR_DEC / DEC[m]
        se = (ONE + AR_DEC) / (ONE + DEC[m])
        ax.scatter([sd], [se], s=40 if m == "Ours" else 24, color=COLOR[m], edgecolor="white", linewidths=0.5, zorder=4)
        # labels stacked to the left in the points' own order (DFlash lowest), 11 pt apart, with short leaders
        dy = {"DFlash": -13, "DDTree": -1, "Ours": 11}[m]
        ax.annotate(f"{m}  {se:.1f}×", (sd, se), xytext=(-14, dy),
                    textcoords="offset points", ha="right", va="center", fontsize=SZ["note"],
                    color=COLOR["Ours"] if m == "Ours" else "#2E2E2E",
                    arrowprops=dict(arrowstyle="-", color="#BDBDBD", lw=0.5, shrinkA=1, shrinkB=3))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 100)
    ax.set_ylim(1, 100)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1×", "10×", "100×"])
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1×", "10×", "100×"])
    ax.set_xlabel("decoding speed-up", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("end-to-end speed-up", fontsize=SZ["label"], labelpad=2)
    fs.save(fig, str(HERE / "p4c_amdahl"), width=CW)


def p4d_stages():
    fig = plt.figure(figsize=(CW, 2.5))
    ax = fig.add_axes([0.17, 0.14, 0.81, 0.70])
    stages = [("preprocess", PRE, PRE), ("vision\nencoder", VIS, VIS), ("prefill", PREF, PREF),
              ("decode", AR_DEC, DEC["Ours"])]
    for i, (name, a, o) in enumerate(stages):
        ax.bar(i - 0.19, a, width=0.36, color="#B0B0B0", linewidth=0, zorder=2)
        ax.bar(i + 0.19, o, width=0.36, color=COLOR["Ours"], linewidth=0, zorder=2)
        ax.text(i - 0.19, a * 1.1, f"{a:.0f}", ha="center", va="bottom", fontsize=7.0)
        ax.text(i + 0.19, o * 1.1, f"{o:.0f}", ha="center", va="bottom", fontsize=7.0, color=COLOR["Ours"])
    ax.set_yscale("log")
    ax.set_ylim(10, 40000)
    ax.set_yticks([10, 100, 1000, 10000])
    ax.set_yticklabels(["10", "100", "1k", "10k"])          # mathtext 10^k shrinks the exponent below 7 pt
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("ms per image (log)", fontsize=SZ["label"], labelpad=2)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s for s, _, _ in stages], fontsize=SZ["tick"], linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=2)
    m2.legend_row(fig, [0.17, 0.88, 0.5, 0.09], [("#B0B0B0", "AR"), (COLOR["Ours"], "Ours")])
    fs.save(fig, str(HERE / "p4d_stages"), width=CW)


def main():
    for fn in (p4a_stack, p4b_share, p4c_amdahl, p4d_stages):
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
