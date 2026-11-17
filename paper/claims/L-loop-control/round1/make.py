"""L -- loop control (component 2), round 1,.

Claim: on top of the lossless speed-up, the loop rule buys another 1.05-1.8x of measured wall clock, and at the
tier we ship (escape-then-stop) the paired quality change is inside its own confidence interval -- on image-to-SVG
it is a small improvement. The same rule is handed to the strongest baseline and to the original model: everyone
gets faster, and we stay first in all eight benchmark cells.

Calibers (do not mix): the rule ladder is 900 benchmark rows (300 for InternSVG image-to-SVG) with paired
bootstrap CIs; the fairness panel is the full official benchmark (8,014 prompts) with measured seconds.
Data: runs/2026-11-04/figdata_b3/l_rules.json (runs/2026-11-04/figdata_b3.py).

Candidates, one chart family each (guide 3.2, 7.3):
  l1_tradeoff   forward passes saved against the paired quality change, with 95% CIs -- which tier is free
  l2_ladder     the rules as a ladder of measured wall clock, one line per target and decoding mode
  l3_fair       the same rule on the strongest baseline and on us, eight benchmark cells
  l4_audit      does the rule cut the right outputs: firings split into true loops and premature stops
  l5_arrows     where the rule moves each system in the (seconds, official quality) plane
Run: python3 paper/claims/L-loop-control/round1/make.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import figstyle as fs  # noqa: E402
from sightline.paths import RUNS

fs.apply_style()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
D = json.loads((RUNS / "2026-11-04/figdata_b3/l_rules.json").read_text())

COL_W, FULL_W = 3.25, 6.875
SZ = dict(tick=8.0, label=9.0, legend=8.5, note=8.5, lead=9.0, letter=12.0)
plt.rcParams.update({"xtick.labelsize": SZ["tick"], "ytick.labelsize": SZ["tick"]})

C_OURS, C_DD, C_AR = "#1D4E89", "#2F6D6D", "#8C7B63"
RULES = ["lossless", "rollback", "escape-then-stop", "stop"]
RULE_C = {"lossless": "#9FB6CC", "rollback": "#E0B86A", "escape-then-stop": "#C98A1E", "stop": "#9A5B12"}
SHIP = "escape-then-stop"                                 # the tier the paper ships
LADDER = D["ladder"]
TARGETS = ["OmniSVG-8B, text-to-SVG", "OmniSVG-8B, image-to-SVG", "InternSVG-8B, text-to-SVG",
           "InternSVG-8B, image-to-SVG"]
SHORT = {t: t.replace("-8B, ", "\n").replace("-to-SVG", "") for t in TARGETS}
MODE_TXT = {"greedy": "greedy", "sample": "official sampling"}
METRIC = {"clip": "CLIP score", "dino": "DINO similarity"}
LAD = {(r["target"], r["mode"], r["rule"]): r for r in LADDER}
SAR = {(r["block"], r["subset"], r["method"]): r for r in D["sarena"]}
BLOCKS = ["OmniSVG-8B, text-to-SVG", "InternSVG-8B, text-to-SVG",
          "OmniSVG-8B, image-to-SVG", "InternSVG-8B, image-to-SVG"]
SUBSETS = ["icon", "illustration"]


def legend(fig, rect, items, bold=("Ours",)):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    texts = [ax.text(0, 0.5, s, fontsize=SZ["legend"], va="center", ha="left",
                     color=c if s in bold else "#2E2E2E", fontweight="bold" if s in bold else "normal")
             for c, s in items]
    rend = fig.canvas.get_renderer()
    aw = ax.get_window_extent(rend).width
    ws = [x.get_window_extent(rend).width / aw for x in texts]
    lead = 0.021
    gap = (0.995 - sum(ws) - len(items) * lead) / max(1, len(items) - 1)
    assert gap > 0.004, ("legend does not fit", gap)
    x = 0.004
    for (c, s), txt, w in zip(items, texts, ws):
        ax.add_patch(Rectangle((x, 0.22), 0.016, 0.56, color=c, linewidth=0))
        txt.set_x(x + lead)
        x += lead + w + gap


# --------------------------------------------------------------------------------------------- l1
def l1_tradeoff():
    """Is the rule free? Forward passes saved against the paired quality change, with its 95% CI."""
    H = 2.75
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0.205, 0.175, 0.760, 0.675])
    ax.axhline(0, color=fs.REF, lw=0.7, zorder=1)
    xs, ys = [], []
    for r in LADDER:
        if r["rule"] == "lossless":
            continue
        # CLIP (~0.27) and DINO (~0.91) live on different scales: express both as % of the lossless run
        ref = LAD[(r["target"], r["mode"], "lossless")]["quality"][r["primary"]][0]
        d, lo, hi = [100 * v / ref for v in r["quality"][r["primary"]]]
        x = r["nfe_vs_lossless"]
        c = RULE_C[r["rule"]]
        ax.plot([x, x], [lo, hi], "-", color=c, lw=1.0, alpha=0.75, zorder=2)
        ax.plot([x], [d], "o" if r["mode"] == "greedy" else "^", ms=5.0, mfc=c, mec="white", mew=0.5, zorder=3)
        xs += [x, x]
        ys += [lo, hi]
    ax.set_xlim(0.97, 1.40)
    ax.set_ylim(-1.15, 1.15)
    fs.assert_inside(ax, xs, ys)
    ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4], ["1.0", "1.1", "1.2", "1.3", "1.4"])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0], ["−1.0", "−0.5", "0", "+0.5", "+1.0"])
    ax.set_xlabel("forward passes saved, × over the lossless run", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("paired quality change, %\nof the lossless run (95% CI)", fontsize=SZ["label"], labelpad=1.5)
    ax.text(1.378, 0.04, "no measurable change", ha="right", va="bottom", fontsize=SZ["note"], color=fs.REF)
    ax.text(1.378, -1.08, "● greedy    ▲ official sampling", ha="right", va="bottom", fontsize=SZ["note"],
            color="#2E2E2E")
    legend(fig, [0.03, 0.905, 0.955, 0.06], [(RULE_C[r], r) for r in RULES[1:]], bold=())
    fs.save(fig, str(HERE / "l1_tradeoff"), width=COL_W)


# --------------------------------------------------------------------------------------------- l2
def l2_ladder():
    """The ladder in measured wall clock: every target climbs when the rule is switched on."""
    H = 2.75
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0.185, 0.215, 0.545, 0.645])
    xs = np.arange(4)
    lines = []
    for r in D["new_targets"]:
        key = ("InternSVG-8B, image-to-SVG" if r["target"] == "i2s" else "StarVector-8B, image-to-SVG",
               r["mode"])
        lines.append((key, r["rule"], r["wall_vs_ar"]))
    series = {}
    for key, rule, v in lines:
        series.setdefault(key, {})[rule] = v
    cmap = {"InternSVG-8B, image-to-SVG": C_OURS, "StarVector-8B, image-to-SVG": "#5E87B5"}
    for (tgt, mode), by in series.items():
        ys = [by[r] for r in RULES]
        ax.plot(xs, ys, marker="o", color=cmap[tgt], lw=1.5 if mode == "greedy" else 1.0, ms=4.4,
                ls="-" if mode == "greedy" else (0, (3, 2)), mfc=cmap[tgt], mec="white", mew=0.5, zorder=3)
        ax.annotate(f"{tgt.split('-8B')[0]}\n{MODE_TXT[mode]}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(5, 0), ha="left", va="center", fontsize=SZ["note"], color=cmap[tgt],
                    linespacing=1.1)
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(4.5, 12.2)
    ax.set_xticks(xs, ["lossless", "rollback", "escape\nthen stop", "stop"], fontsize=SZ["tick"],
                  linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("wall clock, × over the original model", fontsize=SZ["label"], labelpad=1.5)
    fs.save(fig, str(HERE / "l2_ladder"), width=COL_W)


# --------------------------------------------------------------------------------------------- l3
def l3_fair():
    """The same rule on the strongest baseline and on us: measured seconds on the full official benchmark."""
    H = 2.9
    fig = plt.figure(figsize=(FULL_W, H))
    ax = fig.add_axes([0.085, 0.235, 0.905, 0.620])
    cells = [(b, s) for b in BLOCKS for s in SUBSETS]
    xs = np.arange(len(cells))
    w = 0.37
    for k, (m, mf, c) in enumerate((("DDTree", "DDTree (fast)", C_DD), ("Ours", "Ours (fast)", C_OURS))):
        base = [SAR[(b, s, "AR")]["seconds"] / SAR[(b, s, m)]["seconds"] for b, s in cells]
        full = [SAR[(b, s, "AR")]["seconds"] / SAR[(b, s, mf)]["seconds"] for b, s in cells]
        off = (k - 0.5) * w
        ax.bar(xs + off, base, w * 0.92, color=c, linewidth=0, zorder=2)
        ax.bar(xs + off, np.array(full) - np.array(base), w * 0.92, bottom=base, color=c, alpha=0.42,
               linewidth=0, zorder=2)
        for x, v in zip(xs, full):
            ax.text(x + off, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=SZ["note"], color=c)
    for x, (b, s) in zip(xs, cells):
        lead = SAR[(b, s, "DDTree (fast)")]["seconds"] / SAR[(b, s, "Ours (fast)")]["seconds"]
        ax.text(x, 12.3, f"×{lead:.2f}", ha="center", va="bottom", fontsize=SZ["lead"], color=C_OURS)
    ax.text(-0.45, 13.8, "numbers above the bars: our lead over DDTree once both carry the rule", ha="left",
            va="bottom", fontsize=SZ["note"], color=C_OURS)
    ax.set_ylim(0, 15.2)
    ax.set_yticks([0, 4, 8, 12])
    ax.set_ylabel("wall clock, × over\nthe original model", fontsize=SZ["label"], labelpad=1.5)
    fs.two_row_xaxis(ax, xs, [s if s == "icon" else "illus." for _, s in cells],
                     [(b.replace(", ", " ").replace("-to-SVG", ""),
                       [i for i, (bb, _) in enumerate(cells) if bb == b]) for b in BLOCKS],
                     line_down_pt=12.0, name_down_pt=13.0, fs_group=SZ["tick"])
    legend(fig, [0.085, 0.915, 0.905, 0.06],
           [(C_DD, "DDTree, lossless"), ("#A9C4C4", "DDTree + the same loop rule"),
            (C_OURS, "Ours, lossless"), ("#A8BDD6", "Ours + the loop rule")], bold=())
    fs.save(fig, str(HERE / "l3_fair"), width=FULL_W)


# --------------------------------------------------------------------------------------------- l4
def l4_audit():
    """Exact replay on the model's own outputs: of the outputs the rule stops, how many really were in a loop."""
    H = 2.6
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0.270, 0.300, 0.560, 0.560])
    rows = D["audit"]
    ys = np.arange(len(rows))[::-1]
    for y, a in zip(ys, rows):
        tot = a["outputs"]
        tru, pre = 100 * a["true_triggers"] / tot, 100 * a["premature"] / tot
        miss = 100 * a["misses"] / tot
        ax.barh(y, tru, 0.52, color="#C98A1E", linewidth=0, zorder=2)
        ax.barh(y, pre, 0.52, left=tru, color="#9A5B12", linewidth=0, zorder=2)
        ax.barh(y, miss, 0.52, left=tru + pre, color="#DDDDDD", linewidth=0, zorder=2)
        ax.text(tru + pre + miss + 1.2, y, f"{a['true_triggers'] / a['fires']:.0%}", ha="left", va="center",
                fontsize=SZ["note"], color="#2E2E2E")
    ax.set_yticks(ys, [f"{a['model'].split('-8B')[0]}\n{a['subset']}" for a in rows], fontsize=SZ["tick"],
                  linespacing=1.1)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 72)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.set_xlabel("outputs (%)", fontsize=SZ["label"], labelpad=1.5)
    ax.text(72, len(rows) - 0.20, "really a loop", ha="right", va="bottom",
            fontsize=SZ["note"], color="#2E2E2E")
    legend(fig, [0.025, 0.905, 0.962, 0.06],
           [("#C98A1E", "really in a loop"), ("#9A5B12", "stopped early"), ("#DDDDDD", "loop missed")],
           bold=())
    fs.save(fig, str(HERE / "l4_audit"), width=COL_W)


# --------------------------------------------------------------------------------------------- l5
def l5_arrows():
    """What the rule does on the official benchmark: faster to the right, quality as % of the original model."""
    H = 2.35
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0.205, 0.205, 0.770, 0.675])
    ax.axhline(0, color=fs.REF, lw=0.7, zorder=1)
    xs, ys = [], []
    for b in BLOCKS:
        key = "CLIP-T2I" if "text" in b else "DINO-Score"
        n = {s_: SAR[(b, s_, "AR")]["n"] for s_ in SUBSETS}
        tot = sum(n.values())

        def agg(method, field):
            if field == "sec":
                return sum(n[s_] * SAR[(b, s_, method)]["seconds"] for s_ in SUBSETS) / tot
            return sum(n[s_] * SAR[(b, s_, method)]["metrics"][key] for s_ in SUBSETS) / tot

        ar_q, ar_s = agg("AR", key), agg("AR", "sec")
        for m, mf, c in (("DDTree", "DDTree (fast)", C_DD), ("Ours", "Ours (fast)", C_OURS)):
            x0, y0 = ar_s / agg(m, "sec"), 100 * (agg(m, key) / ar_q - 1)
            x1, y1 = ar_s / agg(mf, "sec"), 100 * (agg(mf, key) / ar_q - 1)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5, shrinkA=3.0, shrinkB=1.0,
                                        mutation_scale=10))
            ax.plot([x0], [y0], "o", ms=4.6, mfc="white", mec=c, mew=1.2, zorder=4)
            xs += [x0, x1]
            ys += [y0, y1]
        ax.annotate(b.split("-8B, ")[0] + " " + ("text" if "text" in b else "image"),
                    (ar_s / agg("Ours (fast)", "sec"), 100 * (agg("Ours (fast)", key) / ar_q - 1)),
                    textcoords="offset points", xytext=(5, 0), ha="left", va="center",
                    fontsize=SZ["note"], color=C_OURS)
    ax.set_xlim(4.6, 15.6)
    ax.set_ylim(-0.24, 0.24)
    fs.assert_inside(ax, xs, ys)
    ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2], ["−0.2", "−0.1", "0", "+0.1", "+0.2"])
    ax.set_xticks([5, 7, 9, 11], ["5", "7", "9", "11"])
    ax.set_xlabel("wall clock, × over the original model", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("official quality, % of\nthe original model", fontsize=SZ["label"], labelpad=1.5)
    ax.text(4.75, 0.012, "same as the original model", ha="left", va="bottom", fontsize=SZ["note"],
            color=fs.REF)
    legend(fig, [0.03, 0.905, 0.955, 0.06],
           [(C_DD, "DDTree → + the rule"), (C_OURS, "Ours → + the rule")], bold=())
    fs.save(fig, str(HERE / "l5_arrows"), width=COL_W)


if __name__ == "__main__":
    ship = [r for r in LADDER if r["rule"] == SHIP]
    print(f"{SHIP}: forward savings ×{min(r['nfe_vs_lossless'] for r in ship):.2f}-"
          f"×{max(r['nfe_vs_lossless'] for r in ship):.2f}; CI covers 0 in "
          f"{sum(r['quality'][r['primary']][1] <= 0 <= r['quality'][r['primary']][2] for r in ship)}/{len(ship)}")
    for b in BLOCKS:
        for s in SUBSETS:
            print(f"  {b[:24]:26s} {s:12s} ours fast ×{SAR[(b, s, 'AR')]['seconds'] / SAR[(b, s, 'Ours (fast)')]['seconds']:.2f}"
                  f"  lead over DDTree fast ×{SAR[(b, s, 'DDTree (fast)')]['seconds'] / SAR[(b, s, 'Ours (fast)')]['seconds']:.2f}")
    l1_tradeoff()
    l2_ladder()
    l3_fair()
    l4_audit()
    l5_arrows()
