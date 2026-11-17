"""P5 -- does the gain survive a production engine? round 1,.

Claim (as the data actually reads): in SGLang the head keeps its advantage over the whole latency-sensitive half
of the serving frontier -- at one request it decodes 9.3x faster than the original model and 1.29x faster than the
block-16 baseline, and it is the only arm that can serve a request under 1.2 s. As the batch grows the engine
becomes compute-bound and every speculative method converges towards the original model; there the SHORTER
verification window wins, and the SAME block-32 head run with a 16-token window follows the dedicated block-16
head to within 5-9% without any retraining, keeping 1.78x at 64 concurrent requests.

Known shortfall, drawn as it is: above 8 concurrent requests the dedicated block-16 head is 5-9% faster than our
head with its window turned down. Nothing is hidden; see README for how we propose to handle it.

Setup: SGLang 0.5.22, InternSVG-8B text-to-SVG, one H200, one node; 56 measured cells.
Data: runs/2026-11-04/figdata_b3/p5_sglang.json (runs/2026-10-29/analysis/b2_sweep.json).

Candidates, one chart family each (guide 3.2, 7.3):
  p5a_pareto    the serving frontier: system throughput against per-request latency, concurrency on every point
  p5b_curves    speed-up over the original model against concurrency
  p5c_slo       how much throughput each arm can serve under a per-request latency target
  p5d_accept    accepted tokens per verification step against concurrency -- why the curves fall
  p5e_window    one head, one knob: the two verification windows side by side with the baseline
Run: python3 paper/claims/P5-serving/round1/make.py
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
ROWS = json.loads((RUNS / "2026-11-04/figdata_b3/p5_sglang.json").read_text())["rows"]

COL_W, FULL_W = 3.25, 6.875
SZ = dict(tick=8.0, label=9.0, legend=8.5, note=8.5, lead=9.0, letter=12.0)
plt.rcParams.update({"xtick.labelsize": SZ["tick"], "ytick.labelsize": SZ["tick"]})

AR, B16, W32, W16 = "AR", "DFlash (block 16)", "Ours (window 32)", "Ours (window 16)"
ARMS = [AR, B16, W32, W16]
COLOR = {AR: "#8C7B63", B16: "#9FC8C8", W32: "#1D4E89", W16: "#6E9AC4"}
LABEL = {AR: "original model", B16: "DFlash, block 16", W32: "Ours, window 32", W16: "Ours, window 16"}
CONC = [1, 2, 4, 8, 16, 32, 64]
MODES = ["greedy", "sample"]
MODE_TXT = {"greedy": "greedy", "sample": "official sampling"}
G = {(r["mode"], r["arm"], r["conc"]): r for r in ROWS}


def legend(fig, rect, items, bold=()):
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
    lead = 0.020
    gap = (0.995 - sum(ws) - len(items) * lead) / max(1, len(items) - 1)
    assert gap > 0.004, ("legend does not fit", gap)
    x = 0.004
    for (c, s), txt, w in zip(items, texts, ws):
        ax.add_patch(Rectangle((x, 0.22), 0.015, 0.56, color=c, linewidth=0))
        txt.set_x(x + lead)
        x += lead + w + gap


def two_panels(height, left=0.088, right=0.996, bottom=0.190, top=0.825, gap=0.115):
    fig = plt.figure(figsize=(FULL_W, height))
    pw = (right - left - gap) / 2
    return fig, [fig.add_axes([left + i * (pw + gap), bottom, pw, top - bottom]) for i in range(2)]


# --------------------------------------------------------------------------------------------- p5a
def p5a_pareto():
    """The serving frontier. Up and to the right is better: more tokens a second at a lower per-request wait."""
    fig, axes = two_panels(2.75)
    for ax, mode in zip(axes, MODES):
        pts = [(G[(mode, a, c)]["tok_per_s"], G[(mode, a, c)]["latency_s"], a, c) for a in ARMS for c in CONC]
        front = [p for p in pts if not any(o[0] >= p[0] and o[1] <= p[1] and o[:2] != p[:2] for o in pts)]
        front.sort(key=lambda p: p[0])
        ax.plot([p[0] for p in front], [p[1] for p in front], "-", color="#C9C9C9", lw=3.2, zorder=1,
                solid_capstyle="round")
        for a in ARMS:
            xs = [G[(mode, a, c)]["tok_per_s"] for c in CONC]
            ys = [G[(mode, a, c)]["latency_s"] for c in CONC]
            ax.plot(xs, ys, "-o", color=COLOR[a], lw=1.5 if a == W32 else 1.1, ms=4.6 if a == W32 else 3.8,
                    mfc=COLOR[a], mec="white", mew=0.5, zorder=3)
        for c, dx, dy in ((1, 0, 10), (8, 0, -11), (64, 13, 0)):
            r = G[(mode, W32, c)]
            ax.annotate(f"{c}", (r["tok_per_s"], r["latency_s"]), textcoords="offset points", xytext=(dx, dy),
                        ha="center", va="center", fontsize=SZ["note"], color=COLOR[W32])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("tokens per second, whole engine", fontsize=SZ["label"], labelpad=1.5)
        ax.set_ylabel("wait per request (s)", fontsize=SZ["label"], labelpad=1.5)
        ax.yaxis.set_label_coords(-0.105, 0.5)
        ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
        ax.set_xticks([200, 1000, 5000, 20000], ["200", "1k", "5k", "20k"])
        ax.set_xlim(150, 30000)
        ax.set_title(MODE_TXT[mode], fontsize=SZ["legend"], pad=3.5, color="#2E2E2E")
        ax.text(0.03, 0.06, "concurrency on the blue points", transform=ax.transAxes, fontsize=SZ["note"],
                color=COLOR[W32])
    axes[0].text(0.03, 0.92, "the frontier of all four arms", transform=axes[0].transAxes,
                 fontsize=SZ["note"], color="#9A9A9A")
    legend(fig, [0.088, 0.925, 0.908, 0.06], [(COLOR[a], LABEL[a]) for a in ARMS], bold=(LABEL[W32],))
    fs.save(fig, str(HERE / "p5a_pareto"), width=FULL_W)


# --------------------------------------------------------------------------------------------- p5b
def p5b_curves():
    """Speed-up over the original model at the same concurrency."""
    fig, axes = two_panels(2.6)
    for ax, mode in zip(axes, MODES):
        ax.axhline(1, color=fs.REF, lw=0.7, ls=(0, (4, 3)), zorder=1)
        for a in ARMS[1:]:
            ys = [G[(mode, a, c)]["speedup"] for c in CONC]
            ax.plot(CONC, ys, "-o", color=COLOR[a], lw=1.6 if a == W32 else 1.1, ms=4.6 if a == W32 else 3.8,
                    mfc=COLOR[a], mec="white", mew=0.5, zorder=3,
                    ls="-" if a != W16 else (0, (3.5, 2)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONC, [str(c) for c in CONC])
        ax.set_xlim(0.85, 76)
        ax.set_ylim(0, 10.4)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel("concurrent requests", fontsize=SZ["label"], labelpad=1.5)
        ax.set_ylabel("speed-up over the\noriginal model", fontsize=SZ["label"], labelpad=1.5)
        ax.yaxis.set_label_coords(-0.105, 0.5)
        ax.set_title(MODE_TXT[mode], fontsize=SZ["legend"], pad=3.5, color="#2E2E2E")
    axes[0].text(1.0, 1.35, "no gain", ha="left", va="bottom", fontsize=SZ["note"], color=fs.REF)
    legend(fig, [0.088, 0.925, 0.908, 0.06], [(COLOR[a], LABEL[a]) for a in ARMS[1:]], bold=(LABEL[W32],))
    fs.save(fig, str(HERE / "p5b_curves"), width=FULL_W)


# --------------------------------------------------------------------------------------------- p5c
SLO = [1.2, 1.5, 2.0, 3.0, 5.0]


def p5c_slo():
    """How much an operator can serve under a per-request latency target."""
    H = 2.6
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0.195, 0.215, 0.785, 0.635])
    xs = np.arange(len(SLO))
    w = 0.21
    for k, a in enumerate(ARMS):
        v = []
        for s in SLO:
            ok = [G[("greedy", a, c)]["tok_per_s"] for c in CONC if G[("greedy", a, c)]["latency_s"] <= s]
            v.append(max(ok) / 1000 if ok else 0.0)
        ax.bar(xs + (k - 1.5) * w, v, w * 0.9, color=COLOR[a], linewidth=0, zorder=2)
        for x, y in zip(xs, v):
            if y == 0:
                ax.plot([x + (k - 1.5) * w], [0.45], "x", ms=4.0, mew=1.1, color=COLOR[a], clip_on=False)
    ax.set_ylim(0, 17.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.text(len(SLO) - 0.5, 16.6, "× = the target cannot be met at any concurrency", ha="right", va="top",
            fontsize=SZ["note"], color="#2E2E2E")
    ax.set_ylabel("thousand tokens a second\nunder the target", fontsize=SZ["label"], labelpad=1.5)
    ax.set_xticks(xs, [f"{s:g} s" for s in SLO], fontsize=SZ["tick"])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlabel("target wait per request", fontsize=SZ["label"], labelpad=1.5)
    legend(fig, [0.025, 0.905, 0.962, 0.06], [(COLOR[a], LABEL[a].split(",")[0]) for a in ARMS],
           bold=("Ours",))
    fs.save(fig, str(HERE / "p5c_slo"), width=COL_W)


# --------------------------------------------------------------------------------------------- p5d
def p5d_accept():
    """Why the curves fall: it is not the drafting. Accepted tokens per verification step barely move."""
    H = 2.5
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0.175, 0.215, 0.800, 0.635])
    for a in ARMS[1:]:
        for mode, ls in (("greedy", "-"), ("sample", (0, (3.5, 2)))):
            ys = [G[(mode, a, c)]["accept"] for c in CONC]
            ax.plot(CONC, ys, ls=ls, marker="o", color=COLOR[a], lw=1.5 if a == W32 else 1.0,
                    ms=4.2 if a == W32 else 3.4, mfc=COLOR[a], mec="white", mew=0.5, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONC, [str(c) for c in CONC])
    ax.set_xlim(0.85, 76)
    ax.set_ylim(4, 14.5)
    ax.set_xlabel("concurrent requests", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens accepted per\nverification step", fontsize=SZ["label"], labelpad=1.5)
    ax.text(70, 13.9, "solid greedy, dashed official sampling", ha="right", va="top", fontsize=SZ["note"],
            color="#2E2E2E")
    legend(fig, [0.025, 0.905, 0.962, 0.06], [(COLOR[a], LABEL[a]) for a in ARMS[1:]], bold=(LABEL[W32],))
    fs.save(fig, str(HERE / "p5d_accept"), width=COL_W)


# --------------------------------------------------------------------------------------------- p5e
def p5e_window():
    """One head, one knob: which verification window to use at which load, against the dedicated block-16 head."""
    fig, axes = two_panels(2.6)
    xs = np.arange(len(CONC))
    w = 0.26
    for ax, mode in zip(axes, MODES):
        for k, a in enumerate((B16, W32, W16)):
            v = [G[(mode, a, c)]["speedup"] for c in CONC]
            ax.bar(xs + (k - 1) * w, v, w * 0.9, color=COLOR[a], linewidth=0, zorder=2)
        best = [max(G[(mode, W32, c)]["speedup"], G[(mode, W16, c)]["speedup"]) for c in CONC]
        ref = [G[(mode, B16, c)]["speedup"] for c in CONC]
        for x, bst, rf in zip(xs, best, ref):
            ax.text(x, 9.90, f"{100 * (bst / rf - 1):+.0f}%", ha="center", va="bottom", fontsize=7.5,
                    color=COLOR[W32] if bst >= rf else "#9A5B12")
        ax.axhline(1, color=fs.REF, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(xs, [str(c) for c in CONC], fontsize=SZ["tick"])
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_ylim(0, 12.1)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_xlabel("concurrent requests", fontsize=SZ["label"], labelpad=1.5)
        ax.set_ylabel("speed-up over the\noriginal model", fontsize=SZ["label"], labelpad=1.5)
        ax.yaxis.set_label_coords(-0.105, 0.5)
        ax.set_title(MODE_TXT[mode], fontsize=SZ["legend"], pad=3.5, color="#2E2E2E")
    for ax in axes:
        ax.text(-0.45, 12.0, "our better window, against the block-16 head", ha="left", va="top",
                fontsize=SZ["note"], color="#2E2E2E")
    legend(fig, [0.088, 0.925, 0.908, 0.06], [(COLOR[a], LABEL[a]) for a in (B16, W32, W16)],
           bold=(LABEL[W32],))
    fs.save(fig, str(HERE / "p5e_window"), width=FULL_W)


if __name__ == "__main__":
    for mode in MODES:
        for a in ARMS[1:]:
            v = [G[(mode, a, c)]["speedup"] for c in CONC]
            print(f"{mode:7s} {a:20s} " + " ".join(f"{x:5.2f}" for x in v))
    for s in SLO:
        row = []
        for a in ARMS:
            ok = [G[("greedy", a, c)]["tok_per_s"] for c in CONC if G[("greedy", a, c)]["latency_s"] <= s]
            row.append(f"{a.split(' (')[0]} {max(ok) if ok else 0:6.0f}")
        print(f"  SLO {s:4.1f}s  " + " | ".join(row))
    p5a_pareto()
    p5b_curves()
    p5c_slo()
    p5d_accept()
    p5e_window()
