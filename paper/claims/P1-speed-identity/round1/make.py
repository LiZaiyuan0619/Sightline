"""P1 (+ P2): speed-up = tokens per round x (AR time per token / time per round) -- round 1, night.

Claim: one identity explains the whole ranking. Every method's verify forward costs the same ~12 ms; what differs is
how many tokens a round yields (tokens per round) and what the draft costs. EAGLE-3's tokens per round is decent but its layer-serial draft takes
20.9 of its 35.9 ms per round; ours gets the longest tokens per round (19.0) for about DDTree's round cost (21.5 vs 20.2 ms).
Data: runs/2026-10-30/analysis/timing_split_all.md (OmniSVG-8B text-to-SVG, 40 MMSVG-Bench rows, greedy, one node,
AR 10.92 ms / token). Seven methods (SpS / ViSpec do not run on this block / were not timed).
Candidates: p1a_stack (ms per round, split), p1b_rays (tokens per round vs ms per round, iso-speed-up rays), p1c_pertoken
            (ms per accepted token, split, AR as reference), p1d_gaincost (log gain from tokens per round vs log cost of a round).
Run: python3 paper/claims/P1-speed-identity/round1/make.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("m2r2", ROOT / "paper/claims/M2-block-ceiling/round2/make.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)
fs, SZ, W, COLOR = m2.fs, m2.SZ, m2.W, m2.COLOR
CW = 3.25
METHODS = ["PLD", "REST", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]
PARTS = [("draft", "draft", "#B48EAD"), ("tree", "tree build", "#D9C27A"), ("verify", "verify forward", "#9AAABB"),
         ("other", "other", "#D9D9D9")]


def load():
    rows, ar = {}, None
    for line in (RUNS / "2026-10-30/analysis/timing_split_all.md").read_text().splitlines():
        if line.startswith("AR:"):
            ar = float(line.split()[1])
        if line.startswith("| ") and not line.startswith("| method") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows[c[0]] = dict(tau=float(c[1]), ms=float(c[2]), draft=float(c[3]), tree=float(c[4]), verify=float(c[5]),
                              other=float(c[6]), speedup=float(c[8].rstrip("x")))
    assert ar == 10.92 and set(rows) == set(METHODS), (ar, rows.keys())
    for m, r in rows.items():                       # the identity, checked on the printed numbers
        assert abs(r["tau"] * ar / r["ms"] - r["speedup"]) < 0.06, (m, r["tau"] * ar / r["ms"], r["speedup"])
    return rows, ar


def xaxis(ax):
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=SZ["tick"], rotation=35, ha="right", rotation_mode="anchor")
    for t, m in zip(ax.get_xticklabels(), METHODS):
        t.set_color(COLOR["Ours"] if m == "Ours" else "#2E2E2E")
        t.set_fontweight("bold" if m == "Ours" else "normal")
    ax.tick_params(axis="x", length=0, pad=2)


def p1a_stack(rows, ar):
    fig = plt.figure(figsize=(CW, 2.7))
    ax = fig.add_axes([0.15, 0.19, 0.83, 0.60])
    for i, m in enumerate(METHODS):
        r, b = rows[m], 0.0
        for k, _, col in PARTS:
            ax.bar(i, r[k], bottom=b, width=0.66, color=col, linewidth=0, zorder=2)
            b += r[k]
        ax.text(i, b + 0.6, f"τ {r['tau']:.1f}\n{r['speedup']:.1f}×", ha="center", va="bottom", fontsize=7.0,
                color=COLOR["Ours"] if m == "Ours" else "#2E2E2E", linespacing=1.0)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_ylim(0, 45)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_ylabel("ms per round", fontsize=SZ["label"], labelpad=2)
    xaxis(ax)
    m2.legend_row(fig, [0.15, 0.90, 0.84, 0.07], [(c, t) for _, t, c in PARTS])
    fs.save(fig, str(HERE / "p1a_stack"), width=CW)


def p1b_rays(rows, ar):
    """x = ms per round, y = tokens per round; along each grey ray the speed-up is constant (tokens per round * AR / ms)."""
    fig = plt.figure(figsize=(CW, 2.75))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
    xs = np.linspace(0, 40, 50)
    for s in (2, 4, 6, 8, 10):
        ax.plot(xs, s * xs / ar, color="#DADADA", lw=0.7, zorder=0)
        xe = min(38.5, 21.5 * ar / ar * (20.5 * ar / (s * ar)) if False else 38.5)
        ye = s * xe / ar
        if ye > 21:
            xe, ye = 21 * ar / s, 21
        ax.text(xe, ye, f"{s}×", fontsize=7.0, color="#9A9A9A", ha="left", va="bottom")
    for m in METHODS:
        r = rows[m]
        ours = m == "Ours"
        ax.scatter(r["ms"], r["tau"], s=46 if ours else 26, color=COLOR[m], edgecolor="white", linewidths=0.5,
                   zorder=4)
        dx, dy, ha = {"PLD": (-5, -5, "right"), "REST": (4, 4, "left"), "EAGLE-3": (-5, 0, "right"),
                      "DFlash": (-5, 2, "right"), "DARTree": (5, 2, "left"), "DDTree": (5, -6, "left"),
                      "Ours": (5, 0, "left")}[m]
        ax.annotate(m, (r["ms"], r["tau"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=SZ["legend"], color=COLOR["Ours"] if ours else "#2E2E2E",
                    fontweight="bold" if ours else "normal")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 21.5)
    ax.set_xlabel("ms per round", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens per round τ", fontsize=SZ["label"], labelpad=2)
    ax.text(0.98, 0.03, "grey rays: equal speed-up over AR", transform=ax.transAxes, fontsize=SZ["note"],
            color="#8A8A8A", ha="right", va="bottom",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    fs.save(fig, str(HERE / "p1b_rays"), width=CW)


def p1c_pertoken(rows, ar):
    """What one accepted token costs (ms per round / tokens per round), split the same way; AR's cost per token is the line."""
    fig = plt.figure(figsize=(CW, 2.7))
    ax = fig.add_axes([0.15, 0.19, 0.83, 0.60])
    for i, m in enumerate(METHODS):
        r, b = rows[m], 0.0
        for k, _, col in PARTS:
            v = r[k] / r["tau"]
            ax.bar(i, v, bottom=b, width=0.66, color=col, linewidth=0, zorder=2)
            b += v
        ax.text(i, b + 0.15, f"{b:.1f}", ha="center", va="bottom", fontsize=7.5,
                color=COLOR["Ours"] if m == "Ours" else "#2E2E2E")
    ax.axhline(ar, color=fs.REF, lw=0.8, linestyle=(0, (3, 2)), zorder=3)
    ax.text(len(METHODS) - 0.45, ar + 0.2, f"AR {ar:.1f} ms", fontsize=SZ["note"], color=fs.REF, ha="right",
            va="bottom")
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.set_ylim(0, 12.5)
    ax.set_yticks([0, 4, 8, 12])
    ax.set_ylabel("ms per accepted token", fontsize=SZ["label"], labelpad=2)
    xaxis(ax)
    m2.legend_row(fig, [0.15, 0.90, 0.84, 0.07], [(c, t) for _, t, c in PARTS])
    fs.save(fig, str(HERE / "p1c_pertoken"), width=CW)


def p1d_gaincost(rows, ar):
    """On a log axis the identity is a sum: up by tokens per round (tokens per round), down by the round's cost in AR steps;
    the dot is what is left -- the speed-up."""
    fig = plt.figure(figsize=(CW, 2.7))
    ax = fig.add_axes([0.15, 0.19, 0.83, 0.64])
    for i, m in enumerate(METHODS):
        r = rows[m]
        gain, cost = np.log2(r["tau"]), np.log2(r["ms"] / ar)
        ax.bar(i, gain, width=0.6, color=COLOR[m], alpha=0.9, linewidth=0, zorder=2)
        ax.bar(i, -cost, width=0.6, color="#C9C9C9", linewidth=0, zorder=2)
        ax.scatter([i], [gain - cost], s=34 if m == "Ours" else 22, color="#1A1A1A", zorder=4, marker="D")
    ax.axhline(0, color="#4A4A4A", lw=0.7)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ticks = [-2, -1, 0, 1, 2, 3, 4]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"÷{2 ** -t:g}" if t < 0 else f"×{2 ** t:g}" for t in ticks])
    ax.set_ylim(-2.2, 4.6)
    ax.set_ylabel("factor (log scale)", fontsize=SZ["label"], labelpad=2)
    xaxis(ax)
    m2.legend_row(fig, [0.15, 0.89, 0.84, 0.08], [("#7F9FBF", "tokens per round"), ("#C9C9C9", "round cost in AR steps"),
                                                 ("#1A1A1A", "speed-up (◆)")])
    fs.save(fig, str(HERE / "p1d_gaincost"), width=CW)


def main():
    rows, ar = load()
    print("identity tau * AR / ms reproduces every printed speed-up (checked in load)")
    for fn in (p1a_stack, p1b_rays, p1c_pertoken, p1d_gaincost):
        fn(rows, ar)
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
