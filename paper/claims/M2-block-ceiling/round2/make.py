"""M2 (+ M3): the baselines' rounds are often cut off by their own window; ours rarely are -- round 2, evening.

Claim (loop-free content, i.e. before any loop onset --  decision): a round that fills the method's whole window
was cut off, not wrong. On loop-free content that happens in % of the rounds of DDTree / DARTree / EAGLE-3 and in
only 4-12% of ours (window 32), in all five model x task blocks and both decoding modes; the survival curves show the
cliff at each baseline's window (13, 16, 17) while ours keeps accepting up to 32.

Why round 2: round 1 was all-token (55 / 64 / 27%) and had 3 baselines; the caliber check moved
mechanism figures to the loop-free caliber, and the comparison set is table 1's seven methods (SpS / ViSpec have no
profile). Data: runs/2026-11-03/figdata_b1/m2_healthy.json (runs/2026-11-03/figdata_b1_m2.py; reproduces
runs/2026-11-02/caliber/healthy_vs_all.md cell by cell).

Candidates: m2a_bars (grouped bars), m2b_survival (survival curves, 2 x 5), m2c_rows (per-row strips), m2d_ridges
(histogram ridges of tokens per round).
Run: python3 paper/claims/M2-block-ceiling/round2/make.py
"""
import importlib.util
import json
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
spec = importlib.util.spec_from_file_location("t1r4", ROOT / "paper/claims/T1-main-table/round4/make.py")
t1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t1)
fs, SZ, W = t1.fs, t1.SZ, t1.W
COLOR = dict(t1.COLOR)
COLOR["Ours"] = COLOR[t1.OURS]
METHODS = ["PLD", "REST", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]      # the seven with a profile
BLOCKS = t1.BLOCKS
BLOCK_LABEL = t1.BLOCK_LABEL
MODES = ("greedy", "sample")
MODE_TXT = {"greedy": "greedy", "sample": "official sampling"}
DATA = json.loads((RUNS / "2026-11-03/figdata_b1/m2_healthy.json").read_text())["blocks"]
# anchors from runs/2026-11-02/caliber/healthy_vs_all.md (guide SS8: reproduce a printed number first)
assert round(DATA["OmniSVG-8B, text-to-SVG"]["greedy"]["DDTree"]["healthy"]["limit_share"], 2) == 0.37
assert round(DATA["OmniSVG-8B, text-to-SVG"]["greedy"]["Ours"]["healthy"]["tau"], 2) == 11.91


def legend_row(fig, rect, items):
    """items = [(colour, text)], spaced by measured text width."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    texts = [ax.text(0, 0.5, t, fontsize=SZ["legend"], va="center", ha="left",
                     color=COLOR["Ours"] if t == "Ours" else "#2E2E2E", fontweight="bold" if t == "Ours" else "normal")
             for _, t in items]
    rend = fig.canvas.get_renderer()
    aw = ax.get_window_extent(rend).width
    ws = [t.get_window_extent(rend).width / aw for t in texts]
    lead = 0.029
    gap = (0.995 - sum(ws) - len(items) * lead) / max(1, len(items) - 1)
    gap = min(gap, 0.05)
    x = 0.004
    for (c, _), t, w in zip(items, texts, ws):
        ax.add_patch(Rectangle((x, 0.22), 0.021, 0.56, color=c, linewidth=0))
        t.set_x(x + lead)
        x += lead + w + gap


def method_legend(fig, rect):
    legend_row(fig, rect, [(COLOR[m], m) for m in METHODS])


# --------------------------------------------------------------------------------------------------
def m2a_bars():
    """Share of loop-free rounds that fill the window, 5 block groups x 7 bars, greedy row and sampling row."""
    H = 3.55
    fig = plt.figure(figsize=(W, H))
    left, pw = 0.106, 0.890
    for r, mode in enumerate(MODES):
        ax = fig.add_axes([left, (H - 0.42 - (r + 1) * 1.18 - r * 0.20) / H, pw, 1.18 / H])
        X, Y = [], []
        for b, blk in enumerate(BLOCKS):
            for i, m in enumerate(METHODS):
                v = DATA[blk][mode][m]["healthy"]["limit_share"] * 100
                x = b * 8.2 + i
                ax.bar(x, v, width=0.84, color=COLOR[m], linewidth=0, zorder=2)
                X.append(x)
                Y.append(v)
                if m == "Ours":
                    ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=SZ["note"],
                            color=COLOR["Ours"])
                if m == "DDTree":
                    ax.text(x, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=SZ["note"], color="#3E3E3E")
        ax.set_xlim(-0.8, 4 * 8.2 + 6.8)
        ax.set_ylim(0, 56)
        fs.assert_inside(ax, X, Y)
        ax.set_yticks([0, 20, 40])
        ax.set_ylabel(f"rounds that fill\nthe window (%),\n{MODE_TXT[mode]}", fontsize=SZ["label"], labelpad=2,
                      linespacing=1.05)
        ax.set_xticks([])
        if r == 0:
            for b, blk in enumerate(BLOCKS):
                ax.text(b * 8.2 + 3, 58, BLOCK_LABEL[blk], ha="center", va="bottom", fontsize=SZ["header"],
                        linespacing=1.05, clip_on=False)
    method_legend(fig, [left, 0.03 / H, pw, 0.21 / H])
    fs.save(fig, str(HERE / "m2a_bars"), width=W)


# --------------------------------------------------------------------------------------------------
def m2b_survival():
    """P(a round emits >= k tokens) on loop-free content, one panel per block x mode, seven curves. Each baseline's
    curve stops at its own window; ours runs to 32."""
    H = 3.85
    fig = plt.figure(figsize=(W, H))
    left, right, gap = 0.088, 0.006, 0.012
    rects = t1.grid_rects(left, right, 0.40, H, gap, 1.12, 0.36)
    for r, mode in enumerate(MODES):
        for c, blk in enumerate(BLOCKS):
            ax = fig.add_axes(rects[mode][c])
            for m in METHODS:
                d = DATA[blk][mode][m]
                s = d["healthy"]["survival"]
                ks = np.array(sorted(int(k) for k in s))
                ys = np.array([s[str(k)] for k in ks])
                ours = m == "Ours"
                ax.plot(ks, ys, color=COLOR[m], lw=2.0 if ours else 1.15, zorder=5 if ours else 3,
                        drawstyle="steps-post")
                ax.plot([ks[-1], ks[-1]], [ys[-1], 0], color=COLOR[m], lw=2.0 if ours else 1.15, zorder=5 if ours else 3,
                        linestyle=(0, (2, 1.5)))
            ax.set_xlim(1, 33)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([1, 16, 32])
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(axis="both", pad=1.5)
            if c == 0:
                ax.set_ylabel(f"share of rounds\nemitting ≥ k,\n{MODE_TXT[mode]}", fontsize=SZ["label"], labelpad=2,
                              linespacing=1.05)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("tokens per round k", fontsize=SZ["label"], labelpad=1.5)
            if r == 0:
                ax.set_title(BLOCK_LABEL[blk], fontsize=SZ["header"], pad=3, linespacing=1.08)
    method_legend(fig, [left, 0.03 / H, 0.906, 0.21 / H])
    fs.save(fig, str(HERE / "m2b_survival"), width=W)


# --------------------------------------------------------------------------------------------------
def m2c_rows():
    """Every prompt as one dot: the share of its loop-free rounds that fill the window, per method and block (greedy
    row, sampling row). Dots are jittered strips, the black tick is the median."""
    rng = np.random.default_rng(0)
    H = 3.7
    fig = plt.figure(figsize=(W, H))
    left, pw = 0.106, 0.890
    for r, mode in enumerate(MODES):
        ax = fig.add_axes([left, (H - 0.42 - (r + 1) * 1.2 - r * 0.18) / H, pw, 1.2 / H])
        for b, blk in enumerate(BLOCKS):
            for i, m in enumerate(METHODS):
                v = np.asarray(DATA[blk][mode][m]["healthy"]["per_row_limit"]) * 100
                x = b * 8.2 + i
                ax.scatter(x + rng.uniform(-0.33, 0.33, v.size), v, s=1.3, color=COLOR[m], alpha=0.45,
                           linewidths=0, rasterized=True, zorder=2)
                ax.hlines(np.median(v), x - 0.42, x + 0.42, color="#1A1A1A", lw=1.4, zorder=4)
        ax.set_xlim(-0.8, 4 * 8.2 + 6.8)
        ax.set_ylim(-2, 102)
        ax.set_yticks([0, 50, 100])
        ax.set_ylabel(f"rounds that fill\nthe window (%),\n{MODE_TXT[mode]}", fontsize=SZ["label"], labelpad=2,
                      linespacing=1.05)
        ax.set_xticks([])
        if r == 0:
            for b, blk in enumerate(BLOCKS):
                ax.text(b * 8.2 + 3, 104, BLOCK_LABEL[blk], ha="center", va="bottom", fontsize=SZ["header"],
                        linespacing=1.05, clip_on=False)
    method_legend(fig, [left, 0.03 / H, pw, 0.21 / H])
    fs.save(fig, str(HERE / "m2c_rows"), width=W, dpi=600)


# --------------------------------------------------------------------------------------------------
def m2d_ridges():
    """Histogram of tokens emitted per round (loop-free), one ridge per method, one column per block (greedy). The
    baselines pile up at their window (the spike on the right end of each ridge); ours is spread out to 32."""
    H = 3.9
    fig = plt.figure(figsize=(W, H))
    left, right, gap = 0.075, 0.006, 0.012
    pw = (1 - left - right - 4 * gap) / 5
    for c, blk in enumerate(BLOCKS):
        ax = fig.add_axes([left + c * (pw + gap), 0.14, pw, (H - 0.42 - 0.14 * H) / H])
        for i, m in enumerate(METHODS[::-1]):          # ours on top
            h = DATA[blk]["greedy"][m]["healthy"]["hist"]
            ks = np.array(sorted(int(k) for k in h))
            ys = np.array([h[str(k)] for k in ks])
            base = i * 1.0
            scale = 0.92 / 0.50                              # 50% of rounds = one ridge height: every window spike fits
            yy = np.minimum(ys * scale, 0.92)
            ax.fill_between(ks, base, base + yy, step="mid", color=COLOR[m], alpha=0.85, linewidth=0, zorder=3)
            for k, y in zip(ks, ys):                          # only PLD / REST's one-token rounds exceed 50%
                if y > 0.50:
                    ax.text(k + 0.9, base + 0.62, f"{y:.0%}", fontsize=SZ["note"], color="#5E5E5E", ha="left",
                            va="center")
            ax.axhline(base, color="#D9D9D9", lw=0.5, zorder=1)
        ax.set_xlim(0.5, 32.5)
        ax.set_ylim(0, len(METHODS) + 0.05)
        ax.set_xticks([1, 16, 32])
        ax.set_yticks(np.arange(len(METHODS)) + 0.3)
        if c == 0:
            ax.set_yticklabels(METHODS[::-1], fontsize=SZ["tick"])
            for t, m in zip(ax.get_yticklabels(), METHODS[::-1]):
                t.set_color(COLOR["Ours"] if m == "Ours" else "#2E2E2E")
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_xlabel("tokens per round", fontsize=SZ["label"], labelpad=1.5)
        ax.set_title(BLOCK_LABEL[blk], fontsize=SZ["header"], pad=3, linespacing=1.08)
    fig.text(0.004, 0.012, "Greedy decoding, healthy content. Ridge height: share of rounds (one ridge = 50%).",
             fontsize=SZ["note"], color="#6A6A6A", ha="left", va="bottom")
    fs.save(fig, str(HERE / "m2d_ridges"), width=W)


def main():
    lo = min(DATA[b][m_]["Ours"]["healthy"]["limit_share"] for b in BLOCKS for m_ in MODES)
    hi = max(DATA[b][m_]["Ours"]["healthy"]["limit_share"] for b in BLOCKS for m_ in MODES)
    base = [DATA[b][m_][m]["healthy"]["limit_share"] for b in BLOCKS for m_ in MODES
            for m in ("DDTree", "DARTree", "EAGLE-3")]
    print(f"ours, healthy limit-hit: {lo:.0%}-{hi:.0%}; DDTree/DARTree/EAGLE-3: {min(base):.0%}-{max(base):.0%}")
    for fn in (m2a_bars, m2b_survival, m2c_rows, m2d_ridges):
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
