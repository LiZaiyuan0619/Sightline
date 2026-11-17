"""M1 (+ M4): SVG output is long runs of near-certain structure with uncertain numbers in between -- round 1,.

Claim: a draft gets the structure of an SVG (path commands, markup) right ~% of the time; what it misses are the
numbers (coordinates; curve control points above all), which take % of all rejections while being ~60% of the
tokens. The target model is near-certain on most positions (entropy < 0.01 nats at ~70% of them), and acceptance
follows the target's own entropy. Hence a long speculation window pays off on SVG.
Caliber: loop-free positions (before any loop onset), as computed by analysis/profiles_all.py (caliber).

Data: runs/2026-10-30/analysis/profile_{t2,4b,isvg,i2,i2s}.json -> [mode][method]["classes"]
      $SIGHTLINE_RUNS/r1029/entropy/{t2-ours,t2-ddtree,t2-dflash,i2-ours,isvg-ours}.json (target entropy; 3 blocks only)
      one real row for the strip: $SIGHTLINE_RUNS/tableA/t2-own/ours/D1-top16-N127/shard-0-of-2, gidx 0 (greedy, no loop)
Candidates: m1a_acc (dot plot per class, 7 methods x 5 blocks), m1b_shares (where tokens are vs where rejections are),
            m1c_entropy (entropy spectrum + acceptance by entropy), m1d_strip (one real SVG, token by token).
Run: python3 paper/claims/M1-token-classes/round1/make.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("m2r2", ROOT / "paper/claims/M2-block-ceiling/round2/make.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)
fs, SZ, W, COLOR, METHODS, BLOCKS, BLOCK_LABEL = m2.fs, m2.SZ, m2.W, m2.COLOR, m2.METHODS, m2.BLOCKS, m2.BLOCK_LABEL
sys.path.insert(0, str(ROOT / "analysis"))
from analysis import acceptance_profiles_all as P  # noqa: E402
from sightline.paths import RUNS

PROF = {blk: json.loads((ROOT / f"runs/2026-10-30/analysis/profile_{k}.json").read_text())
        for blk, k in zip(BLOCKS, ("t2", "4b", "isvg", "i2", "i2s"))}
# OmniSVG-4B: the profile read 4B outputs with 8B token ids (96% "other"); use the 4B-vocabulary recompute
_c4 = json.loads((RUNS / "2026-11-03/figdata_b1/m1_classes_4b.json").read_text())
for _mode in ("greedy", "sample"):
    for _m in METHODS:
        PROF[BLOCKS[1]][_mode][_m]["classes"] = _c4[_mode][_m]
ENT = (RUNS / "r1029/entropy")
OMNI = ["cmd", "coord:M", "coord:L", "coord:C-ctrl", "coord:C-end", "coord:Z", "arc/A", "color"]
ISVG = ["path command", "markup", "space-punct", "number", "color"]
NAME = {"cmd": "command", "coord:M": "move-to", "coord:L": "line-to", "coord:C-ctrl": "curve control",
        "coord:C-end": "curve end", "coord:Z": "close", "arc/A": "arc", "color": "color",
        "path command": "command", "markup": "markup", "space-punct": "punctuation", "number": "number"}
GROUP = {"cmd": "structure", "eos": "structure", "other": "structure", "coord:M": "numbers", "coord:L": "numbers",
         "coord:C-ctrl": "numbers", "coord:C-end": "numbers", "coord:Z": "numbers", "arc/A": "numbers",
         "color": "colour", "path command": "structure", "markup": "structure", "space-punct": "structure",
         "other text": "structure", "number": "numbers"}
GCOL = {"structure": "#9AAABB", "numbers": "#E2B86E", "colour": "#B78DAE"}
GTXT = {"structure": "structure (commands, markup)", "numbers": "numbers (coordinates)", "colour": "colours"}
# anchor (guide SS8): the profile printed ours' command accuracy on OmniSVG-8B text, greedy as 0.988
assert round(PROF[BLOCKS[0]]["greedy"]["Ours"]["classes"]["cmd"]["acc"], 3) == 0.988


def classes_of(blk):
    return OMNI if "OmniSVG" in blk else ISVG


# --------------------------------------------------------------------------------------------------
def m1a_acc():
    """P(the draft's token is right | the round got that far), per token class, every method, every block (greedy)."""
    H = 3.3
    fig = plt.figure(figsize=(W, H))
    left, right, gap = 0.092, 0.004, 0.014
    widths = np.array([len(classes_of(b)) for b in BLOCKS], float) + 0.6
    pw_tot = 1 - left - right - 4 * gap
    x0 = left
    for c, blk in enumerate(BLOCKS):
        pw = pw_tot * widths[c] / widths.sum()
        ax = fig.add_axes([x0, 0.33, pw, (H - 0.4 - 0.33 * H) / H])
        x0 += pw + gap
        cls = classes_of(blk)
        for i, m in enumerate(METHODS):
            d = PROF[blk]["greedy"][m]["classes"]
            ys = [d[k]["acc"] if k in d and d[k]["acc"] is not None else np.nan for k in cls]
            xs = np.arange(len(cls)) + (i - 3) * 0.075
            ours = m == "Ours"
            ax.scatter(xs, ys, s=30 if ours else 16, color=COLOR[m], edgecolor="white", linewidths=0.4,
                       zorder=6 if ours else 3)
        ax.set_xlim(-0.6, len(cls) - 0.4)
        ax.set_ylim(0.25, 1.02)
        ax.set_xticks(range(len(cls)))
        ax.set_xticklabels([NAME[k] for k in cls], rotation=40, ha="right", rotation_mode="anchor",
                           fontsize=SZ["tick"])
        for t, k in zip(ax.get_xticklabels(), cls):
            t.set_color("#2E2E2E" if GROUP[k] == "structure" else "#8A5A00")
        ax.tick_params(axis="x", pad=1.5, length=2)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#EDEDED", lw=0.5, zorder=0)
        if c == 0:
            ax.set_ylabel("draft token right\n(given the round got there)", fontsize=SZ["label"], labelpad=2,
                          linespacing=1.05)
        else:
            ax.set_yticklabels([])
        ax.set_title(BLOCK_LABEL[blk], fontsize=SZ["header"], pad=3, linespacing=1.08)
    m2.method_legend(fig, [left, 0.02 / H, 0.90, 0.2 / H])
    fs.save(fig, str(HERE / "m1a_acc"), width=W)


# --------------------------------------------------------------------------------------------------
def group_shares(blk, mode, m, key):
    d = PROF[blk][mode][m]["classes"]
    out = {"structure": 0.0, "numbers": 0.0, "colour": 0.0}
    for k, v in d.items():
        out[GROUP[k]] += v[key]
    return out


def m1b_shares():
    """Where the tokens are and where the rejections are, three groups. Per block: tokens, then rejections of DDTree and
    of ours (greedy). Numbers are ~60% of the tokens and % of the rejections; structure the reverse."""
    H = 2.95
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.075, 0.25, 0.921, (H - 0.45 - 0.25 * H) / H])
    labels, x = [], 0
    for b, blk in enumerate(BLOCKS):
        bars = [("tokens", group_shares(blk, "greedy", "Ours", "share")),
                ("DDTree", group_shares(blk, "greedy", "DDTree", "rej_share")),
                ("ours", group_shares(blk, "greedy", "Ours", "rej_share"))]
        for j, (lab, g) in enumerate(bars):
            bottom = 0.0
            for grp in ("structure", "numbers", "colour"):
                v = g[grp] * 100
                ax.bar(x, v, bottom=bottom, width=0.78, color=GCOL[grp], linewidth=0, zorder=2)
                if grp == "numbers":
                    ax.text(x, bottom + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=SZ["note"],
                            color="#3A2A00")
                bottom += v
            labels.append((x, lab))
            x += 1
        ax.text(x - 2, 104, BLOCK_LABEL[blk], ha="center", va="bottom", fontsize=SZ["header"], linespacing=1.05)
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.plot([x - 2 - 0.42, x - 1 + 0.42], [-0.12, -0.12], transform=trans, color=fs.GROUP_LINE, lw=0.7,
                clip_on=False)
        ax.text(x - 1.5, -0.14, "rejections", transform=trans, ha="center", va="top", fontsize=SZ["tick"],
                color="#2E2E2E")
        x += 0.8
    ax.set_xlim(-0.6, x - 0.6)
    ax.set_ylim(0, 101)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("share (%)", fontsize=SZ["label"], labelpad=2)
    ax.set_xticks([p for p, _ in labels])
    ax.set_xticklabels([l for _, l in labels], fontsize=SZ["tick"])
    ax.tick_params(axis="x", length=0, pad=2)
    m2.legend_row(fig, [0.075, 0.02 / H, 0.75, 0.2 / H], [(GCOL[g], GTXT[g]) for g in GCOL])
    fs.save(fig, str(HERE / "m1b_shares"), width=W)


# --------------------------------------------------------------------------------------------------
def m1c_entropy():
    """(left) how certain the target is at the positions it generates: share of positions per entropy bin;
    (right) acceptance rate per entropy bin. Only three blocks have the target's entropies (GPU run)."""
    files = {"OmniSVG-8B text": ("t2-ours", "t2-ddtree", "t2-dflash"), "OmniSVG-8B image": ("i2-ours",),
             "InternSVG-8B text": ("isvg-ours",)}
    lines = {"OmniSVG-8B text": "-", "OmniSVG-8B image": (0, (4, 2)), "InternSVG-8B text": (0, (1.2, 1.4))}
    H = 2.75
    fig = plt.figure(figsize=(W, H))
    a = fig.add_axes([0.075, 0.30, 0.33, 0.62])
    b = fig.add_axes([0.49, 0.30, 0.25, 0.62])
    bins = None
    for bi, (blk, fl) in enumerate(files.items()):
        for f in fl:
            meth = {"ours": "Ours", "ddtree": "DDTree", "dflash": "DFlash"}[f.split("-")[1]]
            d = json.loads((ENT / f"{f}.json").read_text())
            bk = d[f.split("-")[1]]["buckets"]
            bins = [f"{x['lo']:g}–{x['hi']:g}" if x["hi"] < 1e8 else f">{x['lo']:g}" for x in bk]
            n = np.array([x["n"] for x in bk], float)
            acc = np.array([x["accept"] for x in bk])
            if meth == "Ours":
                a.bar(np.arange(len(bk)) + (bi - 1) * 0.27, n / n.sum() * 100, width=0.26,
                      color=["#4F6D8F", "#8FA6BE", "#C9D3DE"][bi], linewidth=0, zorder=2, label=blk)
            b.plot(np.arange(len(bk)), acc, color=COLOR[meth], lw=1.8 if meth == "Ours" else 1.2,
                   linestyle=lines[blk], marker="o", ms=3.5 if meth == "Ours" else 2.5, zorder=5 if meth == "Ours" else 3,
                   label=f"{meth}, {blk}")
    for ax in (a, b):
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=35, ha="right", rotation_mode="anchor", fontsize=SZ["tick"])
        ax.set_xlabel("target entropy at the position (nats)", fontsize=SZ["label"], labelpad=1.5)
        ax.tick_params(axis="both", pad=1.5)
    a.set_ylabel("positions (%)", fontsize=SZ["label"], labelpad=2)
    a.set_ylim(0, 80)
    a.legend(fontsize=SZ["legend"], frameon=False, loc="upper right", handlelength=1.0)
    b.set_ylabel("draft accepted", fontsize=SZ["label"], labelpad=2)
    ys_all = np.concatenate([ln.get_ydata() for ln in b.get_lines()])
    b.set_ylim(np.floor(ys_all.min() * 20) / 20 - 0.01, 1.0)
    fs.assert_inside(b, np.concatenate([ln.get_xdata() for ln in b.get_lines()]), ys_all)
    b.legend(fontsize=SZ["note"], frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=2.0,
             ncol=1, labelspacing=0.3)
    for ax, s in ((a, "a"), (b, "b")):
        bb = ax.get_position()
        fig.text(bb.x0 - 0.068, bb.y1 + 0.02, s, fontsize=12, fontweight="bold", ha="left", va="top")
    fs.save(fig, str(HERE / "m1c_entropy"), width=W)


# --------------------------------------------------------------------------------------------------
def m1d_strip():
    """One real SVG (OmniSVG-8B text-to-SVG, greedy, ours, no loop), token by token: colour = token group; a gap =
    a new round; a black bar under a token = the round stopped there because the draft was wrong at that token."""
    import glob
    f = sorted(glob.glob(f"{RUNS}/tableA/t2-own/ours/D1-top16-N127/"
                         "shard-0-of-2/results.jsonl"))[0]
    r = next(json.loads(l) for l in open(f) if json.loads(l)["gidx"] == 0 and json.loads(l)["mode"] == "greedy")
    toks, acc, cap = r["spec_tokens"], r["acceptance"], 32
    assert r["onset"] is None and len(toks) == 288 and acc[:5] == [8, 29, 20, 8, 32], (len(toks), acc[:5])
    lab = P.token_classes(toks)
    per_line, cell, gapw = 50, 1.0, 0.45
    n_lines = int(np.ceil(len(toks) / per_line))
    H = 0.30 + n_lines * 0.21 + 0.30
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.01, 0.30 / H, 0.98, (n_lines * 0.21) / H])
    ax.axis("off")
    # round of every position, and where each round ends
    rnd, ends, full = np.zeros(len(toks), int), set(), set()
    s = 1                                          # position 0 comes from the prefill
    for j, e in enumerate(acc):
        rnd[s:s + e] = j
        if e >= cap:
            full.add(j)
        else:
            ends.add(min(s + e - 1, len(toks) - 1))
        s += e
    x = y = 0.0
    xmax = 0.0
    for i, t in enumerate(toks):
        if i and (i % per_line == 0):
            x, y = 0.0, y - 1.0
        elif i and rnd[i] != rnd[i - 1]:
            x += gapw
        g = GROUP.get(lab[i], "structure")
        ax.add_patch(Rectangle((x, y), cell * 0.9, 0.62, facecolor=GCOL[g], edgecolor="none"))
        if i in ends:
            ax.add_patch(Rectangle((x, y - 0.17), cell * 0.9, 0.12, facecolor="#1A1A1A", edgecolor="none"))
        x += cell
        xmax = max(xmax, x)
    ax.set_xlim(-0.3, xmax + 0.3)
    ax.set_ylim(y - 0.35, 0.8)
    stops = [lab[i] for i in sorted(ends)]
    share_num = sum(GROUP.get(l) == "numbers" for l in stops) / len(stops)
    print(f"m1d: {len(acc)} rounds, {len(ends)} stopped by a wrong draft token, {len(full)} filled the window; "
          f"{share_num:.0%} of the stops on a number")
    items = [(GCOL[g], GTXT[g]) for g in GCOL] + [("#1A1A1A", "round stopped here (draft wrong)")]
    m2.legend_row(fig, [0.01, 0.03 / H, 0.98, 0.2 / H], items)
    fs.save(fig, str(HERE / "m1d_strip"), width=W)


def main():
    for blk in BLOCKS:
        g = group_shares(blk, "greedy", "Ours", "share")
        rj = group_shares(blk, "greedy", "Ours", "rej_share")
        print(f"{blk:28s} numbers: {g['numbers']:.0%} of tokens, {rj['numbers']:.0%} of ours' rejections")
    for fn in (m1a_acc, m1b_shares, m1c_entropy, m1d_strip):
        fn()
        print("wrote", fn.__name__)


if __name__ == "__main__":
    main()
