"""Natural-language control figure (TH4), round 1 (main line): single-column candidates on ONE dataset.

Claim (plain words). Two pairs of corpora have the same miss rate for a block drafter (one candidate per position,
block-16 footing): OmniSVG-8B's block-16 head vs Qwen3-8B on HumanEval (6.6% vs 6.6%), its block-32 head vs GSM8K
(7.2% vs 7.0%). The same miss rate is arranged very differently. On vector graphics the misses clump: the runs
between misses are twice as unequal (gap CV 3.6 vs 1.8, 3.0 vs 1.5), one- and two-token runs make up four tenths of
the runs against a quarter, and runs of 64 tokens or more hold almost half of the tokens against a third or less.
And the drafter sees far into the block: at offset 15, with no decision in between, it keeps 97% of its offset-1
accuracy on vector graphics and 84% / 77% on natural language, where decisions are also denser.
Per prompt (numbers.md) the gap CV does NOT separate the corpora (medians 1.5 vs 1.5): the pooled CV is
clumping across drawings (prompts without a miss next to prompts missing one position in five) plus a within-prompt
part concentrated in a few prompts, so the clumping is shown pooled (n1c) and as the per-prompt miss-rate spread
(n1a), not as a per-prompt CV. What the data does NOT say (numbers.md): at the window of 16 the share of tokens in runs >= 16 is about the same
(77% vs 74%), and the gain from doubling the window 8 -> 16 is the same; with 16 candidates natural language's
idealised miss rate is 0.07%, lower than SVG's 1.0%. So the control supports "clumped" and "far horizon", not "more
tokens in long runs at the published window".

Data: build_rows.py (runs between misses, per prompt) and build_dec_rows.py (decisions and the distance term, per
prompt). SVG in the loop-free caliber, natural language all tokens (no loops). Pooled values are asserted against
nl_vs_svg.json before anything is drawn. Vocabulary on the figure: a *run* is the stretch from one miss to the next;
a *stretch* (panel e) is the stretch between two decisions of the target.

Run:  python3 paper/claims/TH4-nl-control/round1/make.py [names...]   (STRICT=0 to draft)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402

H.apply()
SZ, C, INK, MUTED = H.SZ, H.C, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W_PT = H.COL_W * 72
BLUE, GREY = C["ours"], "#8C8C8C"
BLUE_FILL, GREY_FILL = "#BFD0E6", "#DADADA"
MIN_L = 32            # a prompt with fewer labelled positions has no run structure to speak of; not plotted
MIN_GAPS = 5          # per-prompt gap CV only where a prompt has at least this many interior gaps
ORDER = ["svg16", "humaneval", "svg32", "gsm8k"]
NAME = {"svg16": "SVG, block-16 head", "humaneval": "HumanEval", "svg32": "SVG, block-32 head", "gsm8k": "GSM8K"}
SHORT = {"svg16": "SVG", "humaneval": "HumanEval", "svg32": "SVG", "gsm8k": "GSM8K"}
KIND = {"svg16": "svg", "humaneval": "nl", "svg32": "svg", "gsm8k": "nl"}
PAIR = {"svg16": 0, "humaneval": 0, "svg32": 1, "gsm8k": 1}
MARK = {0: "o", 1: "^"}
LS = {0: "-", 1: (0, (3, 1.6))}
COL = {"svg": BLUE, "nl": GREY}
FILL = {"svg": BLUE_FILL, "nl": GREY_FILL}
NL_JSON = Path("runs/2026-11-05/theory/nl_vs_svg.json")
JSON_LABEL = {"svg16": "OmniSVG-8B t2svg, b16 head", "svg32": "OmniSVG-8B t2svg, b32 head",
              "humaneval": "Qwen3-8B HumanEval", "gsm8k": "Qwen3-8B GSM8K"}
JMAX = 15


# ------------------------------------------------------------------------------------------------- data layer
def pieces(x):
    return [g + 1 for g in x["interior"]] + [h + 1 for h in x["heads"]] + x["tails"] + x["free"]


def load():
    ref = json.loads(NL_JSON.read_text())
    matched = {(m["svg"], m["nl"]): m for m in ref["matched"]}
    D = {}
    for c in ORDER:
        j = json.loads((HERE / f"rows_{c}.json").read_text())
        allrows = [r["k"]["1"] for r in j["rows"] if r["k"]["1"]["L"] > 0]
        npos = sum(x["L"] for x in allrows)
        nfail = sum(x["misses"] for x in allrows)
        ev = np.array([g for x in allrows for g in x["interior"]], dtype=np.float64)
        allpieces = np.array([n for x in allrows for n in pieces(x)], dtype=np.float64)
        pooled = dict(lam=nfail / npos, cv=float(ev.std() / ev.mean()),
                      tau={W: npos / sum(x["rounds"][str(W)] for x in allrows) for W in (4, 8, 16)},
                      share={t: float(allpieces[allpieces >= t].sum() / allpieces.sum()) for t in (16, 32, 64)},
                      tiny=float((allpieces <= 2).mean()))
        rk = ref["corpora"][JSON_LABEL[c]]["k"]["1"]
        assert abs(pooled["lam"] - rk["lam"]) < 1e-9 and abs(pooled["cv"] - rk["cv"]) < 1e-6, (c, pooled, rk)
        rows = [x for x in allrows if x["L"] >= MIN_L]
        P = [np.array(pieces(x), dtype=np.float64) for x in rows]
        per = dict(lam=np.array([x["misses"] / x["L"] for x in rows]),
                   mean_run=np.array([x["L"] / x["pieces"] for x in rows]),
                   share16=np.array([p[p >= 16].sum() / p.sum() for p in P]),
                   share64=np.array([p[p >= 64].sum() / p.sum() for p in P]),
                   tiny=np.array([(p <= 2).mean() for p in P]),
                   gain=np.array([x["rounds"]["8"] / x["rounds"]["16"] for x in rows]))
        cv = []
        for x in rows:
            g = np.array(x["interior"], dtype=np.float64)
            cv.append(g.std() / g.mean() if len(g) >= MIN_GAPS and g.mean() > 0 else np.nan)
        per["cv"] = np.array(cv)
        # decisions and the distance term
        jd = json.loads((HERE / f"rows_dec_{c}.json").read_text())
        drows = [x for x in jd["rows"] if x["lim"] >= MIN_L]
        hits0 = np.array([x["hits0"] for x in jd["rows"]], dtype=np.float64)
        n0 = np.array([x["n0"] for x in jd["rows"]], dtype=np.float64)
        g_pooled = hits0.sum(0) / np.maximum(n0.sum(0), 1)
        rg = ref["corpora"][JSON_LABEL[c]]["gj"]
        for jj in range(1, JMAX + 1):
            if str(jj) in rg:
                assert abs(g_pooled[jj] - rg[str(jj)]["acc"]) < 1e-9, (c, jj)
        rng = np.random.default_rng(1107)
        boots = []
        for _ in range(400):
            idx = rng.integers(0, len(hits0), len(hits0))
            boots.append(hits0[idx].sum(0) / np.maximum(n0[idx].sum(0), 1))
        boots = np.array(boots)
        dec = dict(g=g_pooled, g_lo=np.percentile(boots, 2.5, axis=0), g_hi=np.percentile(boots, 97.5, axis=0),
                   density=np.array([x["n_dec"] / x["lim"] for x in drows]),
                   tok16=np.array([(lambda s: s[s >= 16].sum() / s.sum())(np.array(x["stretches"], dtype=np.float64))
                                   for x in drows]),
                   density_pooled=sum(x["n_dec"] for x in jd["rows"]) / sum(x["lim"] for x in jd["rows"]))
        s_all = np.array([v for x in jd["rows"] for v in x["stretches"]], dtype=np.float64)
        dec["tok16_pooled"] = float(s_all[s_all >= 16].sum() / s_all.sum())
        D[c] = dict(rows=len(rows), all_rows=len(allrows), pooled=pooled, per=per, allpieces=allpieces, dec=dec)
    m1 = matched[("OmniSVG-8B t2svg, b16 head", "Qwen3-8B HumanEval")]
    m2 = matched[("OmniSVG-8B t2svg, b32 head", "Qwen3-8B GSM8K")]
    assert abs(round(m1["cv"][0], 2) - 3.62) < 1e-9 and abs(round(m1["cv"][1], 2) - 1.79) < 1e-9
    assert abs(round(m2["cv"][0], 2) - 2.96) < 1e-9 and abs(round(m2["cv"][1], 2) - 1.45) < 1e-9
    assert abs(round(m1["gratio"][0], 3) - 0.967) < 1e-9 and abs(round(m1["gratio"][1], 2) - 0.84) < 1e-9
    assert abs(round(m2["gratio"][0], 3) - 0.975) < 1e-9 and abs(round(m2["gratio"][1], 3) - 0.771) < 1e-9
    return D


D = load()
PAIR_TEXT = [f"miss rate {100 * D['svg16']['pooled']['lam']:.1f}% vs {100 * D['humaneval']['pooled']['lam']:.1f}%",
             f"miss rate {100 * D['svg32']['pooled']['lam']:.1f}% vs {100 * D['gsm8k']['pooled']['lam']:.1f}%"]


def _fig(h_pt):
    return plt.figure(figsize=(W_PT / 72, h_pt / 72))


def _lam_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.012, 0.6)
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:g}%"))


def _violin(ax, xc, vals, color, fill, width=0.38, jitter=0.11, s=6, rng=None, clip=None):
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    rng = rng or np.random.default_rng(1107)
    if len(v) >= 10 and v.std() > 0:
        lo, hi = (v.min(), v.max()) if clip is None else (max(v.min(), clip[0]), min(v.max(), clip[1]))
        g = np.linspace(lo, hi, 200)
        d = gaussian_kde(v, bw_method=0.3)(g)
        d = d / d.max() * width
        f = ax.fill_betweenx(g, xc - d, xc + d, color=fill, lw=0, alpha=0.85, zorder=1)
        f.set_gid("bg")
    vv = v if clip is None else np.clip(v, clip[0], clip[1])
    sc = ax.scatter(xc + rng.uniform(-jitter, jitter, len(vv)), vv, s=s, color=color, alpha=0.55, lw=0, zorder=3)
    sc.set_rasterized(True)
    ax.plot([xc], [np.median(v)], marker="D", ms=6.0, mfc="white", mec=color, mew=1.5, zorder=6)
    return float(np.median(v))


def _group_axis(ax, labels, pair_text):
    """Four groups in two pairs; first row the corpus, second row the pair's matched miss rate under a thin line."""
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=SZ["tick"])
    ax.tick_params(axis="x", length=0, pad=2)
    for (a, b), txt in zip(((0, 1), (2, 3)), pair_text):
        y = -0.16
        ax.plot([a - 0.42, b + 0.42], [y, y], color=H.GROUP_LINE, lw=0.8, clip_on=False,
                transform=ax.get_xaxis_transform(), zorder=1)
        ax.text((a + b) / 2, y - 0.025, txt, ha="center", va="top", fontsize=SZ["tick"] - 0.5, color=MUTED,
                transform=ax.get_xaxis_transform())
    ax.spines["bottom"].set_visible(False)


def _marker_items():
    return [H.item("vector graphics (OmniSVG-8B)", BLUE, kind="marker", marker="o", bold=True, text_color=BLUE),
            H.item("natural language (Qwen3-8B)", GREY, kind="marker", marker="o")]


def _line_items():
    return [H.item("SVG, block-16 head", BLUE, kind="line", lw=1.8, bold=True, text_color=BLUE),
            H.item("HumanEval", GREY, kind="line", lw=1.5),
            H.item("SVG, block-32 head", BLUE, kind="line", lw=1.8, ls=LS[1], bold=True, text_color=BLUE),
            H.item("GSM8K", GREY, kind="line", lw=1.5, ls=LS[1])]


def _label_medians(ax, meds, fmt):
    for i, c in enumerate(ORDER):
        t = ax.text(i + 0.44, meds[c], fmt(meds[c]), ha="left", va="center", fontsize=SZ["tick"] - 0.6,
                    color=COL[KIND[c]], fontweight="bold" if KIND[c] == "svg" else "normal")
        t.set_gid("free")


# ------------------------------------------------------------------------------------------------- candidates
def n1a_rate_violin():
    """Per prompt, the drafter's miss rate (log scale). The two corpora of a pair share the average (short dark bar);
    on vector graphics the prompts spread over more than an order of magnitude, on natural language over a factor of
    two. Prompts without a single miss sit at the floor (count written)."""
    H_PT = 190
    fig = _fig(H_PT)
    L, B = 58, 46
    ax = H.axes_pt(fig, L, B, W_PT - L - 8, H_PT - B - 8)
    FLOOR = 0.003
    rng = np.random.default_rng(1107)
    for i, c in enumerate(ORDER):
        lam = D[c]["per"]["lam"]
        nz = lam[lam > 0]
        v = np.log10(np.maximum(lam, FLOOR))
        if len(nz) >= 10:
            g = np.linspace(np.log10(max(nz.min(), FLOOR)), np.log10(nz.max()), 200)
            d = gaussian_kde(np.log10(nz), bw_method=0.3)(g)
            d = d / d.max() * 0.38
            f = ax.fill_betweenx(10 ** g, i - d, i + d, color=FILL[KIND[c]], lw=0, alpha=0.85, zorder=1)
            f.set_gid("bg")
        sc = ax.scatter(i + rng.uniform(-0.11, 0.11, len(v)), 10 ** v, s=6, color=COL[KIND[c]], alpha=0.55, lw=0,
                        zorder=3)
        sc.set_rasterized(True)
        ax.plot([i], [np.median(lam)], marker="D", ms=6.0, mfc="white", mec=COL[KIND[c]], mew=1.5, zorder=6)
        ax.plot([i - 0.3, i + 0.3], [D[c]["pooled"]["lam"]] * 2, color=INK, lw=1.6, zorder=5)
        nzero = int((lam == 0).sum())
        if nzero:
            t = ax.text(i, FLOOR * 0.78, f"{nzero} with no miss", ha="center", va="top", fontsize=7.0,
                        color=MUTED)
            t.set_gid("free")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.55, 0.7)
    ax.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:g}%"))
    _group_axis(ax, [SHORT[c] for c in ORDER], ["same average miss rate" for _ in range(2)])
    ax.set_ylabel("miss rate per prompt\n(bar = corpus average, white = median)", fontsize=SZ["label"] - 0.5,
                  labelpad=2, linespacing=1.18)
    return fig


def n1c_ccdf():
    """Pooled: share of all tokens that sit in runs at least x tokens long. The four curves meet near the window of
    16 and part beyond it: the vector-graphics tail is heavier."""
    H_PT = 188
    fig = _fig(H_PT)
    bot = H.legend(fig, _line_items(), x_pt=42, top_pt=2, width_pt=W_PT - 44, fontsize=7.8, gap_pt=9)
    L, B = 40, 30
    ax = H.axes_pt(fig, L, B, W_PT - L - 8, H_PT - B - bot - 4)
    for c in ORDER:
        n = np.sort(D[c]["allpieces"])
        tail = np.cumsum((n / n.sum())[::-1])[::-1]
        xs, ys, last = [1.0], [1.0], None
        for x, t in zip(n, tail):
            if x != last:
                xs.append(float(x))
                ys.append(float(t))
                last = x
        ax.step(xs, 100 * np.array(ys), where="post", color=COL[KIND[c]], lw=1.8 if KIND[c] == "svg" else 1.5,
                ls=LS[PAIR[c]], zorder=3 if KIND[c] == "svg" else 2)
    ax.set_xscale("log")
    ax.set_xlim(1, 1500)
    ax.xaxis.set_major_locator(FixedLocator([1, 4, 16, 64, 256, 1024]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    for x in (16, 64):
        H.ref_line(ax, x=x)
        ax.text(x * 1.08, 2, f"{x}", ha="left", va="bottom", fontsize=SZ["tick"] - 0.8, color=MUTED)
    # the values at 64, in the empty upper right (the curves have dropped below 50% there)
    sh = {c: 100 * D[c]["pooled"]["share"][64] for c in ORDER}
    ax.text(1400, 99, "tokens in runs of 64 or more", ha="right", va="top", fontsize=SZ["tick"] - 0.8, color=MUTED)
    ax.text(1400, 88, f"vector graphics {sh['svg16']:.0f}% / {sh['svg32']:.0f}%", ha="right", va="top",
            fontsize=SZ["tick"] - 0.6, color=BLUE, fontweight="bold")
    ax.text(1400, 77, f"natural language {sh['humaneval']:.0f}% / {sh['gsm8k']:.0f}%", ha="right", va="top",
            fontsize=SZ["tick"] - 0.6, color=GREY)
    ax.set_xlabel("run length (tokens)", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens in runs at least\nthis long (%)", fontsize=SZ["label"] - 0.5, labelpad=2, linespacing=1.18)
    return fig


def n1d_horizon():
    """The distance term g(j): top-1 accuracy at offset j when no decision lies between the block start and the
    position, pooled over prompts with a 95% bootstrap band over prompts."""
    H_PT = 190
    fig = _fig(H_PT)
    bot = H.legend(fig, _line_items(), x_pt=42, top_pt=2, width_pt=W_PT - 44, fontsize=7.8, gap_pt=9)
    L, B = 40, 30
    ax = H.axes_pt(fig, L, B, W_PT - L - 8, H_PT - B - bot - 4)
    js = np.arange(1, JMAX + 1)
    for c in ORDER:
        g, lo, hi = (D[c]["dec"][k][1:JMAX + 1] for k in ("g", "g_lo", "g_hi"))
        f = ax.fill_between(js, 100 * lo, 100 * hi, color=FILL[KIND[c]], lw=0, alpha=0.6, zorder=1)
        f.set_gid("bg")
        ax.plot(js, 100 * g, color=COL[KIND[c]], lw=1.8 if KIND[c] == "svg" else 1.5, ls=LS[PAIR[c]],
                marker=MARK[PAIR[c]], ms=3.2, mew=0, zorder=3 if KIND[c] == "svg" else 2)
    ax.set_xlim(0.5, 18.4)
    ax.set_xticks([1, 4, 8, 12, 15])
    ax.set_ylim(60, 101)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_xlabel("offset inside the block (tokens from the start)", fontsize=SZ["label"] - 0.5, labelpad=1.5)
    ax.set_ylabel("accuracy when no decision\nis crossed (%)", fontsize=SZ["label"] - 0.5, labelpad=2,
                  linespacing=1.18)
    ends = {c: 100 * D[c]["dec"]["g"][JMAX] for c in ORDER}
    s_lo, s_hi = sorted((ends["svg16"], ends["svg32"]))
    labels = [((s_lo + s_hi) / 2, f"{s_lo:.0f}–{s_hi:.0f}%" if s_hi - s_lo >= 0.5 else f"{s_hi:.0f}%", BLUE, "bold")]
    labels += [(ends[c], f"{ends[c]:.0f}%", GREY, "normal") for c in ("humaneval", "gsm8k")]
    for y, txt, col, fw in labels:
        ax.text(15.45, y, txt, ha="left", va="center", fontsize=SZ["tick"] - 0.6, color=col, fontweight=fw)
    return fig


def n1e_stretch_violin():
    """Per prompt, the share of its tokens that sit in stretches of at least 16 positions between two decisions of
    the target (entropy above 0.2 nats)."""
    H_PT = 184
    fig = _fig(H_PT)
    L, B = 50, 46
    ax = H.axes_pt(fig, L, B, W_PT - L - 8, H_PT - B - 8)
    meds = {c: _violin(ax, i, 100 * D[c]["dec"]["tok16"], COL[KIND[c]], FILL[KIND[c]], clip=(0, 100))
            for i, c in enumerate(ORDER)}
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    _group_axis(ax, [SHORT[c] for c in ORDER], PAIR_TEXT)
    _label_medians(ax, meds, lambda v: f"{v:.0f}%")
    ax.set_ylabel("tokens in stretches of at least 16\nbetween two decisions (%, per prompt)", fontsize=SZ["label"] - 0.5,
                  labelpad=2, linespacing=1.18)
    return fig


def n1f_density_violin():
    """Per prompt, the share of positions that are decisions of the target (entropy above 0.2 nats)."""
    H_PT = 184
    fig = _fig(H_PT)
    L, B = 50, 46
    ax = H.axes_pt(fig, L, B, W_PT - L - 8, H_PT - B - 8)
    meds = {c: _violin(ax, i, 100 * D[c]["dec"]["density"], COL[KIND[c]], FILL[KIND[c]], clip=(0, 60))
            for i, c in enumerate(ORDER)}
    ax.set_ylim(0, 62)
    ax.set_yticks([0, 15, 30, 45, 60])
    _group_axis(ax, [SHORT[c] for c in ORDER], PAIR_TEXT)
    _label_medians(ax, meds, lambda v: f"{v:.0f}%")
    ax.set_ylabel("positions that are decisions\n(%, per prompt, white = median)", fontsize=SZ["label"] - 0.5,
                  labelpad=2, linespacing=1.18)
    return fig


CANDIDATES = {"n1a_rate_violin": n1a_rate_violin, "n1c_ccdf": n1c_ccdf, "n1d_horizon": n1d_horizon,
              "n1e_stretch_violin": n1e_stretch_violin, "n1f_density_violin": n1f_density_violin}


def report():
    L = ["| corpus | prompts plotted (of) | pooled miss rate | pooled gap CV | median per-prompt CV (n) | "
         "runs of 1-2 tokens (share of runs) | tokens in runs >= 16 / 32 / 64 | gain 8 -> 16 (pooled) | "
         "decision density | tokens in stretches >= 16 | g(15)/g(1) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        d, p, q = D[c], D[c]["per"], D[c]["pooled"]
        m = np.isfinite(p["cv"])
        g = d["dec"]["g"]
        L.append(f"| {NAME[c]} | {d['rows']} ({d['all_rows']}) | {100 * q['lam']:.1f}% | {q['cv']:.2f} | "
                 f"{np.median(p['cv'][m]):.2f} ({int(m.sum())}) | {100 * q['tiny']:.0f}% | "
                 f"{100 * q['share'][16]:.0f} / {100 * q['share'][32]:.0f} / {100 * q['share'][64]:.0f}% | "
                 f"×{q['tau'][16] / q['tau'][8]:.3f} | {100 * d['dec']['density_pooled']:.1f}% | "
                 f"{100 * d['dec']['tok16_pooled']:.0f}% | {g[JMAX] / g[1]:.3f} |")
    txt = "\n".join(L)
    (HERE / "numbers.md").write_text("# numbers behind round 1 (one candidate per position, block-16 footing)\n\n"
                                     + txt + "\n\nAll pooled values equal nl_vs_svg.json (asserted in make.py).\n")
    print(txt)


def main(names):
    report()
    outs = []
    for n in names or list(CANDIDATES):
        fig = CANDIDATES[n]()
        H.save(fig, str(HERE / n), width="col", strict=STRICT)
        plt.close(fig)
        outs.append(str(HERE / f"{n}.png"))
    if not names:
        H.sheet(outs, str(HERE / "_sheet.png"), cols=3, title="TH4 natural-language control, round 1 (11-07)")


if __name__ == "__main__":
    main(sys.argv[1:])
