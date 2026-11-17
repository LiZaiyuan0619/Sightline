"""Per-prompt data for the natural-language control figure (TH4), round 1 (main line). 0 GPU, runs on dev.

The claim the figure has to carry: at the SAME miss rate, a block drafter's misses on vector-graphics streams clump,
so the runs between misses are far more unequal than on natural language and far more tokens sit in runs longer
than the published window. The experiment line's control (runs/2026-11-05/theory/nl_vs_svg.py) found two pairs of
corpora whose miss rates differ by < 2% on the same footing (block 16, offset-1 labels, 1 candidate per position):
    OmniSVG-8B text-to-SVG, block-16 head   vs   Qwen3-8B HumanEval   (lambda 0.0665 vs 0.0658)
    OmniSVG-8B text-to-SVG, block-32 head   vs   Qwen3-8B GSM8K       (lambda 0.0716 vs 0.0703)
Its numbers are pooled over rows; the figure wants one point per prompt. This script cuts every prompt's labelled
stream after each miss (T1's `segments`, read-only) and records the pieces, so any per-prompt statistic can be
formed downstream: miss rate, mean run length, share of tokens in runs of at least 8 / 16 / 32, tokens per round at
W = 4 / 8 / 16 (Lemma 1 per stream), and the gap CV where a prompt has enough interior gaps.

Footing is the control's: offset-1 labels at k = 1 (and k = 16, kept for completeness), windows <= 16 only, SVG in
the loop-free caliber (positions before the loop onset), natural language all-token (no loops at all, ref_onset = -1).
Pooled over rows, the miss rate, the interior-gap CV and the number of gaps must reproduce nl_vs_svg.json for each
corpus (asserted), so the per-prompt points are the control's own data redrawn, not a new estimate.

  python3 paper/claims/TH4-nl-control/round1/build_rows.py [--procs 6]
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
THEORY = RUNS / "2026-11-05/theory"
sys.path.insert(0, str(THEORY))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "h2"))
import stream_gaps as SG  # noqa: E402  (read-only use of the experiment line's helpers)

G2 = (RUNS / "r1105")
HERE = Path(__file__).resolve().parent
WINDOWS = [4, 8, 16]
KS = [1, 16]
CORPORA = {
    "svg16": dict(label="OmniSVG-8B text-to-SVG, block-16 head", kind="svg", cache=G2 / "cache/b16-6ep/greedy",
                  healthy=True, json_label="OmniSVG-8B t2svg, b16 head"),
    "svg32": dict(label="OmniSVG-8B text-to-SVG, block-32 head", kind="svg", cache=G2 / "cache300/b32-6ep/greedy",
                  healthy=True, json_label="OmniSVG-8B t2svg, b32 head"),
    "humaneval": dict(label="Qwen3-8B HumanEval", kind="nl", cache=G2 / "cache-nl/humaneval/greedy",
                      healthy=False, json_label="Qwen3-8B HumanEval"),
    "gsm8k": dict(label="Qwen3-8B GSM8K", kind="nl", cache=G2 / "cache-nl/gsm8k/greedy",
                  healthy=False, json_label="Qwen3-8B GSM8K"),
}
PAIRS = [("svg16", "humaneval"), ("svg32", "gsm8k")]


def load_fields(f, healthy):
    """The control's own convention (nl_vs_svg.load): the limit counts every token of the row, so a row without a loop
    onset is labelled one position further than T1's load_fields (which stops before the last token). Needed to
    reproduce nl_vs_svg.json's miss rates exactly; the difference is one position per row."""
    z = np.load(f, allow_pickle=True)
    n = len(z["tokens"])
    on = int(z["ref_onset"])
    lim = n if (not healthy or on < 0) else min(n, on)
    return dict(gidx=int(z["gidx"]), rr=z["real_rank"], no=z["n_off"].astype(int), n=n, onset=on, lim=max(lim, 0))


def one(args):
    f, healthy = args
    row = load_fields(f, healthy)
    out = dict(gidx=row["gidx"], n=row["n"], lim=row["lim"], onset=row["onset"], k={})
    for k in KS:
        lab, dropped = SG.label_run(row, 1, k)
        s = SG.segments([lab])
        interior, heads = s["interior"], s["heads"]
        tails = [t for t in s["tails"] if t > 0]
        free = list(s["free"])
        closed = [g + 1 for g in interior] + [h + 1 for h in heads]
        allp = closed + tails + free
        L = int(sum(allp))
        assert L == len(lab), (f, k, L, len(lab))          # every labelled position lies in exactly one piece
        out["k"][str(k)] = dict(L=L, misses=int((lab == 0).sum()) if len(lab) else 0, dropped=int(dropped),
                                interior=[int(g) for g in interior], heads=[int(h) for h in heads],
                                tails=[int(t) for t in tails], free=[int(x) for x in free],
                                rounds={str(W): int(sum(math.ceil(n / W) for n in allp)) for W in WINDOWS},
                                pieces=len(allp))
    return out


def build(name, procs, ref):
    c = CORPORA[name]
    files = sorted(str(p) for p in Path(c["cache"]).glob("*.npz"))
    with Pool(procs) as pool:
        rows = pool.map(one, [(f, c["healthy"]) for f in files])
    # pooled checks against the experiment line's control (k = 1 is what the matched pairs use)
    r1 = ref["corpora"][c["json_label"]]
    assert r1["rows"] == len(rows), (name, r1["rows"], len(rows))
    for k in KS:
        if str(k) not in r1["k"]:          # the control skipped k = 16 where a corpus has < 30 interior gaps (NL)
            continue
        rk = r1["k"][str(k)]
        npos = sum(x["k"][str(k)]["L"] for x in rows)
        nfail = sum(x["k"][str(k)]["misses"] for x in rows)
        ev = np.array([g for x in rows for g in x["k"][str(k)]["interior"]], dtype=np.float64)
        lam, cv = nfail / npos, float(ev.std() / ev.mean())
        assert abs(lam - rk["lam"]) < 1e-9, (name, k, lam, rk["lam"])
        assert abs(cv - rk["cv"]) < 1e-6, (name, k, cv, rk["cv"])
        assert len(ev) == rk["n_gaps"], (name, k, len(ev), rk["n_gaps"])
        for W in WINDOWS:   # Lemma 1 pooled = the control's plug-in tokens per round
            tau = npos / sum(x["k"][str(k)]["rounds"][str(W)] for x in rows)
            assert abs(tau - rk["tau"][str(W)]) < 1e-9, (name, k, W, tau, rk["tau"][str(W)])
    out = dict(meta=dict(corpus=name, label=c["label"], kind=c["kind"], cache=str(c["cache"]), rows=len(rows),
                         caliber="healthy (before ref_onset)" if c["healthy"] else "all tokens (no loops)",
                         labels="offset 1", widths=KS, windows=WINDOWS,
                         note="pooled miss rate, interior-gap CV, gap count and tau_W reproduce nl_vs_svg.json (asserted)"),
               rows=rows)
    (HERE / f"rows_{name}.json").write_text(json.dumps(out))
    print(name, len(rows), "rows ->", f"rows_{name}.json")
    return out


def summary(data, ref):
    L = ["# TH4 per-prompt data, round 1 (11-07)\n",
         "Pooled over rows (k = 1, offset-1 labels, block-16 footing). Must equal nl_vs_svg.json (asserted in build).\n",
         "| corpus | rows | positions | miss rate | mean run (tokens) | gap CV | tokens in runs >= 8 / 16 / 32 | tau_4 / tau_8 / tau_16 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, d in data.items():
        rows = d["rows"]
        k1 = [r["k"]["1"] for r in rows]
        npos = sum(x["L"] for x in k1)
        nfail = sum(x["misses"] for x in k1)
        pieces = np.array([n for x in k1 for n in ([g + 1 for g in x["interior"]] + [h + 1 for h in x["heads"]] + x["tails"] + x["free"])])
        ev = np.array([g for x in k1 for g in x["interior"]], dtype=np.float64)
        share = {t: pieces[pieces >= t].sum() / pieces.sum() for t in (8, 16, 32)}
        taus = [npos / sum(x["rounds"][str(W)] for x in k1) for W in WINDOWS]
        L.append(f"| {d['meta']['label']} | {len(rows)} | {npos} | {nfail / npos:.4f} | {pieces.mean():.1f} | "
                 f"{ev.std() / ev.mean():.2f} | {share[8]:.2f} / {share[16]:.2f} / {share[32]:.2f} | "
                 + " / ".join(f"{t:.2f}" for t in taus) + " |")
    L.append("\nMatched pairs (from nl_vs_svg.json): " + "; ".join(
        f"{m['svg']} vs {m['nl']}: lambda {m['lam_svg']:.4f} vs {m['lam_nl']:.4f}, CV {m['cv'][0]:.2f} vs {m['cv'][1]:.2f}"
        for m in ref["matched"]))
    (HERE / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    ref = json.loads((THEORY / "nl_vs_svg.json").read_text())
    data = {name: build(name, a.procs, ref) for name in CORPORA}
    summary(data, ref)
