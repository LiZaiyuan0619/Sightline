"""Per-prompt decision data for the natural-language control figure (TH4), round 1. 0 GPU.

Companion of build_rows.py (runs between misses). This one carries Proposition 2's side of the control, from the
TARGET's own entropy: decision points (H > 0.2 nats), the no-decision stretches between them and the share of tokens
that sit in stretches of at least 16 / 32, and the drafter's distance term g(j): top-1 accuracy at offset j when the
position is not a decision and no decision lies between the block start and it. Per prompt we keep hits and counts
per offset, so the pooled g(j) and its bootstrap band over prompts can be formed downstream.

Same four corpora and footing as build_rows.py (offset 1..15, block-16 footing; SVG loop-free caliber, natural
language all tokens). Pooled over rows, decision density, the token share in stretches >= 16 and g(j) must reproduce
nl_vs_svg.json (asserted).

  python3 paper/claims/TH4-nl-control/round1/build_dec_rows.py [--procs 6]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_rows as BR  # noqa: E402  (corpora, G2)

ETA = 0.2
JMAX = 15
ENTROPY = {"svg16": BR.G2 / "entropy/b16-6ep/greedy", "svg32": BR.G2 / "cache300/entropy-b32-6ep/greedy",
           "humaneval": None, "gsm8k": None}


def one(args):
    f, healthy, edir = args
    z = np.load(f, allow_pickle=True)
    tok = z["tokens"]
    n = len(tok)
    if edir is None:
        Hs = np.asarray(z["H"], dtype=np.float32)
    else:
        y = np.load(Path(edir) / Path(f).name, allow_pickle=True)
        assert int(y["gidx"]) == int(z["gidx"]) and len(y["H"]) == n, f
        Hs = np.asarray(y["H"], dtype=np.float32)
    on = int(z["ref_onset"])
    lim = n if (not healthy or on < 0) else min(n, on)
    lim = max(lim, 0)
    rr, no = z["real_rank"], z["n_off"].astype(int)
    d = Hs[:lim] > ETA
    fidx = np.flatnonzero(d)
    stretches = [int(lim)] if len(fidx) < 2 else [int(v) for v in (np.diff(fidx) - 1)]
    cum = np.concatenate([[0], np.cumsum(d)])
    T = rr.shape[0]
    hits0, n0, hitsall, nall = [0] * (JMAX + 1), [0] * (JMAX + 1), [0] * (JMAX + 1), [0] * (JMAX + 1)
    for j in range(1, JMAX + 1):
        t_end = min(T, lim - j)
        if t_end <= 0:
            break
        t = np.arange(t_end)
        p = t + j
        ok = no[:t_end] >= j
        right = (rr[:t_end, j - 1] == 0)
        sel = ok & (~d[p]) & ((cum[p] - cum[t + 1]) == 0)
        hits0[j], n0[j] = int(right[sel].sum()), int(sel.sum())
        hitsall[j], nall[j] = int(right[ok].sum()), int(ok.sum())
    return dict(gidx=int(z["gidx"]), lim=int(lim), n_dec=int(d.sum()), stretches=stretches,
                hits0=hits0, n0=n0, hitsall=hitsall, nall=nall)


def build(name, procs, ref):
    c = BR.CORPORA[name]
    files = sorted(str(p) for p in Path(c["cache"]).glob("*.npz"))
    with Pool(procs) as pool:
        rows = pool.map(one, [(f, c["healthy"], ENTROPY[name]) for f in files])
    r = ref["corpora"][c["json_label"]]
    dens = sum(x["n_dec"] for x in rows) / sum(x["lim"] for x in rows)
    s = np.array([v for x in rows for v in x["stretches"] if v >= 0], dtype=np.float64)
    tok16 = float(s[s >= 16].sum() / s.sum())
    assert abs(dens - r["dec"]["density"]) < 1e-9, (name, dens, r["dec"]["density"])
    assert abs(tok16 - r["dec"]["tok16"]) < 1e-9, (name, tok16, r["dec"]["tok16"])
    for j in range(1, JMAX + 1):
        h, m = sum(x["hits0"][j] for x in rows), sum(x["n0"][j] for x in rows)
        if str(j) in r["gj"]:
            assert m == r["gj"][str(j)]["n"] and abs(h / m - r["gj"][str(j)]["acc"]) < 1e-9, (name, j, h / m, r["gj"][str(j)])
    out = dict(meta=dict(corpus=name, label=c["label"], rows=len(rows), eta=ETA, jmax=JMAX,
                         note="pooled density, tok16 and g(j) reproduce nl_vs_svg.json (asserted)"), rows=rows)
    (HERE / f"rows_dec_{name}.json").write_text(json.dumps(out))
    print(name, len(rows), "rows; density", round(dens, 4), "tok16", round(tok16, 3),
          "g(15)/g(1)", round((sum(x["hits0"][15] for x in rows) / sum(x["n0"][15] for x in rows))
                              / (sum(x["hits0"][1] for x in rows) / sum(x["n0"][1] for x in rows)), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    ref = json.loads((BR.THEORY / "nl_vs_svg.json").read_text())
    for name in BR.CORPORA:
        build(name, a.procs, ref)
