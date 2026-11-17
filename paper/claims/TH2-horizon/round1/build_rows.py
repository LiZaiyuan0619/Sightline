"""Per-prompt data for the Proposition 2 (horizon in decisions) figure, round 1 (main line). 0 GPU.

One dataset for every candidate, the same prompts as Proposition 1's figure: our block-32 head's 300-prompt greedy
cache with the target's own entropy (r1105/cache300/{b32-6ep, entropy-b32-6ep}/greedy), loop-free caliber.
A decision point is a position where the TARGET's entropy exceeds 0.2 nats (T2's main definition). For a block
drafted from start t, position p = t + j (offset j) is scored like T2 section B: hit = the drafter's first guess at
offset j is the target's token; c = decision points strictly between t and p, capped at 2; only positions that are
not decision points themselves; token class from the OmniSVG vocabulary (command / coordinate / colour / other).
Kept per prompt and per offset band (1-8, 9-16, ): hits and counts by class x c.

Uses the experiment line's T2 helpers read-only (runs/2026-11-05/theory/decision_horizon.py: load_row,
decision_mask, classes_of); writes only here. Pooled over prompts the result must equal T2's published table
(decision_horizon_b32-6ep.json B["0.2"]) -- asserted.

  python3 paper/claims/TH2-horizon/round1/build_rows.py
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
THEORY = RUNS / "2026-11-05/theory"
sys.path.insert(0, str(THEORY))
sys.path.insert(0, str(ROOT / "src"))
import decision_horizon as DH  # noqa: E402  (read-only use)

G2 = (RUNS / "r1105/cache300")
HEAD = "b32-6ep"
CACHE, ENT = G2 / HEAD / "greedy", G2 / f"entropy-{HEAD}" / "greedy"
ETA = "0.2"
BANDS = ((1, 8), (9, 16), (17, 31))
HERE = Path(__file__).resolve().parent


def one(pair):
    cf, ef = pair
    r = DH.load_row(cf, ef, True, 0.9)
    lim, T, K = r["lim"], r["rr"].shape[0], r["rr"].shape[1]
    acc = np.zeros((len(BANDS), 4, 3, 2), dtype=np.int64)          # band x class x c x (miss, hit)
    if lim > 1:
        d = DH.decision_mask(r, ETA, 0.9)
        cum = np.concatenate([[0], np.cumsum(d)])
        for j in range(1, K + 1):
            t_end = min(T, lim - j)
            if t_end <= 0:
                break
            bi = next((i for i, (a, b) in enumerate(BANDS) if a <= j <= b), None)
            if bi is None:
                continue
            t = np.arange(t_end)
            ok = r["no"][:t_end] >= j
            p = t + j
            hit = (r["rr"][:t_end, j - 1] == 0)
            c = np.minimum(cum[p] - cum[t + 1], 2)
            sel = ok & ~d[p]
            if sel.any():
                np.add.at(acc[bi], (r["cls"][p][sel], c[sel], hit[sel].astype(int)), 1)
    return dict(gidx=r["gidx"], lim=int(lim), onset=r["onset"], acc=acc.tolist())


def main():
    cf = sorted(str(p) for p in CACHE.glob("*.npz"))
    pairs = [(f, str(ENT / Path(f).name)) for f in cf]
    with Pool(6) as pool:
        rows = pool.map(one, pairs)
    t2 = json.loads((THEORY / f"decision_horizon_{HEAD}.json").read_text())
    leak = set(int(g) for g in t2["H"]["leak_union"])
    for r in rows:
        r["leak"] = r["gidx"] in leak
    A = np.array([r["acc"] for r in rows]).sum(axis=0)                 # band x class x c x 2
    names = ("cmd", "coord", "color", "other")
    labels = ("1-8", "9-16", "17-31")
    checked = 0
    for bi, bl in enumerate(labels):
        for ci, cl in enumerate(names):
            for c in range(3):
                key = f"{cl}/{bl}/c{c}"
                if key in t2["B"][ETA]:
                    n = int(A[bi, ci, c].sum())
                    ref = t2["B"][ETA][key]
                    assert n == ref["n"] and abs(A[bi, ci, c, 1] / n - ref["acc"]) < 1e-12, (key, n, ref)
                    checked += 1
    out = dict(meta=dict(head=HEAD, cache=str(CACHE), entropy=str(ENT), eta=float(ETA), bands=labels,
                         classes=names, rows=len(rows), leak_rows=len(leak & {r["gidx"] for r in rows}),
                         note=f"pooled counts equal T2 B[{ETA}] in {checked} cells (asserted)"), rows=rows)
    (HERE / f"rows_{HEAD}.json").write_text(json.dumps(out))
    print(f"{len(rows)} prompts, {checked} T2 cells reproduced exactly -> rows_{HEAD}.json")


if __name__ == "__main__":
    main()
