"""Per-prompt data for the Proposition 1 figure, round 2 (main line). 0 GPU.

One fixed dataset for every candidate: a head's 300-row greedy cache (r1105/cache300/<head>/greedy), loop-free
caliber (positions before the row's loop onset, as T1), offset-1 labels at 1 and 16 candidates per position.
For each prompt the labelled stream is cut after every miss into pieces (T1's `segments`): a piece closed by a miss
carries g + 1 tokens (the target writes the missed position), the last piece of a row carries its t correct tokens,
a row without any miss is one piece. A window-W decoder spends ceil(n / W) rounds on a piece of n tokens, so
tau_W = sum(n) / sum(ceil(n / W)) and the upper bound tau_inf = sum(n) / #pieces (Lemma 1, per stream).

Uses the experiment line's T1 helpers read-only (runs/2026-11-05/theory/stream_gaps.py: load_fields, label_run,
segments); writes only into this directory. Pooled over rows the result must equal T1's published identity values
(stream_gaps_<head>.json A["loop-free/j1/k<k>"]["identity"]) -- asserted.
Also flags the rows T2 found degenerate although no loop onset was detected (decision_horizon_<head>.json
H.leak_union), so every candidate can be checked on the strict-loop-free subset too.

  python3 paper/claims/TH1-run-length/round2/build_rows.py [--heads b32-6ep b64-3ep b16-6ep] [--procs 6]
"""
import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
THEORY = RUNS / "2026-11-05/theory"
sys.path.insert(0, str(THEORY))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "h2"))
import stream_gaps as SG  # noqa: E402  (read-only use)

G2 = (RUNS / "r1105")
CACHE = {"b32-6ep": G2 / "cache300/b32-6ep/greedy", "b64-3ep": G2 / "cache300/b64-3ep/greedy",
         "b16-6ep": G2 / "cache/b16-6ep/greedy"}
WINDOWS = [4, 8, 12, 16, 24, 32, 48, 64]
KS = [1, 16]
HERE = Path(__file__).resolve().parent


def pieces_of(labels):
    s = SG.segments([labels])
    closed = [g + 1 for g in s["interior"]] + [h + 1 for h in s["heads"]]
    open_ = [t for t in s["tails"] if t > 0] + list(s["free"])
    return closed, open_


def one(f):
    row = SG.load_fields(f, True)
    out = dict(gidx=row["gidx"], n=row["n"], lim=row["lim"], onset=row["onset"], k={})
    for k in KS:
        lab, dropped = SG.label_run(row, 1, k)
        closed, open_ = pieces_of(lab)
        allp = closed + open_
        L = int(sum(allp))
        out["k"][str(k)] = dict(L=L, dropped=int(dropped), misses=len(closed), closed=closed, open=open_,
                                rounds={str(W): int(sum(math.ceil(n / W) for n in allp)) for W in WINDOWS},
                                pieces=len(allp))
    return out


def build(head, procs):
    files = sorted(str(p) for p in CACHE[head].glob("*.npz"))
    with Pool(procs) as pool:
        rows = pool.map(one, files)
    t1 = json.loads((THEORY / f"stream_gaps_{head}.json").read_text())
    t2 = json.loads((THEORY / f"decision_horizon_{head}.json").read_text())
    leak = set(int(g) for g in t2["H"].get("leak_union_gidx", t2["H"].get("leak_union_list", [])) or [])
    if not leak and isinstance(t2["H"].get("leak_union"), list):
        leak = set(int(g) for g in t2["H"]["leak_union"])
    for r in rows:
        r["leak"] = r["gidx"] in leak
    # pooled identity must reproduce T1 exactly
    for k in KS:
        for W in WINDOWS:
            L = sum(r["k"][str(k)]["L"] for r in rows)
            R = sum(r["k"][str(k)]["rounds"][str(W)] for r in rows)
            ref = t1["A"][f"healthy/j1/k{k}"]["identity"][str(W)]
            assert abs(L / R - ref) < 1e-9, (head, k, W, L / R, ref)
        P = sum(r["k"][str(k)]["pieces"] for r in rows)
        ref = t1["A"][f"healthy/j1/k{k}"]["tau_inf_raw"]
        L = sum(r["k"][str(k)]["L"] for r in rows)
    out = dict(meta=dict(head=head, cache=str(CACHE[head]), rows=len(rows), caliber="healthy (before ref_onset)",
                         labels="offset 1", widths=KS, windows=WINDOWS,
                         leak_rows=len([r for r in rows if r["leak"]]),
                         note="pooled tau_W over rows equals T1 identity (asserted); tau_inf here = L / #pieces"),
               rows=rows)
    p = HERE / f"rows_{head}.json"
    p.write_text(json.dumps(out))
    print(head, len(rows), "rows,", out["meta"]["leak_rows"], "leak rows ->", p.name)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", default=["b32-6ep"])
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    for h in a.heads:
        build(h, a.procs)
