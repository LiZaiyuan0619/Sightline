""" (claim M7): where does the tree budget actually go? For every node the budget buys, record its DEPTH in the
block and its SIBLING RANK at that depth (rank 0 = the drafter's top-1 continuation of its parent), and whether the
target's realized token walked through it. Two readings the acceptance profile cannot give:

  spend    the share of the N nodes that lands at each (depth, rank) - the shape the best-first tree actually takes;
  payoff   the share of ACCEPTED tokens at each (depth, rank) - how often the budget spent off the top-1 path pays off,
           which is the whole argument for a tree over a chain.

Exact, 0 GPU: it replays simulate.tree_accept's pops (the verifier builds its trees with the same pop order,
simulate.grow_tree) over the cached draft distributions of a head (cache_draft.py), against the realized tokens.

  python analysis/tree_budget.py --caches ours=<dir> ddtree=<dir> --budget 127 --topk 16 \
      --out runs/2026-10-31/analysis/tree_budget
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline import design as simulate
from sightline.paths import RUNS


def tree_spend(cands, budget, realized, max_rank):
    """One round. Returns (spend Counter[(depth, rank)], accepted Counter[(depth, rank)], accepted length).
    Mirrors simulate.tree_accept exactly (same heap keys, same pop order), recording every pop."""
    spend, acc = Counter(), Counter()
    k = len(cands)
    m = min(k, len(realized))
    if not k or not cands[0][0]:
        return spend, acc, 0
    heap = [(-cands[0][1][0], 0, 0, -1, True)]
    deepest, pops = 0, 0
    while heap and pops < budget:
        neg, depth, q, parent, parent_ok = heapq.heappop(heap)
        pops += 1
        cum = -neg
        ids, sc = cands[depth]
        cell = (depth, min(q, max_rank))
        spend[cell] += 1
        node_ok = parent_ok and depth < m and ids[q] == realized[depth]
        if node_ok:
            acc[cell] += 1
            deepest = max(deepest, depth + 1)
        if q + 1 < len(ids):
            heapq.heappush(heap, (-(cum - sc[q] + sc[q + 1]), depth, q + 1, parent, parent_ok))
        if depth + 1 < k and cands[depth + 1][0]:
            heapq.heappush(heap, (-(cum + cands[depth + 1][1][0]), depth + 1, 0, pops - 1, node_ok))
    return spend, acc, deepest


def run(cache: Path, design, budget, max_rank):
    spend, acc = Counter(), Counter()
    rounds = emitted = 0
    for f in sorted((cache / "greedy").glob("*.npz")):
        row = simulate.load_row(f)
        tokens = row["tokens"]
        n_gen = len(tokens)
        K = row["top_ids"].shape[1]
        stop = row["onset"] if row["onset"] else n_gen            # loop-free positions, the caliber of every other panel
        t = 0
        while t < n_gen - 1 and t < stop:
            realized = [int(x) for x in tokens[t + 1:t + 1 + K]]
            s, a, L = tree_spend(simulate.block_candidates(row, t, design), budget, realized, max_rank)
            spend += s
            acc += a
            rounds += 1
            e = min(L + 1, n_gen - 1 - t)
            emitted += e
            t += e
    return {"rounds": rounds, "emitted": emitted, "tau": emitted / max(rounds, 1),
            "spend": {f"{d},{r}": c for (d, r), c in sorted(spend.items())},
            "accepted": {f"{d},{r}": c for (d, r), c in sorted(acc.items())},
            "nodes_per_round": sum(spend.values()) / max(rounds, 1),
            "accepted_per_round": sum(acc.values()) / max(rounds, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True, help="name=<cache dir of cache_draft.py>")
    ap.add_argument("--design", default="D1-top16-N127")
    ap.add_argument("--budget", type=int, default=127)
    ap.add_argument("--max-rank", type=int, default=8, help="ranks above this are pooled into one bucket")
    ap.add_argument("--out", type=Path, default=RUNS / "2026-10-31/analysis/tree_budget")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    design = simulate.pick_designs(a.design)[0]
    res = {}
    for spec in a.caches:
        name, path = spec.split("=", 1)
        res[name] = run(Path(path), design, a.budget, a.max_rank)
        print(f"[{name}] rounds {res[name]['rounds']}, tau {res[name]['tau']:.2f}, "
              f"nodes/round {res[name]['nodes_per_round']:.1f}", flush=True)
    (a.out / "tree_budget.json").write_text(json.dumps(res, indent=1))

    L = [f"# Where the tree budget goes ({a.design}, healthy positions, OmniSVG-8B text-to-SVG)\n",
         "spend = share of the budget's nodes at that (depth, sibling rank); accepted = share of accepted draft "
         "tokens there. Rank 0 is the drafter's top-1 continuation of its parent; ranks above "
         f"{a.max_rank} are pooled.\n"]
    for name, r in res.items():
        L += [f"\n## {name} (tau {r['tau']:.2f}, {r['nodes_per_round']:.1f} nodes and "
              f"{r['accepted_per_round']:.2f} accepted per round)\n",
              "| rank | spend | accepted | accepted / spend |", "|---|---|---|---|"]
        tot_s, tot_a = sum(r["spend"].values()), sum(r["accepted"].values())
        by_rank_s, by_rank_a = Counter(), Counter()
        for key, c in r["spend"].items():
            by_rank_s[int(key.split(",")[1])] += c
        for key, c in r["accepted"].items():
            by_rank_a[int(key.split(",")[1])] += c
        for rank in sorted(by_rank_s):
            s, ac = by_rank_s[rank], by_rank_a[rank]
            L.append(f"| {rank}{'+' if rank == a.max_rank else ''} | {100 * s / tot_s:.1f}% | "
                     f"{100 * ac / max(tot_a, 1):.1f}% | {ac / s:.3f} |")
        L += ["", "| depth | spend | accepted | accepted / spend |", "|---|---|---|---|"]
        by_d_s, by_d_a = Counter(), Counter()
        for key, c in r["spend"].items():
            by_d_s[int(key.split(",")[0])] += c
        for key, c in r["accepted"].items():
            by_d_a[int(key.split(",")[0])] += c
        for d in sorted(by_d_s):
            s, ac = by_d_s[d], by_d_a[d]
            L.append(f"| {d + 1} | {100 * s / tot_s:.1f}% | {100 * ac / max(tot_a, 1):.1f}% | {ac / s:.3f} |")
    (a.out / "tree_budget.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
