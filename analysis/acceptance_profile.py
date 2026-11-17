""" the acceptance profile of a decoding run, from the result files only
(0 GPU). Three things every speculative-decoding paper shows and we had the data for but never extracted:

  1. distribution of tokens emitted per round (DDTree Fig 4 style) and the survival curve P(round emits > j tokens);
  2. the share of rounds that hit the speculation limit (the whole block was accepted) - the quantity that says whether
     a longer block would still pay off;
  3. how much of the round's budget is actually used: mean tree nodes per round against mean tokens emitted
     (`mean_tree_nodes` is in the result rows), i.e. nodes spent per accepted token.

  python3 analysis/accept_profile.py --rows <dir> [<dir>...] --labels a b --blocks 32 16 --out <prefix>
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path


def load(root: str, mode: str) -> list[dict]:
    return [r for f in sorted(glob.glob(f"{root}/shard-*/results.jsonl")) for r in map(json.loads, open(f))
            if r and r["mode"] == mode]


def profile(rows: list[dict], block: int) -> dict:
    hist: Counter = Counter()
    nodes, emitted, full = 0.0, 0, 0
    for r in rows:
        for n in r["acceptance"]:
            hist[n] += 1
            emitted += n
            full += int(n >= block)
        if r.get("mean_tree_nodes"):
            nodes += r["mean_tree_nodes"] * len(r["acceptance"])
    tot = sum(hist.values())
    surv = {}
    run = tot
    for j in range(1, block + 2):
        surv[j] = run / max(tot, 1)
        run -= hist.get(j, 0)
    return {"rounds": tot, "tau_all": emitted / max(tot, 1), "full_block_share": full / max(tot, 1),
            "hist": {k: v / max(tot, 1) for k, v in sorted(hist.items())},
            "survival": surv, "mean_nodes": nodes / max(tot, 1) if nodes else None,
            "nodes_per_token": (nodes / emitted) if (nodes and emitted) else None,
            "median_accept": st.median([n for n, c in hist.items() for _ in range(c)]) if tot else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--blocks", nargs="+", type=int, required=True)
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert len(a.rows) == len(a.labels) == len(a.blocks)
    res = {}
    for root, name, blk in zip(a.rows, a.labels, a.blocks):
        rows = load(root, a.mode)
        res[name] = {"block": blk, "n_rows": len(rows), **profile(rows, blk)}
        print(name, res[name]["rounds"], "rounds", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    L = ["# Acceptance profile per round (" + a.mode + ")", "",
         "| method | block | rounds | tau (all tokens) | median | full-block rounds | nodes / round | nodes / token |",
         "|---|---|---|---|---|---|---|---|"]
    for n, v in res.items():
        mn = f"{v['mean_nodes']:.1f}" if v["mean_nodes"] else "-"
        npt = f"{v['nodes_per_token']:.1f}" if v["nodes_per_token"] else "-"
        L.append(f"| {n} | {v['block']} | {v['rounds']} | {v['tau_all']:.2f} | {v['median_accept']:.0f} | "
                 f"{v['full_block_share']:.3f} | {mn} | {npt} |")
    L += ["", "Survival P(round emits > j tokens):", "", "| j | " + " | ".join(res) + " |",
          "|---|" + "---|" * len(res)]
    for j in (1, 2, 4, 8, 12, 16, 24, 32):
        cells = [f"{res[n]['survival'].get(j, 0):.3f}" if j <= res[n]["block"] + 1 else "-" for n in res]
        L.append(f"| {j} | " + " | ".join(cells) + " |")
    a.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("ACCEPT_PROFILE_DONE", a.out.with_suffix(".md"))


if __name__ == "__main__":
    main()
