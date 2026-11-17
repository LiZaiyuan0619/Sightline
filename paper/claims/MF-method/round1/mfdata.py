"""MF method figure, round 1: the real pieces the schematic embeds.

1. One real draft tree: `runs/2026-10-29/analysis/trees/tree-gidx150.json` (OmniSVG-8B text-to-SVG, bench row 150,
   illustration; our block-32 head with the top-16 / 127-node tree; the 6 rounds after the prompt were dumped on
   ). Round 0 is used: 21 tokens accepted, and at positions 11 and 15 the accepted token was the draft's second
   guess, so a chain that only ever follows the first guess would have stopped at 10. TREE / ROUND pick the round;
   `tree-gidx0.json` round 5 is the other readable one (24 accepted, a chain stops at 22, but positions 1-22 are a
   single file and only the last four fan out).
2. The 8 candidates of one text-to-SVG prompt with their rendered CLIP scores (`figures/qualitative/round2/cands_t2.json`,
   rendered on ), via the H4 data layer.
3. One real loop with the rule replayed on it: `runs/2026-11-05/mf_loop_example.json` (built by
   `runs/2026-11-05/mf_loop_example.py` from the same strict InternSVG-8B rows table 1 reports). The rule
   (sightline/process.py loop_fire / loop_rollback 'esc_stop', eval flags --loop-ban 128 --loop-repeats 3
   --loop-min-period 8) fires when the output ends with three exact copies of one P-token cycle, 8 <= P <= 128; the
   first firing bans the token that would start a fourth copy (escape), the next firing ends the output (stop).
   LOOP_GIDX picks the shown row: a 10-token cycle of path coordinates; without the rule that output runs to its
   4,000-token cap.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
TREE = RUNS / "2026-10-29/analysis/trees/tree-gidx150.json"
LOOPS = RUNS / "2026-11-05/mf_loop_example.json"
ROUND = 0
LOOP_GIDX = 863
sys.path.insert(0, str(ROOT / "paper/claims/H4-qualitative/round1"))
import h4data  # noqa: E402


def tree(round_i: int = ROUND) -> dict:
    """Node-link layout: x = position in the block (depth), y = a tidy order (children under their parent, by rank),
    centred per depth. Returns nodes [(depth, y, accepted, rank)], edges [((x0, y0), (x1, y1), accepted)], facts."""
    t = json.loads(TREE.read_text())
    assert t["design"] == "D1-top16-N127" and t["block_size"] == 32, t["design"]
    r = t["rounds"][round_i]
    nodes = r["nodes"]
    assert len(nodes) == 127
    by_depth = defaultdict(list)
    for n in nodes:
        by_depth[n["depth"]].append(n)
    ypos = {}
    for d in sorted(by_depth):
        col = sorted(by_depth[d], key=lambda n: (ypos.get(n["parent"], 0.0), n["rank"]))
        k = len(col)
        for j, n in enumerate(col):
            ypos[n["i"]] = j - (k - 1) / 2
    # keep the accepted path straight: shift every depth so its accepted node (if any) sits at y = 0
    acc_y = {n["depth"]: ypos[n["i"]] for n in nodes if n["accepted"]}
    shift = {d: -acc_y.get(d, acc_y[max(acc_y)] if d > max(acc_y) else 0.0) for d in by_depth}
    out_nodes = [(n["depth"], ypos[n["i"]] + shift[n["depth"]], n["accepted"], n["rank"]) for n in nodes]
    lo = min(y for _, y, _, _ in out_nodes)                 # the tidy layout is not symmetric about the
    hi = max(y for _, y, _, _ in out_nodes)                 # accepted path; keep it, but report the extent
    edges = []
    by_i = {n["i"]: n for n in nodes}
    for n in nodes:
        if n["parent"] >= 0:
            p = by_i[n["parent"]]
            edges.append(((p["depth"], ypos[p["i"]] + shift[p["depth"]]),
                          (n["depth"], ypos[n["i"]] + shift[n["depth"]]), n["accepted"] and p["accepted"]))
    acc = sorted((n for n in nodes if n["accepted"]), key=lambda n: n["depth"])
    assert len(acc) == r["n_accepted"] == 21
    chain = next(n["depth"] for n in acc if n["rank"] > 0) - 1          # the rank-0 chain stops here
    assert chain == 10
    facts = {"accepted": r["n_accepted"], "chain_stops": chain, "max_depth": max(by_depth), "y_lo": lo, "y_hi": hi,
             "widest": max(len(v) for v in by_depth.values()),
             "off_chain": [(n["depth"], n["rank"] + 1) for n in acc if n["rank"] > 0]}
    return {"nodes": out_nodes, "edges": edges, "facts": facts}


def loop() -> dict:
    d = json.loads(LOOPS.read_text())
    e = next(x for x in d if x["gidx"] == LOOP_GIDX)
    P, tok = e["period"], e["tokens"]
    assert e["copies"] == 3 and e["min_period"] == 8 and e["max_period"] == 128
    assert tok[:P] == tok[P:2 * P] == tok[2 * P:3 * P], "the three copies must be identical"
    assert tok[3 * P] == tok[0], "the next token would start a fourth copy"
    e["cycle"] = tok[:P]
    e["saved"] = e["n_tokens"] - e["fire_at"]
    return e


def candidates() -> dict:
    c = h4data.candidates_t2()[0]
    assert c["caption"] == "A mountain with a flag on top"
    return c


if __name__ == "__main__":
    tr = tree()
    print(tr["facts"])
    ys = [y for _, y, _, _ in tr["nodes"]]
    print("y range", min(ys), max(ys))
    c = candidates()
    print(c["caption"], [round(x["clip"], 3) for x in c["cands"]], c["t_sd8"])
    e = loop()
    print("loop", e["gidx"], "P", e["period"], "".join(e["cycle"]), "fires at", e["fire_at"],
          "of", e["n_tokens"], "saved", e["saved"])
