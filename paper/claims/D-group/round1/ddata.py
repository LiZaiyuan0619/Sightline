"""Data for the design-choice row (D1 recipe, D2 block length, D3 tree), round 1 of the merged figure, night.

Read-only views of the batch-4 data pack plus the tree-budget matrices; every block asserts a number that is
already published somewhere (guide SS8: a new reader must reproduce a printed number before it draws).

  D1  runs/2026-11-04/figdata_b4/d1_ladder.json   the 2 x 2 (block 16 / 32 x chain / top-16 tree), loop-free tokens per round
  D2  runs/2026-11-04/figdata_b4/d2_block.json    block 16 / 32 / 64, all-token / loop-free / loop-only tokens per round
  D3  runs/2026-10-31/analysis/tree_budget/tree_budget.json
      node spend and accepted draft tokens per (position in the block, sibling rank), ours vs DDTree, both at
      127 nodes per round, OmniSVG-8B text-to-SVG, loop-free positions only
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import numpy as np
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[4]
_B4 = RUNS / "2026-11-04/figdata_b4"

# ---------------------------------------------------------------------------------------------- D1
_d1 = json.loads((_B4 / "d1_ladder.json").read_text())["blocks"]
BLOCKS = ["OmniSVG-8B, text-to-SVG", "InternSVG-8B, text-to-SVG", "OmniSVG-8B, image-to-SVG"]
SHORT = {"OmniSVG-8B, text-to-SVG": "OmniSVG-8B text", "InternSVG-8B, text-to-SVG": "InternSVG-8B text",
         "OmniSVG-8B, image-to-SVG": "OmniSVG-8B image"}
ARMS = ["16-chain", "16-tree", "32-chain", "32-tree"]


def tau1(b, arm):
    """Loop-free tokens accepted per round (the caliber for ablations)."""
    return _d1[b][arm]["tau_healthy"]


def gain1(b, arm):
    return tau1(b, arm) / tau1(b, "16-chain")


def d1_summary():
    g = {b: {a: gain1(b, a) for a in ARMS} for b in BLOCKS}
    for b in BLOCKS:
        g[b]["product"] = g[b]["16-tree"] * g[b]["32-chain"]
        g[b]["block_on_tree"] = g[b]["32-tree"] / g[b]["16-tree"]
        g[b]["tree_on_long"] = g[b]["32-tree"] / g[b]["32-chain"]
        g[b]["compound"] = g[b]["32-tree"] / g[b]["product"]
    return g


G1 = d1_summary()
_o = G1["OmniSVG-8B, text-to-SVG"]
assert (round(_o["16-tree"], 2), round(_o["32-chain"], 2), round(_o["32-tree"], 2)) == (1.37, 1.13, 1.70), _o
assert all(G1[b]["32-tree"] > G1[b]["product"] for b in BLOCKS)          # together > product, every block
assert [round(G1[b]["32-tree"], 2) for b in BLOCKS] == [1.70, 1.73, 1.67]


def rng(key, fmt="%.2f"):
    vals = [G1[b][key] for b in BLOCKS]
    return (fmt % min(vals)) + "–" + (fmt % max(vals))


# ---------------------------------------------------------------------------------------------- D2
_d2 = json.loads((_B4 / "d2_block.json").read_text())["cells"]
D2_CELL = {16: "16|6ep|tree", 32: "32|3ep|tree", 64: "64|3ep|tree"}   # as round 1 / round 2 (b2_merged)
LENGTHS = [16, 32, 64]
MODES = ["greedy", "sample"]


def tau2(L, mode, kind):
    """kind: 'all' (table 1's count), 'loop-free' (loop-free), 'loop' (inside repetition loops only)."""
    return _d2[f"{D2_CELL[L]}|{mode}"]["tau_" + kind]


def share_loop2(L, mode):
    return _d2[f"{D2_CELL[L]}|{mode}"]["loop_token_share"]


assert round(tau2(32, "greedy", "healthy"), 2) == 12.36 and round(tau2(64, "greedy", "healthy"), 2) == 12.24
assert round(tau2(32, "greedy", "all"), 2) == 16.17 and round(tau2(64, "greedy", "all"), 2) == 17.01
assert round(tau2(32, "greedy", "loop"), 2) == 24.06 and round(tau2(64, "greedy", "loop"), 2) == 30.92

# ---------------------------------------------------------------------------------------------- D3
_tb = json.loads((RUNS / "2026-10-31/analysis/tree_budget/tree_budget.json").read_text())
RANK_NAMES = ["top-1", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th+"]


def tree_mats(method):
    """(nodes, accepted) per round, each an array [rank, position-1]; method 'ours' or 'ddtree'."""
    v = _tb[method]
    sp, ac = v["spend"], v["accepted"]
    depth = max(int(k.split(",")[0]) for k in sp) + 1
    ranks = max(int(k.split(",")[1]) for k in sp) + 1
    S, A = np.zeros((ranks, depth)), np.zeros((ranks, depth))
    for k, x in sp.items():
        d, r = map(int, k.split(","))
        S[r, d] = x
    for k, x in ac.items():
        d, r = map(int, k.split(","))
        A[r, d] = x
    return S / v["rounds"], A / v["rounds"]


S_OURS, A_OURS = tree_mats("ours")
S_DDT, A_DDT = tree_mats("ddtree")
assert abs(S_OURS.sum() - 127) < 1e-6 and abs(S_DDT.sum() - 127) < 1e-6            # same budget
assert round(A_OURS.sum(), 2) == 14.47 and round(A_DDT.sum(), 2) == 9.68
assert round(100 * S_OURS[0].sum() / 127, 1) == 69.3 and round(100 * A_OURS[0].sum() / A_OURS.sum(), 1) == 95.7
assert S_OURS.shape[1] == 31 and S_DDT.shape[1] == 15                              # positions 1..31 / 1..15
BEYOND16 = A_OURS[:, 15:].sum() / A_OURS.sum()       # accepted at positions 16..31: a 16-block cannot draft them
assert round(BEYOND16, 3) == 0.374, BEYOND16

if __name__ == "__main__":
    for b in BLOCKS:
        print(b, {k: round(v, 3) for k, v in G1[b].items()})
    for L in LENGTHS:
        print(L, [(m, round(tau2(L, m, "all"), 2), round(tau2(L, m, "healthy"), 2), round(tau2(L, m, "loop"), 2))
                  for m in MODES])
    print("ours nodes / accepted per round", S_OURS.sum(), A_OURS.sum(), "beyond 16:", round(BEYOND16, 3))
    print("ddtree nodes / accepted per round", S_DDT.sum(), A_DDT.sum())
    print("yield ours", (A_OURS.sum(0) / S_OURS.sum(0)).round(3))
    print("yield ddt ", (A_DDT.sum(0) / S_DDT.sum(0)).round(3))
    print("nodes/pos ours", S_OURS.sum(0).round(2))
    print("nodes/pos ddt ", S_DDT.sum(0).round(2))
