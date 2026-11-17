"""CPU unit tests for simulate.py (no GPU, no cache needed).

1. tree_accept (lazy best-first growth) == glance.tree.build_budget_tree + walk, on random
   uniform-width candidate lists with continuous scores (no ties), many budgets/widths.
2. walk / accept_length on a synthetic row: chain acceptance and per-round clipping at the
   stream end; a rank-2 realized token is rescued by top-2 trees only with enough budget.
3. records: the trunc/loop split matches eval_b3.spec_records' rule.

Usage: python3 test_simulate.py   (any env with numpy)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline import design as sim                                               # noqa: E402
from sightline.tree_utils import build_budget_tree, children_of               # noqa: E402

COORD = sim.COORD_START


def ref_accept(ids, logp, budget, realized):
    nodes = build_budget_tree(ids, logp, budget)
    ch = children_of(nodes)
    cur, depth = -1, 0
    while depth < len(realized):
        nxt = [c for c in ch[cur] if nodes[c].token == realized[depth]]
        if not nxt:
            break
        cur = nxt[0]
        depth += 1
    return depth


def test_tree_vs_glance(n_cases=3000, seed=0):
    rng = random.Random(seed)
    for case in range(n_cases):
        depth = rng.randint(1, 15)
        width = rng.randint(1, 8)
        budget = rng.choice([1, 2, 3, 7, 15, 31, 63, 127])
        ids, logp = [], []
        for _ in range(depth):
            toks = rng.sample(range(50), width)
            sc = sorted((-rng.expovariate(1.0) for _ in range(width)), reverse=True)
            ids.append(toks)
            logp.append(sc)
        # realized path: follow candidates with prob 0.7 per step, else an outside token
        realized = []
        for d in range(depth):
            realized.append(rng.choice(ids[d]) if rng.random() < 0.7 else 99)
        cands = [(ids[d], logp[d]) for d in range(depth)]
        got = sim.tree_accept(cands, budget, realized)
        want = ref_accept(ids, logp, budget, realized)
        assert got == want, (case, got, want, budget, width, depth)
        # grow_tree: same node set as glance, parents first, and walking it gives the same depth
        nodes = sim.grow_tree(cands, budget)
        gl = build_budget_tree(ids, logp, budget)
        assert len(nodes) == len(gl)
        def paths(ns, tok_of, parent_of):
            out = set()
            for i in range(len(ns)):
                pth, c = [], i
                while c != -1:
                    pth.append(tok_of(ns[c])); c = parent_of(ns[c])
                out.add(tuple(reversed(pth)))
            return out
        assert paths(nodes, lambda n: n[0], lambda n: n[2]) == paths(gl, lambda n: n.token, lambda n: n.parent)
        assert all(n[2] < i for i, n in enumerate(nodes))
        cur, dpt = -1, 0
        while dpt < len(realized):
            nxt = [i for i, n in enumerate(nodes) if n[2] == cur and n[0] == realized[dpt]]
            if not nxt:
                break
            cur, dpt = nxt[0], dpt + 1
        assert dpt == got, (case, dpt, got)
    print(f"[ok] tree_accept / grow_tree == glance build_budget_tree on {n_cases} random cases")


def synthetic_row(tokens, wrong_at, rank2_at=(), n_off=15, onset=None, eos=True):
    """Cache-like row: top-1 = realized token except at (t, j) in wrong_at; at (t, j) in
    rank2_at the realized token sits at rank 2 (top-2 rescues it)."""
    T, K = len(tokens) - 1, 15
    top_ids = np.full((T, K, 32), 7, dtype=np.int32)
    top_lp = np.tile(np.linspace(-0.05, -9, 32, dtype=np.float16), (T, K, 1))
    for t in range(T):
        for j in range(K):
            y = tokens[t + 1 + j] if t + 1 + j < len(tokens) else 5
            top_ids[t, j, 0] = y
            top_ids[t, j, 1] = 11
            if (t, j) in wrong_at:
                top_ids[t, j, 0] = 13
            if (t, j) in rank2_at:
                top_ids[t, j, 0] = 13
                top_ids[t, j, 1] = y
    row = {"tokens": np.asarray(tokens, dtype=np.int32),
           "n_off": np.full(T, n_off, dtype=np.int8), "top_ids": top_ids, "top_lp": top_lp,
           "onset": onset, "ref_eos": eos}
    return row


def ref_walk(row, b, budget):
    """Reference walk: every round builds glance's budget tree over the per-offset top-b."""
    tokens = [int(x) for x in row["tokens"]]
    out, t = [], 0
    while t < len(tokens) - 1:
        k = int(row["n_off"][t])
        ids = [[int(x) for x in row["top_ids"][t, j, :b]] for j in range(k)]
        lp = [[float(x) for x in row["top_lp"][t, j, :b]] for j in range(k)]
        realized = tokens[t + 1:t + 1 + k]
        L = ref_accept(ids, lp, budget, realized) if b > 1 else next(
            (j for j in range(len(realized)) if ids[j][0] != realized[j]), len(realized))
        e = min(L + 1, len(tokens) - 1 - t)
        out.append(e)
        t += e
    return out


def test_walk():
    toks = list(range(1000, 1040))                 # 40 tokens, all distinct
    # chain: wrong at (0, 3) -> round 0 accepts 3 drafts, commits 4; next block at t=4 is clean
    row = synthetic_row(toks, wrong_at={(0, 3)})
    acc = sim.walk(row, sim.Design("D0-chain", 15))
    assert acc[0] == 4, acc
    assert acc[1] == 16, acc                         # full block: 15 drafts + bonus
    assert sum(acc) == len(toks) - 1, (sum(acc), acc)
    assert acc == ref_walk(row, 1, 15)
    # rank-2 rescue: the realized token sits at rank 2 at (0, 3) and (4, 0)
    row = synthetic_row(toks, wrong_at={(20, 5)}, rank2_at={(0, 3), (4, 0)})
    chain = sim.walk(row, sim.Design("D0-chain", 15))
    for budget in (4, 15, 31, 63, 127):
        got = sim.walk(row, sim.Design("x", budget, ("top2",)))
        assert got == ref_walk(row, 2, budget), (budget, got[:4])
        assert sum(got) == len(toks) - 1
    big = sim.walk(row, sim.Design("y", 127, ("top2",)))
    assert big[0] > chain[0], (big[:2], chain[:2])
    print(f"[ok] walk vs glance reference walk; chain {chain[:3]} -> top2/N127 {big[:3]}")


def test_records():
    acc = [4, 16, 16, 3]                              # rounds start at t = 0, 4, 20, 36
    cs = sim.records(acc, onset=21, eos=False)
    # rounds' first committed positions: 1, 5, 21, 37 -> two before the onset
    assert cs.trunc.steps == 2 and cs.trunc.emitted == 4 + 16, cs.trunc
    assert cs.loop.steps == 2 and cs.loop.emitted == 16 + 3, cs.loop
    assert sim.records(acc, onset=2, eos=False).trunc is None    # onset <= MIN_HEALTHY_ONSET
    print("[ok] records(): trunc/loop split")


if __name__ == "__main__":
    test_tree_vs_glance()
    test_walk()
    test_records()
    print("ALL_OK")
