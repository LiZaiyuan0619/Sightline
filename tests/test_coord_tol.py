"""CPU tests for the geometric tolerance (coord_tol) of sightline/verify.py and sightline/tree_spec.py.

1. verify_block, hand-made full-vocab logits: snaps within r (greedy and sampling), rejects
   beyond r, across a grid-row boundary (adjacent ids, distance 199), and for non-coordinates;
   coord_tol=0 gives the strict result.
2. tree_spec_generate on the tiny fp32 target of test_tree_spec.py with its scripted drafter
   (which puts an x-neighbour of the true coordinate on top 10% of the time), greedy:
   - coord_tol=0: output == AR (strict path unchanged);
   - coord_tol=r: snaps happen, and every emitted token equals the target's own greedy choice
     at its prefix or is a coordinate within r of it (teacher-forced replay of the output).

Usage (the dflash environment, CPU):  python sightline/test_coord_tol.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline import design as sim                                              # noqa: E402
from sightline import tree_spec as ts                                 # noqa: E402
from sightline.ar import ar_generate                                  # noqa: E402
from sightline.process import apply_chain                             # noqa: E402
from sightline.verify import ChainParams, coord_cheb, verify_block    # noqa: E402
from sightline.svgtok.types import CMD_LINE, CMD_MOVE, COORD_START, EOS, GRID  # noqa: E402
from test_tree_spec import ScriptedDrafter, tiny_target             # noqa: E402

V = 197000


def coord(x, y):
    return COORD_START + y * GRID + x


def peaked(tok, boost=30.0):
    logits = torch.zeros(V)
    logits[tok] = boost
    return logits


def check(name, ok):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    assert ok, name


def test_verify_block():
    g = ChainParams(greedy=True)
    c = coord(50, 60)
    bonus = coord(10, 10)
    r = verify_block([coord(51, 60)], [peaked(c), peaked(bonus)], [1, 2], g)
    check("strict: 1 px off is rejected, target emitted", r.n_accepted == 0 and r.emitted == [c])
    r0 = verify_block([coord(51, 60)], [peaked(c), peaked(bonus)], [1, 2], g, coord_tol=0)
    check("coord_tol=0 == strict", r0.emitted == r.emitted and r0.n_accepted == r.n_accepted)
    r = verify_block([coord(51, 59)], [peaked(c), peaked(bonus)], [1, 2], g, coord_tol=1)
    check("tol 1: diagonal neighbour snaps, drafted coord + bonus emitted",
          r.n_accepted == 1 and r.emitted == [coord(51, 59), bonus] and r.per_position[0]["snapped"])
    r = verify_block([coord(52, 60)], [peaked(c), peaked(bonus)], [1, 2], g, coord_tol=1)
    check("tol 1: 2 px off is rejected", r.n_accepted == 0 and r.emitted == [c])
    r = verify_block([coord(52, 60)], [peaked(c), peaked(bonus)], [1, 2], g, coord_tol=2)
    check("tol 2: 2 px off snaps", r.n_accepted == 1 and r.emitted[0] == coord(52, 60))
    a, b = coord(199, 10), coord(0, 11)
    check("row boundary: adjacent ids are 199 px apart", b - a == 1 and coord_cheb(a, b) == 199)
    r = verify_block([b], [peaked(a), peaked(bonus)], [1, 2], g, coord_tol=2)
    check("tol 2: row-boundary neighbour id is rejected", r.n_accepted == 0 and r.emitted == [a])
    r = verify_block([CMD_LINE], [peaked(CMD_MOVE), peaked(bonus)], [1, 2], g, coord_tol=4)
    check("tol 4: command mismatch is rejected", r.n_accepted == 0 and r.emitted == [CMD_MOVE])
    r = verify_block([coord(5, 5)], [peaked(EOS), peaked(bonus)], [1, 2], g, coord_tol=4)
    check("tol 4: coord draft vs EOS target is rejected", r.n_accepted == 0 and r.emitted == [EOS])
    # chain of snaps: the second position is judged on the history that holds the snapped token
    drafts = [coord(51, 60), coord(71, 80)]
    logits = [peaked(c), peaked(coord(70, 80)), peaked(bonus)]
    r = verify_block(drafts, logits, [1, 2], g, coord_tol=1)
    check("tol 1: two snaps in a row, both drafted coords committed",
          r.n_accepted == 2 and r.emitted == drafts + [bonus])
    s = ChainParams(greedy=False, temperature=1.0, top_k=50, top_p=0.9)
    gen = torch.Generator().manual_seed(0)
    r = verify_block([coord(51, 60)], [peaked(c), peaked(bonus)], [1, 2], s, gen, coord_tol=1)
    check("sampling, tol 1: the drawn coord snaps to the draft",
          r.n_accepted == 1 and r.emitted[0] == coord(51, 60) and r.per_position[0]["target"] == c)


def replay_greedy_ok(model, prompt, out, params, tol):
    """Every out[i] equals the target's greedy draw at prompt + out[:i], or is a coordinate
    within tol of it."""
    with torch.no_grad():
        logits = model(torch.tensor([prompt + out])).logits[0]
    n_p = len(prompt)
    bad = 0
    for i, t in enumerate(out):
        hist = prompt + out[:i]
        p = apply_chain(logits[n_p + i - 1], hist, penalty=params.repetition_penalty,
                        temperature_=params.temperature, top_k_=params.top_k, top_p_=params.top_p,
                        greedy=True)
        y = int(torch.argmax(p))
        if t != y:
            d = coord_cheb(t, y)
            if d is None or d > tol:
                bad += 1
    return bad


def test_tree_walk():
    model = tiny_target()
    params = ChainParams(greedy=True, repetition_penalty=1.05)
    designs = [sim.Design("D0-chain", 15), sim.Design("D1-top4-N63", 63, ("top4",))]
    rng = random.Random(7)
    total_snaps = 0
    for case in range(3):
        prompt = [rng.randrange(152000) for _ in range(rng.randint(5, 20))]
        ar = ar_generate(model, prompt, 60, [EOS], params).tokens
        for d in designs:
            res0 = ts.tree_spec_generate(model, ScriptedDrafter(ar, len(prompt), 11 + case), prompt,
                                         60, [EOS], params, d, coord_tol=0)
            check(f"case {case} {d.name}: coord_tol=0 output == AR", res0.tokens == ar)
            for tol in (1, 2):
                res = ts.tree_spec_generate(model, ScriptedDrafter(ar, len(prompt), 11 + case),
                                            prompt, 60, [EOS], params, d, coord_tol=tol)
                n_snap = sum(b["n_snap"] for b in res.per_block)
                total_snaps += n_snap
                bad = replay_greedy_ok(model, prompt, res.tokens, params, tol)
                check(f"case {case} {d.name} tol {tol}: {n_snap} snaps, every token is the "
                      f"target's choice or within {tol} px of it", bad == 0)
    check(f"snaps exercised ({total_snaps} in total)", total_snaps > 0)


def main():
    print("verify_block:")
    test_verify_block()
    print("tree_spec_generate (tiny fp32 target):")
    test_tree_walk()
    print("ALL_OK")


if __name__ == "__main__":
    main()
