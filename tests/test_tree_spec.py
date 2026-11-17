"""CPU test for sightline/tree_spec.py with a tiny random fp32 target and a scripted drafter.

The scripted drafter knows the target's own continuation and places the true next tokens at
rank 1, rank 2 or outside its candidates by a fixed rule, so trees accept multi-token paths.
Checks, greedy and seeded sampling:
  1. tree_spec_generate's output == the AR output (fp32: exact). Exercises the ancestor mask,
     packed positions, KV gathering down to the accepted path and the node-by-node walk;
  2. per-round accepted counts == simulate.walk on a cache built from the same draft
     distributions (the verifier and the simulator agree exactly);
  3. the chain design reproduces sightline.spec.spec_generate's acceptance with the same drafts.

Usage (the dflash environment, CPU):  python sightline/test_tree_spec.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline import design as sim                                              # noqa: E402
from sightline import tree_spec as ts                                 # noqa: E402
from sightline.ar import ar_generate                                  # noqa: E402
from sightline.verify import ChainParams                              # noqa: E402
from sightline.svgtok.types import COORD_START, EOS, GRID                     # noqa: E402

V = 197000


def tiny_target():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=False)
    m = Qwen2ForCausalLM(cfg).eval()
    # make coordinates likely so the spatial sources fire: boost the coordinate rows of lm_head
    with torch.no_grad():
        m.lm_head.weight[COORD_START:COORD_START + GRID * GRID] *= 3.0
    return m


class ScriptedDrafter(ts.TreeDrafter):
    """Draft log-probs built from the known continuation `truth` (generated part only)."""

    def __init__(self, truth: list[int], n_prompt: int, seed: int):
        self.truth, self.n_prompt, self.seed = truth, n_prompt, seed
        self.needs_hidden = True

    def prefill_done(self, hidden_states):
        pass

    def update_hidden_kept(self, hidden_states, keep):
        pass

    def rollback(self, new_start, old_start):
        pass

    def draft_logprobs(self, start: int, block_first: int, verify_size: int) -> torch.Tensor:
        t = start - self.n_prompt
        k = verify_size - 1
        logits = torch.full((k, V), -30.0)
        for j in range(k):
            r = random.Random(self.seed * 1000003 + t * 101 + j)
            idx = t + 1 + j
            y = self.truth[idx] if idx < len(self.truth) else EOS
            filler = [r.randrange(V) for _ in range(6)]
            u = r.random()
            if u < 0.55:
                order = [y] + filler
            elif u < 0.8:
                order = [filler[0], y] + filler[1:]
            elif u < 0.9 and COORD_START <= y < COORD_START + GRID * GRID:
                nb = y + 1 if (y - COORD_START) % GRID < GRID - 1 else y - 1   # x-neighbour on top
                order = [nb] + filler
            else:
                order = filler
            for q, tok in enumerate(order):
                logits[j, tok] = 2.0 - 1.1 * q + 0.01 * r.random()
        return torch.log_softmax(logits, -1)


def cache_from_drafter(drafter: ScriptedDrafter, prompt: list[int], tokens: list[int], max_new: int,
                       K: int = 15) -> dict:
    """What cache_draft would store, built from the scripted distributions."""
    n_p = len(prompt)
    T = len(tokens) - 1
    parts = []
    for t in range(T):
        vs = min(K + 1, max_new - t)
        lp = drafter.draft_logprobs(n_p + t, tokens[t], vs)
        committed = prompt + tokens[:t + 1]
        parts.append(ts.summarize_block(lp, ts.recent_coords(committed, n_p), K))
    row = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}
    row["tokens"] = np.asarray(tokens, dtype=np.int32)
    return row


def run_case(model, prompt, params, seed, max_new, designs):
    gen = torch.Generator().manual_seed(seed)
    ar = ar_generate(model, prompt, max_new, [EOS], params, generator=gen)
    truth = ar.tokens
    n_p = len(prompt)
    row = cache_from_drafter(ScriptedDrafter(truth, n_p, seed), prompt, truth, max_new)
    out = {}
    for d in designs:
        gen = torch.Generator().manual_seed(seed)
        res = ts.tree_spec_generate(model, ScriptedDrafter(truth, n_p, seed), prompt, max_new, [EOS],
                                    params, d, generator=gen)
        assert res.tokens == truth, (d.name, "output differs from AR",
                                     next(i for i, (a, b) in enumerate(zip(res.tokens, truth)) if a != b))
        simulated = sim.walk(row, d)
        assert res.acceptance == simulated, (d.name, res.acceptance[:12], simulated[:12])
        out[d.name] = sum(res.acceptance) / len(res.acceptance)
    return out, len(truth)


def test_copy_branch_merge():
    """add_copy_branch follows existing children that carry the cycle, then appends the rest."""
    from sightline.tree_utils import TreeNode
    prompt = [5, 6]
    committed = prompt + [9, 1, 2, 3, 1, 2, 3]              # generated part ends with 2 x (1 2 3)
    assert ts.detect_period(committed, len(prompt)) == 3
    nodes = [TreeNode(1, 1, -1, 0), TreeNode(7, 1, -1, 1), TreeNode(2, 2, 0, 0)]
    out, P, added = ts.add_copy_branch(nodes, committed, len(prompt), 8)
    assert P == 3 and added == 6, (P, added)              # 1 2 matched in the tree, 6 appended
    path = [n.token for n in out[3:]]
    assert path == [3, 1, 2, 3, 1, 2] and out[3].parent == 2 and out[3].depth == 3, out[3:]
    assert ts.add_copy_branch(nodes, prompt + [9, 1, 2, 3], len(prompt), 8)[1:] == (0, 0)
    print("[ok] add_copy_branch: trie merge, depths, no period -> no branch")


def copy_and_ban_cases(model):
    """Long period-copy branch and loop escape through the real tree decoder (fp32: exact)."""
    rng = random.Random(5)
    prompts = [[rng.randrange(152000) for _ in range(rng.randint(5, 20))] for _ in range(4)]
    designs = [sim.Design("D0-chain", 15), sim.Design("D1-top4-N63", 63, ("top4",))]
    long_rounds = 0
    for name, params in (
            ("greedy", ChainParams(greedy=True)),
            ("greedy+ban", ChainParams(greedy=True, loop_ban=64)),
            ("sample+ban", ChainParams(greedy=False, temperature=0.8, top_k=50, top_p=0.9, loop_ban=64)),
            ("greedy+ban3", ChainParams(greedy=True, loop_ban=128, loop_repeats=3, loop_min_period=2)),
            ("greedy+stop3", ChainParams(greedy=True, loop_ban=128, loop_repeats=3, loop_min_period=2,
                                         loop_action="eos"))):
        for ci, prompt in enumerate(prompts):
            seed = 21 + ci
            gen = torch.Generator().manual_seed(seed)
            truth = ar_generate(model, prompt, 150, [EOS], params, generator=gen).tokens
            n_p = len(prompt)
            row = cache_from_drafter(ScriptedDrafter(truth, n_p, seed), prompt, truth, 150)
            for d in designs:
                for C in (0, 48):
                    gen = torch.Generator().manual_seed(seed)
                    res = ts.tree_spec_generate(model, ScriptedDrafter(truth, n_p, seed), prompt, 150,
                                                [EOS], params, d, generator=gen, copy_long=C)
                    assert res.tokens == truth, (name, ci, d.name, C, "output differs from AR")
                    dc = sim.replace(d, copy_long=C)
                    assert res.acceptance == sim.walk(row, dc), (name, ci, d.name, C)
                    if C and max(res.acceptance) > 16:
                        long_rounds += 1
    assert long_rounds > 0, "the copy branch never ran past the block"
    print(f"[ok] copy_long + loop_ban: tree output == AR of the same (modified) target, verifier == "
          f"simulator; {long_rounds} runs had rounds longer than the block")


def rollback_cases(model):
    """ rollback escape (process.loop_rollback) through the real tree decoder: output == AR of
    the same modified target (fp32: exact), with and without the copy branch; rollbacks must occur."""
    rng = random.Random(7)
    prompts = [[rng.randrange(152000) for _ in range(rng.randint(5, 20))] for _ in range(6)]
    designs = [sim.Design("D0-chain", 15), sim.Design("D1-top4-N63", 63, ("top4",))]
    n_rb = n_runs = n_es = 0
    for name, params in (
            ("greedy+rb1", ChainParams(greedy=True, loop_ban=128, loop_repeats=3, loop_min_period=2,
                                       loop_action="rollback", loop_max_rb=1)),
            ("greedy+rb3", ChainParams(greedy=True, loop_ban=128, loop_repeats=3, loop_min_period=2,
                                       loop_action="rollback", loop_max_rb=3)),
            ("sample+rb2", ChainParams(greedy=False, temperature=0.8, top_k=50, top_p=0.9, loop_ban=64,
                                       loop_repeats=2, loop_min_period=2, loop_action="rollback",
                                       loop_max_rb=2)),
            ("greedy+escstop1", ChainParams(greedy=True, loop_ban=128, loop_repeats=3, loop_min_period=2,
                                            loop_action="esc_stop", loop_max_rb=1)),
            ("sample+escstop2", ChainParams(greedy=False, temperature=0.8, top_k=50, top_p=0.9, loop_ban=64,
                                            loop_repeats=2, loop_min_period=2, loop_action="esc_stop",
                                            loop_max_rb=2))):
        for ci, prompt in enumerate(prompts):
            seed = 31 + ci
            gen = torch.Generator().manual_seed(seed)
            truth = ar_generate(model, prompt, 150, [EOS], params, generator=gen).tokens
            n_p = len(prompt)
            for d in designs:
                for C in (0, 48):
                    gen = torch.Generator().manual_seed(seed)
                    res = ts.tree_spec_generate(model, ScriptedDrafter(truth, n_p, seed), prompt, 150,
                                                [EOS], params, d, generator=gen, copy_long=C)
                    assert res.tokens == truth, (name, ci, d.name, C, "output differs from AR")
                    n_rb += sum(1 for b in res.per_block if b.get("rollback"))
                    n_runs += 1
                    if name.startswith(("greedy+escstop", "sample+escstop")) and d.name == "D0-chain" and C == 0:
                        n_es += truth[-1] == EOS and len(truth) < 150
    assert n_rb > 0, "no rollback happened in the test cases"
    assert n_es > 0, "escape-then-stop never ended a stream in the test cases"
    print(f"[ok] rollback escape: tree output == AR of the same modified target in {n_runs} runs "
          f"({n_rb} rollbacks; escape-then-stop ended {n_es} streams)")


def main():
    test_copy_branch_merge()
    model = tiny_target()
    rollback_cases(model)
    copy_and_ban_cases(model)
    designs = [sim.Design("D0-chain", 15),
               sim.Design("D1-top2-N31", 31, ("top2",)),
               sim.Design("D1-top4-N63", 63, ("top4",)),
               sim.Design("D3-top2+nb1-N63", 63, ("top2", "nb1")),
               sim.Design("D9-top4+copy-N31", 31, ("top4", "copy"))]
    rng = random.Random(5)
    for mode, params in (("greedy", ChainParams(greedy=True, repetition_penalty=1.05)),
                         ("sample", ChainParams(greedy=False, temperature=0.8, top_k=50, top_p=0.9,
                                                repetition_penalty=1.05))):
        for case in range(3):
            prompt = [rng.randrange(152000) for _ in range(rng.randint(5, 20))]
            taus, n = run_case(model, prompt, params, seed=11 + case, max_new=60, designs=designs)
            print(f"[ok] {mode} case {case}: {n} tokens, output == AR, verifier == simulator; tau "
                  + ", ".join(f"{k} {v:.2f}" for k, v in taus.items()))
    print("ALL_OK")


if __name__ == "__main__":
    main()
