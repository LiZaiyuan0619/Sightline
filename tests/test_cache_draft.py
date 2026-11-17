"""CPU unit test for cache_draft.py with tiny random fp32 models (no weights needed).

1. Chunked-context alignment: replay the harness drafter (DFlashDrafter.propose +
   update_hidden semantics) with random accept lengths over the same teacher-forced features;
   at every block start it visits, its drafts must equal the cached top-1 — including the
   shortened blocks in the last 15 positions of a capped stream.
2. End to end: a real harness spec_generate (greedy, tiny target) — the cached top-1 at each
   visited block start must equal the drafts the loop proposed (context from block forwards
   there, from one teacher-forced forward in the cache; fp32, so equal).
3. Brute-force checks of the derived fields (real rank, 7x7 window, x/y grid, copy list,
   type masses) on random entries.

Usage (the dflash environment, CPU):  python test_cache_tiny.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline import cache_draft as cd
from sightline.svgtok.types import (ARC_END, ARC_START, CMD_CLOSE, CMD_MOVE, COLOR_END,  # noqa: E402
                          COLOR_START, COORD_START, EOS, GRID)

V = 197000


def tiny_models():
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config
    from dflash.model import DFlashDraftModel
    torch.manual_seed(0)
    tcfg = Qwen2Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=28,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                       tie_word_embeddings=False)
    target = Qwen2ForCausalLM(tcfg).eval()
    dcfg = Qwen3Config(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                       max_position_embeddings=4096, rope_theta=1e6,
                       layer_types=["full_attention"] * 2)
    dcfg.dflash_config = {"mask_token_id": 196900, "target_layer_ids": [1, 7, 13, 19, 25]}
    dcfg.block_size = 16
    dcfg.num_target_layers = 28
    head = DFlashDraftModel(dcfg).eval()
    return target, head


def svg_like(rng: random.Random, n: int) -> list[int]:
    out = []
    for _ in range(n):
        u = rng.random()
        if u < 0.6:
            out.append(COORD_START + rng.randrange(GRID * GRID) if rng.random() < 0.5
                       else (out[-1] if out and COORD_START <= out[-1] < COORD_START + GRID * GRID
                             else COORD_START + 5 * GRID + 7))
        elif u < 0.75:
            out.append(rng.randrange(CMD_MOVE, CMD_CLOSE + 1))
        elif u < 0.9:
            out.append(rng.randrange(COLOR_START, COLOR_END + 1))
        else:
            out.append(rng.randrange(0, 152000))
    return out


@torch.inference_mode()
def teacher_feats(target, head, prompt, tokens):
    from dflash.model import extract_context_feature
    ids = torch.tensor([prompt + tokens])
    out = target(ids, output_hidden_states=True, logits_to_keep=1)
    return extract_context_feature(out.hidden_states, head.target_layer_ids)


@torch.inference_mode()
def test_chunked(target, head, max_new: int, n_gen: int, seed: int):
    from sightline.spec import DFlashDrafter
    rng = random.Random(seed)
    prompt = svg_like(rng, 23)
    tokens = svg_like(rng, n_gen)
    arr = cd.cache_row(target, head, prompt, tokens, max_new=max_new)
    feats = teacher_feats(target, head, prompt, tokens)
    n_p = len(prompt)
    drafter = DFlashDrafter(head, target, 16)
    drafter.target_hidden = feats[:, :n_p]                    # = prefill_done
    t = visits = short = 0
    while t < n_gen - 1:
        vs = min(16, max_new - t)
        start = n_p + t
        drafts = drafter.propose(n_p + t + 1, start, tokens[t], vs, None, None)
        cached = arr["top_ids"][t, :vs - 1, 0].tolist()
        assert drafts == cached, (t, drafts[:4], cached[:4])
        assert int(arr["n_off"][t]) == vs - 1
        produced = rng.randint(1, min(vs, n_gen - 1 - t))
        drafter.target_hidden = feats[:, start:start + produced]   # = update_hidden(produced)
        t += produced
        visits += 1
        short += vs < 16
    return visits, short, arr, prompt, tokens


@torch.inference_mode()
def test_end_to_end(target, head):
    from sightline.spec import DFlashDrafter, spec_generate
    from sightline.verify import ChainParams
    rng = random.Random(7)
    prompt = svg_like(rng, 31)
    drafter = DFlashDrafter(head, target, 16)
    res = spec_generate(target, drafter, prompt, 48, [EOS],
                        ChainParams(greedy=True, repetition_penalty=1.05), block_size=16)
    arr = cd.cache_row(target, head, prompt, res.tokens, max_new=48)
    n_p = len(prompt)
    for blk in res.per_block:
        t = blk["start"] - n_p
        k = len(blk["drafts"])
        assert blk["drafts"] == arr["top_ids"][t, :k, 0].tolist(), (t, blk["drafts"][:4])
    return len(res.per_block), len(res.tokens)


@torch.inference_mode()
def test_fields(target, head, prompt, tokens, max_new, n_checks=60, seed=3):
    """Check fill_block's derived fields against brute force on the SAME log-probs it saw
    (a near-uniform random draft makes ranks hypersensitive to recomputation noise)."""
    seen = {}
    orig = cd.fill_block

    def spy(st, t, lp, realized, copy_ids, nb_dydx):
        seen[t] = (lp.clone(), realized.clone(), copy_ids.clone())
        return orig(st, t, lp, realized, copy_ids, nb_dydx)

    cd.fill_block = spy
    try:
        arr = cd.cache_row(target, head, prompt, tokens, max_new=max_new)
    finally:
        cd.fill_block = orig
    rng = random.Random(seed)
    n_gen = len(tokens)
    for _ in range(n_checks):
        t = rng.choice(sorted(seen))
        lp, realized, copy_ids = seen[t]
        k = int(arr["n_off"][t])
        assert k == lp.shape[0]
        j = rng.randrange(k)
        row = lp[j]
        top = row.topk(32)
        assert arr["top_ids"][t, j].tolist() == top.indices.tolist()
        if t + 1 + j < n_gen:
            y = tokens[t + 1 + j]
            assert int(realized[j]) == y
            assert int(arr["real_rank"][t, j]) == int((row > row[y]).sum())
            assert abs(float(arr["real_lp"][t, j]) - float(row[y])) < 1e-2 * max(1, abs(float(row[y])))
        else:
            assert int(arr["real_rank"][t, j]) == -1
        c = row[COORD_START:COORD_START + GRID * GRID]
        b = int(c.argmax())
        assert int(arr["coord_best"][t, j]) == COORD_START + b
        bx, by = b % GRID, b // GRID
        for q, (dy, dx) in enumerate([(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4)]):
            x, y2 = bx + dx, by + dy
            v = float(arr["nb_lp"][t, j, q])
            if 0 <= x < GRID and 0 <= y2 < GRID:
                assert abs(v - float(c[y2 * GRID + x])) < 2e-2, (q, v)
            else:
                assert math.isnan(v)
        c2 = c.view(GRID, GRID)
        xt = torch.logsumexp(c2, 0).topk(8).indices.tolist()
        yt = torch.logsumexp(c2, 1).topk(8).indices.tolist()
        assert arr["xy_x"][t, j].tolist() == xt and arr["xy_y"][t, j].tolist() == yt
        a, bb = rng.randrange(8), rng.randrange(8)
        assert abs(float(arr["xy_lp"][t, j, a, bb]) - float(c2[yt[a], xt[bb]])) < 2e-2
        rec = []
        for tok in tokens[:t + 1]:
            if COORD_START <= tok < COORD_START + GRID * GRID:
                if tok in rec:
                    rec.remove(tok)
                rec.append(tok)
        rec = rec[::-1][:16]
        got = [x for x in arr["copy_ids"][t].tolist() if x >= 0]
        assert got == rec, (t, got[:4], rec[:4])
        for q, tok in enumerate(rec):
            assert abs(float(arr["copy_lp"][t, j, q]) - float(row[tok])) < 2e-2
        masses = np.exp(arr["type_lse"][t, j].astype(np.float64))
        ref = [float(torch.logsumexp(c, 0).exp()),
               float(row[CMD_MOVE:CMD_CLOSE + 1].exp().sum()),
               float(row[COLOR_START:COLOR_END + 1].exp().sum()),
               float(row[ARC_START:ARC_END + 1].exp().sum()),
               float(row[EOS].exp())]
        assert np.allclose(masses[:5], ref, rtol=2e-2, atol=1e-6), (masses, ref)
        assert abs(masses.sum() - 1) < 2e-2, masses.sum()
    return len(seen)


def main():
    target, head = tiny_models()
    v1, s1, arr, prompt, tokens = test_chunked(target, head, max_new=200, n_gen=90, seed=1)
    print(f"[ok] chunked alignment, EOS-like stream: {v1} block starts")
    v2, s2, arr2, p2, t2 = test_chunked(target, head, max_new=40, n_gen=40, seed=2)
    assert s2 > 0
    print(f"[ok] chunked alignment, capped stream: {v2} block starts ({s2} shortened blocks)")
    nb, nt = test_end_to_end(target, head)
    print(f"[ok] end to end vs harness spec_generate: {nb} blocks, {nt} tokens")
    n1 = test_fields(target, head, prompt, tokens, max_new=200)
    n2 = test_fields(target, head, p2, t2, max_new=40, seed=4)
    assert n1 == len(tokens) - 1 and n2 == len(t2) - 1
    print("[ok] derived fields (rank, 7x7, x/y grid, copy list, type masses)")
    print("ALL_OK")


if __name__ == "__main__":
    main()
