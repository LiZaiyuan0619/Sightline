""" evening (C2): stop at the first "</svg>" (the official generate_im2svg stopping criterion) in both arms of the
StarVector evaluation, as copies of the two decode loops (the shared library is not edited):
  - ar_generate = vlm.py vlm_ar_generate + the stop;
  - tree_spec_generate = sightline/tree_spec.py tree_spec_generate + the stop: a round's emitted tokens are cut at the
    token that completes "</svg>", exactly as at an EOS.
Why: on 2,000 StarVector-8B greedy outputs ($SIGHTLINE_RUNS/starvec/gen-8b icon units 0 and 9), 17% close </svg> and run on to the
4,000 cap; tokens after the first </svg> = 27% of all generated tokens (median tail 8, p90 2,713). 1B: tail <= 1 token.
"</svg>" is found in the decoded stream (per-token decode, concatenated, as build_sv_trainset.end_index),
since its tokenization depends on context. Both arms stop at the same token, so lossless outputs stay identical; the
stop is reported as stop_reason "eos" (a finished SVG), so the records treat it like an EOS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.ar import GenResult, _pick                                   # noqa: E402
from sightline.process import draw, eos_only, loop_rollback                 # noqa: E402
from sightline.spec import SpecResult, _chain_draw                          # noqa: E402
from sightline.tree_spec import (StreamPositions, add_copy_branch, build_tree, children_of,  # noqa: E402
                               coord_cheb, gather_cache, tree_mask)

CLOSE = "</svg>"
WINDOW = 16            # tokens decoded around a candidate; "</svg>" has 6 characters


class SvgStop:
    """completes(seq, n_prompt): does seq's last token complete the first "</svg>" of the response seq[n_prompt:]?
    Called after every committed token, so the first completion is the one that fires."""

    def __init__(self, which: str):
        from tokenizers import Tokenizer
        from starvec.build_sv_trainset import TOK
        t = Tokenizer.from_file(TOK[which])
        n = t.get_vocab_size(with_added_tokens=True)
        self.piece = [t.decode([i], skip_special_tokens=False) for i in range(n)]
        self.gt = frozenset(i for i, p in enumerate(self.piece) if ">" in p)

    def completes(self, seq: list[int], n_prompt: int) -> bool:
        if seq[-1] not in self.gt:                       # the completing token holds the final ">"
            return False
        tail = [self.piece[x] if x < len(self.piece) else "" for x in seq[max(n_prompt, len(seq) - WINDOW):]]
        text = "".join(tail)
        k = text.find(CLOSE, max(0, len(text) - len(tail[-1]) - len(CLOSE) + 1))
        return k >= 0 and k + len(CLOSE) > len(text) - len(tail[-1])

    def cut(self, tokens: list[int]) -> int | None:
        """index of the first token of a response that completes "</svg>" (None: never closed). The prompt ends with
        "<svg", which cannot start a "</svg>", so the response alone decides."""
        for i in range(len(tokens)):
            if self.completes(tokens[:i + 1], 0):
                return i
        return None


def ar_generate(model, inp, max_new_tokens: int, eos_ids: list[int], params, generator=None,
                stop: SvgStop | None = None) -> GenResult:
    """vlm.py vlm_ar_generate + the "</svg>" stop (stop=None: the original loop)."""
    from transformers import DynamicCache
    dev = next(model.parameters()).device
    ids = torch.tensor([inp.prompt_ids], device=dev)
    cache = DynamicCache()
    out = model(ids, past_key_values=cache, use_cache=True, **inp.prefill_kwargs())
    nfe = 1
    history = list(inp.prompt_ids)
    n_p = len(inp.prompt_ids)
    logits = out.logits[0, -1]
    n_rb, ban = 0, None
    reason = "max_new_tokens"
    while True:
        act, new_len, btok = loop_rollback(history, params, n_rb, n_p)
        if act == "rollback":
            history = history[:new_len]
            cache.crop(new_len - 1)
            pos = torch.tensor([[new_len - 1]], device=dev)
            out = model(torch.tensor([[history[-1]]], device=dev), position_ids=inp(pos),
                        past_key_values=cache, use_cache=True)
            nfe += 1
            logits = out.logits[0, -1]
            n_rb, ban = n_rb + 1, (new_len, btok)
            continue
        if act == "eos":
            nxt = int(draw(eos_only(logits.float(), eos_ids[0]), params.greedy, generator))
        elif act == "ban":
            nxt = int(_pick(logits, history, params, generator, extra_ban=btok))
            n_rb += 1
        else:
            eb = ban[1] if ban is not None and ban[0] == len(history) else None
            nxt = int(_pick(logits, history, params, generator, extra_ban=eb))
        history.append(nxt)
        if nxt in eos_ids or (stop is not None and stop.completes(history, n_p)):
            reason = "eos"
            break
        if len(history) - n_p >= max_new_tokens:
            break
        pos = torch.tensor([[len(history) - 1]], device=dev)
        out = model(torch.tensor([[nxt]], device=dev), position_ids=inp(pos),
                    past_key_values=cache, use_cache=True)
        nfe += 1
        logits = out.logits[0, -1]
    return GenResult(tokens=history[n_p:], nfe=nfe, stop_reason=reason)


def tree_spec_generate(model, drafter, prompt_ids: list[int], max_new_tokens: int,
                       eos_ids: list[int], params, design,
                       generator: torch.Generator | None = None, block_size: int = 16,
                       positions=None, prefill_kwargs: dict | None = None,
                       coord_tol: int = 0, copy_long: int = 0, stop: SvgStop | None = None) -> SpecResult:
    """sightline/tree_spec.py tree_spec_generate + the "</svg>" stop (stop=None: the original loop)."""
    from transformers import DynamicCache
    positions = positions or StreamPositions()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_prompt = len(prompt_ids)
    max_len = n_prompt + max_new_tokens
    K = block_size - 1
    cache = DynamicCache()
    nfe = 0
    prompt = torch.tensor([prompt_ids], device=dev)
    kw = prefill_kwargs if prefill_kwargs is not None else positions.prefill_kwargs(n_prompt, dev)
    out = model(prompt, past_key_values=cache, use_cache=True, output_hidden_states=True, **kw)
    nfe += 1
    first = int(_chain_draw(out.logits[0, -1], list(prompt_ids), params, generator))
    drafter.prefill_done(out.hidden_states)
    committed = list(prompt_ids) + [first]
    acceptance: list[int] = []
    per_block: list[dict] = []
    stop_reason = "max_new_tokens"
    if first in eos_ids or (stop is not None and stop.completes(committed, n_prompt)):
        return SpecResult(tokens=[first], nfe=nfe, acceptance=[1], per_block=[], stop_reason="eos")
    n_rb, ban = 0, None

    while len(committed) < max_len:
        verify_size = min(block_size, max_len - len(committed) + 1)
        start = len(committed) - 1
        if hasattr(drafter, "propose_tree"):
            nodes = drafter.propose_tree(committed, start, verify_size - 1)
        else:
            lp = drafter.draft_logprobs(start, committed[-1], verify_size)
            nodes = build_tree(design, lp, committed, n_prompt, K)
        copy_P = n_copy = 0
        if copy_long:
            nodes, copy_P, n_copy = add_copy_branch(nodes, committed, n_prompt,
                                                    min(copy_long, max_len - len(committed)))
        packed = torch.tensor([[committed[-1]] + [n.token for n in nodes]], device=dev)
        packed_pos = torch.tensor([[start] + [start + n.depth for n in nodes]], device=dev)
        n_prefix = cache.get_seq_length()
        out = model(packed, position_ids=positions(packed_pos), past_key_values=cache,
                    use_cache=True, attention_mask=tree_mask(n_prefix, nodes, dev, dtype),
                    output_hidden_states=True)
        nfe += 1
        logits = out.logits[0]
        children = children_of(nodes)
        history = list(committed)
        path: list[int] = []
        emitted: list[int] = []
        cursor = -1
        n_snap = 0
        rollback = None
        while True:
            pos_logits = logits[0 if cursor == -1 else cursor + 1]
            act, new_len, btok = loop_rollback(history, params, n_rb, n_prompt)
            if act == "rollback":
                rollback = (len(history), new_len, btok)
                break
            if act == "eos":
                y = int(draw(eos_only(pos_logits.float(), eos_ids[0]), params.greedy, generator))
            elif act == "ban":
                y = int(_chain_draw(pos_logits, history, params, generator, extra_ban=btok))
                n_rb += 1
            else:
                eb = ban[1] if ban is not None and ban[0] == len(history) else None
                y = int(_chain_draw(pos_logits, history, params, generator, extra_ban=eb))
            emitted.append(y)
            kids = children.get(cursor, [])
            hit = next((c for c in kids if nodes[c].token == y), None)
            if hit is None and coord_tol > 0:
                near = [(coord_cheb(nodes[c].token, y), k, c) for k, c in enumerate(kids)]
                near = [t for t in near if t[0] is not None and t[0] <= coord_tol]
                if near:
                    hit = min(near)[2]
                    emitted[-1] = nodes[hit].token
                    n_snap += 1
            if hit is None:
                break
            path.append(hit)
            history.append(nodes[hit].token)
            cursor = hit
        n_acc = len(path)
        truncated = False
        seq = list(committed)
        for i, t in enumerate(emitted):
            seq.append(t)
            if t in eos_ids or (stop is not None and stop.completes(seq, n_prompt)):
                emitted = emitted[:i + 1]
                truncated = True
                break
        if len(committed) + len(emitted) > max_len:
            emitted = emitted[:max_len - len(committed)]
        committed += emitted
        acceptance.append(len(emitted))
        per_block.append({"start": start, "n_nodes": len(nodes), "n_accepted": n_acc,
                          "emitted": emitted, "n_snap": n_snap, "copy_P": copy_P,
                          "n_copy": n_copy,
                          "rollback": list(rollback[:2]) if rollback and not truncated else None})
        if truncated or len(committed) >= max_len:
            stop_reason = "eos" if truncated else "max_new_tokens"
            break
        keep = [0] + [i + 1 for i in path]
        gather_cache(cache, n_prefix, keep)
        drafter.update_hidden_kept(out.hidden_states, keep)
        if rollback is not None:
            _, new_len, btok = rollback
            committed = committed[:new_len]
            cache.crop(new_len - 1)
            drafter.rollback(new_len - 1, start)
            n_rb, ban = n_rb + 1, (new_len, btok)
    return SpecResult(tokens=committed[n_prompt:], nfe=nfe, acceptance=acceptance,
                      per_block=per_block, stop_reason=stop_reason)
