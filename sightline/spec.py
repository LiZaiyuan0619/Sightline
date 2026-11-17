"""Speculative decoding loop for the strict verifier.

Structure mirrors the official dflash_generate (the dflash environment/.../dflash/model.py), with two
differences:
  1. verification uses sightline/verify.py: the full HF-order processing chain
     (repetition penalty -> temperature -> top-k -> top-p) with the correct history;
  2. every block produces a per-position acceptance log (for the three τ calibers).

Invariant: the target KV cache always holds the KVs of `committed[:-1]`; each block
forward feeds [committed[-1]] + drafts, exactly like the official implementation.

Losslessness: in greedy mode the emitted token at every position is the target chain's
argmax; in sampling mode it is the target chain's own draw (sample-and-compare). Either
way the emitted sequence follows the AR distribution exactly, for ANY drafter.

Drafters (pluggable):
  SelfDrafter(model):   target drafts its own greedy continuation (self-check 2; τ = K+1)
  RandomDrafter(...):   uniform random tokens (self-check 3; τ ~ 1)
  DFlashDrafter(...):   official DFlashDraftModel b16 (self-check 1 vs official; speed test)
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.process import apply_chain, draw
from sightline.verify import BlockResult, ChainParams, verify_block


@dataclass
class SpecResult:
    tokens: list[int]                 # generated tokens (prompt excluded)
    nfe: int                          # target forward passes (prefill + one per block)
    acceptance: list[int]             # committed tokens per block (produced, EOS-truncated)
    per_block: list[dict] = field(default_factory=list)
    stop_reason: str = "max_new_tokens"

    def tau(self, lo: int = 0, hi: int | None = None) -> float:
        """Acceptance length recomputed over generated positions [lo, hi).

        Three-caliber support: pass (0, None) for all tokens,
        (0, eos_pos) for EOS-only, (0, loop_onset) for truncated-at-loop. Position i
        counts as produced by the block that emitted it; a block is included when its
        first TOKEN lands inside [lo, hi)."""
        hi = hi if hi is not None else len(self.tokens)
        produced = 0
        blocks = 0
        pos = 0
        for acc, blk in zip(self.acceptance, self.per_block):
            start_tok = pos
            pos += acc
            if start_tok >= hi or pos <= lo:
                continue
            blocks += 1
            produced += max(0, min(pos, hi) - max(start_tok, lo))
        return produced / blocks if blocks else 0.0


def _chain_draw(logits, history, params, generator, extra_ban=None):
    processed = apply_chain(logits, history, penalty=params.repetition_penalty,
                            temperature_=params.temperature, top_k_=params.top_k,
                            top_p_=params.top_p, greedy=params.greedy,
                            loop_ban_=getattr(params, "loop_ban", 0),
                            loop_repeats_=getattr(params, "loop_repeats", 2),
                            loop_min_period_=getattr(params, "loop_min_period", 4),
                            loop_action_=getattr(params, "loop_action", "ban"),
                            extra_ban=extra_ban)
    return draw(processed, params.greedy, generator)


# ---------------------------------------------------------------- drafters

class SelfDrafter:
    """Drafts the target's own chain-processed continuation (greedy) from a cache clone."""
    needs_hidden = False

    def __init__(self, model):
        self.model = model

    def propose(self, cache, committed, K, params, generator):
        clone = copy.deepcopy(cache)
        dev = next(self.model.parameters()).device
        hist = list(committed)
        prev = committed[-1]
        drafts = []
        with torch.inference_mode():
            for _ in range(K):
                pos = torch.tensor([[len(hist) - 1]], device=dev)
                out = self.model(torch.tensor([[prev]], device=dev), position_ids=pos,
                                 past_key_values=clone, use_cache=True)
                t = int(_chain_draw(out.logits[0, -1], hist, params, generator))
                drafts.append(t)
                hist.append(t)
                prev = t
        return drafts


class RandomDrafter:
    needs_hidden = False

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def propose(self, cache, committed, K, params, generator):
        g = generator if generator is not None else torch.Generator(device="cpu")
        return torch.randint(0, self.vocab_size, (K,), generator=g).tolist()


class DFlashDrafter:
    """Official DFlash draft model wrapper (mirrors dflash.model.dflash_generate)."""
    needs_hidden = True

    def __init__(self, draft_model, target, block_size):
        self.model = draft_model
        self.target = target
        self.block_size = block_size
        self.cache = self._new_cache()
        self.target_hidden = None
        self.mask_token_id = draft_model.mask_token_id

    def _new_cache(self):
        from dflash.model import _make_cache
        return _make_cache(self.model.config)

    def prefill_done(self, hidden_states):
        from dflash.model import extract_context_feature
        self.target_hidden = extract_context_feature(hidden_states, self.model.target_layer_ids)

    def propose(self, committed_len, start, block_first, verify_size, params, generator):
        """Mirror official: block = [last_committed, mask...]; drafts = argmax(draft logits)."""
        from dflash.model import _raw_input_embeddings, _output_head, _crop_to
        dev = next(self.model.parameters()).device
        K = verify_size - 1
        block_ids = torch.tensor([[block_first] + [self.mask_token_id] * K], device=dev)
        position_ids = torch.arange(start - (self.target_hidden.shape[1]),
                                    start + verify_size, device=dev).unsqueeze(0)
        noise_embedding = _raw_input_embeddings(self.target, block_ids, 1.0)
        draft_hidden = self.model(
            target_hidden=self.target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )[:, 1 - verify_size:, :]
        _crop_to(self.cache, start)
        draft_logits = self.model.compute_logits(draft_hidden, _output_head(self.target))
        return torch.argmax(draft_logits, dim=-1)[0].tolist()

    def update_hidden(self, hidden_states, produced):
        from dflash.model import extract_context_feature
        self.target_hidden = extract_context_feature(
            hidden_states, self.model.target_layer_ids)[:, :produced, :]


# ---------------------------------------------------------------- main loop

@torch.inference_mode()
def spec_generate(model, drafter, prompt_ids, max_new_tokens, eos_ids, params: ChainParams,
                  generator: torch.Generator | None = None, block_size: int = 16,
                  stop_token_ids: list[int] | None = None,
                  on_block=None, coord_tol: int = 0) -> SpecResult:
    """Block-speculative generation loop. Returns tokens, NFE, per-block acceptance logs.

    on_block(block_info, raw_logits, result): optional analysis hook called after each
    block's verification (before commit/EOS truncation). block_info = {"start", "drafts",
    "block"}; raw_logits = the K+1 raw logit tensors of the block forward (raw_logits[j-1]
    predicts draft position j). Default None: no overhead, behavior unchanged."""
    if getattr(params, "loop_action", "ban") == "rollback":
        raise NotImplementedError("rollback escape is implemented in sightline.tree_spec / sightline.ar only")
    dev = next(model.parameters()).device
    stop_ids = stop_token_ids if stop_token_ids is not None else eos_ids
    n_prompt = len(prompt_ids)
    max_len = n_prompt + max_new_tokens

    prompt = torch.tensor([prompt_ids], device=dev)
    cache = DynamicCache()
    nfe = 0

    # prefill
    out = model(prompt, position_ids=torch.arange(n_prompt, device=dev).unsqueeze(0),
                past_key_values=cache, use_cache=True,
                output_hidden_states=drafter.needs_hidden)
    nfe += 1
    first = int(_chain_draw(out.logits[0, -1], list(prompt_ids), params, generator))
    if drafter.needs_hidden:
        drafter.prefill_done(out.hidden_states)

    committed = list(prompt_ids) + [first]
    acceptance: list[int] = []
    per_block: list[dict] = []
    stop_reason = "max_new_tokens"

    if first in stop_ids:                       # AR parity: stop immediately on EOS
        return SpecResult(tokens=[first], nfe=nfe, acceptance=[1],
                          per_block=[], stop_reason="eos")

    while len(committed) < max_len:
        verify_size = min(block_size, max_len - len(committed) + 1)  # slots incl. the anchor token
        start = len(committed) - 1                                   # global position of committed[-1]
        if drafter.needs_hidden:
            drafts = drafter.propose(len(committed), start, committed[-1], verify_size,
                                     params, generator)
        else:
            drafts = drafter.propose(cache, committed, verify_size - 1, params, generator)
        block = [committed[-1]] + drafts
        positions = torch.arange(start, start + len(block), device=dev).unsqueeze(0)
        out = model(torch.tensor([block], device=dev), position_ids=positions,
                    past_key_values=cache, use_cache=True,
                    output_hidden_states=drafter.needs_hidden)
        nfe += 1

        raw_logits = [out.logits[0, j] for j in range(len(block))]
        res: BlockResult = verify_block(drafts, raw_logits, committed, params, generator,
                                        coord_tol=coord_tol)
        if on_block is not None:
            on_block({"start": start, "drafts": drafts, "block": block}, raw_logits, res)

        # commit; truncate at the first stop token (AR parity: stop token included, then stop)
        emitted = res.emitted
        truncated = False
        for i, t in enumerate(emitted):
            if t in stop_ids:
                emitted = emitted[:i + 1]
                truncated = True
                break
        if len(committed) + len(emitted) > max_len:
            emitted = emitted[:max_len - len(committed)]
        committed += emitted
        produced = len(emitted)
        acceptance.append(produced)
        per_block.append({"start": start, "drafts": drafts, "emitted": emitted,
                          "n_accepted": res.n_accepted, "per_position": res.per_position})
        # crop cache to keep KVs of committed[:-1] (negative = remove N from the end)
        remove = cache.get_seq_length() - (len(committed) - 1)
        if remove:
            cache.crop(-remove)
        if drafter.needs_hidden:
            drafter.update_hidden(out.hidden_states, produced)
        if truncated:
            stop_reason = "eos"
            break
    return SpecResult(tokens=committed[n_prompt:], nfe=nfe, acceptance=acceptance,
                      per_block=per_block, stop_reason=stop_reason)
