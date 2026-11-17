"""Same-stack AR loop for the strict verifier.

- batch 1, KV cache (DynamicCache), one target forward per emitted token
- the SAME processing chain as the verifier (sightline/process.py): repetition penalty
  applied with history = prompt + generated-so-far (HF semantics), then warpers
- counts NFE (target forward passes) — this is the speedup denominator
- greedy and sampling modes; sampling takes an explicit torch.Generator for determinism

This is the correctness reference: speculative output must equal this loop token-by-token
(in greedy mode) for the "strictly lossless" claim.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.process import apply_chain, draw, eos_only, loop_rollback


@dataclass
class GenResult:
    tokens: list[int]          # newly generated tokens (prompt excluded)
    nfe: int                   # target forward passes used
    stop_reason: str           # "eos" | "max_new_tokens"


@dataclass
class ChainParams:
    greedy: bool = True
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    loop_ban: int = 0               #  loop escape (process.loop_ban): max period, 0 = off
    loop_repeats: int = 2
    loop_min_period: int = 4
    loop_action: str = "ban"        # "ban" the continuation | "eos" stop at the detected loop
                                    # | "rollback" / "esc_stop" (process.loop_rollback; decoder-level)
    loop_max_rb: int = 1            # rollbacks / escapes before the next firing stops with EOS


def load_target(model_path: str, device: str = "cuda:0"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                 device_map=device)
    model.eval()
    return model


@torch.inference_mode()
def ar_generate(model, input_ids: list[int], max_new_tokens: int, eos_token_ids: list[int],
                params: ChainParams, generator: torch.Generator | None = None) -> GenResult:
    """Greedy/sampling AR generation, one forward per token, NFE counted."""
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    cache = DynamicCache()
    nfe = 0

    out = model(ids, past_key_values=cache, use_cache=True)
    nfe += 1
    n_p = len(input_ids)
    history = list(input_ids)
    logits = out.logits[0, -1]
    n_rb, ban = 0, None                  # rollback escape state (process.loop_rollback)
    stop = "max_new_tokens"
    while True:
        act, new_len, btok = loop_rollback(history, params, n_rb, n_p)
        if act == "rollback":            # keep one cycle, recompute the logits at the new end
            history = history[:new_len]
            cache.crop(new_len - 1)
            inp = torch.tensor([[history[-1]]], dtype=torch.long, device=device)
            out = model(inp, past_key_values=cache, use_cache=True)
            nfe += 1
            logits = out.logits[0, -1]
            n_rb, ban = n_rb + 1, (new_len, btok)
            continue
        if act == "eos":
            next_tok = int(draw(eos_only(logits.float(), eos_token_ids[0]), params.greedy, generator))
        elif act == "ban":                # escape-then-stop: ban the cycle's continuation here
            next_tok = int(_pick(logits, history, params, generator, extra_ban=btok))
            n_rb += 1
        else:
            eb = ban[1] if ban is not None and ban[0] == len(history) else None
            next_tok = int(_pick(logits, history, params, generator, extra_ban=eb))
        history.append(next_tok)
        if next_tok in eos_token_ids:
            stop = "eos"
            break
        if len(history) - n_p >= max_new_tokens:
            break
        inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        out = model(inp, past_key_values=cache, use_cache=True)
        nfe += 1
        logits = out.logits[0, -1]
    return GenResult(tokens=history[n_p:], nfe=nfe, stop_reason=stop)


def _pick(logits: torch.Tensor, history: list[int], params: ChainParams,
          generator: torch.Generator | None, extra_ban: int | None = None) -> torch.Tensor:
    processed = apply_chain(logits, history, penalty=params.repetition_penalty,
                            temperature_=params.temperature, top_k_=params.top_k,
                            top_p_=params.top_p, greedy=params.greedy,
                            loop_ban_=params.loop_ban, loop_repeats_=params.loop_repeats,
                            loop_min_period_=params.loop_min_period, loop_action_=params.loop_action,
                            extra_ban=extra_ban)
    return draw(processed, params.greedy, generator)
