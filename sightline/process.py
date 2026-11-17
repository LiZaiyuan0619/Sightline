"""HF-consistent logits processing chain for the strict verifier.

Order matches HF `generate`: repetition penalty -> temperature -> top-k -> top-p.
Each function is a faithful reimplementation, unit-tested against transformers'
own processor classes (see test_process.py).

Subtlety for block verification (harness spec): when processing position i of a drafted
block, the repetition-penalty history must be:

    prompt + already-generated (accepted) tokens + draft tokens [0..i-1] of this block

i.e. the same history the token would have in a plain AR run. The caller is responsible
for assembling `history_ids`; these functions are pure.

Greedy mode (HF semantics): only the repetition penalty applies, then argmax.
Warpers (temperature/top-k/top-p) are no-ops for argmax, so they are skipped — but the
penalty DOES change argmax, so it must be applied.
"""
from __future__ import annotations

import torch

NEG_INF = float("-inf")
_EOS = [196999]          # the loop rule's EOS (action "eos"); OmniSVG by default, set_eos for other targets


def set_eos(eos_id: int):
    """Target-specific EOS for the loop rule's stop action (e.g. InternSVG-8B: 151645)."""
    _EOS[0] = int(eos_id)


def repetition_penalty(logits: torch.Tensor, history_ids, penalty: float) -> torch.Tensor:
    """HF RepetitionPenaltyLogitsProcessor semantics: once per UNIQUE token in history,
    logit>0 -> /penalty, logit<0 -> *penalty. Penalty is applied PER BATCH ROW.

    logits: (vocab,) or (batch, vocab). history_ids: flat list/tensor for batch 1,
    or (batch, seq) / list of per-row lists for batches.
    """
    if penalty == 1.0:
        return logits
    dev = logits.device
    if isinstance(history_ids, torch.Tensor):
        hist = history_ids.to(dev)
    else:
        history_ids = list(history_ids)
        if history_ids and isinstance(history_ids[0], (list, tuple, torch.Tensor)):
            hist = torch.stack([torch.as_tensor(h, dtype=torch.long, device=dev) for h in history_ids])
        else:
            hist = torch.as_tensor(history_ids, dtype=torch.long, device=dev).unsqueeze(0)
    if hist.dim() == 1:
        hist = hist.unsqueeze(0)

    if logits.dim() == 1:                      # batch-1 fast path
        idx = torch.unique(hist[0])
        vals = logits[idx]
        out = logits.clone()
        out[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
        return out

    out = logits.clone()
    for b in range(out.shape[0]):
        idx = torch.unique(hist[b] if hist.shape[0] > 1 else hist[0])
        vals = out[b, idx]
        out[b, idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


def temperature(logits: torch.Tensor, t: float) -> torch.Tensor:
    return logits if t == 1.0 else logits / t


def top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """HF TopKLogitsWarper semantics: remove logits strictly below the k-th largest
    (ties at the threshold are kept)."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return torch.where(logits < threshold, NEG_INF, logits)


def top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """HF TopPLogitsWarper semantics (4.x): sort ascending, cumsum of softmax probs,
    remove tokens with cumulative prob <= 1 - p (ascending order = the tail), always
    keep the largest token."""
    if p >= 1.0 or p <= 0.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=False, dim=-1)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumulative <= (1.0 - p)
    remove[..., -1:] = False
    sorted_logits = torch.where(remove, NEG_INF, sorted_logits)
    out = torch.empty_like(logits)
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


def loop_fire(history_ids, max_period: int, repeats: int = 2, min_period: int = 4) -> int:
    """The loop rule: the smallest P (min_period <= P <= max_period) such that the history ends
    with `repeats` exact copies of one P-token cycle; 0 if none (or max_period <= 0)."""
    if max_period <= 0:
        return 0
    h = history_ids.tolist() if isinstance(history_ids, torch.Tensor) else list(history_ids)
    n = len(h)
    for P in range(min_period, max_period + 1):
        if repeats * P > n:
            break
        tail = h[n - P:]
        if all(h[n - (r + 1) * P:n - r * P] == tail for r in range(1, repeats)):
            return P
    return 0


def loop_ban(logits: torch.Tensor, history_ids, max_period: int, repeats: int = 2,
             min_period: int = 4, action: str = "ban", eos_id: int | None = None) -> torch.Tensor:
    """Loop escape (; changes the target distribution, NOT the HF chain): when the history
    ends with `repeats` exact copies of one P-token cycle (min_period <= P <= max_period, the
    smallest such P), the token that would continue the cycle, history[-P], gets logit -inf.
    Every decoder (AR, chain, tree) applies it with the same per-position history, so
    speculative decoding stays exact with respect to this modified target. max_period = 0: off.
    action "eos": instead of banning the continuation, force EOS (stop at the detected loop).
    action "rollback" is not a per-position rule: the decoders handle it (loop_rollback)."""
    if action in ("rollback", "esc_stop"):
        return logits
    P = loop_fire(history_ids, max_period, repeats, min_period)
    if not P:
        return logits
    if action == "eos":
        return eos_only(logits, _EOS[0] if eos_id is None else eos_id)
    h = history_ids.tolist() if isinstance(history_ids, torch.Tensor) else list(history_ids)
    out = logits.clone()
    out[..., h[len(h) - P]] = NEG_INF
    return out


def eos_only(logits: torch.Tensor, eos_id: int) -> torch.Tensor:
    out = torch.full_like(logits, NEG_INF)
    out[..., eos_id] = 0.0
    return out


def loop_rollback(history: list[int], params, n_rollbacks: int, n_prompt: int) -> tuple[str, int, int]:
    """Rollback escape (; a decoder-level modification of the target, LoopGuard-style).
    Before drawing the token at position len(history): if the loop rule fires with period P and
    the kept cycle lies in the generated part, either roll back -- drop the last (repeats - 1)
    copies, keeping one cycle, and ban the token that would restart the cycle at the new
    position -- or, once params.loop_max_rb rollbacks are used, stop with EOS.
    Returns ("none" | "rollback" | "eos", new_length, banned_token). A deterministic function of
    (history, n_rollbacks), applied identically by every decoder (AR, tree)."""
    action = getattr(params, "loop_action", "ban")
    if action not in ("rollback", "esc_stop"):
        return "none", 0, -1
    P = loop_fire(history, params.loop_ban, params.loop_repeats, params.loop_min_period)
    if not P:
        return "none", 0, -1
    if action == "esc_stop":
        # escape-then-stop: the first loop_max_rb firings ban the cycle's continuation in
        # place (= the stateless "ban" rule), the next firing stops with EOS
        if n_rollbacks >= getattr(params, "loop_max_rb", 1):
            return "eos", 0, -1
        return "ban", len(history), history[len(history) - P]
    new_len = len(history) - (params.loop_repeats - 1) * P
    if new_len - P < n_prompt:
        return "none", 0, -1
    if n_rollbacks >= getattr(params, "loop_max_rb", 1):
        return "eos", 0, -1
    return "rollback", new_len, history[new_len - P]


def apply_chain(logits: torch.Tensor, history_ids, penalty: float = 1.0,
                temperature_: float = 1.0, top_k_: int = 0, top_p_: float = 1.0,
                greedy: bool = False, loop_ban_: int = 0, loop_repeats_: int = 2,
                loop_min_period_: int = 4, loop_action_: str = "ban",
                extra_ban: int | None = None) -> torch.Tensor:
    """Full processing chain. Greedy skips the warpers (HF greedy path).

    IMPORTANT: all processing happens in float32 — HF `generate` casts the raw logits
    to fp32 (`next_token_logits.to(dtype=torch.float32)`) before running processors.
    Processing in bf16 flips argmax on near-ties: e.g. logit 29.25 (= 234/8) divided by
    penalty 1.05 rounds back to exactly 27.875 in bf16, tying an unpenalized 27.875 and
    argmax then picks the lower index — observed on Qwen3-8B 2026-10-08. Keep fp32.
    """
    logits = logits.float()
    logits = repetition_penalty(logits, history_ids, penalty)
    logits = loop_ban(logits, history_ids, loop_ban_, loop_repeats_, loop_min_period_, loop_action_)
    if extra_ban is not None:                      # rollback escape: the token that would restart the cycle
        logits = logits.clone()
        logits[..., extra_ban] = NEG_INF
    if not greedy:
        logits = temperature(logits, temperature_)
        logits = top_k(logits, top_k_)
        logits = top_p(logits, top_p_)
    return logits


def draw(logits: torch.Tensor, greedy: bool, generator: torch.Generator | None = None):
    """Argmax or multinomial draw from PROCESSED logits."""
    if greedy:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator
                             ).reshape(logits.shape[:-1])
