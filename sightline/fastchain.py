""" opt-in faster logits processing for the strict decoders. enable patches sightline.process in place, so AR
(sightline.ar) and every SD method (sightline.spec / tree_spec, the baselines' drafters) get the same processing chain;
sightline/process.py itself is unchanged (queued jobs import it). Per position (H200, V = 197k, history 1.5k):
repetition penalty 0.147 ms, top-p after top-k 0.159 ms, out of a 0.357 ms sampling chain.

  - repetition_penalty: the penalized set is a GPU boolean mask over the vocabulary, extended with the history's new
    tokens; the cached prefix is checked by content (history[:n] == cached), so a new list object or a rollback
    rebuilds it. The values are the original ones (x / p if x > 0 else x * p, fp32, on the same token set): exact.
  - top_p (1-D, at most FASTCHAIN_K finite logits, i.e. after top-k): the nucleus is computed on the finite values
    only. Same rule as process.top_p / HF (ascending cumulative <= 1 - p removed, the largest kept); the probabilities
    come from one softmax over the finite values instead of the whole vector, so a token at the exact nucleus edge can
    differ in the last bit (checked on real decoding). FASTCHAIN_TOPP=0 keeps the original top_p.
  - draw, top_k, temperature, loop rules: unchanged (same random stream: the multinomial still sees the full vector).
Batched inputs (sightline.batch_spec) fall through to the originals.
"""
from __future__ import annotations

import os

import torch

from sightline import process as P

_ORIG = {"repetition_penalty": P.repetition_penalty, "top_p": P.top_p}
K_MAX = int(os.environ.get("FASTCHAIN_K", "256"))
_state: dict = {}


def repetition_penalty(logits: torch.Tensor, history_ids, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    if (logits.dim() != 1 or isinstance(history_ids, torch.Tensor)
            or (len(history_ids) and isinstance(history_ids[0], (list, tuple, torch.Tensor)))):
        return _ORIG["repetition_penalty"](logits, history_ids, penalty)
    h = history_ids if isinstance(history_ids, list) else list(history_ids)
    key = (logits.device, logits.shape[0])
    st = _state.get(key)
    if st is None:
        st = _state[key] = {"toks": [], "mask": torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)}
    n = len(st["toks"])
    if len(h) < n or h[:n] != st["toks"]:
        st["mask"].zero_()
        st["toks"] = []
        n = 0
    if len(h) > n:
        st["mask"][torch.tensor(h[n:], dtype=torch.long, device=logits.device)] = True
        st["toks"].extend(h[n:])
    return torch.where(st["mask"], torch.where(logits > 0, logits / penalty, logits * penalty), logits)


def top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0 or p <= 0.0:
        return logits
    if logits.dim() != 1:
        return _ORIG["top_p"](logits, p)
    k = min(K_MAX, logits.shape[0])
    vals, idx = torch.topk(logits, k)                        # descending; -inf last
    if bool(torch.isfinite(vals[-1])):                       # more than K finite logits: the original
        return _ORIG["top_p"](logits, p)
    asc_v, asc_i = vals.flip(0), idx.flip(0)                 # ascending, -inf first (prob 0, as in the full sort)
    cum = asc_v.softmax(dim=-1).cumsum(dim=-1)
    remove = cum <= (1.0 - p)
    remove[-1:] = False
    out = logits.clone()
    out.scatter_(0, asc_i, torch.where(remove, P.NEG_INF, asc_v))   # every finite logit is in asc_i: no host sync
    return out


def enable(topp: bool | None = None):
    """Patch sightline.process (module globals that apply_chain resolves at call time)."""
    P.repetition_penalty = repetition_penalty
    if (os.environ.get("FASTCHAIN_TOPP", "0") == "1") if topp is None else topp:   # off: measured slower (0.176 vs 0.159 ms)
        P.top_p = top_p


def disable():
    P.repetition_penalty, P.top_p = _ORIG["repetition_penalty"], _ORIG["top_p"]
    _state.clear()
