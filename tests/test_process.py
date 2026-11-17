"""Unit test: our processing chain must match HF's own processors exactly.

Run:  <env>/bin/python sightline/test_process.py
Works under both transformers 4.51 and 5.15 (class names are stable).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline import process as P

from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper,
)


def make_case(seed, vocab=200, batch=2):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, vocab, generator=g) * 3
    hist = torch.randint(0, vocab, (batch, 37), generator=g)
    return logits, hist


def check(name, ours, theirs, atol=1e-5):
    """Compare allowing -inf in both (masked entries); finite entries must match."""
    both_inf = (ours == float("-inf")) & (theirs == float("-inf"))
    either_inf = torch.isinf(ours) | torch.isinf(theirs)
    if (either_inf & ~both_inf).any():
        print(f"  [FAIL] {name}: inf pattern differs "
              f"({(either_inf & ~both_inf).sum().item()} positions)")
        return False
    fin = ~either_inf
    ok = True
    if fin.any():
        ok = torch.allclose(ours[fin], theirs[fin], atol=atol)
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        print(f"        max finite diff {(ours[fin] - theirs[fin]).abs().max().item():.6g}")
    return ok


def main():
    all_ok = True
    for seed in range(5):
        logits, hist = make_case(seed)
        print(f"case seed={seed}")

        # 1. repetition penalty (HF: unique tokens, sign-dependent scaling)
        for penalty in (1.05, 1.2, 2.0):
            ours = P.repetition_penalty(logits, hist, penalty)
            theirs = RepetitionPenaltyLogitsProcessor(penalty)(hist, logits.clone())
            all_ok &= check(f"repetition_penalty {penalty}", ours, theirs)

        # 2. temperature
        for t in (0.5, 0.9, 1.7):
            ours = P.temperature(logits, t)
            theirs = TemperatureLogitsWarper(t)(hist, logits.clone())
            all_ok &= check(f"temperature {t}", ours, theirs)

        # 3. top-k (ties: ours keeps == threshold, HF same)
        for k in (1, 5, 50):
            ours = P.top_k(logits, k)
            theirs = TopKLogitsWarper(k)(hist, logits.clone())
            all_ok &= check(f"top_k {k}", ours, theirs)

        # 4. top-p
        for p in (0.5, 0.9, 0.99):
            ours = P.top_p(logits, p)
            theirs = TopPLogitsWarper(p)(hist, logits.clone())
            all_ok &= check(f"top_p {p}", ours, theirs)

        # 5. full chain order: penalty -> temperature -> top-k -> top-p
        ours = P.apply_chain(logits, hist, penalty=1.05, temperature_=0.7, top_k_=40, top_p_=0.9)
        cur = logits.clone()
        cur = RepetitionPenaltyLogitsProcessor(1.05)(hist, cur)
        cur = TemperatureLogitsWarper(0.7)(hist, cur)
        cur = TopKLogitsWarper(40)(hist, cur)
        cur = TopPLogitsWarper(0.9)(hist, cur)
        all_ok &= check("full chain", ours, cur)

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


def test_loop_ban():
    """ loop escape: ban history[-P] once the history ends with 2 copies of a P-cycle."""
    logits = torch.zeros(50)
    h = [1, 2, 9, 8, 7, 6, 9, 8, 7, 6]                    # 2 x (9 8 7 6)
    out = P.loop_ban(logits, h, 64)
    assert out[9] == float("-inf") and torch.isfinite(out).sum() == 49, out
    assert torch.equal(P.loop_ban(logits, h, 0), logits)            # off
    assert torch.equal(P.loop_ban(logits, h, 64, repeats=3), logits)  # needs 3 copies
    assert torch.equal(P.loop_ban(logits, [1, 2, 3, 3], 64), logits)  # P=1 < min_period
    assert torch.equal(P.loop_ban(logits, [4, 5, 6, 7, 4, 5, 6, 8], 64), logits)
    h8 = [1] + [3, 4, 5, 6, 7, 8, 9, 10] * 2                # 2 x an 8-cycle
    assert torch.equal(P.loop_ban(logits, h8, 64, min_period=9), logits)
    assert P.loop_ban(logits, h8, 64, min_period=8)[3] == float("-inf")
    stop = P.loop_ban(torch.zeros(197000), h, 64, action="eos")
    assert stop[196999] == 0.0 and torch.isinf(stop).sum() == 196999   # only EOS left
    # through the chain: greedy argmax moves off the banned token
    lg = torch.zeros(50); lg[9] = 5.0; lg[3] = 4.0
    assert int(P.draw(P.apply_chain(lg, h, greedy=True, loop_ban_=64), True)) == 3
    print("[ok] loop_ban")


if __name__ == "__main__":
    test_loop_ban()
    raise SystemExit(main())
