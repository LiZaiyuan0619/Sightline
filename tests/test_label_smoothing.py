"""CPU unit test for the SLS patch (patches/specforge-sls.patch) in SpecForge's DFlash
objective: _spatial_label_smoothing must equal a dense soft-label cross entropy, in value and
gradient, on a small vocabulary (coord ids at an offset, edges and corners included), and the
patch must be off when SPECFORGE_SLS_EPS is unset.

Usage (the specforge environment, CPU):  python test_sls.py
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

os.environ.pop("SPECFORGE_SLS_EPS", None)
import specforge.algorithms.common.dflash_family_model as m   # noqa: E402

CS, G, V = 10, 20, 10 + 20 * 20 + 7          # coord ids 10..409, a few ids after


def dense_ref(logits, target, eps, sigma, radius):
    logp = torch.log_softmax(logits.float(), -1).reshape(-1, logits.shape[-1])
    t = target.reshape(-1)
    out = []
    for i in range(t.numel()):
        y = int(t[i])
        if CS <= y < CS + G * G:
            cx, cy = (y - CS) % G, (y - CS) // G
            soft = torch.zeros(logits.shape[-1])
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x2, y2 = cx + dx, cy + dy
                    if 0 <= x2 < G and 0 <= y2 < G:
                        soft[CS + y2 * G + x2] = math.exp(-(dx * dx + dy * dy) / (2 * sigma ** 2))
            soft = eps * soft / soft.sum()
            soft[y] += 1 - eps
            out.append(-(soft * logp[i]).sum())
        else:
            out.append(-logp[i, y])
    return torch.stack(out).reshape(target.shape)


def main():
    assert m._SLS_EPS == 0, "patch must be off without SPECFORGE_SLS_EPS"
    torch.manual_seed(0)
    B, N, S = 2, 3, 16
    target = torch.randint(0, V, (B, N, S))
    # force edge / corner coordinates and some non-coordinates
    target[0, 0, :6] = torch.tensor([CS, CS + G - 1, CS + (G - 1) * G, CS + G * G - 1,
                                     CS + 5 * G + 7, 3])
    for eps, sigma, radius in ((0.1, 1.0, 2), (0.3, 1.0, 2), (0.3, 0.7, 1), (0.0, 1.0, 2)):
        logits = (torch.randn(B, N, S, V) * 3).requires_grad_(True)
        nlq = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none"
                              ).reshape(target.shape)
        got = m._spatial_label_smoothing(logits, target, nlq, eps, sigma, radius, CS, G)
        w = torch.rand(target.shape)
        (got * w).sum().backward()
        g_got = logits.grad.clone()
        logits.grad = None
        ref = dense_ref(logits, target, eps, sigma, radius)
        (ref * w).sum().backward()
        g_ref = logits.grad.clone()
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-5), (eps, (got - ref).abs().max())
        assert torch.allclose(g_got, g_ref, atol=1e-5, rtol=1e-4), (eps, (g_got - g_ref).abs().max())
        print(f"[ok] eps={eps} sigma={sigma} r={radius}: value max|d|={float((got - ref).abs().max()):.2e} "
              f"grad max|d|={float((g_got - g_ref).abs().max()):.2e}")
    # bf16 logits (the training dtype): value close, ordering of hard vs soft preserved
    logits = (torch.randn(B, N, S, V) * 3).to(torch.bfloat16).requires_grad_(True)
    nlq = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none").reshape(target.shape)
    got = m._spatial_label_smoothing(logits, target, nlq, 0.3, 1.0, 2, CS, G)
    ref = dense_ref(logits.detach().float(), target, 0.3, 1.0, 2)
    rel = float(((got.float() - ref).abs() / ref.abs().clamp(min=1)).max())
    assert rel < 2e-2, rel
    got.float().sum().backward()
    assert torch.isfinite(logits.grad.float()).all()
    print(f"[ok] bf16 logits: max relative error {rel:.2e}, finite grads")
    print("ALL_OK")


if __name__ == "__main__":
    main()
