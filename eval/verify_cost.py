"""A2 step 7: verification cost curve. Also used by PB-1 with --model JanusCoderV-7B --contexts <median> --ks 1,8,16,32,64,128.

Measures, at batch 1 with a KV cache, the wall time of ONE target forward that
verifies K drafted tokens (the block-verification forward is over the last
committed token + K drafts = K+1 positions):

    t_verify(K) for K in --ks, contexts --contexts, plus t_AR (single-token
    forward) as the reference.

Environment: the dflash environment (torch 2.13 / transformers 5.15) — the locked eval
environment (bf16-numerics memo: same env, same forward structure as src/harness).

Usage (GPU):
  python verify_cost.py                       # A2 defaults
  python verify_cost.py --model.../JanusCoderV-7B \
      --contexts 3000 --ks 1,8,16,32,64,128                          # PB-1
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import argparse
import json
import statistics as st
import time
from pathlib import Path
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
MODEL = MODELS

CONTEXTS = [500, 1000, 1500]
KS = [1, 2, 4, 8, 16, 32, 64]
TRIALS = 30
WARMUP = 5


def bench(model, torch, ctx_len: int, k: int, seed: int = 0) -> float:
    """Median seconds of one (k+1)-token forward at position ctx_len.

    Runs under inference_mode: without it every forward builds an autograd graph (the
    weights require grad), which inflated all timings by a constant overhead
    and kept the prefill activations alive (OOM at ctx 10000 for JanusCoderV, )."""
    g = torch.Generator().manual_seed(seed)
    from transformers import DynamicCache
    with torch.inference_mode():
        cache = DynamicCache()
        vocab = model.get_input_embeddings().num_embeddings
        prompt = torch.randint(0, vocab, (1, ctx_len), generator=g).cuda()
        model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)
        times = []
        block = torch.randint(0, vocab, (1, k + 1), generator=g).cuda()
        for t in range(TRIALS + WARMUP):
            pos = ctx_len + t * (k + 1)
            cache_position = torch.arange(pos, pos + k + 1, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=block, past_key_values=cache, use_cache=True,
                  cache_position=cache_position)
            torch.cuda.synchronize()
            if t >= WARMUP:
                times.append(time.perf_counter() - t0)
    return st.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL / "OmniSVG1.1_8B-qwen2-text")
    ap.add_argument("--out", type=Path, default=RUNS / "2026-10-09-verify-cost")
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    import torch
    import transformers
    # resolve the class from config.architectures: covers both the plain
    # Qwen2ForCausalLM (A2) and VL classes like Qwen2_5_VLForConditionalGeneration
    # (PB-1) — transformers 5.x no longer maps VL through AutoModelForCausalLM.
    arch = json.loads((args.model / "config.json").read_text())["architectures"][0]
    cls = getattr(transformers, arch)
    model = cls.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    results = {"model": str(args.model), "dtype": "bfloat16", "batch": 1,
               "trials": TRIALS, "grad": "inference_mode", "contexts": {}}
    for ctx in contexts:
        t_ar = bench(model, torch, ctx, 0)
        row = {"t_ar_s": t_ar}
        for k in ks:
            row[f"t_verify_{k}_s"] = bench(model, torch, ctx, k)
            row[f"ratio_vs_ar_{k}"] = row[f"t_verify_{k}_s"] / t_ar
        results["contexts"][ctx] = row
        print(f"ctx={ctx}: t_AR={t_ar*1e3:.2f} ms | " +
              " ".join(f"K={k}:{row[f't_verify_{k}_s']*1e3:.2f}ms"
                       f"({row[f'ratio_vs_ar_{k}']:.2f}x)" for k in ks))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verify_cost.json").write_text(json.dumps(results, indent=2))
    print("wrote", args.out / "verify_cost.json")


if __name__ == "__main__":
    main()
