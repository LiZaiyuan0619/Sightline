"""A2 step 8 lossless audit: fp32 bit-level check of our block
speculative decoder with a trained OmniSVG DFlash head, after GLANCE's
experiments/audit_lossless.py.

In bf16 the block forward (spec) and the per-token forward (AR) can swap argmax at
near-ties, so B3's greedy spot-check reports divergences with margins. Here target
and head run in fp32 on 10 MMSVGBench text2svg prompts (5 icon + 5 illustration,
greedy = official chain incl. repetition penalty, then argmax): spec must reproduce
AR greedy token for token wherever AR reproduces itself (the control GLANCE reports).

Usage (GPU, the dflash environment):
  python audit_fp32.py --drafter <head dir> --out <dir> \
      [--n-per-subtype 5] [--max-new 1536]
Output: <out>/audit_fp32.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from t7.eval_b3 import EOS, MAX_NEW, MODEL, chain_params, load_rows   # noqa: E402
from sightline.ar import ar_generate                                     # noqa: E402
from sightline.spec import spec_generate, DFlashDrafter                  # noqa: E402


def first_div(a: list[int], b: list[int]):
    n = min(len(a), len(b))
    d = next((j for j in range(n) if a[j] != b[j]), None)
    return d if d is not None or len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-subtype", type=int, default=5)
    ap.add_argument("--gidx", default=None,
                    help="comma-separated bench gidx to audit instead (e.g. early bf16 divergences)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--target", type=Path, default=MODEL / "OmniSVG1.1_8B-qwen2-text")
    ap.add_argument("--block-size", type=int, default=None,
                    help="default: the head's own block_size (B32 heads, 10-11)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel
    torch.backends.cuda.matmul.allow_tf32 = False          # true fp32 matmuls (the default)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    head = DFlashDraftModel.from_pretrained(args.drafter, torch_dtype=torch.float32).cuda().eval()
    if args.block_size is None:
        args.block_size = int(head.block_size)

    bench = load_rows(["bench"])
    rows = []
    if args.gidx:
        want = [int(g) for g in args.gidx.split(",")]
        rows = [r for g in want for r in bench if r["gidx"] == g]
    else:
        for subtype in ("icon", "illustration"):
            rows += [r for r in bench if r["subtype"] == subtype][:args.n_per_subtype]

    report = {"drafter": str(args.drafter), "dtype": "float32", "mode": "greedy",
              "max_new_tokens": args.max_new, "block_size": args.block_size, "rows": []}
    n_match = n_self = 0
    for r in rows:
        params = chain_params("greedy", r["subtype"])
        t0 = time.perf_counter()
        ref = ar_generate(target, r["prompt_input_ids"], args.max_new, [EOS], params).tokens
        again = ar_generate(target, r["prompt_input_ids"], args.max_new, [EOS], params).tokens
        drafter = DFlashDrafter(head, target, args.block_size)
        spec = spec_generate(target, drafter, r["prompt_input_ids"], args.max_new, [EOS],
                             params, block_size=args.block_size)
        d_spec, d_self = first_div(ref, spec.tokens), first_div(ref, again)
        n_match += d_spec is None
        n_self += d_self is None
        tau = sum(spec.acceptance) / max(len(spec.acceptance), 1)
        report["rows"].append({"id": r["id"], "gidx": r["gidx"], "subtype": r["subtype"],
                               "n_ar": len(ref), "n_spec": len(spec.tokens),
                               "first_div_spec": d_spec, "first_div_self": d_self,
                               "tau": round(tau, 3), "wall_s": round(time.perf_counter() - t0, 1)})
        print(f"[audit] {r['subtype']} gidx={r['gidx']}: n={len(ref)} spec_div={d_spec} "
              f"self_div={d_self} tau={tau:.2f}", flush=True)
    report["total"] = {"n": len(rows), "identical_to_ar": n_match, "ar_self_consistent": n_self}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit_fp32.json").write_text(json.dumps(report, indent=2))
    print(f"audit: {n_match}/{len(rows)} identical to fp32 AR greedy; "
          f"AR self-consistent on {n_self}/{len(rows)}")
    if n_match < n_self:
        print("AUDIT_FAILED: spec diverged where AR reproduces itself")
        return 1
    print("AUDIT_FP32_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
