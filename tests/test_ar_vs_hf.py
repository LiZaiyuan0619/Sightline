"""Small-scale check: our AR loop must reproduce HF generate token-by-token.

Runs on a GPU host (an interactive GPU node). 3 short prompts, greedy, repetition_penalty 1.0 and 1.05.

Usage: <the dflash environment>/bin/python sightline/test_ar_vs_hf.py [--model PATH] [--n 32]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.ar import load_target, ar_generate, ChainParams
from transformers import AutoTokenizer
from sightline.paths import MODELS

DEFAULT_MODEL = f"{MODELS}/Qwen3-8B"

PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    "def fibonacci(n):",
    "The capital of France is",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_target(args.model)
    print(f"loaded {args.model}")

    all_ok = True
    for penalty in (1.0, 1.05):
        print(f"--- repetition_penalty={penalty}")
        for pi, prompt in enumerate(PROMPTS):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
            if not isinstance(ids, list):          # transformers 5.x returns a dict
                ids = ids["input_ids"]
            params = ChainParams(greedy=True, repetition_penalty=penalty)
            ours = ar_generate(model, ids, args.n, [tok.eos_token_id], params,
                               generator=None)

            # HF reference
            inp = torch.tensor([ids], device="cuda:0")
            with torch.inference_mode():
                ref = model.generate(inp, do_sample=False, max_new_tokens=args.n,
                                     repetition_penalty=penalty if penalty != 1.0 else None,
                                     eos_token_id=tok.eos_token_id)
            ref_tokens = ref[0, inp.shape[1]:].tolist()

            n = min(len(ours.tokens), len(ref_tokens))
            first_div = next((i for i in range(n) if ours.tokens[i] != ref_tokens[i]), None)
            if first_div is None and len(ours.tokens) == len(ref_tokens):
                print(f"  prompt {pi}: MATCH ({len(ours.tokens)} tokens, NFE {ours.nfe}, "
                      f"stop={ours.stop_reason})")
            else:
                all_ok = False
                print(f"  prompt {pi}: DIVERGE@{first_div} "
                      f"(ours {len(ours.tokens)} vs ref {len(ref_tokens)})")
                print(f"    ours: {ours.tokens[first_div:first_div+8] if first_div is not None else 'len'}")
                print(f"    ref : {ref_tokens[first_div:first_div+8] if first_div is not None else 'len'}")
    print("ALL MATCH" if all_ok else "SOME DIVERGE — investigate")


if __name__ == "__main__":
    main()
