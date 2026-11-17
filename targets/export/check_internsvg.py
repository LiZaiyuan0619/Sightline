""" is the exported InternSVG-8B text model (export_internsvg_text.py) the same function as the
original InternVLForConditionalGeneration on text-only prompts?

fp32, same env (the glance environment, transformers 5.15): prompt logits max |diff| and greedy continuations of --new tokens
on --n SArena text2svg prompts (half icon, half illustration), token by token.

  python check_internsvg.py [--n 4] [--new 256] [--dtype float32]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from targets.internsvg_generate import MODEL, SYSTEM, IM_END, rows  # noqa: E402
from sightline.paths import RUNS

EXPORT = MODEL.parent / "InternSVG-8B-qwen2-text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    dt = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(EXPORT)
    prompts = [r["question"] for r in rows("text2svg", "icon", a.n // 2)] + \
              [r["question"] for r in rows("text2svg", "illustration", a.n - a.n // 2)]
    texts = [f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n" for q in prompts]
    outs = {}
    for name in ("orig", "export"):
        if name == "orig":
            m = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=dt, attn_implementation="sdpa").cuda().eval()
        else:
            m = AutoModelForCausalLM.from_pretrained(EXPORT, dtype=dt, attn_implementation="sdpa").cuda().eval()
        res = []
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                logits = m(input_ids=ids).logits[0].float().cpu()
                g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=a.new,
                               eos_token_id=IM_END, pad_token_id=151643)[0, ids.shape[1]:].tolist()
            res.append((logits, g, ids.shape[1]))
        outs[name] = res
        del m
        torch.cuda.empty_cache()
    report = []
    for i, ((lo, go, n), (le, ge, _)) in enumerate(zip(outs["orig"], outs["export"])):
        d = (lo - le).abs().max().item()
        first = next((j for j in range(min(len(go), len(ge))) if go[j] != ge[j]), None)
        report.append({"i": i, "prompt_len": n, "logit_maxdiff": d, "len_orig": len(go), "len_export": len(ge),
                       "identical": go == ge, "first_diff": first})
        print(report[-1], flush=True)
    ok = all(r["identical"] for r in report)
    print("ALL_IDENTICAL" if ok else "DIFFERENCES", a.dtype)
    out = RUNS / "2026-10-15-harvest/internsvg-export-check.json"
    out.write_text(json.dumps({"dtype": a.dtype, "new": a.new, "rows": report}, indent=1))


if __name__ == "__main__":
    main()
