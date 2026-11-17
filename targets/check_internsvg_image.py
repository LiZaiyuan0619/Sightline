""" self-checks of vlm_i2s.py before anything is trained or timed on it (table 1's 5th block).
  1. prompt ids: vlm_i2s.prompt_ids_of(prompt) == the prompt_token_ids vLLM recorded for the training data;
  2. our harness AR loop (vlm_ar_generate, greedy) == HF generate (greedy, no penalty) token for token;
  3. the vLLM training targets vs our bf16 AR: identical prefix length (vLLM and HF bf16 kernels differ, so late
     near-tie flips are expected; an early or systematic divergence would mean a preprocessing mismatch);
  4. target_features shapes for the DFlash layers [1, 7, 13, 19, 25] and the EAGLE-3 layers [1, 13, 24] (+ logits);
  5. (--text-check) the VLM's language model on a text-only prompt == model/InternSVG-8B-qwen2-text (the target of the
     text heads we warm-start from): greedy tokens and max |logit diff|.
  python check_i2s.py --n 4 --max-new 600 [--text-check] --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.verify import ChainParams  # noqa: E402
from targets import internsvg as V  # noqa: E402
from sightline.paths import MODELS, RUNS

GEN = (RUNS / "isvg-data/i2s")


def records(n: int) -> list[dict]:
    out = []
    for cls in ("icon", "illustration"):
        prompts = {json.loads(x)["gidx"]: json.loads(x) for x in open(GEN / f"prompts-{cls}.jsonl") if x.strip()}
        recs = [json.loads(x) for x in open(GEN / "gen" / cls / "unit-0000" / "results.jsonl") if x.strip()]
        for r in recs[:n]:
            out.append({**r, "prompt": prompts[r["gidx"]]["prompt"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="records per class")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--text-check", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    proc = V.load_processor()
    rows = records(a.n)
    res = {"prompt_ids": [], "hf": [], "vllm": [], "features": {}}
    for r in rows:
        res["prompt_ids"].append({"gidx": r["gidx"], "identical": V.prompt_ids_of(proc, r["prompt"]) == r["prompt_input_ids"]})
    print("[1] prompt ids", sum(x["identical"] for x in res["prompt_ids"]), "/", len(rows), flush=True)
    model, note = V.load_target(torch.bfloat16)
    print("[load]", note, flush=True)
    greedy = ChainParams(greedy=True)
    for r in rows:
        inp = V.build_inputs(proc, model, {"prompt": r["prompt"], "image_path": r["image_path"]})
        ours = V.vlm_ar_generate(model, inp, a.max_new, [V.EOS], greedy)
        ids = torch.tensor([inp.prompt_ids], device="cuda")
        with torch.inference_mode():
            hf = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inp.pixel_values,
                                max_new_tokens=a.max_new, do_sample=False, eos_token_id=V.EOS, pad_token_id=V.PAD)
        hf_tok = hf[0, ids.shape[1]:].tolist()
        hf_tok = hf_tok[:hf_tok.index(V.EOS) + 1] if V.EOS in hf_tok else hf_tok
        div = next((j for j in range(min(len(ours.tokens), len(hf_tok))) if ours.tokens[j] != hf_tok[j]), None)
        res["hf"].append({"gidx": r["gidx"], "subtype": r["subtype"], "n_ours": len(ours.tokens), "n_hf": len(hf_tok),
                          "identical": ours.tokens == hf_tok, "first_div": div})
        vt = r["token_ids"][:len(ours.tokens)]
        vdiv = next((j for j in range(min(len(ours.tokens), len(vt))) if ours.tokens[j] != vt[j]), None)
        res["vllm"].append({"gidx": r["gidx"], "subtype": r["subtype"], "n_compared": min(len(ours.tokens), len(vt)),
                            "first_div": vdiv, "n_vllm": len(r["token_ids"])})
        print("[2]", json.dumps(res["hf"][-1]), "[3]", json.dumps(res["vllm"][-1]), flush=True)
    r = rows[0]
    seq = r["prompt_input_ids"] + r["token_ids"][:200]
    h5 = V.target_features(model, proc, seq, r["image_path"], [1, 7, 13, 19, 25])
    h3, lg = V.target_features(model, proc, seq, r["image_path"], [1, 13, 24], logits=True)
    res["features"] = {"dflash": list(h5.shape), "eagle3": list(h3.shape), "logits": list(lg.shape),
                       "finite": bool(torch.isfinite(h5).all() and torch.isfinite(lg).all())}
    print("[4]", res["features"], flush=True)
    if a.text_check:
        from transformers import AutoModelForCausalLM
        text = AutoModelForCausalLM.from_pretrained(MODELS / "InternSVG-8B-qwen2-text",
                                                    dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        q = "Generate an SVG illustration of a red bicycle leaning against a wall."
        tids = proc.tokenizer(f"<|im_start|>system\n{V.SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n"
                              "<|im_start|>assistant\n", add_special_tokens=False).input_ids
        x = torch.tensor([tids], device="cuda")
        with torch.inference_mode():
            l1 = model(x).logits[0].float()
            l2 = text(x).logits[0].float()
            g1 = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=200, do_sample=False,
                                eos_token_id=V.EOS, pad_token_id=V.PAD)[0, x.shape[1]:].tolist()
            g2 = text.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=200, do_sample=False,
                               eos_token_id=V.EOS, pad_token_id=V.PAD)[0, x.shape[1]:].tolist()
        res["text"] = {"max_abs_logit_diff": float((l1 - l2).abs().max()), "greedy_identical": g1 == g2, "n": len(g1)}
        print("[5]", res["text"], flush=True)
    (a.out / "check_i2s.json").write_text(json.dumps(res, indent=1))
    print("CHECK_I2S", "prompt", sum(x["identical"] for x in res["prompt_ids"]), "hf", sum(x["identical"] for x in res["hf"]),
          "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
