""" (C1): is the exported OmniLottie text model (export_lottie_text.py) the same function as the official
LottieDecoder on Text-to-Lottie prompts? = check_omnisvg4b.py for OmniLottie, in the official environment
(the omnisvg environment, transformers 4.51 = OmniLottie's 4.51.3), fp32 by default:
  - official: refs/OmniLottie/decoder_hf.LottieDecoder.from_pretrained(model/OmniLottie), the base built from the
    Qwen2.5-VL-3B config only (no base weights offline; every weight comes from OmniLottie), and the official inference
    chain of refs/OmniLottie/inference_hf.py: chat template (system "You are a Lottie animation expert.", user
    "Generate Lottie code: <simplified text>"), LEFT padding to 1500 with PAD + attention mask, get_rope_index positions
    x mask, greedy generate, EOS 192399;
  - export: Qwen2ForCausalLM on the unpadded prompt ids, greedy generate.
Reports prompt-logit max |diff| (last position) and token-by-token identity on --n MMLottieBench Text-to-Lottie prompts
(half Real, half Synthetic); also writes the prompt ids (the harness uses them).
  python check_lottie.py [--n 4] [--new 512] [--bf16]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from sightline.paths import MODELS, REFS, RUNS

ROOT = Path(__file__).resolve().parents[1]
M = MODELS
EXPORT, OFFICIAL, BASE = M / "OmniLottie-qwen2-text", M / "OmniLottie", M / "Qwen2.5-VL-3B-Instruct-configs"
BENCH = M / "MMLottieBench/data"
SYSTEM_PROMPT = "You are a Lottie animation expert."
LOTTIE_BOS, LOTTIE_EOS, PAD = 192398, 192399, 151643


def simplify_to_animation_description(text):            # = refs/OmniLottie/inference_hf.py
    if not text or not isinstance(text, str):
        return text
    for pattern in [r'^The video features?\s+', r'^The scene shows?\s+', r'^An animation of\s+', r'^There is\s+',
                    r'^It shows?\s+']:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


def bench_rows(n: int) -> list[dict]:
    import pyarrow.parquet as pq
    out = []
    for sub in ("real", "synthetic"):
        t = pq.read_table(BENCH / f"{sub}-00000-of-00001.parquet", columns=["id", "text", "task_type"]).to_pylist()
        out += [{"id": r["id"], "text": r["text"], "subset": sub} for r in t if r["task_type"] == "Text-to-Lottie"][:n // 2]
    return out


def official_model(dtype):
    sys.path.insert(0, str(REFS / "OmniLottie"))
    import decoder_hf
    cls = decoder_hf.Qwen2_5_VLForConditionalGeneration

    def from_config_only(path, config=None, torch_dtype=None, attn_implementation=None, ignore_mismatched_sizes=None, **kw):
        return cls._from_config(config, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    decoder_hf.Qwen2_5_VLForConditionalGeneration = type("Qwen2_5_VLFromConfig", (cls,),
                                                          {"from_pretrained": staticmethod(from_config_only)})
    from configuration_lottie_decoder import LottieDecoderConfig
    cfg = LottieDecoderConfig.from_pretrained(OFFICIAL)
    cfg.base_model_path = str(BASE)
    m = decoder_hf.LottieDecoder.from_pretrained(OFFICIAL, config=cfg)
    return m.to(dtype).cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--new", type=int, default=512)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", type=Path, default=RUNS / "2026-10-28/lottie-export-check.json")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if a.bf16 else torch.float32
    from transformers import AutoProcessor, AutoTokenizer
    proc = AutoProcessor.from_pretrained(BASE)
    rows = bench_rows(a.n)
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": f"Generate Lottie code: "
                                                                      f"{simplify_to_animation_description(r['text'])}"}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        r["prompt_text"] = text
        r["prompt_input_ids"] = proc(text=[text], padding=False, return_tensors="pt")["input_ids"][0].tolist()
    tok = AutoTokenizer.from_pretrained(EXPORT)
    for r in rows:                                        # the export's own tokenizer gives the same ids
        assert tok(r["prompt_text"], add_special_tokens=False).input_ids == r["prompt_input_ids"], r["id"]
    outs = {}
    for name in ("official", "export"):
        if name == "official":
            m = official_model(dtype)
        else:
            from transformers import AutoConfig, Qwen2ForCausalLM
            from safetensors.torch import load_file
            cfg = AutoConfig.from_pretrained(EXPORT)
            cfg._attn_implementation = "sdpa"
            m = Qwen2ForCausalLM(cfg)
            missing, unexpected = m.load_state_dict(load_file(str(EXPORT / "model.safetensors")), strict=False)
            assert not unexpected and not missing, (missing, unexpected)
            m = m.to(dtype).cuda().eval()
        print(f"[{name}] loaded", flush=True)
        res = []
        for r in rows:
            ids = torch.tensor([r["prompt_input_ids"]])
            with torch.no_grad():
                if name == "official":
                    n_pad = 1500 - ids.shape[1]
                    pid = torch.cat([torch.full((1, n_pad), PAD), ids], 1).cuda()
                    am = torch.cat([torch.zeros((1, n_pad), dtype=torch.long), torch.ones_like(ids)], 1).cuda()
                    tr = m.transformer
                    tr.rope_deltas = None
                    pos, _ = tr.get_rope_index(input_ids=pid, attention_mask=am, image_grid_thw=None, video_grid_thw=None)
                    pos = pos * am[None, ]
                    logits = tr(input_ids=pid, attention_mask=am, position_ids=pos).logits[0, -1].float().cpu()
                    tr.rope_deltas = None
                    g = tr.generate(input_ids=pid, attention_mask=am, position_ids=pos, max_new_tokens=a.new,
                                    eos_token_id=LOTTIE_EOS, pad_token_id=PAD, use_cache=True, do_sample=False,
                                    num_beams=1)[0, pid.shape[1]:].tolist()
                else:
                    ids = ids.cuda()
                    logits = m(input_ids=ids).logits[0, -1].float().cpu()
                    g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.new,
                                   eos_token_id=LOTTIE_EOS, pad_token_id=PAD, do_sample=False)[0, ids.shape[1]:].tolist()
            res.append((logits, g))
        outs[name] = res
        del m
        torch.cuda.empty_cache()
    report = []
    for r, (lo, go), (le, ge) in zip(rows, outs["official"], outs["export"]):
        first = next((j for j in range(min(len(go), len(ge))) if go[j] != ge[j]), None)
        report.append({"id": r["id"], "subset": r["subset"], "prompt_len": len(r["prompt_input_ids"]),
                       "logit_maxdiff": (lo - le).abs().max().item(), "len_official": len(go), "len_export": len(ge),
                       "identical": go == ge, "first_diff": first, "starts_with_bos": bool(go) and go[0] == LOTTIE_BOS,
                       "eos": LOTTIE_EOS in go})
        print(report[-1], flush=True)
    print("ALL_IDENTICAL" if all(x["identical"] for x in report) else "DIFFERENCES", "dtype", dtype)
    a.out.write_text(json.dumps({"new": a.new, "dtype": str(dtype), "rows": report,
                                 "prompts": [{k: r[k] for k in ("id", "subset", "text", "prompt_text", "prompt_input_ids")}
                                             for r in rows]}, indent=1))


if __name__ == "__main__":
    main()
