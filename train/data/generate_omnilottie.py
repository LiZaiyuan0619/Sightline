""" (C1): OmniLottie Text-to-Lottie drafter training data with vLLM (= gen_vllm.py for OmniLottie).
  - model: the exported text model model/OmniLottie-qwen2-text (export_lottie_text.py; fp32-identical to the
    official LottieDecoder incl. its left padding, check_lottie.py), vocab 192,400, rope_scaling None;
  - prompts: MMLottie-2M rows (the 5 downloaded shards: Lottie 0-2, Lottie_SVG 0-1 = 100k), text = the official
    simplify_to_animation_description(desc_en), in the official chat format (system "You are a Lottie animation expert.",
    user "Generate Lottie code: <text>"; = the Qwen2.5-VL processor's chat template, checked by --check-template);
  - decoding: the official default, greedy, max 4096 new tokens (inference_hf.py --max_tokens), EOS 192399, no penalty;
  - records: prompt ids + output token ids (the first output token is Lottie BOS 192398, as the official model emits it).
Units of --unit-size prompts -> OUT/unit-XXXX/results.jsonl + DONE; re-runs skip DONE units. The vLLM env has no
pyarrow: the prompt list is extracted once (--build-prompts, the svgmetrics environment) to $SIGHTLINE_RUNS/lottie/prompts.jsonl, in shard order.
  python gen_vllm_lottie.py --build-prompts --out <any>
  python gen_vllm_lottie.py \
      --units 0-9 --out <dir> [--limit N] [--gpu-mem 0.88]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import argparse
import glob
import json
import re
import time
from pathlib import Path
from sightline.paths import MODELS, OUT, RUNS

M = MODELS
EXPORT = M / "OmniLottie-qwen2-text"
SHARDS = sorted(glob.glob(str(M / "MMLottie-2M/data/Lottie/*.parquet"))) + \
    sorted(glob.glob(str(M / "MMLottie-2M/data/Lottie_SVG/*.parquet")))
SYSTEM_PROMPT = "You are a Lottie animation expert."
LOTTIE_EOS = 192399
PROMPTS = (RUNS / "lottie/prompts.jsonl")


def simplify_to_animation_description(text):            # = refs/OmniLottie/inference_hf.py
    if not text or not isinstance(text, str):
        return text
    for pattern in [r'^The video features?\s+', r'^The scene shows?\s+', r'^An animation of\s+', r'^There is\s+',
                    r'^It shows?\s+']:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


def chat(text: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nGenerate Lottie code: "
            f"{simplify_to_animation_description(text)}<|im_end|>\n<|im_start|>assistant\n")


def build_prompts():
    import pyarrow.parquet as pq
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with PROMPTS.open("w") as f:
        for sh in SHARDS:
            for r in pq.read_table(sh, columns=["id", "source", "desc_en", "token_length"]).to_pylist():
                if r["desc_en"] and r["desc_en"].strip():
                    f.write(json.dumps({"id": r["id"], "source": r["source"], "text": r["desc_en"],
                                        "token_length": r["token_length"], "shard": Path(sh).name}) + "\n")
                    n += 1
    print("wrote", n, "prompts to", PROMPTS)


def load_items() -> list[dict]:
    with PROMPTS.open() as f:
        return [json.loads(x) for x in f if x.strip()]


def parse_units(spec: str, n_units: int) -> list[int]:
    if spec == "all":
        return list(range(n_units))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default="all")
    ap.add_argument("--unit-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N prompts of each unit")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--check-template", type=Path, default=None,
                    help="runs/2026-10-28/lottie-export-check.json: assert chat() == the processor's template")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build-prompts", action="store_true")
    a = ap.parse_args()
    if a.build_prompts:
        build_prompts()
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EXPORT)
    if a.check_template:
        for p in json.loads(a.check_template.read_text())["prompts"]:
            assert chat(p["text"]) == p["prompt_text"], (chat(p["text"]), p["prompt_text"])
            assert tok(chat(p["text"]), add_special_tokens=False).input_ids == p["prompt_input_ids"]
        print("template check OK", flush=True)
    items = load_items()
    n_units = (len(items) + a.unit_size - 1) // a.unit_size
    print(f"{len(items)} prompts in {n_units} units", flush=True)
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(EXPORT), dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=1536 + a.max_tokens,
              enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop_token_ids=[LOTTIE_EOS], detokenize=False)
    for u in parse_units(a.units, n_units):
        d = a.out / f"unit-{u:04d}"
        if (d / "DONE").exists():
            print("skip", d, flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        part = items[u * a.unit_size:(u + 1) * a.unit_size][:a.limit]
        ids = [tok(chat(x["text"]), add_special_tokens=False).input_ids for x in part]
        keep = [i for i, x in enumerate(ids) if len(x) <= 1500]          # the official text_len
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": ids[i]} for i in keep], sp, use_tqdm=False)
        dt = time.time() - t0
        n_tok = n_eos = 0
        with (d / "results.jsonl").open("w") as f:
            for i, o in zip(keep, outs):
                g = list(o.outputs[0].token_ids)
                eos = bool(g) and g[-1] == LOTTIE_EOS
                n_tok += len(g)
                n_eos += eos
                f.write(json.dumps({**part[i], "gidx": u * a.unit_size + i, "prompt_input_ids": ids[i],
                                    "token_ids": g, "eos": eos, "n_gen": len(g)}) + "\n")
        stats = {"n": len(keep), "skipped_long": len(part) - len(keep), "tokens": n_tok, "seconds": round(dt, 1),
                 "tok_per_s": round(n_tok / max(dt, 1e-6), 1), "eos": n_eos}
        (d / "DONE").write_text(json.dumps(stats))
        print(d, stats, flush=True)
    print("GEN_VLLM_LOTTIE_DONE", a.units)


if __name__ == "__main__":
    main()
