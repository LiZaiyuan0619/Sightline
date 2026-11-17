""" HiVG-3B drafter data (copy of gen_vllm_fam.py with the hivg3b family + top-k; new file because queued
jobs import gen_vllm_fam.py). Same pools / unit format. Original docstring: third-family SVG LLMs with vLLM, on the same 300k
MMSVG-2M captions as the InternSVG drafters (gen_vllm.py pools: mmsvg = 50k per class, mmsvg-ext = 100k
per class), each family's official prompt and sampling. Units of --unit-size prompts -> OUT/<class>/unit-XXXX/
results.jsonl + DONE (same record format as gen_vllm.py, so build_trainset_fam.py reads both).
  python gen_vllm_fam.py --family svgen7b \
      --pool mmsvg --class icon --units 0-19 --out <dir> [--limit N] [--kv-gb 30 (shared GPU: skip memory profiling)]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from fam.families import FAMILIES  # noqa: E402
from sightline.paths import MODELS, OUT, RUNS

HIVG = f"{MODELS}/HiVG-3B-Base-qwen2-text"
# refs/HiVG/hivg_infer/pipeline.py: text2svg prompt, the model's chat template, GenerationConfig T 0.7 / top-p 0.9 / top-k 50
FAMILIES["hivg3b"] = {"model": Path(HIVG), "eos": [151645, 151643], "stop": 151645, "mask": 151662, "pad": 151643,
                      "temperature": 0.7, "top_p": 0.9, "top_k": 50, "max_new": 4000}


def encode(tok, fam: str, desc: str) -> list[int]:
    assert fam == "hivg3b", fam
    msgs = [{"role": "user", "content": f"[T2ST] Generate SVG tokens from text.\n{desc}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids

POOL = RUNS / "2026-10-09-prompt-pool/train-pool"
POOL_EXT = (RUNS / "t6-ext/train-ids")


def parse_units(spec: str, n_units: int) -> list[int]:
    if spec == "all":
        return list(range(n_units))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return [u for u in out if u < n_units]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["hivg3b"], required=True)
    ap.add_argument("--pool", choices=["mmsvg", "mmsvg-ext"], required=True)
    ap.add_argument("--class", dest="cls", choices=["icon", "illustration"], required=True)
    ap.add_argument("--units", default="all")
    ap.add_argument("--unit-size", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N prompts of each unit")
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--kv-gb", type=float, default=None, help="fixed KV cache size (shared GPUs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    F = FAMILIES[a.family]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(F["model"])
    src = (POOL if a.pool == "mmsvg" else POOL_EXT) / f"{a.cls}.jsonl"
    pool = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    n_units = (len(pool) + a.unit_size - 1) // a.unit_size
    units = parse_units(a.units, n_units)
    kw = dict(model=str(F["model"]), dtype="bfloat16", max_model_len=4096 + 512, seed=a.seed,
              enable_prefix_caching=True)
    if a.kv_gb:
        kw.update(gpu_memory_utilization=0.5, kv_cache_memory_bytes=int(a.kv_gb * 1024 ** 3))
    else:
        kw.update(gpu_memory_utilization=a.gpu_mem)
    llm = LLM(**kw)
    sp = SamplingParams(temperature=F["temperature"], top_p=F["top_p"], top_k=F.get("top_k", -1), max_tokens=F["max_new"],
                        stop_token_ids=F["eos"], detokenize=False, seed=a.seed)
    for u in units:
        d = a.out / a.cls / f"unit-{u:04d}"
        if (d / "DONE").exists():
            print("skip", d, flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        part = pool[u * a.unit_size:(u + 1) * a.unit_size][:a.limit]
        ids = [encode(tok, a.family, r["text"]) for r in part]
        t0 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp, use_tqdm=False)
        dt = time.time() - t0
        n_tok = n_eos = 0
        with (d / "results.jsonl").open("w") as f:
            for r, p, o in zip(part, ids, outs):
                g = list(o.outputs[0].token_ids)
                eos = bool(g) and g[-1] in F["eos"]
                n_tok += len(g)
                n_eos += eos
                f.write(json.dumps({"id": r["id"], "subtype": a.cls, "question": r["text"], "prompt_input_ids": p,
                                    "token_ids": g, "eos": eos, "n_gen": len(g)}) + "\n")
        stats = {"n": len(part), "tokens": n_tok, "seconds": round(dt, 1), "tok_per_s": round(n_tok / dt, 1),
                 "temperature": F["temperature"], "top_p": F["top_p"], "eos": n_eos, "family": a.family}
        (d / "DONE").write_text(json.dumps(stats))
        print(f"unit {u} {stats}", flush=True)


if __name__ == "__main__":
    main()
