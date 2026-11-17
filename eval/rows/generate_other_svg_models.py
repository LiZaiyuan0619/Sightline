""" table 2's reference rows: other SVG LLMs (SVGen-7B / 3B, HiVG-3B) on the FULL SArena text2svg test sets
(Icon 6,013 + Illustration 2,001; the descriptions of $SIGHTLINE_RUNS/tableA/xbench/omnisvg-sarena-full.jsonl), each family's official
prompt, sampling and length cap (SVGen T 0.2 / top-p 0.7 / 4,000 new tokens; HiVG T 0.7 / top-p 0.9 / top-k 50 /
4,096 = refs/HiVG/hivg_infer/pipeline.py), with vLLM: these outputs are only scored for quality (official SArena metrics,
runs/2026-10-24/sarena_eval_fam.py); per-image time is measured separately in HF eager on the node (jobs_1024f.sh).
Prompts = the drafter-data prompts (families.encode; HiVG: gen_vllm_hivg.encode); request seed = 42 + gidx.
  python gen_vllm_sarena.py --family svgen7b \
      --out $SIGHTLINE_RUNS/tableB-fam/svgen7b.jsonl [--limit N] [--kv-gb 40 (shared GPU: skip memory profiling)]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
import gen_vllm_hivg as G  # noqa: E402  (defines FAMILIES["hivg3b"] and encode for all three families)
from sightline.paths import RUNS

ROWS = (RUNS / "tableA/xbench/omnisvg-sarena-full.jsonl")
CAP = {"svgen7b": 4000, "svgen3b": 4000, "hivg3b": 4096}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(CAP))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--kv-gb", type=float, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    F = G.FAMILIES[a.family]
    tok = AutoTokenizer.from_pretrained(F["model"])
    rows = [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()][:a.limit]
    from fam.families import encode as enc_svgen
    enc = G.encode if a.family == "hivg3b" else enc_svgen        # gen_vllm_hivg.encode is HiVG-only
    ids = [enc(tok, a.family, r["text"]) for r in rows]
    kw = dict(model=str(F["model"]), dtype="bfloat16", seed=0, max_model_len=max(len(p) for p in ids) + CAP[a.family] + 8,
              enable_prefix_caching=True)
    if a.kv_gb:
        kw.update(gpu_memory_utilization=0.5, kv_cache_memory_bytes=int(a.kv_gb * 1024 ** 3))
    else:
        kw.update(gpu_memory_utilization=a.gpu_mem)
    llm = LLM(**kw)
    sps = [SamplingParams(temperature=F["temperature"], top_p=F["top_p"], top_k=F.get("top_k", -1), max_tokens=CAP[a.family],
                          stop_token_ids=F["eos"], seed=42 + r["gidx"], detokenize=False) for r in rows]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sps, use_tqdm=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            f.write(json.dumps({"id": r["id"], "subtype": r["subtype"], "gidx": r["gidx"], "token_ids": list(c.token_ids),
                                "finish_reason": c.finish_reason}) + "\n")
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    cap = sum(o.outputs[0].finish_reason == "length" for o in outs)
    print(f"GEN_DONE {a.family} {len(rows)} prompts, {n} tokens, {cap} at the cap, {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
