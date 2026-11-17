""" InternSVG-8B text2svg generation with vLLM (training data for its DFlash head, and fast reference runs).

Model: the exported text-only Qwen2ForCausalLM (model/InternSVG-8B-qwen2-text; fp32-identical to the InternVL
checkpoint on text prompts, check_internsvg.py). Protocol = the official SArena generation eval: greedy,
max 4000 new tokens, EOS <|im_end|>, prompt in the SFT template (run.py). Prompts are tokenized here with
the HF tokenizer (default regex, = training-time transformers 4.52) and passed to vLLM as token ids.

Pools:
  mmsvg   runs/2026-10-09-prompt-pool/train-pool/<class>.jsonl (50k captions per class, the OmniSVG main100k captions),
          each wrapped in one of SArena's 10 text2svg instruction templates (seeded per prompt index);
  mmsvg-ext the 200k extension captions of OmniSVG's main300k ($SIGHTLINE_RUNS/t6-ext/train-ids/<class>.jsonl, 100k per class);
          use --seed 142 so the template draws differ from the main pool;
  sarena the SArena test rows that run.py samples (seed 0), for checking against the HF runs.
Units of --unit-size prompts write OUT/<class>/unit-XXXX/results.jsonl + DONE; re-runs skip DONE units.

  python gen_vllm.py \
      --pool mmsvg --class icon --units 0-9 --out <dir> [--limit N] [--gpu-mem 0.85]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from targets.internsvg_generate import SARENA, FILES, SYSTEM, IM_END, rows as sarena_rows  # noqa: E402
from sightline.paths import MODELS, OUT, RUNS

EXPORT = MODELS / "InternSVG-8B-qwen2-text"
POOL = RUNS / "2026-10-09-prompt-pool/train-pool"
POOL_EXT = (RUNS / "t6-ext/train-ids")


def templates() -> list[str]:
    """The 10 instruction prefixes of SArena text2svg (identical across icon / illustration)."""
    seen = []
    for sub in ("icon", "illustration"):
        for line in (SARENA / FILES[("text2svg", sub)]).read_text().splitlines():
            if line.strip():
                p = json.loads(line)["conversations"][0]["value"].split("Instruction:")[0]
                if p not in seen:
                    seen.append(p)
    assert len(seen) == 10, len(seen)
    return sorted(seen)


def parse_units(spec: str, n_units: int) -> list[int]:
    if spec == "all":
        return list(range(n_units))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def chat(q: str) -> str:
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["mmsvg", "mmsvg-ext", "sarena"], required=True)
    ap.add_argument("--class", dest="cls", choices=["icon", "illustration"], required=True)
    ap.add_argument("--units", default="all")
    ap.add_argument("--unit-size", type=int, default=500)
    ap.add_argument("--n", type=int, default=150, help="sarena pool: rows sampled as in run.py")
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N prompts of each unit")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42, help="template choice: Random(seed + prompt index)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="10-21: 0 = greedy (the official SArena protocol, the 10-15 data); e.g. 0.6 with --top-p 0.9 "
                         "for sampled drafter data matching the T=0.6 best-of-N caliber")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(EXPORT)
    if a.pool in ("mmsvg", "mmsvg-ext"):
        src = (POOL if a.pool == "mmsvg" else POOL_EXT) / f"{a.cls}.jsonl"
        pool = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        T = templates()
        items = [{"id": r["id"], "question": f"{T[random.Random(a.seed + i).randrange(10)]}Instruction: {r['text']}"
                  + ("\n" if a.cls == "illustration" else "")} for i, r in enumerate(pool)]
    else:
        items = [{"id": r["id"], "question": r["question"]} for r in sarena_rows("text2svg", a.cls, a.n)]
    n_units = (len(items) + a.unit_size - 1) // a.unit_size
    units = parse_units(a.units, n_units)
    llm = LLM(model=str(EXPORT), dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096 + 512,
              enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, stop_token_ids=[IM_END],
                        detokenize=False)
    for u in units:
        d = a.out / a.cls / f"unit-{u:04d}"
        if (d / "DONE").exists():
            print("skip", d, flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        part = items[u * a.unit_size:(u + 1) * a.unit_size][:a.limit]
        ids = [tok(chat(x["question"]), add_special_tokens=False).input_ids for x in part]
        t0 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ids], sp, use_tqdm=False)
        dt = time.time() - t0
        n_tok = 0
        with (d / "results.jsonl").open("w") as f:
            for x, p, o in zip(part, ids, outs):
                g = list(o.outputs[0].token_ids)
                eos = bool(g) and g[-1] == IM_END
                n_tok += len(g)
                f.write(json.dumps({"id": x["id"], "subtype": a.cls, "question": x["question"], "prompt_input_ids": p,
                                    "token_ids": g, "eos": eos, "n_gen": len(g)}) + "\n")
        stats = {"n": len(part), "tokens": n_tok, "seconds": round(dt, 1), "tok_per_s": round(n_tok / dt, 1),
                 "temperature": a.temperature, "top_p": a.top_p,
                 "eos": sum(1 for o in outs if o.outputs[0].token_ids and o.outputs[0].token_ids[-1] == IM_END)}
        (d / "DONE").write_text(json.dumps(stats))
        print(d, stats, flush=True)
    print("GEN_VLLM_DONE", a.pool, a.cls, a.units)


if __name__ == "__main__":
    main()
