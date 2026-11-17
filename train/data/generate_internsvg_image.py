""" InternSVG-8B image-to-SVG generation with vLLM = the drafter training data of table 1's 5th block (and fast
reference runs). = gen_vllm.py for images:
  - model: the full InternVL checkpoint (model/InternSVG-8B, transformers-native InternVLForConditionalGeneration; vLLM 0.16
    runs it as InternS1ForConditionalGeneration, image placeholder "<IMG_CONTEXT>" -> <img> + 256 features + </img>);
  - prompts: build_i2s_prompts.py ($SIGHTLINE_RUNS/isvg-data/i2s/prompts-<class>.jsonl: SArena img2svg template + image +
    MMSVG description), in the SFT chat format of run.py (InternVL system prompt, no BOS);
  - images: i2s_image.load_i2s_image (white background, RGB, 448);
  - decoding: the official SArena protocol, greedy, max 4000 new tokens, EOS <|im_end|> (refs/InternSVG metrics/inference);
  - records: vLLM's own prompt_token_ids (image tokens expanded) + output token ids, so training replays exactly these ids.
Units of --unit-size prompts write OUT/<class>/unit-XXXX/results.jsonl + DONE; re-runs skip DONE units.
 fix: --model model/InternSVG-8B-vllm-norope (the checkpoint with text_config.rope_scaling removed). vLLM's
'dynamic' NTK rope scales the base statically at every length (1e6 -> ~3.05e6) while HF scales only beyond 32,768,
so the data (default --model) came from a different RoPE than the target: teacher-forced agreement with the HF
target only.8% per stream, first divergence at token  (check_i2s.py).
  python gen_vllm_i2s.py \
      --class icon --units 0-9 --out <dir> [--limit N] [--gpu-mem 0.85]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the script directory holds profile.py, which would shadow the stdlib `profile` that vLLM's cProfile
# import needs: drop it from sys.path; our modules are imported as the `internsvg` package
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from internsvg.i2s_image import load_i2s_image  # noqa: E402
from targets.internsvg_generate import SYSTEM, IM_END  # noqa: E402
from sightline.paths import MODELS, OUT, RUNS

MODEL = MODELS / "InternSVG-8B"
PROMPTS = (RUNS / "isvg-data/i2s")
PLACEHOLDER = "<IMG_CONTEXT>"


def chat(prompt: str) -> str:
    assert prompt.count("<image>") == 1, prompt
    q = prompt.replace("<image>", PLACEHOLDER)
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


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
    ap.add_argument("--class", dest="cls", choices=["icon", "illustration"], required=True)
    ap.add_argument("--units", default="all")
    ap.add_argument("--unit-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N prompts of each unit")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--first", type=int, default=0, help="smoke: skip the first N prompts of each unit")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    items = [json.loads(x) for x in open(PROMPTS / f"prompts-{a.cls}.jsonl") if x.strip()]   # split on \n only
    n_units = (len(items) + a.unit_size - 1) // a.unit_size
    units = parse_units(a.units, n_units)
    llm = LLM(model=str(a.model), dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4000 + 640,
              limit_mm_per_prompt={"image": 1}, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop_token_ids=[IM_END], detokenize=False)
    pool = ThreadPoolExecutor(16)
    for u in units:
        d = a.out / a.cls / f"unit-{u:04d}"
        if (d / "DONE").exists():
            print("skip", d, flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        part = items[u * a.unit_size:(u + 1) * a.unit_size][a.first:][:a.limit]
        images = list(pool.map(lambda x: load_i2s_image(x["image_path"]), part))
        reqs = [{"prompt": chat(x["prompt"]), "multi_modal_data": {"image": im}} for x, im in zip(part, images)]
        t0 = time.time()
        outs = llm.generate(reqs, sp, use_tqdm=False)
        dt = time.time() - t0
        n_tok = 0
        with (d / "results.jsonl").open("w") as f:
            for x, o in zip(part, outs):
                g = list(o.outputs[0].token_ids)
                eos = bool(g) and g[-1] == IM_END
                n_tok += len(g)
                f.write(json.dumps({**{k: x[k] for k in ("gidx", "id", "subtype", "image_path", "template", "caption")},
                                    "prompt_input_ids": list(o.prompt_token_ids), "token_ids": g, "eos": eos,
                                    "n_gen": len(g)}) + "\n")
        stats = {"n": len(part), "tokens": n_tok, "seconds": round(dt, 1), "tok_per_s": round(n_tok / max(dt, 1e-6), 1),
                 "eos": sum(1 for o in outs if o.outputs[0].token_ids and o.outputs[0].token_ids[-1] == IM_END)}
        (d / "DONE").write_text(json.dumps(stats))
        print(d, stats, flush=True)
    print("GEN_VLLM_I2S_DONE", a.cls, a.units)


if __name__ == "__main__":
    main()
