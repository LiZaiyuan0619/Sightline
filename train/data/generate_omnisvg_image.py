"""H3a S3: self-generate image-to-SVG training data with the OFFICIAL OmniSVG stack
(the omnisvg environment, transformers 4.51 — the stack T1 reproduced the image2svg metrics with; 4.51
derives the M-RoPE grid positions from input_ids + image_grid_thw + attention_mask itself).

Images: $SIGHTLINE_RUNS/h3a-data/train_pool-<class>.jsonl (build_sets.py), official preprocessing
(preprocess_image_for_svg, replace_background=True -> 448 px RGB) and the official image2svg
prompt (prepare_inputs' messages), batched with left padding. Official image2svg sampling:
T 0.3, top-p 0.90, top-k 50, repetition penalty 1.05; max_new 1536; one sample per image.

Units of --unit-size images per class, resumable like gen_data.py (DONE / PARTIAL per
unit), output schema = T6 (candidates[0].raw_token_ids / token_ids / eos_found) plus image_path,
so the C1 loop rule and the analysis tools apply unchanged.

Usage (the omnisvg environment, GPU):
  python gen_i2svg.py --class icon --units 0-9 [--batch-size 32]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from sightline.paths import MODELS, REFS, RUNS

ROOT = Path(__file__).resolve().parents[1]
REF = REFS / "OmniSVG"
MODEL = MODELS
DATA = (RUNS / "h3a-data")
PARAMS = {"temperature": 0.3, "top_p": 0.90, "top_k": 50, "repetition_penalty": 1.05}
INSTRUCTION = "Generate SVG code that accurately represents this image:"


def load_official():
    sys.path.insert(0, str(REF))
    prev = Path.cwd()
    try:
        os.chdir(REF)
        official = importlib.import_module("inference")
    finally:
        os.chdir(prev)
    official.CONFIG_PATH = str(REF / "config.yaml")
    return official


def parse_units(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def batch_inputs(official, images):
    """prepare_inputs('image-to-svg',...) for a batch of preprocessed PIL images."""
    from qwen_vl_utils import process_vision_info
    msgs = [[{"role": "system", "content": official.SYSTEM_PROMPT},
             {"role": "user", "content": [{"type": "text", "text": INSTRUCTION},
                                          {"type": "image", "image": im}]}] for im in images]
    texts = [official.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in msgs]
    image_inputs, _ = process_vision_info(msgs)
    return official.processor(text=texts, images=image_inputs, padding=True, truncation=True,
                              return_tensors="pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", choices=["icon", "illustration"], required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--unit-size", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=1042)
    ap.add_argument("--out", type=Path, default=DATA / "gen")
    args = ap.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GPU required")
    official = load_official()
    official.load_models("8B", str(MODEL / "OmniSVG1.1_8B"), str(MODEL / "Qwen2.5-VL-7B-Instruct"))
    model = official.sketch_decoder.transformer
    dev = official.get_model_input_device()
    pool = [json.loads(l) for l in (DATA / f"train_pool-{args.cls}.jsonl").read_text().split("\n")
            if l.strip()]
    out_root = args.out / args.cls
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = dict(do_sample=True, max_new_tokens=args.max_new_tokens, num_return_sequences=1,
               use_cache=True, eos_token_id=official.EOS_TOKEN_ID, pad_token_id=official.PAD_TOKEN_ID,
               bos_token_id=official.BOS_TOKEN_ID, no_repeat_ngram_size=0, **PARAMS)
    for u in parse_units(args.units):
        rows = pool[u * args.unit_size:(u + 1) * args.unit_size]
        unit_dir = out_root / f"unit-{u:04d}"
        if not rows or (unit_dir / "DONE").exists():
            continue
        unit_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed + u)
        t0 = time.perf_counter()
        n_fail = n_tok = n_eos = 0
        with (unit_dir / "results.jsonl").open("w") as log:
            for b0 in range(0, len(rows), args.batch_size):
                batch = rows[b0:b0 + args.batch_size]
                try:
                    images = [official.preprocess_image_for_svg(r["image_path"], replace_background=True)[0]
                              for r in batch]
                    inputs = batch_inputs(official, images)
                    inputs = {k: (v.to(dev, dtype=official.DTYPE) if k == "pixel_values" else v.to(dev))
                              for k, v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(**inputs, **cfg)
                    L = inputs["input_ids"].shape[1]
                    gen = out[:, L:].cpu().tolist()
                    ids = inputs["input_ids"].cpu()
                    att = inputs["attention_mask"].cpu()
                except Exception as e:                      # noqa: BLE001 - log & continue
                    with (out_root / "failures.log").open("a") as fl:
                        fl.write(f"{args.cls} unit {u} batch {b0}: {e!r}\n")
                    n_fail += len(batch)
                    continue
                for i, r in enumerate(batch):
                    raw = gen[i]
                    eos = official.EOS_TOKEN_ID in raw
                    tok = raw[:raw.index(official.EOS_TOKEN_ID) + 1] if eos else raw
                    prompt = ids[i][att[i].bool()].tolist()          # strip left padding
                    n_tok += len(raw)
                    n_eos += int(eos)
                    log.write(json.dumps({
                        "id": r["id"], "task": "image2svg", "subtype": args.cls,
                        "image_path": r["image_path"], "seed": args.seed + u,
                        "generation_config": {"engine": "official-omnisvg-4.51", **PARAMS,
                                              "max_new_tokens": args.max_new_tokens,
                                              "batch_size": len(batch)},
                        "prompt_input_ids": prompt,
                        "candidates": [{"index": 0, "raw_token_ids": raw, "token_ids": tok,
                                        "length": len(tok), "eos_found": eos}]}) + "\n")
        wall = time.perf_counter() - t0
        marker = "PARTIAL" if n_fail else "DONE"
        (unit_dir / marker).write_text(json.dumps({"class": args.cls, "unit": u, "n_rows": len(rows),
                                                   "n_failed": n_fail, "eos": n_eos, "tokens": n_tok,
                                                   "wall_s": round(wall, 1)}) + "\n")
        print(f"[{marker.lower()}] {args.cls} unit {u}: {len(rows)} rows ({n_fail} failed), "
              f"{n_eos} eos, {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
