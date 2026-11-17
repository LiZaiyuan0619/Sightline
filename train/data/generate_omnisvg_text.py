"""A2 step 2 (T6): self-generate text2svg training data on GPU.

Engine: the exported pure-text Qwen2ForCausalLM (bit-identical to the target's
language side; T2-verified against the official stack in this env) with HF
generate, batched over single-subtype batches. Sampling params are the OFFICIAL
per-subclass defaults (refs/OmniSVG config.yaml; also the design notes step 2):
  icon:         temperature 0.5, top_p 0.88, top_k 50, repetition_penalty 1.05
  illustration: temperature 0.6, top_p 0.90, top_k 60, repetition_penalty 1.03
max_new_tokens 1536, one sample per prompt. Training data only needs to come from
the target's distribution, so the engine/env choice is free.

Units & resume: the pool is split into units of --unit-size prompts (500 by
default, so each class has 100 units = 50k prompts). Each unit writes
OUT/<class>/unit-XXXX/results.jsonl + DONE (json summary). Re-runs skip DONE
units; a failed batch is logged to failures.log and the run continues, and the
unit then gets PARTIAL instead of DONE, so a re-run regenerates it.

Output schema matches T1 (candidates[0].raw_token_ids/token_ids/eos_found) so
sightline/sightline/svgtok/loops.py and the analysis tools work unchanged.

Cost (an interactive GPU node 2026-10-09): HF static batching runs every batch to 1536 steps, so
the cost is ~1.29 s per prompt at batch 64 regardless of output length
(20k prompts ~7.2 GPU-h, 100k ~36 GPU-h).

Usage (the omnisvg environment, GPU):
  python gen_data.py --class icon --units 0-19    # first 10k icons
  python gen_data.py --class illustration --units all
  # batch-4 extension (200k): units restart at 0 in the new pool; seeds 142+u never reuse
  # the first 100k's 42..141
  ... gen_data.py --class icon --units 0-19 --pool $SIGHTLINE_RUNS/t6-ext/train-ids --out $SIGHTLINE_RUNS/t6-ext/data --seed 142
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import argparse
import json
import time
from pathlib import Path
from sightline.paths import MODELS, OUT, RUNS

ROOT = Path(__file__).resolve().parents[1]
MODEL = MODELS
POOL = RUNS / "2026-10-09-prompt-pool"

BOS, EOS, PAD = 196998, 196999, 151643
# Official per-subclass sampling defaults (refs/OmniSVG config.yaml text_to_svg_*).
SAMPLE_PARAMS = {
    "icon":         {"temperature": 0.5, "top_p": 0.88, "top_k": 50,  "repetition_penalty": 1.05},
    "illustration": {"temperature": 0.6, "top_p": 0.90, "top_k": 60,  "repetition_penalty": 1.03},
}


def parse_units(spec: str, n_units: int) -> list[int]:
    if spec == "all":
        return list(range(n_units))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [u for u in out if 0 <= u < n_units]


def run_units(cls: str, units: list[int], args, model, torch):
    src = Path(args.pool) / f"{cls}.jsonl"
    rows = [json.loads(l) for l in src.read_text().split(chr(10)) if l.strip()]
    n_units = (len(rows) + args.unit_size - 1) // args.unit_size
    out_root = Path(args.out) / cls
    out_root.mkdir(parents=True, exist_ok=True)
    params = SAMPLE_PARAMS[cls]

    for u in units:
        unit_dir = out_root / f"unit-{u:04d}"
        unit_rows = rows[u * args.unit_size:(u + 1) * args.unit_size]
        if not unit_rows:
            continue
        if (unit_dir / "DONE").exists():
            print(f"[skip] {cls} unit {u} (DONE)")
            continue
        unit_dir.mkdir(parents=True, exist_ok=True)
        seed = args.seed + u
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        n_tok = 0
        n_eos = 0
        n_fail = 0
        with (unit_dir / "results.jsonl").open("w") as log:
            for b0 in range(0, len(unit_rows), args.batch_size):
                batch = unit_rows[b0:b0 + args.batch_size]
                try:
                    bt0 = time.perf_counter()
                    maxlen = max(len(r["prompt_input_ids"]) for r in batch)
                    input_ids = torch.full((len(batch), maxlen), PAD, dtype=torch.long)
                    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
                    for i, r in enumerate(batch):        # left padding
                        p = r["prompt_input_ids"]
                        input_ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
                        attn[i, maxlen - len(p):] = 1
                    input_ids, attn = input_ids.cuda(), attn.cuda()
                    cfg = dict(do_sample=True, max_new_tokens=args.max_new_tokens,
                               num_return_sequences=1, use_cache=True,
                               eos_token_id=EOS, pad_token_id=PAD, bos_token_id=BOS,
                               no_repeat_ngram_size=0, **params)
                    with torch.no_grad():
                        out = model.generate(input_ids=input_ids, attention_mask=attn, **cfg)
                    gen = out[:, maxlen:].cpu().tolist()
                    wall = time.perf_counter() - bt0
                except Exception as e:                    # noqa: BLE001 - log & continue
                    with (out_root / "failures.log").open("a") as fl:
                        fl.write(f"{cls} unit {u} batch {b0}: {e!r}\n")
                    n_fail += len(batch)
                    continue
                for i, r in enumerate(batch):
                    raw = gen[i]
                    eos_found = EOS in raw
                    tok = raw[:raw.index(EOS) + 1] if eos_found else raw
                    n_tok += len(raw)
                    n_eos += int(eos_found)
                    log.write(json.dumps({
                        "id": r["id"], "task": "text2svg", "subtype": cls,
                        "text": r["text"], "seed": seed,
                        "generation_config": {"engine": "omnisvg1.1_8B-qwen2-text-hf",
                                              **cfg, "batch_size": len(batch)},
                        "prompt_input_ids": r["prompt_input_ids"],
                        "candidates": [{"index": 0, "raw_token_ids": raw,
                                        "token_ids": tok, "length": len(tok),
                                        "eos_found": eos_found}],
                        "timing": {"wall_s": round(wall, 3), "new_tokens": len(raw)},
                    }, ensure_ascii=False) + "\n")
        wall = time.perf_counter() - t0
        # "tokens" counts padded slots (batch x steps), i.e. the compute spent.
        marker = "PARTIAL" if n_fail else "DONE"
        (unit_dir / marker).write_text(json.dumps(
            {"class": cls, "unit": u, "n_rows": len(unit_rows), "n_failed": n_fail,
             "eos": n_eos, "tokens": n_tok, "wall_s": round(wall, 1),
             "tok_per_s": round(n_tok / wall, 1) if wall else None}) + "\n")
        print(f"[{marker.lower()}] {cls} unit {u}: {len(unit_rows)} rows "
              f"({n_fail} failed), {n_tok} tok, {n_eos} eos, {wall:.0f}s "
              f"({n_tok/wall:.0f} tok/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="cls", choices=["icon", "illustration"], required=True)
    ap.add_argument("--units", default="all", help="e.g. 0-39, all, 3,5,7")
    ap.add_argument("--batch-size", type=int, default=64)    # 64: +19% over 32 (an interactive GPU node )
    ap.add_argument("--unit-size", type=int, default=500)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=Path, default=MODEL / "OmniSVG1.1_8B-qwen2-text")
    ap.add_argument("--out", type=Path, default=RUNS / "2026-10-09-t6-data")
    # batch 4: the 200k extension lives on the user disk (extend_pool.py)
    ap.add_argument("--pool", type=Path, default=POOL / "train-ids",
                    help="dir with {icon,illustration}.jsonl prompt ids")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GPU required")
    from transformers import AutoModelForCausalLM
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    print(f"model loaded in {time.perf_counter()-t0:.0f}s")
    # warm-up (excluded from unit timings)
    warm = torch.full((1, 8), PAD, dtype=torch.long).cuda()
    with torch.no_grad():
        model.generate(input_ids=warm, attention_mask=torch.ones_like(warm),
                       max_new_tokens=4, do_sample=False,
                       eos_token_id=EOS, pad_token_id=PAD, bos_token_id=BOS)

    units = parse_units(args.units, 10 ** 9)
    run_units(args.cls, units, args, model, torch)


if __name__ == "__main__":
    main()
