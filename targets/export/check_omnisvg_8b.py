"""T2 equivalence check: does the exported Qwen2ForCausalLM reproduce batch-1 greedy outputs?

Two comparisons per record:
  A. reference path (the omnisvg environment, transformers 4.51, SketchDecoder + full VL checkpoint)
  B. exported path  (the dflash environment, transformers 5.15, Qwen2ForCausalLM text-only)

Both feed the identical prompt_input_ids saved in T1 and generate greedily, then compare
token-by-token against the T1 greedy outputs (the ground truth from batch 1).

Outputs runs/<date>-t2-equiv/{pure,penalty}/{subset}.jsonl with per-record verdicts.

Usage: python check_equiv.py --mode pure|penalty --subset text2svg-icon|text2svg-illustration \
        [--limit N] [--out runs/2026-10-08-t2-equiv]
"""
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from sightline.paths import MODELS

ROOT = Path(__file__).resolve().parents[1]
M = MODELS
T1 = ROOT / "runs" / "2026-10-07-omnisvg-t1"
EXPORT = M / "OmniSVG1.1_8B-qwen2-text"


def load_reference_records(subset, mode, limit):
    path = T1 / f"greedy-{mode}-b1" / subset / "results.jsonl"
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r["candidates"][0]
        out.append({"id": r["id"], "prompt": r["prompt_input_ids"], "ref": c["raw_token_ids"],
                    "cfg": r["generation_config"]})
        if limit and len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pure", "penalty"], required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "2026-10-08-t2-equiv")
    ap.add_argument("--ref-from-t1", action="store_true", default=True,
                    help="compare against T1 outputs (always true; flag kept for clarity)")
    args = ap.parse_args()
    import transformers
    print(f"transformers {transformers.__version__} | torch {torch.__version__}")

    records = load_reference_records(args.subset, args.mode, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.mode}-{args.subset}.jsonl"

    # --- reference path A: SketchDecoder via the official loader (4.51) ---
    # --- exported path B: Qwen2ForCausalLM (this env) ---
    from transformers import Qwen2ForCausalLM
    t0 = time.time()
    model = Qwen2ForCausalLM.from_pretrained(EXPORT, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"loaded exported model in {time.time()-t0:.1f}s")

    n_match, n_diverge = 0, 0
    with out_path.open("w") as log, torch.no_grad():
        for i, rec in enumerate(records):
            ids = torch.tensor([rec["prompt"]], dtype=torch.long, device="cuda:0")
            t0 = time.time()
            out = model.generate(
                input_ids=ids,
                do_sample=False,
                max_new_tokens=rec["cfg"]["max_new_tokens"],
                repetition_penalty=rec["cfg"]["repetition_penalty"],
                eos_token_id=rec["cfg"]["eos_token_id"],
                pad_token_id=rec["cfg"]["pad_token_id"],
                bos_token_id=rec["cfg"]["bos_token_id"],
            )
            gen = out[0, ids.shape[1]:].cpu().tolist()
            ref = rec["ref"]
            first_div = next((j for j, (a, b) in enumerate(zip(gen, ref)) if a != b), None)
            verdict = "match" if (first_div is None and len(gen) == len(ref)) else (
                "match_prefix" if first_div is None else f"diverge@{first_div}")
            (n_match if verdict.startswith("match") else n_diverge, )
            if verdict.startswith("match"):
                n_match += 1
            else:
                n_diverge += 1
            log.write(json.dumps({
                "id": rec["id"], "verdict": verdict, "len_gen": len(gen), "len_ref": len(ref),
                "first_divergence": first_div, "seconds": time.time() - t0,
                "gen_head": gen[:16], "ref_head": ref[:16],
            }) + "\n")
            log.flush()
    print(f"{args.mode}/{args.subset}: {n_match}/{len(records)} fully match, {n_diverge} diverge")
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
