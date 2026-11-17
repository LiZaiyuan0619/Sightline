"""A2 step 4 smoke data: SpecForge prompts_path JSONL from the
T1 sample-b8 text2svg outputs.

SMOKE ONLY: these prompts are MMSVGBench evaluation prompts — the task card
explicitly limits them to pipeline bring-up, never real training or evaluation.
Loop handling follows the same C1 rule as build_trainset (truncate at onset + 2
periods; mask-token streams filtered).

Usage (dev CPU):
  python3 make_smoke_data.py [--n-rows 50] [--out runs/2026-10-09-t7-smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from build_trainset import kept_gen, MASK_TOKEN          # noqa: E402  (C1 loop rule)
from sightline.paths import RUNS

T1 = RUNS / "2026-10-07-omnisvg-t1/sample-b8"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=50, help="prompt rows per subtype")
    ap.add_argument("--out", type=Path, default=RUNS / "2026-10-09-t7-smoke")
    args = ap.parse_args()

    stats = Counter()
    records = []
    for cls in ("icon", "illustration"):
        cell = T1 / f"text2svg-{cls}/results.jsonl"
        rows = [json.loads(l) for l in cell.read_text().split("\n") if l.strip()]
        for r in rows[:args.n_rows]:
            for cand in r["candidates"]:
                gen, label, cut = kept_gen(cand["token_ids"], cand["eos_found"])
                stats[f"{cls}_{label}"] += 1
                stats["tokens_before"] += len(cand["token_ids"])
                stats["tokens_cut"] += cut
                if MASK_TOKEN in cand["raw_token_ids"] or MASK_TOKEN in r["prompt_input_ids"]:
                    stats["mask_filtered"] += 1
                    continue
                if len(gen) <= 1:
                    stats["empty"] += 1
                    continue
                stats["kept"] += 1
                stats["tokens_after"] += len(gen)
                records.append({"input_ids": r["prompt_input_ids"] + gen,
                                "loss_mask": [0] * len(r["prompt_input_ids"]) + [1] * len(gen)})

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "smoke_prompts.jsonl"
    with out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    lens = [sum(rec["loss_mask"]) for rec in records]
    print(json.dumps({**{k: v for k, v in stats.items()},
                      "n_records": len(records),
                      "gen_len_med": sorted(lens)[len(lens) // 2] if lens else 0,
                      "total_tokens": sum(lens)}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
