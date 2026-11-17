"""Batch 4 TR: extend the text2svg training pool by N prompts per
class for the 300k fair-baseline run.

The universe files written by prompt_pool.py are already shuffled once (seed 42)
and bench-filtered; that script took eval = rows [0, 300) and train = rows [300, 50300).
The extension takes the NEXT rows [50300, 50300 + N), so it is disjoint from the eval pool,
the first 100k training prompts and MMSVGBench by construction. Both facts are re-checked
here against the stored eval-pool / train-pool files before anything is written.

Prompt ids are built with the OFFICIAL prepare_inputs via build_prompts.py (its T1
self-check runs first and must report 0 mismatches).

Output (user disk; gen_data.py reads it via --pool):
  <out>/train-ids/{icon,illustration}.jsonl   {id, text, subtype, prompt_input_ids}
  <out>/manifest.json

Usage (the omnisvg environment, dev CPU):
  python extend_pool.py [--n 100000] [--out <output dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
import build_prompts as bp   # noqa: E402
from sightline.paths import RUNS

POOL = RUNS / "2026-10-09-prompt-pool"
EVAL_N, TRAIN_N = 300, 50000          # the split prompt_pool.py made (manifest.json)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000, help="new prompts per class")
    ap.add_argument("--out", type=Path, default=(RUNS / "t6-ext"))
    args = ap.parse_args()

    manifest = json.loads((POOL / "manifest.json").read_text())
    assert manifest["eval_n"] == EVAL_N and manifest["train_n"] == TRAIN_N, manifest

    official = bp.load_official_text_only()
    n, bad = bp.check_against_t1(official)
    print(f"self-check: {n} bench prompts compared, {bad} mismatches", flush=True)
    if bad:
        raise SystemExit("PROMPT FORMAT DRIFT — do not use")

    (args.out / "train-ids").mkdir(parents=True, exist_ok=True)
    out_manifest = {"source": str(POOL), "rows": [EVAL_N + TRAIN_N, EVAL_N + TRAIN_N + args.n],
                    "classes": {}}
    for cls in ("icon", "illustration"):
        universe = read_jsonl(POOL / f"universe-{cls}.jsonl")
        eval_ids = [r["id"] for r in read_jsonl(POOL / "eval-pool" / f"{cls}.jsonl")]
        train_ids = [r["id"] for r in read_jsonl(POOL / "train-pool" / f"{cls}.jsonl")]
        # the universe order must reproduce the stored split exactly
        assert [r["id"] for r in universe[:EVAL_N]] == eval_ids, f"{cls}: eval split drift"
        assert [r["id"] for r in universe[EVAL_N:EVAL_N + TRAIN_N]] == train_ids, \
            f"{cls}: train split drift"
        ext = universe[EVAL_N + TRAIN_N:EVAL_N + TRAIN_N + args.n]
        if len(ext) < args.n:
            raise SystemExit(f"{cls}: only {len(ext)} rows left, need {args.n}")
        used = set(eval_ids) | set(train_ids)
        assert not any(r["id"] in used for r in ext), f"{cls}: overlap with used prompts"

        t0 = time.perf_counter()
        out = args.out / "train-ids" / f"{cls}.jsonl"
        lens = []
        with out.open("w") as f:
            for i, r in enumerate(ext):
                ids = bp.build_prompt_ids(official, r["text"])
                lens.append(len(ids))
                f.write(json.dumps({"id": r["id"], "text": r["text"], "subtype": cls,
                                    "prompt_input_ids": ids}, ensure_ascii=False) + "\n")
                if (i + 1) % 20000 == 0:
                    print(f"  {cls}: {i + 1}/{len(ext)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        lens.sort()
        stats = {"n": len(ext), "len_min": lens[0], "len_med": lens[len(lens) // 2],
                 "len_max": lens[-1], "universe": len(universe),
                 "left_after": len(universe) - EVAL_N - TRAIN_N - len(ext)}
        out_manifest["classes"][cls] = stats
        print(f"{out}: {json.dumps(stats)}", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(out_manifest, indent=2) + "\n")
    print("wrote", args.out / "manifest.json")


if __name__ == "__main__":
    main()
