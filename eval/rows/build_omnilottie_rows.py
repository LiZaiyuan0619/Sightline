""" (C1): evaluation rows for OmniLottie = MMLottieBench's Text-to-Lottie items (Real 150 + Synthetic 150), prompt
ids in the official chat format (gen_vllm_lottie.chat: the official simplify_to_animation_description + the
Qwen2.5-VL chat template; identical to the processor's, checked there). Row schema of eval_tree.py:
    {id, set "mmlottie-bench", subtype real | synthetic, text, prompt_input_ids, gidx 0-299 (Real first)}
-> $SIGHTLINE_RUNS/tableA/xbench/omnilottie-bench.jsonl.
  python build_lottie_rows.py
"""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from gen_vllm_lottie import EXPORT, chat  # noqa: E402
from sightline.paths import MODELS, RUNS

BENCH = (MODELS / "MMLottieBench/data")
OUT = (RUNS / "tableA/xbench/omnilottie-bench.jsonl")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EXPORT)
    rows = []
    for sub in ("real", "synthetic"):
        t = pq.read_table(BENCH / f"{sub}-00000-of-00001.parquet", columns=["id", "text", "task_type"]).to_pylist()
        for r in t:
            if r["task_type"] == "Text-to-Lottie":
                rows.append({"id": r["id"], "set": "mmlottie-bench", "subtype": sub, "text": r["text"],
                             "prompt_input_ids": tok(chat(r["text"]), add_special_tokens=False).input_ids})
    for i, r in enumerate(rows):
        r["gidx"] = i
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "rows ->", OUT, {s: sum(r["subtype"] == s for r in rows) for s in ("real", "synthetic")},
          "max prompt", max(len(r["prompt_input_ids"]) for r in rows))


if __name__ == "__main__":
    main()
