""" InternSVG-8B profile for the SD evaluators (eval_tree.py --profile internsvg).

Replaces the OmniSVG specifics of eval_b3.py:
  - target: the exported text model (model/InternSVG-8B-qwen2-text), EOS <|im_end|> 151645, cap 4000 new tokens;
  - processing chain: greedy = plain argmax (the official SArena protocol: temperature 0, no penalty);
    sample = pure sampling at temperature 1.0 (no official sampling setting exists; robustness caliber only);
  - rows: SArena text2svg. bench = the 150 + 150 rows of run.py (seed 0; the loop survey),
    extra = 300 + 300 further rows (seed 1, disjoint from bench); prompt = the SFT template (run.py), HF tokenizer;
  - loop detector at InternSVG's cap (tail window 1024, max period 512), as the survey.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.verify import ChainParams           # noqa: E402
from targets.internsvg_generate import FILES, SARENA, SYSTEM  # noqa: E402
from sightline.paths import MODELS, RUNS

EOS = 151645
MAX_NEW = 4000
TARGET = MODELS / "InternSVG-8B-qwen2-text"
AR_CACHE = (RUNS / "isvg-eval/ar-cache")
LOOP_KW = {"tail_window": 1024, "max_period": 512}
N_BENCH, N_EXTRA = 150, 300


def chain_params(mode: str, subtype: str) -> ChainParams:
    if mode == "greedy":
        return ChainParams(greedy=True)
    return ChainParams(greedy=False, temperature=1.0)


def _chat(q: str) -> str:
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def load_rows(sets: list[str]) -> list[dict]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET)
    rows = []
    for set_name in ("bench", "extra"):
        for sub in ("icon", "illustration"):
            path = SARENA / FILES[("text2svg", sub)]
            allr = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            bench = sorted(random.Random(0).sample(range(len(allr)), N_BENCH))
            if set_name == "bench":
                pick = bench
            else:
                rest = sorted(set(range(len(allr))) - set(bench))
                pick = sorted(random.Random(1).sample(rest, N_EXTRA))
            if set_name not in sets:
                continue
            for i in pick:
                q = allr[i]["conversations"][0]["value"]
                rows.append({"id": int(allr[i]["id"]), "text": q, "subtype": sub, "set": set_name,
                             "prompt_input_ids": tok(_chat(q), add_special_tokens=False).input_ids})
    order = {("bench", "icon"): 0, ("bench", "illustration"): 1, ("extra", "icon"): 2, ("extra", "illustration"): 3}
    rows.sort(key=lambda r: order[(r["set"], r["subtype"])])
    # stable global index over bench + extra (as eval_b3.load_rows): gidx does not depend on the --sets choice
    full = {("bench", "icon"): 0, ("bench", "illustration"): N_BENCH, ("extra", "icon"): 2 * N_BENCH,
            ("extra", "illustration"): 2 * N_BENCH + N_EXTRA}
    counters = {}
    for r in rows:
        k = (r["set"], r["subtype"])
        r["gidx"] = full[k] + counters.get(k, 0)
        counters[k] = counters.get(k, 0) + 1
    return rows
