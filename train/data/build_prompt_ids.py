"""A2 step 2 prep (T6): build prompt input_ids for the prompt pools on dev CPU.

Calls the OFFICIAL prepare_inputs (refs/OmniSVG/inference.py) with the official
processor injected (no models loaded) — byte-exact by construction. The official
prompt strings have non-obvious whitespace (trailing space in SYSTEM_PROMPT line 1,
an 8-space line inside the instruction wrapper); hardcoding them drifts, so we
don't.

Self-check (required before use): rebuild the 300 MMSVGBench text2svg prompts and
compare token-by-token against the prompt_input_ids T1 stored in results.jsonl.

Usage (the omnisvg environment, CPU):
  python build_prompts.py --check-only     # self-check
  python build_prompts.py                  # build pools
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
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from sightline.paths import MODELS, REFS, RUNS

ROOT = Path(__file__).resolve().parents[1]
REF = REFS / "OmniSVG"
MODEL = MODELS
POOL = RUNS / "2026-10-09-prompt-pool"
T1 = RUNS / "2026-10-07-omnisvg-t1"
INPUTS = RUNS / "2026-10-07-omnisvg-inputs"


def load_official_text_only():
    """Import refs/OmniSVG/inference.py (run.py's dance) and inject the processor
    without loading any model weights."""
    import torch
    sys.path.insert(0, str(REF))
    previous = Path.cwd()
    try:
        os.chdir(REF)  # Upstream inference.py opens./config.yaml at import time.
        probe = (nullcontext() if torch.cuda.is_available()
                 else patch("torch.cuda.is_bf16_supported", return_value=False))
        with probe:
            official = importlib.import_module("inference")
    finally:
        os.chdir(previous)
    official.CONFIG_PATH = str(REF / "config.yaml")

    from transformers import AutoProcessor
    official.processor = AutoProcessor.from_pretrained(
        MODEL / "Qwen2.5-VL-7B-Instruct", padding_side="left", trust_remote_code=True)
    official.processor.tokenizer.padding_side = "left"
    return official


def build_prompt_ids(official, text: str) -> list[int]:
    inputs = official.prepare_inputs("text-to-svg", text)
    return inputs["input_ids"][0].tolist()


def check_against_t1(official) -> tuple[int, int]:
    """Rebuild bench prompts, compare with T1's stored ids. Returns (n, mismatches)."""
    t1_prompts = {}          # text -> prompt_input_ids (from any T1 text2svg cell)
    for cell in T1.glob("*/text2svg-*/results.jsonl"):
        for line in cell.read_text().split(chr(10)):
            if not line.strip():
                continue
            r = json.loads(line)
            t1_prompts.setdefault(r["text"], r["prompt_input_ids"])
    n = bad = 0
    for sub in ("text2svg-icon", "text2svg-illustration"):
        for line in (INPUTS / sub / "manifest.jsonl").read_text().split(chr(10)):
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            if text not in t1_prompts:
                bad += 1
                print(f"  [check] bench prompt not found in T1 store: {text[:60]!r}")
                continue
            n += 1
            mine = build_prompt_ids(official, text)
            if mine != t1_prompts[text]:
                bad += 1
                print(f"  [check] MISMATCH {sub} {text[:60]!r}")
    return n, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    official = load_official_text_only()
    n, bad = check_against_t1(official)
    print(f"self-check: {n} bench prompts compared, {bad} mismatches")
    if bad:
        raise SystemExit("PROMPT FORMAT DRIFT — do not use")
    if args.check_only:
        return

    for group in ("train-pool", "eval-pool"):
        for cls in ("icon", "illustration"):
            src = POOL / group / f"{cls}.jsonl"
            rows = [json.loads(l) for l in src.read_text().split(chr(10)) if l.strip()]
            out = POOL / f"{group.replace('-pool', '')}-ids" / f"{cls}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w") as f:
                for r in rows:
                    f.write(json.dumps({"id": r["id"], "text": r["text"],
                                        "subtype": cls,
                                        "prompt_input_ids": build_prompt_ids(official, r["text"])},
                                       ensure_ascii=False) + "\n")
            lens = [len(json.loads(l)["prompt_input_ids"])
                    for l in out.read_text().split(chr(10)) if l.strip()]
            print(f"{out}: {len(lens)} prompts, len min/med/max = "
                  f"{min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")


if __name__ == "__main__":
    main()
