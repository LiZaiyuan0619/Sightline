""" (C2): the StarVector target behind the interface of vlm_i2s.py (load_processor, load_target,
build_inputs, vlm_ar_generate, EOS, MAX_NEW, PAD), so the image-to-SVG evaluator runs on it unchanged except for its
import (eval_sv.py). The model is chosen by $STARVEC (1b | 8b). Target = target.py; prompt =
[image prefix (Q positions)] ++ "<svg" (official generate_im2svg); total length cap 4000 (README) -> MAX_NEW = 4000 - Q - 2;
EOS 0 (<|endoftext|>).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from starvec import target as T  # noqa: E402
from vlm import vlm_ar_generate  # noqa: E402,F401  (vlm.py; re-exported)

WHICH = os.environ.get("STARVEC", "1b")
EOS, PAD = T.EOS, T.PAD
PROMPT = T.prompt_ids(WHICH)
MAX_NEW = 4000 - len(PROMPT)
IMAGE_TOKEN_ID = PAD


def load_processor():
    return T.processor(WHICH)


def load_target(dtype=torch.bfloat16):
    return T.load(WHICH, dtype), f"StarVector-{WHICH} (target.py) {dtype}"


def build_inputs(proc, model, row: dict, dtype=torch.bfloat16) -> T.SVInputs:
    from PIL import Image
    return T.build_inputs(model, proc, Image.open(row["image_path"]), PROMPT)
