""" the InternSVG-8B image-to-SVG target in the strict harness (the glance environment, transformers 5.15) = vlm.py for
table 1's 5th block. InternSVG-8B is a transformers-native InternVLForConditionalGeneration (model/InternSVG-8B: InternViT
448 px + a 28-layer Qwen2 LM). Differences from the OmniSVG VLM:
  - positions are 1-D (Qwen2 RoPE; its dynamic NTK factor 2 only acts above 32,768 positions, i.e. never here), so after
    the prefill the harness's StreamPositions semantics apply: I2SInputs(pos) = pos, delta = 0;
  - one 448 x 448 tile (processor crop_to_patches false) = 256 <IMG_CONTEXT> (151667) between <img> (151665) and </img>
    (151666), the expansion vLLM made for the training data (gen_vllm_i2s.py), so prompt ids are identical;
  - prompt: the SFT chat format of run.py (InternVL system prompt, no BOS) around the row's own instruction
    (SArena img2svg template with one "<image>");
  - pixels: i2s_image.load_i2s_image (white background, RGB, 448) -> the model's GotOcr2ImageProcessorFast.
Decoding protocol (official SArena inference): greedy without penalty, EOS <|im_end|> 151645, at most 4000 new tokens.
API as vlm.py: load_processor (for load_official), load_target, build_inputs, vlm_ar_generate (reused unchanged:
it only uses prompt_ids, prefill_kwargs and the position map), plus target_features for the draft trainers.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from internsvg.i2s_image import load_i2s_image  # noqa: E402
from targets.internsvg_generate import SYSTEM  # noqa: E402
from vlm import vlm_ar_generate  # noqa: E402,F401  (vlm.py; re-exported)
from sightline.paths import MODELS

MODEL = MODELS / "InternSVG-8B"
IMAGE_TOKEN_ID, IMG_START_ID, IMG_END_ID = 151667, 151665, 151666
N_IMAGE_TOKENS = 256
EOS, MAX_NEW, PAD, MASK = 151645, 4000, 151643, 151662
IMAGE_TEXT = "<img>" + "<IMG_CONTEXT>" * N_IMAGE_TOKENS + "</img>"


def chat(prompt: str) -> str:
    """the row's instruction (one "<image>") in the SFT chat format, image placeholder expanded as vLLM did"""
    assert prompt.count("<image>") == 1, prompt
    q = prompt.replace("<image>", IMAGE_TEXT)
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def load_processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL)


def load_target(dtype=torch.bfloat16):
    from transformers import InternVLForConditionalGeneration
    model = InternVLForConditionalGeneration.from_pretrained(MODEL, dtype=dtype, attn_implementation="sdpa")
    model = model.cuda().eval()
    model.requires_grad_(False)
    return model, f"InternVLForConditionalGeneration {MODEL.name} {dtype}"


def prompt_ids_of(proc, prompt: str) -> list[int]:
    ids = proc.tokenizer(chat(prompt), add_special_tokens=False).input_ids
    assert ids.count(IMAGE_TOKEN_ID) == N_IMAGE_TOKENS, ids.count(IMAGE_TOKEN_ID)
    return ids


def pixel_values(proc, image_path: str, dtype=torch.bfloat16) -> torch.Tensor:
    pv = proc.image_processor(images=[load_i2s_image(image_path)], return_tensors="pt")["pixel_values"]
    assert tuple(pv.shape) == (1, 3, 448, 448), tuple(pv.shape)
    return pv.cuda().to(dtype)


@dataclass
class I2SInputs:
    prompt_ids: list
    pixel_values: torch.Tensor
    delta: int = 0                       # no M-RoPE: the stream position is the RoPE position

    def prefill_kwargs(self) -> dict:
        n = len(self.prompt_ids)
        return {"position_ids": torch.arange(n, device=self.pixel_values.device).unsqueeze(0),
                "pixel_values": self.pixel_values}

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        return pos


def build_inputs(proc, model, row: dict, dtype=torch.bfloat16) -> I2SInputs:
    """row: {prompt (instruction with one <image>) | prompt_input_ids, image_path}"""
    ids = row.get("prompt_input_ids") or prompt_ids_of(proc, row["prompt"])
    return I2SInputs(prompt_ids=list(ids), pixel_values=pixel_values(proc, row["image_path"], dtype))


@torch.inference_mode()
def target_features(model, proc, input_ids: list[int], image_path: str, layers: list[int], logits: bool = False):
    """teacher-forced frozen forward over prompt + response with the image: concatenated hidden states of `layers`
    (HF hidden_states[l + 1]) -> (L, len(layers) * 3584); with logits=True also the full logits (L, vocab)."""
    ids = torch.tensor([input_ids], device="cuda")
    pv = pixel_values(proc, image_path, next(model.parameters()).dtype)
    out = model(ids, pixel_values=pv, output_hidden_states=True, logits_to_keep=0 if logits else 1)
    hs = torch.cat([out.hidden_states[l + 1][0] for l in layers], dim=-1)
    return (hs, out.logits[0]) if logits else hs
