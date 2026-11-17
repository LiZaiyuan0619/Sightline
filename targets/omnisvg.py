"""H3a S1: the OmniSVG VLM target in the strict harness (the glance environment, transformers 5.15).

- load_target: OmniSVG 1.1 8B as Qwen2_5_VLForConditionalGeneration (T8 key remap,
  src/t8/s2_vlm_path.py).
- I2SVGInputs: official preprocessing (preprocess_image_for_svg + prepare_inputs, as T1 / T8 S2)
  and M-RoPE positions computed ONCE per prompt with get_rope_index (the 5.15 signature with
  mm_token_type_ids, VLMPositions515). Every forward then passes positions explicitly:
  the prefill its 3-D grid positions, later steps (pos + rope delta) broadcast over the three
  axes. The AR loop and the speculative loops use the same positions, so neither depends on
  5.15's implicit path (which silently falls back to 1-D positions without mm_token_type_ids).
- vlm_ar_generate: the harness AR loop (same processing chain, NFE counted) with VLM prefill.
- The draft head keeps 1-D positions throughout (it never sees the grid; GLANCE glance/vl.py),
  so sightline.tree_spec.tree_spec_generate runs unchanged given prefill_kwargs + positions.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.ar import GenResult, _pick                       # noqa: E402
from sightline.process import draw, eos_only, loop_rollback     # noqa: E402

IMAGE_TOKEN_ID = 151655


def load_official():
    from s2_vlm_path import load_official_module
    return load_official_module()


def load_target(dtype=torch.bfloat16):
    from s2_vlm_path import load_omnisvg_vlm
    model, note = load_omnisvg_vlm()
    if dtype != torch.bfloat16:
        model = model.to(dtype)
    return model, note


@dataclass
class I2SVGInputs:
    prompt_ids: list
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    prefill_positions: torch.Tensor     # (3, 1, n_prompt)
    delta: int

    def prefill_kwargs(self) -> dict:
        return {"position_ids": self.prefill_positions, "pixel_values": self.pixel_values,
                "image_grid_thw": self.image_grid_thw}

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        """1-D stream positions (1, n) -> M-RoPE positions (3, 1, n) after the prefill."""
        return (pos + self.delta).unsqueeze(0).expand(3, -1, -1)


def build_inputs(official, model, image_path: str, tmp_png: Path, dtype=torch.bfloat16) -> I2SVGInputs:
    """Official preprocessing (as T1 / T8 S2) + M-RoPE bind (VLMPositions515.bind)."""
    image, _ = official.preprocess_image_for_svg(image_path, replace_background=True)
    tmp_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(tmp_png)
    inputs = official.prepare_inputs("image-to-svg", str(tmp_png))
    dev = next(model.parameters()).device
    ids = inputs["input_ids"].to(dev)
    mm = inputs.get("mm_token_type_ids")
    mm = (ids == IMAGE_TOKEN_ID).long() if mm is None else mm.to(dev)
    thw = inputs["image_grid_thw"].to(dev)
    model.model.rope_deltas = None
    positions, deltas = model.model.get_rope_index(ids, mm, thw, None, None)
    return I2SVGInputs(prompt_ids=ids[0].tolist(),
                       pixel_values=inputs["pixel_values"].to(dev, dtype=dtype),
                       image_grid_thw=thw, prefill_positions=positions,
                       delta=int(deltas[0, 0]))


@torch.inference_mode()
def vlm_ar_generate(model, inp: I2SVGInputs, max_new_tokens: int, eos_ids: list[int], params,
                    generator: torch.Generator | None = None) -> GenResult:
    from transformers import DynamicCache
    dev = next(model.parameters()).device
    ids = torch.tensor([inp.prompt_ids], device=dev)
    cache = DynamicCache()
    out = model(ids, past_key_values=cache, use_cache=True, **inp.prefill_kwargs())
    nfe = 1
    history = list(inp.prompt_ids)
    n_p = len(inp.prompt_ids)
    logits = out.logits[0, -1]
    n_rb, ban = 0, None                  # rollback escape state, as in sightline.ar.ar_generate
    stop = "max_new_tokens"
    while True:
        act, new_len, btok = loop_rollback(history, params, n_rb, n_p)
        if act == "rollback":            # keep one cycle; re-feed the last kept token at its position
            history = history[:new_len]
            cache.crop(new_len - 1)
            pos = torch.tensor([[new_len - 1]], device=dev)
            out = model(torch.tensor([[history[-1]]], device=dev), position_ids=inp(pos),
                        past_key_values=cache, use_cache=True)
            nfe += 1
            logits = out.logits[0, -1]
            n_rb, ban = n_rb + 1, (new_len, btok)
            continue
        if act == "eos":
            nxt = int(draw(eos_only(logits.float(), eos_ids[0]), params.greedy, generator))
        elif act == "ban":                # escape-then-stop: ban the cycle's continuation here
            nxt = int(_pick(logits, history, params, generator, extra_ban=btok))
            n_rb += 1
        else:
            eb = ban[1] if ban is not None and ban[0] == len(history) else None
            nxt = int(_pick(logits, history, params, generator, extra_ban=eb))
        history.append(nxt)
        if nxt in eos_ids:
            stop = "eos"
            break
        if len(history) - n_p >= max_new_tokens:
            break
        pos = torch.tensor([[len(history) - 1]], device=dev)
        out = model(torch.tensor([[nxt]], device=dev), position_ids=inp(pos),
                    past_key_values=cache, use_cache=True)
        nfe += 1
        logits = out.logits[0, -1]
    return GenResult(tokens=history[n_p:], nfe=nfe, stop_reason=stop)
