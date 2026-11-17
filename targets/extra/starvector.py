""" (C2): StarVector-1B / 8B as a strict-harness target (the glance environment or the dflash environment, transformers 5.15), rebuilt from
the official pieces so the decode loop is the same HF causal LM the harness already drives:
  - image encoder: 1B = the official CLIP ViT-L/14 @224 (refs/StarVector clip_model.VisionTransformer, 23 layers) +
    ln_vision, 257 positions (CLS + 16 x 16); 8B = SigLIP-large-384 (transformers SiglipVisionModel, config from the
    local HF cache), last_hidden_state, 576 positions;
  - adapter: the official Adapter (c_fc -> swish -> c_proj -> norm; 1B BatchNorm1d over the 257 positions, 8B LayerNorm);
  - LLM: transformers' GPTBigCodeForCausalLM (1B; StarCoder defaults + StarVector's sizes, vocab 49,156, lm_head tied to
    wte) / Starcoder2ForCausalLM (8B; starcoder2-7b config, vocab of the checkpoint);
  - every weight from the StarVector checkpoint (keys model.image_encoder.* / model.image_projection.* /
    model.svg_transformer.transformer.*).
The prompt is the official one: [image prefix embeddings] ++ embeddings of "<svg" (generate_im2svg). For the harness the
prompt ids are [PAD] x Q ++ ids("<svg") and the prefill passes prefix_embeds (Q positions) that replace the PAD
embeddings; later steps are plain token ids (1-D positions; GPTBigCode's learned wpe, Starcoder2's RoPE).
Pixels: the official processors (1B SimpleStarVectorProcessor transform; 8B SiglipImageProcessor of the model dir).
Lengths stay below 4,096 (cap 4,000 total, README), so Starcoder2's 4,096 sliding window never acts.
"""
from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
import torch.nn as nn
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
M = MODELS
MODELS = {"1b": M / "starvector-1b-im2svg", "8b": M / "starvector-8b-im2svg"}
CACHE = M / "hf-configs-cache"
SC2 = CACHE / "models--bigcode--starcoder2-7b/snapshots/bb9afde76d7945da5745592525db122d4d729eb1"
SIGLIP = CACHE / "models--google--siglip-large-patch16-384/snapshots/ce005573a40965dfd21fd937fbdeeebf2439fc35"
PROMPT = "<svg"
PROMPT_IDS = {"1b": [46, 3672], "8b": [65, 3693]}   # tokenizer("<svg") of each model (1B: the official dump; 8B: starcoder2)
EOS, PAD = 0, 49152
Q = {"1b": 257, "8b": 576}


def _official_modules():
    """clip_model / adapter of refs/StarVector without its training-only fairscale import"""
    if "fairscale" not in sys.modules:
        fs = types.ModuleType("fairscale")
        sub = types.ModuleType("fairscale.nn.checkpoint.checkpoint_activations")
        sub.checkpoint_wrapper = lambda module, **kw: module
        sys.modules.update({"fairscale": fs, "fairscale.nn": types.ModuleType("fairscale.nn"),
                            "fairscale.nn.checkpoint": types.ModuleType("fairscale.nn.checkpoint"),
                            "fairscale.nn.checkpoint.checkpoint_activations": sub})
    sys.path.insert(0, str(REFS / "StarVector"))
    from starvector.model.adapters.adapter import Adapter
    from starvector.model.image_encoder.clip_model import LayerNorm, VisionTransformer
    return VisionTransformer, LayerNorm, Adapter


def _state_dict(which: str) -> dict:
    from safetensors.torch import load_file
    idx = json.loads((MODELS[which] / "model.safetensors.index.json").read_text())["weight_map"]
    sd = {}
    for shard in sorted(set(idx.values())):
        sd.update(load_file(str(MODELS[which] / shard)))
    return sd


class StarVectorTarget(nn.Module):
    def __init__(self, which: str):
        super().__init__()
        VisionTransformer, LayerNorm, Adapter = _official_modules()
        cfg = json.loads((MODELS[which] / "config.json").read_text())
        sd = _state_dict(which)
        self.which, self.q = which, Q[which]
        if which == "1b":
            from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM
            lc = GPTBigCodeConfig(n_embd=2048, n_layer=24, n_head=16, n_positions=8192, multi_query=True,
                                  vocab_size=cfg["vocab_size"], bos_token_id=EOS, eos_token_id=EOS, pad_token_id=PAD)
            lc._attn_implementation = "sdpa"
            self.lm = GPTBigCodeForCausalLM(lc)
            self.vision = VisionTransformer(input_resolution=224, patch_size=14, width=1024, layers=23, heads=16,
                                            use_grad_checkpointing=False)
            self.ln_vision = LayerNorm(1024)
            vis_h = 1024
        else:
            from transformers import AutoConfig, SiglipVisionConfig, SiglipVisionModel, Starcoder2ForCausalLM
            lc = AutoConfig.from_pretrained(SC2)
            lc.vocab_size = sd["model.svg_transformer.transformer.model.embed_tokens.weight"].shape[0]
            lc.eos_token_id, lc.pad_token_id = EOS, PAD
            lc._attn_implementation = "sdpa"
            self.lm = Starcoder2ForCausalLM(lc)
            vc = SiglipVisionConfig(**json.loads((SIGLIP / "config.json").read_text())["vision_config"])
            self.vision = SiglipVisionModel(vc)        # 5.15: flat (= 4.49's SiglipVisionTransformer, same param names)
            self.ln_vision = None
            vis_h = vc.hidden_size
        self.adapter = Adapter(vis_h, lc.hidden_size, adapter_norm=cfg.get("adapter_norm", "layer_norm"),
                               query_length=self.q, dropout_prob=0.0)
        lm_sd = {k.removeprefix("model.svg_transformer.transformer."): v for k, v in sd.items()
                 if k.startswith("model.svg_transformer.transformer.")}
        missing, unexpected = self.lm.load_state_dict(lm_sd, strict=False)
        assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
        if "lm_head.weight" in missing:                      # no lm_head in either checkpoint: tied to the input embeddings
            self.lm.lm_head.weight = self.lm.get_input_embeddings().weight
        vis_sd = {k.removeprefix("model.image_encoder.visual_encoder."): v for k, v in sd.items()
                  if k.startswith("model.image_encoder.visual_encoder.")}
        missing, unexpected = self.vision.load_state_dict(vis_sd, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        if self.ln_vision is not None:
            self.ln_vision.load_state_dict({k.removeprefix("model.image_encoder.ln_vision."): v for k, v in sd.items()
                                            if k.startswith("model.image_encoder.ln_vision.")})
        self.adapter.load_state_dict({k.removeprefix("model.image_projection."): v for k, v in sd.items()
                                      if k.startswith("model.image_projection.")})
        self.config = self.lm.config
        self.lm_head = self.lm.lm_head
        self.eval()
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    @torch.no_grad()
    def image_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(1, 3, H, W) -> (1, Q, hidden): adapter(image_encoder(pixels)) as generate_im2svg builds it"""
        pv = pixel_values.to(next(self.vision.parameters()).dtype)
        if self.which == "1b":
            x = self.ln_vision(self.vision(pv))
        else:
            out = self.vision(pv)
            x = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        return self.adapter(x)

    def forward(self, input_ids=None, prefix_embeds=None, **kw):
        emb = self.lm.get_input_embeddings()(input_ids)
        if prefix_embeds is not None:
            q = prefix_embeds.shape[1]
            emb = torch.cat([prefix_embeds.to(emb.dtype), emb[:, q:]], dim=1)
        return self.lm(inputs_embeds=emb, **kw)


def _on_white(img):
    """RGBA / LA / P (the MMSVG training previews: transparent background, RGB 0 underneath) -> RGB on white, as every
    other image-to-SVG pipeline here (OmniSVG replace_background, i2s_image.py). RGB inputs (SArena) pass
    unchanged, so the official preprocessing of the evaluation images is untouched. The 1B official transform alone would
    turn the transparent background black (img.convert("RGB")), the 8B's SigLIP processor rejects 4 channels."""
    from PIL import Image
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return img


def processor(which: str):
    """pixel transform of the official processor of each model, after _on_white"""
    if which == "1b":
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode, pad

        def pad_sq(img):
            w, h = img.size
            d = max(w, h)
            p = [(d - w) // 2, (d - h) // 2]
            p += [d - w - p[0], d - h - p[1]]
            return pad(img, p, fill=255)
        tf = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode == "RGBA" else img),
            transforms.Lambda(pad_sq),
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        return lambda img: tf(_on_white(img)).unsqueeze(0)
    from transformers import SiglipImageProcessor
    # built from the model dir's preprocessor_config.json (384 x 384, mean / std 0.5, bicubic): in transformers 5.15
    # SiglipImageProcessor.from_pretrained(<that dir>) lets its processor_config.json reset the size to the 224 default
    cfg = json.loads((MODELS["8b"] / "preprocessor_config.json").read_text())
    sp = SiglipImageProcessor(**{k: v for k, v in cfg.items() if k not in ("image_processor_type", "processor_class")})
    assert sp.size["height"] == sp.size["width"] == 384, sp.size
    return lambda img: sp(_on_white(img), return_tensors="pt")["pixel_values"]


@dataclass
class SVInputs:
    prompt_ids: list
    prefix: torch.Tensor
    delta: int = 0

    def prefill_kwargs(self) -> dict:
        n = len(self.prompt_ids)
        return {"position_ids": torch.arange(n, device=self.prefix.device).unsqueeze(0), "prefix_embeds": self.prefix}

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        return pos


def load(which: str, dtype=torch.bfloat16) -> StarVectorTarget:
    return StarVectorTarget(which).to(dtype).cuda()


def prompt_ids(which: str) -> list[int]:
    """[PAD] x Q (replaced by the image prefix at prefill) ++ the ids of "<svg" (fixed: loading the 1B tokenizer through
    AutoTokenizer asks an interactive trust-remote-code question, which would hang a batch job)"""
    return [PAD] * Q[which] + PROMPT_IDS[which]


def build_inputs(model: StarVectorTarget, proc, image, prompt: list[int]) -> SVInputs:
    pv = proc(image).cuda()
    return SVInputs(prompt_ids=list(prompt), prefix=model.image_prefix(pv))
