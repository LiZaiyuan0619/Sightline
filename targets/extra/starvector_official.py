""" (C2): load the official StarVector models offline (refs/StarVector code, model/starvector-{1b,8b}-im2svg weights).
The StarVector checkpoints hold every weight (image encoder, adapter, LLM); the official code only fetches the bases'
configs / tokenizers / pretrained weights at construction. Here, without touching the official files:
  - LLM: AutoModelForCausalLM.from_pretrained(<base>, config=...) -> from_config (the checkpoint then fills it);
    1B base bigcode/starcoderbase-1b is gated (no config offline): GPTBigCodeConfig with transformers' defaults (they are
    StarCoder's: gelu_pytorch_tanh, multi-query, eps 1e-5, fp32 softmax) + the sizes of StarVector's own config
    (n_embd 2048, 24 layers, 16 heads, 8192 positions); its tokenizer = the one in the StarVector-1B dir (base vocab +
    [PAD] <svg-start> <image-start> <caption-start> = 49,156);
    8B base bigcode/starcoder2-7b: config + tokenizer from the local HF cache model/hf-configs-cache (downloaded );
  - 8B image encoder: AutoModel.from_pretrained("google/siglip-large-patch16-384") -> from_config (cache config).
Generation (official generate_im2svg): image -> encoder -> adapter -> + embeddings of the prompt "<svg" -> generate.
  the starvector environment (transformers 4.49, the official pin), PYTHONPATH=refs/StarVector
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
M = MODELS
MODELS = {"1b": M / "starvector-1b-im2svg", "8b": M / "starvector-8b-im2svg"}
HF_CACHE = M / "hf-configs-cache"


def _setup():
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if str(REFS / "StarVector") not in sys.path:
        sys.path.insert(0, str(REFS / "StarVector"))


def _patch(which: str):
    """Construction-time patches (module attributes of the official starvector package)."""
    import transformers
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, GPTBigCodeConfig

    def llm_from_config(name, config=None, torch_dtype=None, **kw):
        return AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype, trust_remote_code=True)

    class AutoModelForCausalLMLocal(AutoModelForCausalLM):
        from_pretrained = staticmethod(llm_from_config)

    if which == "1b":
        import starvector.model.llm.starcoder as sc

        def cfg_1b(name, **kw):
            assert name == "bigcode/starcoderbase-1b", name
            return GPTBigCodeConfig(n_embd=2048, n_layer=24, n_head=16, n_positions=8192, vocab_size=49152,
                                    multi_query=True)

        def tok_1b(name, **kw):
            assert name == "bigcode/starcoderbase-1b", name
            return AutoTokenizer.from_pretrained(MODELS["1b"], **kw)

        class AutoConfigLocal(AutoConfig):
            from_pretrained = staticmethod(cfg_1b)

        class AutoTokenizerLocal(AutoTokenizer):
            from_pretrained = staticmethod(tok_1b)
        sc.AutoConfig, sc.AutoTokenizer, sc.AutoModelForCausalLM = AutoConfigLocal, AutoTokenizerLocal, AutoModelForCausalLMLocal
    else:
        import starvector.model.llm.starcoder2 as sc2
        sc2.AutoModelForCausalLM = AutoModelForCausalLMLocal

        def vision_from_config(name, torch_dtype=None, **kw):
            return AutoModel.from_config(AutoConfig.from_pretrained(name), torch_dtype=torch_dtype)
        orig = transformers.AutoModel.from_pretrained
        transformers.AutoModel.from_pretrained = staticmethod(
            lambda name, *a, **kw: vision_from_config(name, **kw) if "siglip" in str(name) else orig(name, *a, **kw))


def load(which: str, dtype=torch.bfloat16):
    """official StarVectorForCausalLM (eval, cuda) with every weight from the StarVector checkpoint"""
    _setup()
    _patch(which)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODELS[which], trust_remote_code=True, torch_dtype=dtype)
    return m.cuda().eval()
