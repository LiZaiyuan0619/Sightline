"""T2: export OmniSVG's language side as a standard Qwen2ForCausalLM (vocab 197,000).

The OmniSVG checkpoint stores a Qwen2_5_VLForConditionalGeneration under a `transformer.` prefix
(SketchDecoder wraps it as `.transformer`). The language side is a plain Qwen2 stack:

    transformer.model.embed_tokens.weight        -> model.embed_tokens.weight
    transformer.model.layers.{i}.*               -> model.layers.{i}.*
    transformer.model.norm.weight                -> model.norm.weight
    transformer.lm_head.weight                   -> lm_head.weight
    transformer.visual.*                          (dropped, 391 tensors)

12 tensors per layer (q/k/v with bias, o/gate/up/down without, two RMSNorms) = 12*28+3 = 339
tensors total, no Qwen3-style q_norm/k_norm. Verified on the real checkpoint 2026-10-08.

Runs on dev: streams tensor-by-tensor with mmap so peak memory stays ~1 tensor.
"""
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from sightline.paths import MODELS

ROOT = Path(__file__).resolve().parents[1]  #.../SpeculativeDecoding
MODEL_ROOT = MODELS
SRC = MODEL_ROOT / "OmniSVG1.1_8B" / "pytorch_model.bin"
DST = MODEL_ROOT / "OmniSVG1.1_8B-qwen2-text"
# values from model/OmniSVG1.1_8B/config.json (scalar fields only; visual fields dropped)
CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 196998,
    "eos_token_id": 196999,
    "pad_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "rope_scaling": None,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 197000,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DST)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # mmap-backed real CPU tensors (NOT map_location='meta': that yields storage-less
    # tensors and torch.save would write a metadata-only file). Saving an mmap-backed
    # dict streams each storage sequentially, so anonymous RSS stays near zero.
    sd = torch.load(SRC, map_location="cpu", mmap=True, weights_only=True)
    keys = sorted(sd.keys())

    lang = [k for k in keys if not k.startswith("transformer.visual")]
    n_layers = sum(1 for k in lang if k.endswith("input_layernorm.weight"))
    assert len(lang) == 12 * n_layers + 3, f"unexpected language tensor count {len(lang)}"
    print(f"{len(lang)} language tensors over {n_layers} layers; dropping {len(keys) - len(lang)} visual tensors")

    out_state = {}
    for k in lang:
        t = sd[k]  # mmap-backed; this materialises one tensor at a time
        assert t.dtype == torch.bfloat16, f"{k} is {t.dtype}"
        out_state[k.removeprefix("transformer.")] = t
    (args.out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    torch.save(out_state, args.out / "pytorch_model.bin")
    print(f"saved {len(out_state)} tensors to {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
