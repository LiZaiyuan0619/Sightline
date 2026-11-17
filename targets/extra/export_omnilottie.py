""" (C1, OmniLottie): export OmniLottie's language side as a standard Qwen2ForCausalLM, as
export_qwen2_text_any.py did for OmniSVG-4B (same Qwen2.5-VL-3B base). OmniLottie (model/OmniLottie, HF format)
is LottieDecoder = Qwen2_5_VLForConditionalGeneration with the vocabulary resized to 192,400 (Lottie tokens 151,665+,
BOS 192398 / EOS 192399); weights are `transformer.*`, the vision tower `transformer.visual.*` (dropped here).
  - text fields of the Qwen2 config from the base config (model/Qwen2.5-VL-3B-Instruct-configs: 36 layers, hidden 2048,
    16 / 2 heads, rope_theta 1e6); vocab / BOS / EOS from OmniLottie's config;
  - rope_scaling None: for text-only prompts Qwen2.5-VL's M-RoPE gives the three axes the same position, i.e. plain
    1-D RoPE (checked numerically by check_lottie.py);
  - the checkpoint has no lm_head (tied to embed_tokens, as the base): lm_head is written as a copy of embed_tokens and
    tie_word_embeddings False (numerically the same; the OmniSVG exports are written the same way);
  - tokenizer + chat template files of the base (Qwen2.5 vocabulary; Lottie ids are output-only).
  python export_lottie_text.py [--dst model/OmniLottie-qwen2-text]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sightline.paths import MODELS

M = MODELS
SRC, BASE = M / "OmniLottie", M / "Qwen2.5-VL-3B-Instruct-configs"
TOK_FILES = ["chat_template.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=Path, default=M / "OmniLottie-qwen2-text")
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src_cfg = json.loads((SRC / "config.json").read_text())
    base = json.loads((BASE / "config.json").read_text())
    cfg = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "pad_token_id": src_cfg["pad_token_id"],
           "max_position_embeddings": 32768, "rope_scaling": None, "sliding_window": None, "use_sliding_window": False,
           "tie_word_embeddings": False, "torch_dtype": "bfloat16", "use_cache": True,
           "vocab_size": src_cfg["vocab_size"], "bos_token_id": src_cfg["bos_token_id"],
           "eos_token_id": src_cfg["eos_token_id"]}
    for k in ("attention_dropout", "hidden_act", "hidden_size", "initializer_range", "intermediate_size",
              "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta"):
        cfg[k] = base[k]
    cfg["max_window_layers"] = cfg["num_hidden_layers"]
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    lang = sorted(k for k in idx if not k.startswith("transformer.visual"))
    n_layers = sum(1 for k in lang if k.endswith("input_layernorm.weight"))
    assert n_layers == cfg["num_hidden_layers"] and len(lang) == 12 * n_layers + 2, (n_layers, len(lang))
    assert "transformer.lm_head.weight" not in idx, "unexpected separate lm_head"
    out = {}
    for shard in sorted(set(idx[k] for k in lang)):
        with safe_open(str(SRC / shard), framework="pt") as f:
            for k in lang:
                if idx[k] == shard:
                    t = f.get_tensor(k)
                    assert t.dtype == torch.bfloat16, (k, t.dtype)
                    out[k.removeprefix("transformer.")] = t.contiguous()
    emb = out["model.embed_tokens.weight"]
    assert tuple(emb.shape) == (cfg["vocab_size"], cfg["hidden_size"]), tuple(emb.shape)
    out["lm_head.weight"] = emb.clone()
    (a.dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file(out, str(a.dst / "model.safetensors"), metadata={"format": "pt"})
    for f in TOK_FILES:
        shutil.copy2(BASE / f, a.dst / f)
    print(f"saved {len(out)} tensors ({n_layers} layers, hidden {cfg['hidden_size']}, vocab {cfg['vocab_size']}) "
          f"to {a.dst} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
