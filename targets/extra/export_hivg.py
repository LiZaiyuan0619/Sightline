""" export HiVG-3B-Base's language side (Qwen2.5-VL-3B layout: model.* + lm_head, visual.* dropped) as a standard
Qwen2ForCausalLM, like export_qwen2_text_any.py does for OmniSVG. Vocab 152,960 (Qwen2.5 + 1,306 HiVG SVG
tokens up to id 152948); lm_head and embed_tokens are both stored, so tie_word_embeddings is written False. On text-only
prompts Qwen2.5-VL's M-RoPE uses equal (t, h, w) positions, i.e. plain RoPE, so the export decodes identically
(checked by check_hivg_export.py). Tokenizer / chat template files are copied from the source.
  python export_hivg_text.py
"""
import json
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sightline.paths import MODELS

SRC = (MODELS / "HiVG-3B-Base")
DST = (MODELS / "HiVG-3B-Base-qwen2-text")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    src_cfg = json.loads((SRC / "config.json").read_text())
    tc = src_cfg["text_config"]
    cfg = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "pad_token_id": 151643,
           "bos_token_id": 151643, "eos_token_id": 151645, "max_position_embeddings": 32768, "rope_scaling": None,
           "sliding_window": None, "use_sliding_window": False, "tie_word_embeddings": False,
           "torch_dtype": "bfloat16", "use_cache": True}
    for k in ("attention_dropout", "hidden_act", "hidden_size", "initializer_range", "intermediate_size",
              "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "rope_theta",
              "vocab_size"):
        cfg[k] = tc[k]
    cfg["max_window_layers"] = cfg["num_hidden_layers"]
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    keep = sorted(k for k in idx if not k.startswith("visual."))
    sd = {}
    for k in keep:
        with safe_open(str(SRC / idx[k]), "pt") as f:
            sd[k] = f.get_tensor(k).to(torch.bfloat16).contiguous()
    n_layers = len({k.split(".")[2] for k in sd if k.startswith("model.layers.")})
    assert n_layers == cfg["num_hidden_layers"], n_layers
    assert sd["lm_head.weight"].shape[0] == cfg["vocab_size"] == sd["model.embed_tokens.weight"].shape[0]
    save_file(sd, str(DST / "model.safetensors"), metadata={"format": "pt"})
    (DST / "config.json").write_text(json.dumps(cfg, indent=1))
    (DST / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": 151643, "eos_token_id": [151645, 151643], "pad_token_id": 151643, "do_sample": False}, indent=1))
    for fn in ("added_tokens.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "chat_template.jinja"):
        if (SRC / fn).exists():
            shutil.copy(SRC / fn, DST / fn)
    print(f"exported {len(sd)} tensors, {n_layers} layers, vocab {cfg['vocab_size']} -> {DST}")


if __name__ == "__main__":
    main()
