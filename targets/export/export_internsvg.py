""" export InternSVG-8B's language side as a standard Qwen2ForCausalLM (vocab 152,126).

InternSVG-8B is an HF-native InternVLForConditionalGeneration (InternViT-300M + Qwen2.5-7B-style LM):

    language_model.model.embed_tokens.weight   -> model.embed_tokens.weight
    language_model.model.layers.{i}.*          -> model.layers.{i}.*
    language_model.model.norm.weight           -> model.norm.weight
    language_model.lm_head.weight              -> lm_head.weight
    vision_tower.* / multi_modal_projector.*      (dropped)

12 tensors per layer x 28 + 3 = 339 tensors. For text-only inputs the InternVL forward is exactly this Qwen2 stack
(1-D RoPE; image tokens are only substituted when pixel values are given).

rope_scaling: the checkpoint's text_config has {"rope_type": "dynamic", "factor": 2.0} with max_position_embeddings
32768. HF's dynamic NTK only rescales once a sequence exceeds 32768 positions, so for our lengths (prompt + <= 4000
new tokens) it is the identity. vLLM / SGLang may apply NTK scaling statically, which would change the model, so the
exported config drops rope_scaling (identical to HF below 32768 positions; checked by check_internsvg.py).

  python3 export_internsvg_text.py [--out <dir>]      (CPU; streams tensor by tensor)
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
from safetensors import safe_open
from safetensors.torch import save_file
from sightline.paths import MODELS

MODEL_ROOT = MODELS
SRC = MODEL_ROOT / "InternSVG-8B"
DST = MODEL_ROOT / "InternSVG-8B-qwen2-text"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.jinja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DST)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src_cfg = json.loads((SRC / "config.json").read_text())
    tc = dict(src_cfg["text_config"])
    cfg = {k: tc[k] for k in ("hidden_act", "hidden_size", "initializer_range", "intermediate_size",
                              "max_position_embeddings", "num_attention_heads", "num_hidden_layers",
                              "num_key_value_heads", "rms_norm_eps", "rope_theta", "vocab_size", "attention_dropout")}
    cfg.update({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "bos_token_id": 151643,
                "eos_token_id": 151645, "pad_token_id": 151643, "max_window_layers": tc["num_hidden_layers"],
                "rope_scaling": None, "sliding_window": None, "use_sliding_window": False,
                "tie_word_embeddings": False, "torch_dtype": "bfloat16", "use_cache": True,
                "_exported_from": str(SRC), "_dropped": "rope_scaling dynamic x2 (identity below 32768 positions)"})
    index = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]
    lang = sorted(k for k in index if k.startswith("language_model."))
    n_layers = sum(1 for k in lang if k.endswith("input_layernorm.weight"))
    assert n_layers == cfg["num_hidden_layers"] and len(lang) == 12 * n_layers + 3, (n_layers, len(lang))
    by_file = {}
    for k in lang:
        by_file.setdefault(index[k], []).append(k)
    new_index, shard_id = {}, 0
    for fname, keys in sorted(by_file.items()):
        shard_id += 1
        out_name = f"model-{shard_id:05d}-of-{len(by_file):05d}.safetensors"
        tensors = {}
        with safe_open(SRC / fname, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                assert str(t.dtype) == "torch.bfloat16", (k, t.dtype)
                nk = "lm_head.weight" if k == "language_model.lm_head.weight" else k.replace("language_model.model.", "model.", 1)
                tensors[nk] = t.contiguous()
                new_index[nk] = out_name
        save_file(tensors, str(args.out / out_name), metadata={"format": "pt"})
        print(f"{out_name}: {len(tensors)} tensors from {fname} ({time.time() - t0:.0f}s)", flush=True)
    emb = None
    with safe_open(SRC / index["language_model.model.embed_tokens.weight"], framework="pt") as f:
        emb = f.get_slice("language_model.model.embed_tokens.weight").get_shape()
    assert emb[0] == cfg["vocab_size"], emb
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_index}, indent=1))
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (args.out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": 151643, "eos_token_id": 151645, "pad_token_id": 151643, "do_sample": False}, indent=2) + "\n")
    for f in TOKENIZER_FILES:
        if (SRC / f).exists():
            shutil.copy2(SRC / f, args.out / f)
    print(f"done: {len(new_index)} tensors, vocab {emb}, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
