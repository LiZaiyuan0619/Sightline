""" export an OmniSVG checkpoint's language side as a standard Qwen2ForCausalLM (generalises
export_qwen2_text.py, which hard-codes the 8B; first use: OmniSVG1.1_4B for the cross-size rows of the
main table). Same tensor renaming (transformer.* -> *, transformer.visual.* dropped); the Qwen2 config's scalar fields
come from the source config.json. OmniSVG-4B stores lm_head and embed_tokens separately although its config says
tied; they are byte-identical (checked ), so both are kept and tie_word_embeddings is written False, as for 8B.
Tokenizer files are copied from the 8B export (same Qwen2.5 vocabulary + the same 197,000-id SVG range).
  python export_qwen2_text_any.py --src model/OmniSVG1.1_4B --dst model/OmniSVG1.1_4B-qwen2-text
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from sightline.paths import MODELS

MODEL_ROOT = MODELS
TOK_FROM = MODEL_ROOT / "OmniSVG1.1_8B-qwen2-text"
TOK_FILES = ["chat_template.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    src_cfg = json.loads((a.src / "config.json").read_text())
    cfg = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "pad_token_id": 151643,
           "max_position_embeddings": 32768, "rope_scaling": None, "sliding_window": None,
           "use_sliding_window": False, "tie_word_embeddings": False, "torch_dtype": "bfloat16", "use_cache": True}
    for k in ("attention_dropout", "bos_token_id", "eos_token_id", "hidden_act", "hidden_size", "initializer_range",
              "intermediate_size", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps",
              "rope_theta", "vocab_size"):
        cfg[k] = src_cfg[k]
    cfg["max_window_layers"] = cfg["num_hidden_layers"]
    t0 = time.time()
    sd = torch.load(a.src / "pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True)
    lang = sorted(k for k in sd if not k.startswith("transformer.visual"))
    n_layers = sum(1 for k in lang if k.endswith("input_layernorm.weight"))
    assert n_layers == cfg["num_hidden_layers"], (n_layers, cfg["num_hidden_layers"])
    assert len(lang) == 12 * n_layers + 3, f"unexpected language tensor count {len(lang)}"
    assert torch.equal(sd["transformer.lm_head.weight"], sd["transformer.model.embed_tokens.weight"]) \
        or not src_cfg.get("tie_word_embeddings"), "config says tied but lm_head != embed_tokens"
    out = {}
    for k in lang:
        assert sd[k].dtype == torch.bfloat16, f"{k} is {sd[k].dtype}"
        out[k.removeprefix("transformer.")] = sd[k]
    # write safetensors with independent tensors. A torch.save'd.bin keeps the tied lm_head / embed_tokens on
    # ONE storage, and transformers 4.51's.bin loader then re-read the file ~100 times (1 TB of reads per load).
    from safetensors.torch import save_file
    out = {k: (v.clone() if k == "lm_head.weight" else v).contiguous() for k, v in out.items()}
    (a.dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file(out, str(a.dst / "model.safetensors"), metadata={"format": "pt"})
    (a.dst / "pytorch_model.bin").unlink(missing_ok=True)
    for f in TOK_FILES:
        shutil.copy2(TOK_FROM / f, a.dst / f)
    print(f"saved {len(out)} tensors ({n_layers} layers, hidden {cfg['hidden_size']}) to {a.dst} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
