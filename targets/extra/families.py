""" third-family SVG LLMs for the transfer matrix, one place for their
official prompt formats, sampling settings and special ids. Sources: refs/SVGen/inference.py (gitcat-404/SVGen,
ACM MM'25), refs/HiVG/hivg_infer/pipeline.py (ximinng/HiVG). Models in $SIGHTLINE_MODELS.
  svgen7b: SVGen-Qwen2.5-Coder-7B-Instruct (Qwen2ForCausalLM, 28 layers, hidden 3584), prompt_type 'qwen'
  svgen3b: SVGen-Llama-3.2-3B-Instruct (LlamaForCausalLM, 28 layers, hidden 3072), prompt_type 'llama'
Both SVGen models answer "<think>...</think>\n```svg\n...\n```"; the official extraction is the ```svg block; official
sampling = temperature 0.2, top-p 0.7 (inference.py), max_model_len 4096.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
from pathlib import Path
from sightline.paths import MODELS

M = MODELS
SVGEN_SYS = ("Please generate the reasoning process and final SVG code separately, according to the following format:"
             "<think>\n...\n</think>\n```svg\n...\n```")
SVGEN_USER = "Please generate an SVG icon that meets the following description: {desc}"

FAMILIES = {
    "svgen7b": {"model": M / "SVGen-Qwen2.5-Coder-7B-Instruct", "eos": [151645, 151643], "stop": 151645,
                "mask": 151662, "pad": 151643, "temperature": 0.2, "top_p": 0.7, "max_new": 4000,
                "prompt": lambda d: (f"<|im_start|>system\n{SVGEN_SYS}<|im_end|>\n<|im_start|>user\n"
                                     f"{SVGEN_USER.format(desc=d)}<|im_end|>\n<|im_start|>assistant\n")},
    "svgen3b": {"model": M / "SVGen-Llama-3.2-3B-Instruct", "eos": [128001, 128008, 128009], "stop": 128009,
                "mask": 128002, "pad": 128004, "temperature": 0.2, "top_p": 0.7, "max_new": 4000,
                "prompt": lambda d: (f"<|start_header_id|>system<|end_header_id|>\n\n{SVGEN_SYS}\n<|eot_id|>\n"
                                     f"<|start_header_id|>user<|end_header_id|>\n\n{SVGEN_USER.format(desc=d)}<|eot_id|>\n"
                                     f"<|start_header_id|>assistant<|end_header_id|>\n\n")},
}


def encode(tok, fam: str, desc: str) -> list[int]:
    """Prompt ids as the official vLLM call builds them (string prompt, tokenizer defaults: Llama gets its BOS)."""
    return tok(FAMILIES[fam]["prompt"](desc)).input_ids
