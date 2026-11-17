""" (C2): reference dump of the official StarVector inference (the starvector environment, transformers 4.49 = the official pin;
official.py) for the harness-side reimplementation (target.py, transformers 5.15) to match:
for --n SArena img2svg inputs (half icon, half illustration of $SIGHTLINE_RUNS/tableA/xbench/omnisvg-i2-sarena.jsonl) it saves
  pixel_values (the model's own processor), prefix embeddings (adapter(image_encoder(pixels)) ++ embeddings of the
  official prompt "<svg"), the prompt ids of "<svg", and the official greedy continuation ids
  (= generate_im2svg's call with use_nucleus_sampling False, num_beams 1, max_length --max-length, its "</svg>" stopping
  criterion), in fp32 (--dtype) -> <out>/<which>-<dtype>.pt
  python dump_official.py --which 1b --n 4 --max-length 1200 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from starvec import official  # noqa: E402
from sightline.paths import RUNS

XR = (RUNS / "tableA/xbench/omnisvg-i2-sarena.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["1b", "8b"], required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1200)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = getattr(torch, a.dtype)
    m = official.load(a.which, dtype).to(dtype)      # the official image encoder casts itself to config.torch_dtype
    core = m.model
    core.model_precision = dtype                       # (fp16 for 1B): one dtype everywhere for the reference
    proc, tok = core.processor, core.svg_transformer.tokenizer
    from PIL import Image
    from transformers.generation.stopping_criteria import StoppingCriteriaList
    from starvector.model.models.starvector_base import StoppingCriteriaSub
    rows = [json.loads(x) for x in open(XR) if x.strip()]
    pick = [r for r in rows if r["subtype"] == "icon"][:a.n // 2] + \
           [r for r in rows if r["subtype"] == "illustration"][:a.n - a.n // 2]
    out = []
    for r in pick:
        pv = proc(Image.open(r["image_path"]), return_tensors="pt")["pixel_values"]
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
        pv = pv.cuda().to(dtype)
        with torch.no_grad():
            emb, att, ptok = core._prepare_generation_inputs({"image": pv}, None, pv.device)
            end = tok("</svg>", add_special_tokens=False)["input_ids"]
            g = core.svg_transformer.transformer.generate(
                inputs_embeds=emb, attention_mask=att, do_sample=False, num_beams=1, max_length=a.max_length,
                min_length=1, repetition_penalty=1.0, length_penalty=1.0, use_cache=True,
                stopping_criteria=StoppingCriteriaList([StoppingCriteriaSub(stops=[end])]),
                pad_token_id=tok.pad_token_id)
        out.append({"gidx": r["gidx"], "subtype": r["subtype"], "image_path": r["image_path"], "pixel_values": pv.cpu(),
                    "prefix_embeds": emb.float().cpu(), "prompt_ids": ptok.input_ids[0].tolist(),
                    "official_ids": g[0].tolist()})
        print(r["gidx"], r["subtype"], "prefix", tuple(emb.shape), "prompt", ptok.input_ids[0].tolist(),
              "generated", len(g[0]), repr(tok.decode(g[0][-12:]))[:80], flush=True)
    meta = {"which": a.which, "dtype": a.dtype, "max_length": a.max_length, "eos": tok.eos_token_id,
            "pad": tok.pad_token_id, "end_seq": tok("</svg>", add_special_tokens=False)["input_ids"],
            "vocab": len(tok), "query_length": core.query_length}
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "rows": out}, a.out / f"{a.which}-{a.dtype}.pt")
    print("DUMP_DONE", meta, flush=True)


if __name__ == "__main__":
    main()
