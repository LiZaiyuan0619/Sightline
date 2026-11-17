""" InternSVG-8B (ICLR'26, InternVL3-8B + SVG tokens) generation with HF transformers, for the loop survey
(does a second SVG LLM also loop / hit the token cap?).

Protocol = the official SArena generation eval (refs/InternSVG/scripts/inference/gen/demo.sh): greedy (temperature 0),
max 4000 new tokens, prompt = SArena conversations[0].value. The official eval goes through LMDeploy; here the prompt is
built in the SFT template the model was trained with (LLaMA-Factory 'intern_vl': default system prompt, no BOS,
'<image>' -> <img><IMG_CONTEXT>*N</img> by the HF processor). Rows: a seeded random sample of --n rows per test file.
Each output keeps the raw generated ids, EOS flag, length, time and the SVG text; loop labels are computed offline
(sightline/sightline/svgtok/loops.py with cap 4000).

  python run.py --task text2svg|img2svg --subset icon|illustration --n 150 \
      [--shard i/n] [--limit k] --out <dir>
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import argparse
import json
import random
import sys
import time
from pathlib import Path
from sightline.paths import MODELS

ROOT = Path(__file__).resolve().parents[1]
MODEL = MODELS / "InternSVG-8B"
SARENA = Path(f"{MODELS}/datasets--InternSVG--SArena/snapshots/"
              "b6b08d734616a8abf67fd221eced1dfa7c0868b9")
FILES = {("text2svg", "icon"): "Icon/generation/text2svg.jsonl", ("img2svg", "icon"): "Icon/generation/img2svg.jsonl",
         ("text2svg", "illustration"): "illustration/text2svg.jsonl",
         ("img2svg", "illustration"): "illustration/img2svg.jsonl"}
# InternSVG's released system prompt (that of InternVL), verbatim: the model is trained with it.
SYSTEM = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
IM_END = 151645


def rows(task: str, subset: str, n: int) -> list[dict]:
    path = SARENA / FILES[(task, subset)]
    allr = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    pick = sorted(random.Random(0).sample(range(len(allr)), min(n, len(allr))))
    out = []
    for i in pick:
        r = allr[i]
        out.append({"id": int(r["id"]), "question": r["conversations"][0]["value"],
                    "image": str(path.parent / r["image"]) if "image" in r else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["text2svg", "img2svg"], required=True)
    ap.add_argument("--subset", choices=["icon", "illustration"], required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=4000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    si, sn = map(int, args.shard.split("/"))
    todo = rows(args.task, args.subset, args.n)[si::sn]
    if args.limit:
        todo = todo[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    res = args.out / f"results-{si}-of-{sn}.jsonl"
    done = {json.loads(x)["id"] for x in res.read_text().splitlines() if x.strip()} if res.exists() else set()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    (args.out / "protocol.json").write_text(json.dumps({
        "model": str(MODEL), "task": args.task, "subset": args.subset, "n": args.n, "row_seed": 0,
        "decoding": "greedy (official temperature 0)", "max_new_tokens": args.max_new_tokens,
        "template": "LLaMA-Factory intern_vl (system prompt, no BOS)", "image_token": proc.image_token}, indent=1))
    with res.open("a") as log:
        for r in todo:
            if r["id"] in done:
                continue
            q = r["question"]
            images = None
            if r["image"]:
                q = q.replace("<image>", proc.image_token)
                images = [Image.open(r["image"]).convert("RGB")]
            text = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            inputs = proc(text=[text], images=images, return_tensors="pt").to("cuda")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            n_prompt = inputs["input_ids"].shape[1]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                     eos_token_id=IM_END, pad_token_id=proc.tokenizer.pad_token_id)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            gen = out[0, n_prompt:].tolist()
            eos = IM_END in gen
            ids = gen[:gen.index(IM_END) + 1] if eos else gen
            svg = proc.tokenizer.decode(ids, skip_special_tokens=False).replace("<|im_end|>", "")
            log.write(json.dumps({"id": r["id"], "task": args.task, "subset": args.subset, "n_prompt": n_prompt,
                                  "n_gen": len(ids), "eos": eos, "t_gen": dt, "token_ids": ids, "svg": svg}) + "\n")
            log.flush()
            print(r["id"], len(ids), eos, f"{dt:.1f}s", flush=True)
    print("INTERNSVG_DONE", args.task, args.subset, args.shard)


if __name__ == "__main__":
    main()
