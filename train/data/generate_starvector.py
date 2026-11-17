""" (C2): StarVector image-to-SVG drafter training data, batched HF generation on the harness target itself
(target.py, transformers 5.15, bf16): every prompt is [image prefix (Q positions)] ++ "<svg", the same length
for all images, so a batch needs no padding; greedy, EOS 0, total length cap 4000 (the README's generate_im2svg call),
i.e. at most 4000 - Q - 2 new tokens. Images: the OmniSVG image-to-SVG training pool ($SIGHTLINE_RUNS/h3a-data/train_pool-{icon,
illustration}.jsonl, 50k each; the images of table 1's other image-to-SVG heads), pixels by the official processor of
each model. Units of 1000 images -> OUT/<class>/unit-XXXX/results.jsonl + DONE (re-runs skip DONE units):
    {gidx, id, subtype, image_path, prompt_input_ids ([PAD] x Q ++ ids("<svg")), token_ids, eos, n_gen}
Smoke  (1B, batch 64, H200): ~6,000 tokens/s, mean 2,992 tokens, 21 / 64 EOS (greedy StarVector loops to the cap
often; the train-set build cuts loops by the C1 rule).
  python gen_hf.py --which 1b --class icon --units 0-9 --batch 64 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from starvec import target as T  # noqa: E402
from sightline.paths import OUT, RUNS

POOL = (RUNS / "h3a-data")
CAP = 4000


def parse_units(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["1b", "8b"], required=True)
    ap.add_argument("--class", dest="cls", choices=["icon", "illustration"], required=True)
    ap.add_argument("--units", default="all")
    ap.add_argument("--unit-size", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    from PIL import Image
    items = [json.loads(x) for x in open(POOL / f"train_pool-{a.cls}.jsonl") if x.strip()]
    n_units = (len(items) + a.unit_size - 1) // a.unit_size
    m = T.load(a.which, torch.bfloat16)
    proc, pids = T.processor(a.which), T.prompt_ids(a.which)
    tail_ids = torch.tensor([pids[T.Q[a.which]:]], device="cuda")
    max_new = CAP - len(pids)
    k = 0 if a.cls == "icon" else 100000
    pool = ThreadPoolExecutor(16)
    for u in parse_units(a.units, n_units):
        d = a.out / a.cls / f"unit-{u:04d}"
        if (d / "DONE").exists():
            print("skip", d, flush=True)
            continue
        d.mkdir(parents=True, exist_ok=True)
        part = items[u * a.unit_size:(u + 1) * a.unit_size][:a.limit]
        t0, n_tok, n_eos = time.time(), 0, 0
        with (d / "results.jsonl").open("w") as f:
            for b0 in range(0, len(part), a.batch):
                chunk = part[b0:b0 + a.batch]
                pv = torch.cat(list(pool.map(lambda r: proc(Image.open(r["image_path"])), chunk))).cuda()
                with torch.no_grad():
                    pre = m.image_prefix(pv)
                    tail = m.get_input_embeddings()(tail_ids.expand(len(chunk), -1))
                    emb = torch.cat([pre.to(tail.dtype), tail], 1)
                    g = m.lm.generate(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long,
                                                                                   device="cuda"),
                                      max_new_tokens=max_new, do_sample=False, eos_token_id=T.EOS, pad_token_id=T.PAD)
                for j, (r, row) in enumerate(zip(chunk, g.tolist())):
                    toks = row[:row.index(T.EOS) + 1] if T.EOS in row else [t for t in row if t != T.PAD]
                    eos = bool(toks) and toks[-1] == T.EOS
                    n_tok += len(toks)
                    n_eos += eos
                    f.write(json.dumps({"gidx": k + u * a.unit_size + b0 + j, "id": r.get("id"), "subtype": a.cls,
                                        "image_path": r["image_path"], "prompt_input_ids": pids, "token_ids": toks,
                                        "eos": eos, "n_gen": len(toks)}) + "\n")
        dt = time.time() - t0
        stats = {"n": len(part), "tokens": n_tok, "seconds": round(dt, 1), "tok_per_s": round(n_tok / max(dt, 1e-6), 1),
                 "eos": n_eos}
        (d / "DONE").write_text(json.dumps(stats))
        print(d, stats, flush=True)
    print("GEN_HF_STARVECTOR_DONE", a.which, a.cls, a.units, flush=True)


if __name__ == "__main__":
    main()
