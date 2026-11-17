""" InternSVG-8B image-to-SVG drafter training set (table 1's 5th block) = build_trainset.py for the
image units of gen_vllm_i2s.py: the same C1 loop rule at InternSVG's cap (build_trainset.kept: loop onset +
2 periods kept, detector cap 4000 / tail 1024 / max period 512; EOS kept when the stream ended with it and was not cut),
rows with the DFlash mask id 151662 dropped. Each row also carries its image, which the trainers feed to the frozen
InternVL forward (train_i2s.py, train_eagle3_i2s.py):
    {input_ids = vLLM's expanded prompt ids + kept response, loss_mask, image_path, gidx, subtype}
written with ASCII escapes (json default) and read line by line (U+2028 in captions;  lesson).
  python3 build_i2s_trainset.py --roots <gen root> --out <dir> [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from internsvg.build_trainset import MASK, kept  # noqa: E402


def one_unit(arg):
    cls, d = arg
    rows, labels = [], Counter()
    gen_tokens = kept_tokens = d_mask = d_short = n_eos = 0
    with (Path(d) / "results.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k, label = kept(r["token_ids"], r["eos"])
            labels[f"{cls}/{label}"] += 1
            gen_tokens += len(r["token_ids"])
            n_eos += bool(r["eos"])
            if MASK in k or MASK in r["prompt_input_ids"]:
                d_mask += 1
                continue
            if len(k) < 2:
                d_short += 1
                continue
            kept_tokens += len(k)
            rows.append(json.dumps({"input_ids": r["prompt_input_ids"] + k,
                                    "loss_mask": [0] * len(r["prompt_input_ids"]) + [1] * len(k),
                                    "image_path": r["image_path"], "gidx": r["gidx"], "subtype": r["subtype"]}))
    return rows, labels, gen_tokens, kept_tokens, d_mask, d_short, n_eos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--classes", default="icon,illustration")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--units", type=int, default=None, help="smoke: only the first N DONE units per class")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    todo = []
    for root in a.roots:
        for cls in a.classes.split(","):
            ds = [str(d) for d in sorted((root / cls).glob("unit-*")) if (d / "DONE").exists()]
            todo += [(cls, d) for d in ds[:a.units]]
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            results = pool.map(one_unit, todo, chunksize=1)
    else:
        results = [one_unit(x) for x in todo]
    rows, stats = [], {"labels": Counter(), "units": len(todo), "dropped_mask": 0, "dropped_short": 0, "eos": 0}
    gen_tokens = kept_tokens = 0
    for rs, labels, g, k, dm, dsh, ne in results:          # fixed unit order -> reproducible shuffle
        rows += rs
        stats["labels"].update(labels)
        gen_tokens += g
        kept_tokens += k
        stats["dropped_mask"] += dm
        stats["dropped_short"] += dsh
        stats["eos"] += ne
    random.Random(a.seed).shuffle(rows)
    with (a.out / "train.jsonl").open("w") as f:
        for x in rows:
            f.write(x + "\n")
    stats.update({"rows": len(rows), "generated_tokens": gen_tokens, "kept_tokens": kept_tokens,
                  "roots": [str(r) for r in a.roots], "labels": dict(stats["labels"])})
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
