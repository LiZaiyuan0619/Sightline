""" drafter training set for a third-family SVG LLM (families.py), the prompts format of
train_dflash.py --task t2svg ({input_ids, loss_mask} per stream). Same rules as build_trainset.py
(the InternSVG drafters): C1 loop rule (keep the loop onset + 2 periods; detector sightline/svgtok/loops.analyze with the
family's cap, tail window 1024, max period 512), the EOS kept when the stream ended with one and was not cut, rows that
contain the DFlash mask id dropped. Only the special ids differ per family.
  python3 build_trainset_fam.py --family svgen7b --roots <gen root> [...] --out <dir> [--workers 16]
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
from sightline.svgtok.loops import analyze  # noqa: E402

PERIODS_KEPT = 2
CFG = {}


def kept(token_ids: list[int], eos: bool) -> tuple[list[int], str]:
    eos_ids, cap = CFG["eos"], CFG["cap"]
    last = token_ids[-1] if eos and token_ids and token_ids[-1] in eos_ids else None
    gen = token_ids[:-1] if last is not None else token_ids
    info = analyze(gen, eos, cap=cap, tail_window=1024, max_period=512)
    if info.label in ("strict-loop", "approx-loop") and info.onset is not None:
        return gen[:min(len(gen), info.onset + PERIODS_KEPT * (info.period or 1))], info.label
    return (gen + [last]) if last is not None else gen, info.label


def one_unit(arg):
    cls, d, cfg = arg
    CFG.update(cfg)
    rows, labels = [], Counter()
    gen_tokens = kept_tokens = d_mask = d_short = 0
    for line in (Path(d) / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k, label = kept(r["token_ids"], r["eos"])
        labels[f"{cls}/{label}"] += 1
        gen_tokens += len(r["token_ids"])
        if cfg["mask"] in k or cfg["mask"] in r["prompt_input_ids"]:
            d_mask += 1
            continue
        if len(k) < 2:
            d_short += 1
            continue
        kept_tokens += len(k)
        rows.append(json.dumps({"input_ids": r["prompt_input_ids"] + k,
                                "loss_mask": [0] * len(r["prompt_input_ids"]) + [1] * len(k)}))
    return rows, labels, gen_tokens, kept_tokens, d_mask, d_short


def main():
    from fam.families import FAMILIES
    # hivg3b lives in gen_vllm_hivg.py (families.py is imported by queued jobs, left untouched): same ids here
    FAMILIES.setdefault("hivg3b", {"eos": [151645, 151643], "mask": 151662, "max_new": 4000})
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--classes", default="icon,illustration")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    F = FAMILIES[a.family]
    cfg = {"eos": list(F["eos"]), "mask": F["mask"], "cap": F["max_new"]}
    a.out.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {"labels": Counter(), "units": 0, "dropped_mask": 0, "dropped_short": 0}
    gen_tokens = kept_tokens = 0
    todo = [(cls, str(d), cfg) for root in a.roots for cls in a.classes.split(",")
            for d in sorted((root / cls).glob("unit-*")) if (d / "DONE").exists()]
    stats["units"] = len(todo)
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            results = pool.map(one_unit, todo, chunksize=1)
    else:
        results = [one_unit(x) for x in todo]
    for rs, labels, g, k, dm, dsh in results:
        rows += rs
        stats["labels"].update(labels)
        gen_tokens += g
        kept_tokens += k
        stats["dropped_mask"] += dm
        stats["dropped_short"] += dsh
    random.Random(a.seed).shuffle(rows)
    with (a.out / "train.jsonl").open("w") as f:
        for x in rows:
            f.write(x + "\n")
    stats.update({"family": a.family, "rows": len(rows), "generated_tokens": gen_tokens, "kept_tokens": kept_tokens,
                  "roots": [str(r) for r in a.roots], "labels": dict(stats["labels"])})
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
