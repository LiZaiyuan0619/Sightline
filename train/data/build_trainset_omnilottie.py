""" (C1): OmniLottie Text-to-Lottie drafter training set = build_trainset.py with OmniLottie's ids and
cap: the C1 loop rule (loop onset + 2 periods kept; detector at the 4096 cap, tail window 1024, max period 512), EOS
192399 kept when the stream ended with it and was not cut, rows containing the DFlash mask id 151662 dropped. The
leading Lottie BOS 192398 the model emits stays in the stream (it is the target's first output token).
    {input_ids = prompt ids + kept response, loss_mask}   (the format of train_dflash.py --task t2svg)
  python3 build_lottie_trainset.py --root <gen dir> --out <dir> [--workers 16] [--units N]
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

EOS, MASK, CAP, PERIODS_KEPT = 192399, 151662, 4096, 2


def kept(token_ids: list[int], eos: bool) -> tuple[list[int], str]:
    gen = token_ids[:-1] if eos and token_ids and token_ids[-1] == EOS else token_ids
    info = analyze(gen, eos, cap=CAP, tail_window=1024, max_period=512)
    if info.label in ("strict-loop", "approx-loop") and info.onset is not None:
        return gen[:min(len(gen), info.onset + PERIODS_KEPT * (info.period or 1))], info.label
    return (gen + [EOS]) if eos else gen, info.label


def one_unit(d: str):
    rows, labels = [], Counter()
    g_tok = k_tok = d_mask = d_short = 0
    with (Path(d) / "results.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k, label = kept(r["token_ids"], r["eos"])
            labels[label] += 1
            g_tok += len(r["token_ids"])
            if MASK in k or MASK in r["prompt_input_ids"]:
                d_mask += 1
                continue
            if len(k) < 2:
                d_short += 1
                continue
            k_tok += len(k)
            rows.append(json.dumps({"input_ids": r["prompt_input_ids"] + k,
                                    "loss_mask": [0] * len(r["prompt_input_ids"]) + [1] * len(k)}))
    return rows, labels, g_tok, k_tok, d_mask, d_short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--units", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [str(d) for d in sorted(a.root.glob("unit-*")) if (d / "DONE").exists()][:a.units]
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            res = pool.map(one_unit, todo, chunksize=1)
    else:
        res = [one_unit(d) for d in todo]
    rows, labels = [], Counter()
    st = {"units": len(todo), "generated_tokens": 0, "kept_tokens": 0, "dropped_mask": 0, "dropped_short": 0}
    for rs, lb, g, k, dm, dsh in res:
        rows += rs
        labels.update(lb)
        st["generated_tokens"] += g
        st["kept_tokens"] += k
        st["dropped_mask"] += dm
        st["dropped_short"] += dsh
    random.Random(a.seed).shuffle(rows)
    with (a.out / "train.jsonl").open("w") as f:
        for x in rows:
            f.write(x + "\n")
    st.update({"rows": len(rows), "labels": dict(labels), "root": str(a.root)})
    (a.out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
