""" (C2): StarVector drafter training set from gen_hf.py units. Response kept:
  - up to and including the token that completes the first "</svg>" (the official generate_im2svg stops there; the 8B
    goes on with <svg-end> / <file_sep>... before EOS, which our evaluation never decodes). "</svg>" is found in the
    decoded text (per-token decode, concatenated), since its tokenization depends on context;
  - else (no closing tag: the stream ran to the cap) the C1 loop rule at the cap (loop onset + 2 periods; detector tail
    window 1024, max period 512, as the InternSVG / OmniLottie sets);
rows containing the DFlash mask id (<fim_pad> = 4) are dropped. Rows: {input_ids = prompt ids ([PAD] x Q ++ "<svg") +
kept response, loss_mask, image_path} (ASCII-escaped JSON, read line by line).
  python build_sv_trainset.py --which 1b --root <gen dir> --out <dir> [--workers 16]
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
from sightline.paths import MODELS

MASK, PERIODS_KEPT, CAP = 4, 2, 4000
TOK = {"1b": f"{MODELS}/starvector-1b-im2svg/tokenizer.json",
       "8b": f"{MODELS}/hf-configs-cache/models--bigcode--starcoder2-7b/snapshots/"
             "bb9afde76d7945da5745592525db122d4d729eb1/tokenizer.json"}
_tok = {}


def end_index(which: str, ids: list[int]) -> int | None:
    """index of the token that completes the first "</svg>" in the decoded stream, else None"""
    if which not in _tok:
        from tokenizers import Tokenizer
        _tok[which] = Tokenizer.from_file(TOK[which])
    t = _tok[which]
    text, ends = "", []
    for x in ids:
        text += t.decode([x], skip_special_tokens=False)
        ends.append(len(text))
    k = text.find("</svg>")
    if k < 0:
        return None
    stop = k + len("</svg>")
    return next(i for i, e in enumerate(ends) if e >= stop)


def kept(which: str, ids: list[int], eos: bool, n_prompt: int) -> tuple[list[int], str]:
    j = end_index(which, ids)
    if j is not None:
        return ids[:j + 1], "closed"
    info = analyze(ids, eos, cap=CAP - n_prompt, tail_window=1024, max_period=512)
    if info.label in ("strict-loop", "approx-loop") and info.onset is not None:
        return ids[:min(len(ids), info.onset + PERIODS_KEPT * (info.period or 1))], "loop-cut"
    return ids, info.label


def one_unit(arg):
    which, d = arg
    rows, labels = [], Counter()
    g_tok = k_tok = drop = 0
    with (Path(d) / "results.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k, lab = kept(which, r["token_ids"], r["eos"], len(r["prompt_input_ids"]))
            labels[lab] += 1
            g_tok += len(r["token_ids"])
            if MASK in k or len(k) < 2:
                drop += 1
                continue
            k_tok += len(k)
            rows.append(json.dumps({"input_ids": r["prompt_input_ids"] + k,
                                    "loss_mask": [0] * len(r["prompt_input_ids"]) + [1] * len(k),
                                    "image_path": r["image_path"]}))
    return rows, labels, g_tok, k_tok, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["1b", "8b"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    todo = [(a.which, str(d)) for cls in ("icon", "illustration")
            for d in sorted((a.root / cls).glob("unit-*")) if (d / "DONE").exists()]
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            res = pool.map(one_unit, todo, chunksize=1)
    else:
        res = [one_unit(x) for x in todo]
    rows, labels = [], Counter()
    st = {"units": len(todo), "generated_tokens": 0, "kept_tokens": 0, "dropped": 0}
    for rs, lb, g, k, dr in res:
        rows += rs
        labels.update(lb)
        st["generated_tokens"] += g
        st["kept_tokens"] += k
        st["dropped"] += dr
    random.Random(a.seed).shuffle(rows)
    with (a.out / "train.jsonl").open("w") as f:
        for x in rows:
            f.write(x + "\n")
    st.update({"rows": len(rows), "labels": dict(labels), "root": str(a.root)})
    (a.out / "stats.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
