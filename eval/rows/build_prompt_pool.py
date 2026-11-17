"""A2 step 1: build the text2svg prompt pool on CPU.

Source: MMSVG-Icon + MMSVG-Illustration v2.0 `description` fields (the training
corpora of the target model — same distribution the model was trained on).
Exclusions: MMSVGBench prompts (exact + near-duplicate), so evaluation prompts
never leak into training.

Near-duplicate rule (documented for the data card):
  normalize = lowercase + collapse whitespace;
  exact: normalized equality against a bench text;
  near: word 3-gram containment |A∩B| / min(|A|,|B|) >= NEAR_TH against any
              bench text (candidates pre-filtered via a 3-gram inverted index).

Outputs (deterministic, seed 42):
  universe-{icon,illustration}.jsonl: all kept rows {id, text, token_len}
  eval-pool/{icon,illustration}.jsonl: EVAL_N each (held out, never train)
  train-pool/{icon,illustration}.jsonl: TRAIN_N each (subsequence of universe)
  manifest.json: counts + dedup/exclusion stats + thresholds + seed

Usage:
  python prompt_pool.py [--eval-n 300] [--train-n 50000]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import pyarrow.parquet as pq
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
MODEL = MODELS
SOURCES = {
    "icon": MODEL / "MMSVG-Icon/data",
    "illustration": MODEL / "MMSVG-Illustration/data",
}
BENCH = RUNS / "2026-10-07-omnisvg-inputs"
OUT = RUNS / "2026-10-09-prompt-pool"

SEED = 42
NEAR_TH = 0.6          # 3-gram containment threshold vs bench texts
MIN_LEN, MAX_LEN = 8, 1000   # characters, after strip


def scrub(text: str) -> str:
    """Remove line-separator chars (U+2028/U+2029/U+0085, \r) that json.dumps
    (ensure_ascii=False) leaves raw but str.splitlines splits on -- they break
    JSONL round-trips on scraped web text."""
    for ch in ("\u2028", "\u2029", "\u0085", "\r"):
        text = text.replace(ch, " ")
    return text


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def trigrams(text: str) -> set:
    w = norm(text).split()
    return {hash(" ".join(w[i:i + 3])) for i in range(len(w) - 2)}


def load_class(cls: str) -> list[dict]:
    rows = []
    for shard in sorted(SOURCES[cls].glob("train-*.parquet")):
        t = pq.read_table(shard, columns=["id", "description", "token_len"])
        for r in t.to_pylist():
            rows.append({"id": r["id"], "text": scrub((r["description"] or "").strip()),
                         "token_len": r["token_len"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--train-n", type=int, default=50000)
    args = ap.parse_args()

    # ---- bench texts (all 600: text2svg prompts + image2svg descriptions) ----
    bench_texts = []
    for sub in ("text2svg-icon", "text2svg-illustration",
                "image2svg-icon", "image2svg-illustration"):
        for line in (BENCH / sub / "manifest.jsonl").read_text().splitlines():
            bench_texts.append(json.loads(line)["text"])
    bench_norm = {norm(t) for t in bench_texts}
    bench_tri = [trigrams(t) for t in bench_texts]
    inv = defaultdict(list)                     # 3-gram hash -> bench indices
    for i, tri in enumerate(bench_tri):
        for h in tri:
            inv[h].append(i)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "near_th": NEAR_TH,
                "eval_n": args.eval_n, "train_n": args.train_n, "classes": {}}

    for cls in ("icon", "illustration"):
        rows = load_class(cls)
        stats = Counter(total=len(rows))

        # ---- filters: non-empty, sane length, in-class exact dedup ----
        seen_norm = set()
        kept = []
        for r in rows:
            t = r["text"]
            if not (MIN_LEN <= len(t) <= MAX_LEN):
                stats["bad_len"] += 1
                continue
            n = norm(t)
            if n in seen_norm:
                stats["dup_in_class"] += 1
                continue
            seen_norm.add(n)
            kept.append(r)
        stats["kept_after_dedup"] = len(kept)

        # ---- bench exclusion: exact then near-dup ----
        final = []
        for r in kept:
            if norm(r["text"]) in bench_norm:
                stats["bench_exact"] += 1
                continue
            tri = trigrams(r["text"])
            if tri:
                hits = Counter()
                for h in tri:
                    for i in inv.get(h, ()):
                        hits[i] += 1
                near = False
                for i, c in hits.items():
                    if c < 2:
                        continue
                    contain = c / min(len(tri), len(bench_tri[i]))
                    if contain >= NEAR_TH:
                        near = True
                        break
                if near:
                    stats["bench_near"] += 1
                    continue
            final.append(r)
        stats["kept_final"] = len(final)

        # ---- deterministic split: shuffle once, eval first, train next ----
        rng = random.Random(SEED)
        rng.shuffle(final)
        eval_pool = final[:args.eval_n]
        train_pool = final[args.eval_n:args.eval_n + args.train_n]

        (OUT / f"universe-{cls}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final))
        for name, subset in (("eval-pool", eval_pool), ("train-pool", train_pool)):
            (OUT / name).mkdir(exist_ok=True)
            (OUT / name / f"{cls}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
        stats["eval_pool"] = len(eval_pool)
        stats["train_pool"] = len(train_pool)
        stats["universe_leftover"] = len(final) - len(eval_pool) - len(train_pool)
        manifest["classes"][cls] = dict(stats)
        print(cls, json.dumps(dict(stats)))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
