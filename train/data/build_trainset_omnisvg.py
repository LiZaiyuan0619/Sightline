"""A2 step 3: build the SpecForge training set from T6 data.

C1 decision (2026-10-09, §6 Q1 option a) on loop segments:
  - strict-loop / approx-loop streams: truncate at loop onset + 2 periods
    (keep the loop-free prefix plus a short tail so the drafter still learns to
    continue a loop — useful at deployment);
  - EOS-ended / no-period / short streams: keep in full.
--keep-loops builds the ablation set instead: every stream in full, loops
included (what an as-is DFlash recipe would train on).

Filters:
  - drop streams containing MASK_TOKEN (196900, our DFlash mask id) — see
    README.md; expected a handful per 100k;
  - drop streams with no generated tokens beyond BOS.

Output (SpecForge prompts_path format, one JSONL line per stream):
    {"input_ids": prompt + kept_generation, "loss_mask": 0*|prompt| + 1*|gen|}
loss covers ONLY the SVG output part (BOS... [EOS]) as required by the task
card. Metadata (loop labels, truncation stats) goes to a side file + data card,
never into the training JSONL.

Usage (dev CPU; 500 prompts per unit):
  python3 build_trainset.py                       # all DONE units
  python3 build_trainset.py --units-icon 0-19 --units-illu 0-19   # first 20k
  python3 build_trainset.py --units-icon 0-19 --units-illu 0-19 --keep-loops \
      --out runs/2026-10-09-t6-trainset-keeploops                       # ablation
  python3 build_trainset.py --units-icon 0-99 --units-illu 0-99 \
      --extra-root $SIGHTLINE_RUNS/t6-ext/data --out <dir>   # 300k

Streams are processed one at a time and only the final JSONL line is kept in memory (the 300k set would need ~16 GB as parsed records); for the 20k/100k sets the output is
byte-identical to the original builder (same record order, same shuffle).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.svgtok.loops import analyze
from sightline.paths import RUNS

MASK_TOKEN = 196900          # see README.md — must never appear in training data
PERIODS_KEPT = 2             # C1 decision: keep ~2 periods after loop onset
SEED = 42


def kept_gen(token_ids: list[int], eos: bool, cap: int = 1536,
             keep_loops: bool = False) -> tuple[list[int], str, int]:
    """Apply the C1 loop rule (or keep everything). Returns (kept_ids, label, n_cut)."""
    info = analyze(token_ids, eos, cap=cap)
    if keep_loops:
        return token_ids, info.label, 0
    if info.label in ("strict-loop", "approx-loop") and info.onset is not None:
        cut = min(len(token_ids), info.onset + PERIODS_KEPT * (info.period or 1))
        return token_ids[:cut], info.label, len(token_ids) - cut
    return token_ids, info.label, 0


def iter_streams(data_root: Path, cls: str, units: list[int] | None):
    """Yield the records of the DONE units of one class. units=None -> all DONE units."""
    unit_dirs = sorted((data_root / cls).glob("unit-*"))
    for d in unit_dirs:
        if not (d / "DONE").exists():
            continue
        u = int(d.name.split("-")[1])
        if units is not None and u not in units:
            continue
        with (d / "results.jsonl").open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def build(data_root: Path, out_dir: Path, units_icon, units_illu, val_n: int,
          keep_loops: bool = False, extra_roots: list[Path] | None = None):
    stats = {}
    records = []
    for cls, units in (("icon", units_icon), ("illustration", units_illu)):
        sources = [(data_root, units)] + [(root, None) for root in (extra_roots or [])]
        c = Counter(n_streams=0)
        for r in (rec for root, us in sources for rec in iter_streams(root, cls, us)):
            c["n_streams"] += 1
            cand = r["candidates"][0]
            gen, label, cut = kept_gen(cand["token_ids"], cand["eos_found"], keep_loops=keep_loops)
            c[f"label_{label}"] += 1
            c["tokens_before"] += len(cand["token_ids"])
            c["tokens_cut"] += cut
            if MASK_TOKEN in cand["raw_token_ids"] or MASK_TOKEN in r["prompt_input_ids"]:
                c["mask_id_filtered"] += 1
                continue
            if len(gen) <= 1:
                c["empty_filtered"] += 1
                continue
            c["kept"] += 1
            c["tokens_after"] += len(gen)
            prompt = r["prompt_input_ids"]
            records.append({"id": r["id"], "subtype": cls, "label": label,
                            "n_prompt": len(prompt), "n_gen": len(gen),
                            "line": json.dumps({"input_ids": prompt + gen,
                                                "loss_mask": [0] * len(prompt) + [1] * len(gen)})})
        stats[cls] = dict(c)

    rng = random.Random(SEED)
    rng.shuffle(records)                      # icon/illustration interleaved
    val = records[:val_n] if val_n else []
    train = records[val_n:]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train.jsonl", train), ("val.jsonl", val)):
        with (out_dir / name).open("w") as f:
            for rec in recs:
                f.write(rec["line"] + "\n")
    with (out_dir / "meta.jsonl").open("w") as f:
        for rec in records:
            f.write(json.dumps({"id": rec["id"], "subtype": rec["subtype"],
                                "loop_label": rec["label"],
                                "n_prompt": rec["n_prompt"], "n_gen": rec["n_gen"]}) + "\n")

    total_tokens = sum(s["tokens_after"] for s in stats.values())
    card = {
        "mask_token": MASK_TOKEN, "seed": SEED,
        "loop_rule": "keep all (ablation)" if keep_loops else f"truncate at onset + {PERIODS_KEPT} periods",
        "n_train": len(train), "n_val": len(val), "total_gen_tokens": total_tokens,
        "avg_gen_tokens": round(total_tokens / max(1, len(records)), 1),
        "classes": stats,
        "data_roots": [str(data_root)] + [str(r) for r in (extra_roots or [])],
    }
    (out_dir / "data_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=RUNS / "2026-10-09-t6-data")
    ap.add_argument("--out", type=Path, default=RUNS / "2026-10-09-t6-trainset")
    ap.add_argument("--units-icon", default=None, help="e.g. 0-39; default all DONE units")
    ap.add_argument("--units-illu", default=None)
    ap.add_argument("--val-n", type=int, default=0)
    ap.add_argument("--keep-loops", action="store_true", help="ablation: no loop truncation")
    ap.add_argument("--extra-root", type=Path, action="append", default=[],
                    help="another T6 data root; all its DONE units are added (repeatable)")

    def parse(spec):
        if spec is None:
            return None
        out = []
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return set(out)

    a = ap.parse_args()
    build(a.data_root, a.out, parse(a.units_icon), parse(a.units_illu), a.val_n,
          keep_loops=a.keep_loops, extra_roots=a.extra_root)
