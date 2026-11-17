"""tokens per round versus training epoch for one B3 run, on the rows every head was evaluated on
(greedy x MMSVGBench text2svg, 300 rows: mid-run heads only got those; the final head
got everything, so its rows are filtered the same way).

Recomputes the calibers from the shard results (as eval_b3_merge.py does) and
puts them next to the training-set proxy of train_curve.py.

Usage: python3 epoch_curve.py <run_id> [--subtype icon|illustration]
Reads EVAL_ROOT/<run_id>/step*/shard-*/results.jsonl and TRAIN_ROOT/<run_id>/plan.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from analysis.calibers import CaliberStream, StreamRecord, caliber_table   # noqa: E402
from sightline.paths import RUNS

TRAIN_ROOT = (RUNS / "b3-train")
EVAL_ROOT = (RUNS / "b3-eval")


def rec(x):
    return None if x is None else StreamRecord(steps=x[0], emitted=x[1])


def head_table(step_dir: Path, subtype: str | None):
    bundles, n_div, n = [], 0, 0
    for f in sorted(step_dir.glob("shard-*/results.jsonl")):
        for line in f.read_text().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["mode"] != "greedy" or r["set"] != "bench" or (subtype and r["subtype"] != subtype):
                continue
            n += 1
            n_div += r["first_div_vs_ar"] is not None
            bundles.append(CaliberStream(full=rec(r["rec_full"]), eos=r["eos"], onset=r["onset"],
                                         trunc=rec(r["rec_trunc"]), loop=rec(r["rec_loop"])))
    return (caliber_table(bundles), n, n_div) if bundles else (None, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--subtype", default=None)
    args = ap.parse_args()
    plan = json.loads((TRAIN_ROOT / args.run_id / "plan.json").read_text())
    tc_path = TRAIN_ROOT / args.run_id / "train_curve.json"
    proxy = {}
    if tc_path.exists():
        proxy = {e["epoch"]: e for e in json.loads(tc_path.read_text())["epochs"]}
    steps = sorted(int(m.group(1)) for d in (EVAL_ROOT / args.run_id).glob("step*")
                   if (m := re.fullmatch(r"step(\d+)", d.name)))
    print(f"== {args.run_id}: greedy x bench{' x ' + args.subtype if args.subtype else ''} "
          f"(per-stream / merged)")
    print("| epoch | step | rows | τ_all | τ_eos | τ_trunc | τ_loop | greedy div | train proxy (chain) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in steps:
        t, n, n_div = head_table(EVAL_ROOT / args.run_id / f"step{s}", args.subtype)
        if t is None:
            continue
        ep = round(s / plan["steps_per_epoch"], 2)
        c = t["calibers"]
        cell = lambda k: (f"{c[k]['per_stream']:.2f} / {c[k]['merged']:.2f}"
                          if c[k]["n"] else "—")
        p = proxy.get(round(ep), {}).get("proxy_len")
        print(f"| {ep} | {s} | {n} | {cell('all')} | {cell('eos')} | {cell('trunc')} | "
              f"{cell('loop')} | {n_div} | {p if p is not None else '—'} |")


if __name__ == "__main__":
    main()
