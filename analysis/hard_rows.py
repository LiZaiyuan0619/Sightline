""" where does the method do WORST, and why (HSD appendix style).

From the result files only (0 GPU): rank the rows of a run by speed-up, report the bottom and top decile, and for each
bucket give the things that could explain it - output length, how often a round hit the speculation limit, the
acceptance distribution, whether the AR output looped or hit the cap, and the prompt text. Writes the per-row table so
a figure or an appendix listing can pick real examples by id.

  python3 analysis/hard_cases.py --rows <design dir> --xrows <row file> --block 32 --out <prefix>
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path


def rows_of(root: str, mode: str):
    return [r for f in sorted(glob.glob(f"{root}/shard-*/results.jsonl")) for r in map(json.loads, open(f))
            if r and r["mode"] == mode]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--xrows", type=Path, default=None)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--k", type=int, default=10, help="how many rows to list per end")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    texts = {}
    if a.xrows:
        for x in a.xrows.read_text().splitlines():
            if x.strip():
                q = json.loads(x)
                texts[q["gidx"]] = q.get("text", "")
    R = []
    for r in rows_of(a.rows, a.mode):
        if not r.get("t_spec") or not r.get("t_ar"):
            continue
        acc = r["acceptance"]
        R.append({"gidx": r["gidx"], "subtype": r["subtype"],
                  "speedup": r["t_ar"] / r["t_spec"], "tau": sum(acc) / max(len(acc), 1),
                  "n_spec": r["n_spec"], "n_ar": r["n_ar"], "eos": r["eos"], "eos_ar": r["eos_ar"],
                  "full_block": sum(1 for n in acc if n >= a.block) / max(len(acc), 1),
                  "rounds": len(acc), "max_accept": r.get("max_accept"),
                  "text": texts.get(r["gidx"], "")[:120]})
    R.sort(key=lambda x: x["speedup"])
    k = a.k
    out = {"n": len(R), "median_speedup": st.median(x["speedup"] for x in R) if R else None,
           "worst": R[:k], "best": R[-k:][::-1]}

    def agg(rs, name):
        return {"bucket": name, "n": len(rs),
                "speedup": st.mean(x["speedup"] for x in rs), "tau": st.mean(x["tau"] for x in rs),
                "len": st.mean(x["n_spec"] for x in rs), "full_block": st.mean(x["full_block"] for x in rs),
                "eos_rate": st.mean(x["eos"] for x in rs), "ar_capped": st.mean(not x["eos_ar"] for x in rs)}
    d = max(1, len(R) // 10)
    out["deciles"] = [agg(R[:d], "bottom 10%"), agg(R[d:-d] or R, "middle"), agg(R[-d:], "top 10%")]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(out, indent=1))
    L = [f"# Hard and easy rows ({a.rows.split('/')[-2]}/{a.rows.split('/')[-1]}, {a.mode})", "",
         f"{out['n']} rows, median speed-up {out['median_speedup']:.2f}x", "",
         "| bucket | n | speed-up | tau | output tokens | full-block rounds | ended with EOS | AR hit the cap |",
         "|---|---|---|---|---|---|---|---|"]
    for b in out["deciles"]:
        L.append(f"| {b['bucket']} | {b['n']} | {b['speedup']:.2f} | {b['tau']:.2f} | {b['len']:.0f} | "
                 f"{b['full_block']:.3f} | {b['eos_rate']:.2f} | {b['ar_capped']:.2f} |")
    for key, title in (("worst", "Slowest rows"), ("best", "Fastest rows")):
        L += ["", f"## {title}", "", "| gidx | subtype | speed-up | tau | tokens | prompt |", "|---|---|---|---|---|---|"]
        for x in out[key]:
            L.append(f"| {x['gidx']} | {x['subtype']} | {x['speedup']:.2f} | {x['tau']:.2f} | {x['n_spec']} | "
                     f"{x['text']} |")
    a.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("HARD_CASES_DONE", a.out.with_suffix(".md"))


if __name__ == "__main__":
    main()
