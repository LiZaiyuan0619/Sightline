"""H4 qualitative comparison, round 1: the data layer.

Source: the qualitative data tables `figures/qualitative/round2/data_{i2,t2,isvg}.json` (written by that round's
prep.py from the fair-frontier best-of-N runs `$SIGHTLINE_RUNS/bon-decode/*-bench300-front`): per bench-300 prompt and decode, the
chosen output's render (official renderer, white background, already on disk), the per-prompt wall time (decode +
render scoring, one H200), its quality scores, and (text-to-SVG) the VLM-judge votes. Nothing is re-rendered here.
`cands_t2.json` (same directory) holds all 8 of our candidates for two text-to-SVG prompts (rendered on ).

Rows shown, all with the same loop control (the fair frontier of Q1; Q2's judge pairs are against the same ar1s):
  ar1s original model + loop control, 1 sample
  ar8s original model + loop control, best of 8
  sd8   Ours, best of 8
Selection, symmetric in the two best-of-8 rows (the per-prompt times are shown as they are):
  image-to-SVG  both best-of-8 outputs cut LPIPS against the input by >= 0.05 versus the 1-sample output
  text-to-SVG   every verdict (2 judges x 2 orders) prefers each best-of-8 output over the 1-sample output
then a fixed-seed random draw per sub-type. Text prompts longer than MAX_WORDS words are skipped before the draw so
the prompt fits under its column (a layout rule, independent of the results).
Two selection variants (the user decides on the rendered figures):
  rule "first"  (default, the rule) also asks that Ours finishes before the 1-sample output on that prompt
                (212 / 172 / 152 of 300 prompts do). Disclosed in the caption; the appendix gets an unconditioned draw.
  rule "any"    no condition on time. The unconditioned seed-0 draw has 3 of 6 image-to-SVG prompts where the
                1-sample output finishes first (base rate 29%): prompts that one sample gets wrong are often ones
                where it drew too little, which is also quick.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import json
import random
import statistics as st
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
SRC = ROOT / "figures/qualitative/round2"
ROWS = ("ar1s", "ar8s", "sd8")
TASKS = ("i2", "t2", "isvg")
TASK_NAME = {"i2": ("OmniSVG-8B", "image-to-SVG"), "t2": ("OmniSVG-8B", "text-to-SVG"),
             "isvg": ("InternSVG-8B", "text-to-SVG")}
MAX_WORDS = 12
SEED = 0

# published means (STATUS 1.3, Q1 data pack): seconds per prompt, decode + render scoring
PUBLISHED = {"i2": {"ar1": 13.1, "ar1s": 7.8, "sd8": 5.0, "ar8s": 19.2},
             "t2": {"ar1": 11.6, "ar1s": 6.9, "sd8": 4.7, "ar8s": 18.2}}


def load(task: str) -> list[dict]:
    d = json.loads((SRC / f"data_{task}.json").read_text())
    assert len(d) == 300, (task, len(d))
    for m, v in PUBLISHED.get(task, {}).items():
        got = st.mean(r["methods"][m]["t"] for r in d)
        assert abs(got - v) < 0.051, f"{task} {m}: mean {got:.3f} s, published {v}"
    return d


def eligible(task: str, d: list[dict]) -> list[dict]:
    if task == "i2":
        def ok(r):
            m = r["methods"]
            if any(m[k]["lpips"] is None for k in ROWS):
                return False
            return all(m[k]["lpips"] <= m["ar1s"]["lpips"] - 0.05 for k in ("sd8", "ar8s"))
    else:
        def ok(r):
            v = r["votes"]
            return all(v[k] and len(v[k]) == 4 and min(v[k]) == 1.0 for k in ("sd8:ar1s", "ar8s:ar1s"))
    return [r for r in d if ok(r)]


def picks(task: str, n_per_subtype: int, rule: str = "first") -> list[dict]:
    d = load(task)
    el = eligible(task, d)
    if rule == "first":
        el = [r for r in el if r["methods"]["sd8"]["t"] < r["methods"]["ar1s"]["t"]]
    if task != "i2":
        el = [r for r in el if len(r["caption"].split()) <= MAX_WORDS]
    rng = random.Random(SEED)
    out = []
    for s in ("icon", "illustration"):
        pool = sorted((r for r in el if r["subtype"] == s), key=lambda r: r["gidx"])
        out += rng.sample(pool, n_per_subtype)
    return out


def picks_wide(task: str, rule: str = "first") -> list[dict]:
    """The 3 + 3 standard picks plus one more icon and one more illustration, drawn with seed 1 from the rest of the
    same eligible pool (so the 8-column grid keeps the 6 examples of the 6-column one)."""
    base = picks(task, 3, rule)
    d = load(task)
    el = eligible(task, d)
    if rule == "first":
        el = [r for r in el if r["methods"]["sd8"]["t"] < r["methods"]["ar1s"]["t"]]
    taken = {r["gidx"] for r in base}
    rng = random.Random(SEED + 1)
    out = {"icon": [r for r in base if r["subtype"] == "icon"],
           "illustration": [r for r in base if r["subtype"] == "illustration"]}
    for s in ("icon", "illustration"):
        pool = sorted((r for r in el if r["subtype"] == s and r["gidx"] not in taken), key=lambda r: r["gidx"])
        out[s].append(rng.choice(pool))
    return out["icon"] + out["illustration"]


def example(r: dict, task: str) -> dict:
    m = r["methods"]
    e = {"task": task, "gidx": r["gidx"], "subtype": r["subtype"],
         "input": r.get("input"), "caption": r.get("caption"),
         "rows": {k: {"png": m[k]["png"], "t": m[k]["t"]} for k in ROWS}}
    if task == "i2":
        for k in ROWS:
            e["rows"][k]["lpips"] = m[k]["lpips"]
    else:
        for k in ROWS:
            e["rows"][k]["clip"] = m[k]["clip"]
    return e


def all_picks(rule: str = "first") -> dict:
    """The one set of examples every candidate draws from (guide SS2.5: one data set per direction)."""
    out = {"i2": [example(r, "i2") for r in picks("i2", 3, rule)],
           "t2": [example(r, "t2") for r in picks("t2", 1, rule)],
           "isvg": [example(r, "isvg") for r in picks("isvg", 1, rule)]}
    counts = {}
    for t in TASKS:
        d = load(t)
        el = eligible(t, d)
        counts[t] = {"eligible": len(el), "of": len(d),
                     "eligible_and_ours_first": sum(r["methods"]["sd8"]["t"] < r["methods"]["ar1s"]["t"] for r in el),
                     "ours_faster_than_1_sample": sum(r["methods"]["sd8"]["t"] < r["methods"]["ar1s"]["t"] for r in d),
                     "ours_faster_than_best_of_8": sum(r["methods"]["sd8"]["t"] < r["methods"]["ar8s"]["t"] for r in d)}
    out["counts"] = counts
    return out


def candidates_t2() -> list[dict]:
    """All 8 of our candidates for the two text-to-SVG prompts rendered on  (selection rule of that round:
    every verdict prefers sd8 over ar1s and sd8 faster than ar1s; seed 0). Order = sampling order."""
    c = json.loads((SRC / "cands_t2.json").read_text())
    for e in c:
        assert sum(x["chosen"] for x in e["cands"]) == 1
        best = max(range(8), key=lambda k: e["cands"][k]["clip"] if e["cands"][k]["valid"] else -1)
        assert e["cands"][best]["chosen"], e["gidx"]     # the chosen one is the best valid by rendered CLIP
    return c


if __name__ == "__main__":
    sel = {}
    for rule in ("first", "any"):
        p = all_picks(rule)
        print("rule", rule, json.dumps(p["counts"]) if rule == "first" else "")
        for t in TASKS:
            for e in p[t]:
                r = e["rows"]
                print(" ", t, e["gidx"], e["subtype"], (e["caption"] or "")[:50],
                      " ".join(f"{k}={r[k]['t']:.1f}s" for k in ROWS))
        sel[rule] = {t: [e["gidx"] for e in p[t]] for t in TASKS}
        sel["counts"] = p["counts"]
    (Path(__file__).resolve().parent / "selection.json").write_text(json.dumps(sel, indent=1))
