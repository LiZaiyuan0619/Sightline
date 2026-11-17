""" how common decoding loops are, with ONE ruler over every target we train a draft
for. The numbers mixed two definitions (some targets were reported as "share of outputs that hit the cap",
others as "share with a periodic tail"), which the figure plan flagged as a coverage decision; this prints both for
all of them, from each target's own AR outputs under its official decoding protocol.

Ruler = sightline/svgtok/loops.analyze on the AR token stream: EOS-ended -> no loop; shorter than the cap -> no loop;
otherwise the tail window is tested for a period (the detector of the loop-control rules). Per target we report
  ended      share of rows the target finished itself (EOS)
  capped     share that ran into the official token cap
  periodic   share whose capped tail is periodic (= the loops component 2 removes)
  tok. in loop share of all generated tokens inside a periodic tail (what the rules can save)
Rows come from the AR of the table-1 / family jobs (inline ar_tokens, or the ar-*/ar.jsonl caches), de-duplicated by
(mode, set, subtype, gidx); the cap is the one the run recorded.

  python3 analysis/loop_share.py [--out runs/2026-10-30/analysis/loop_share.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.svgtok.loops import analyze  # noqa: E402
from sightline.paths import RUNS

U = RUNS
_CAP = {}
LONG = {"tail_window": 1024, "max_period": 512}        # targets with a 4k cap (InternSVG, OmniLottie, StarVector)

# (label, globs, detector kwargs, subset labels to keep)
TARGETS = [
    ("OmniSVG-8B, text-to-SVG", [U / "tableA/t2-own/ours/*/shard-*/results.jsonl"], {}, None),
    ("OmniSVG-4B, text-to-SVG", [U / "tableA/t2-4b/own/ours/*/shard-*/results.jsonl"], {}, None),
    ("InternSVG-8B, text-to-SVG", [U / "tableA/isvg-own/ours/*/shard-*/results.jsonl"], LONG, None),
    ("OmniSVG-8B, image-to-SVG", [U / "tableA/i2-own-b127/ours/*/shard-*/results.jsonl"], {}, None),
    ("InternSVG-8B, image-to-SVG", [U / "tableA/isvg-i2/ar-*/ar.jsonl"], LONG, None),
    ("OmniLottie-3B, text-to-Lottie", [U / "tableA/lottie/s*/ours/*/shard-*/results.jsonl"], LONG, None),
    ("StarVector-1B, image-to-SVG", [U / "tableA/fam-sv1b/ar-*/ar.jsonl"], LONG, None),
    ("StarVector-8B, image-to-SVG", [U / "tableA/fam-sv8b/ar-*/ar.jsonl"], LONG, None),
]


def proto_cap(results_file):
    """the run's own token cap (inline AR rows do not carry it; the ar.jsonl caches do)"""
    f = Path(results_file).parent / "protocol.json"
    if f not in _CAP:
        try:
            _CAP[f] = json.loads(f.read_text()).get("max_new_tokens")
        except OSError:
            _CAP[f] = None
    return _CAP[f]


def ar_rows(globs):
    """(mode, set, subtype, gidx) -> (tokens, ended, cap) from inline AR rows or ar.jsonl caches."""
    out = {}
    for g in globs:
        for f in sorted(glob.glob(str(g))):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                toks = r.get("ar_tokens")
                if not toks:
                    continue
                key = (r.get("mode"), r.get("set"), r.get("subtype"), r.get("gidx"))
                if "stop_reason" in r:                            # ar.jsonl cache
                    ended = r["stop_reason"] == "eos"
                elif "eos_ar" in r:                               # inline AR of an eval row
                    ended = bool(r["eos_ar"])
                else:
                    continue
                cap = r.get("max_new") or r.get("cap") or proto_cap(f) or len(toks)
                out.setdefault(key, (toks, ended, cap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    lines = ["# Decoding loops across every target, one ruler (sightline/svgtok/loops.analyze on the target's own AR output)\n",
             "`periodic` = capped output whose tail repeats with a period in [1, max_period] (the detector the loop "
             "rules use); `tok. in loop` = share of all generated tokens inside such a tail.\n",
             "| target | mode | subset | rows | cap | ended (EOS) | capped | periodic | tok. in loop |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, globs, kw, keep in TARGETS:
        rows = ar_rows(globs)
        if not rows:
            lines.append(f"| {label} | – | – | 0 | – | – | – | – | – |")
            continue
        by = defaultdict(list)
        for (mode, st, sb, _), v in rows.items():
            by[(mode, st, sb)].append(v)
        for (mode, st, sb), vs in sorted(by.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))):
            if keep and sb not in keep:
                continue
            n = len(vs)
            ended = sum(1 for _, e, _ in vs if e)
            capped = sum(1 for t, e, c in vs if not e and len(t) >= c)
            per, loop_tok, all_tok = 0, 0, 0
            for t, e, c in vs:
                info = analyze(list(t), e, cap=c, **kw)
                all_tok += len(t)
                if info.label in ("strict-loop", "approx-loop"):
                    per += 1
                    loop_tok += info.loop_tokens
            cap = vs[0][2]
            lines.append(f"| {label} | {mode} | {st}/{sb} | {n} | {cap} | {ended / n:.0%} | {capped / n:.0%} | "
                         f"{per / n:.0%} | {loop_tok / max(1, all_tok):.0%} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
