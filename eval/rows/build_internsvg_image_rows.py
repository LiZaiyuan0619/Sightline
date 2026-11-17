""" evaluation rows of table 1's 5th block (InternSVG-8B image-to-SVG) = the image inputs of the OmniSVG-8B
image-to-SVG block (same images, same gidx, so the two blocks share columns), with InternSVG's SArena img2svg prompt
    <instruction template> + "\\n Image: <image>\\n Instruction: " + <description of the image>
  own  ($SIGHTLINE_RUNS/h3a-data/eval_rows.jsonl): set "extra" = MMSVG-2M held-out images (300 + 300): description = the MMSVG
       `description` of that id, template drawn per row from SArena's img2svg templates of the subtype (Random(1000 + gidx),
       $SIGHTLINE_RUNS/isvg-data/i2s_templates.json, as the training prompts of build_i2s_prompts.py); set "bench" =
       MMSVG-Bench images (150 + 150), which ship WITHOUT text: only written when --bench-captions <jsonl {id, caption}>
       is given (otherwise left out);
  x    ($SIGHTLINE_RUNS/tableA/xbench/omnisvg-i2-sarena.jsonl, the 900 SArena rows): SArena's own img2svg prompt of that id.
Writes $SIGHTLINE_RUNS/tableA/xbench/internsvg-i2-own.jsonl and internsvg-i2-sarena.jsonl (eval_rows schema + "prompt").
  python build_i2s_rows.py [--bench-captions <jsonl>]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import pyarrow.parquet as pq
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
U = RUNS
M = MODELS
X = U / "tableA/xbench"
MMSVG = {"icon": M / "MMSVG-Icon/data", "illustration": M / "MMSVG-Illustration/data"}
SARENA = Path(f"{MODELS}/datasets--InternSVG--SArena/snapshots/"
              "b6b08d734616a8abf67fd221eced1dfa7c0868b9")
SARENA_I2 = {"icon": SARENA / "Icon/generation/img2svg.jsonl", "illustration": SARENA / "illustration/img2svg.jsonl"}


def descriptions(sub: str, want: set[str]) -> dict[str, str]:
    got = {}
    for f in sorted(glob.glob(str(MMSVG[sub] / "*.parquet"))):
        t = pq.read_table(f, columns=["id", "description"])
        for i, d in zip(t.column("id").to_pylist(), t.column("description").to_pylist()):
            if i in want and d:
                got[i] = d.strip()
        if len(got) == len(want):
            break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-captions", type=Path, default=None)
    a = ap.parse_args()
    tpl = json.loads((U / "isvg-data/i2s_templates.json").read_text())
    rows = [json.loads(x) for x in open(U / "h3a-data/eval_rows.jsonl") if x.strip()]
    caps = {}
    if a.bench_captions:
        caps = {json.loads(x)["id"]: json.loads(x)["caption"].strip() for x in open(a.bench_captions) if x.strip()}
    own, n_skip = [], 0
    for sub in ("icon", "illustration"):
        desc = descriptions(sub, {r["id"] for r in rows if r["subtype"] == sub and r["set"] == "extra"})
        for r in rows:
            if r["subtype"] != sub:
                continue
            d = desc.get(r["id"]) if r["set"] == "extra" else caps.get(r["id"])
            if not d:
                n_skip += 1
                continue
            t = random.Random(1000 + r["gidx"]).choice(tpl["per_subtype"][sub])
            own.append({**r, "prompt": t + tpl["sep"] + d})
    own.sort(key=lambda r: r["gidx"])
    with (X / "internsvg-i2-own.jsonl").open("w") as f:
        for r in own:
            f.write(json.dumps(r) + "\n")
    sar = {}
    for sub, p in SARENA_I2.items():
        with p.open() as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    sar[(sub, int(r["id"]))] = r["conversations"][0]["value"]
    xr = [json.loads(x) for x in open(X / "omnisvg-i2-sarena.jsonl") if x.strip()]
    xo = []
    for r in xr:
        i = int(r["id"].rsplit("-", 1)[1])
        assert r["image_path"].endswith(f"/images/{i}.png"), r
        xo.append({**r, "prompt": sar[(r["subtype"], i)]})
    with (X / "internsvg-i2-sarena.jsonl").open("w") as f:
        for r in xo:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in own + xo:
        by[(r["set"], r["subtype"])] = by.get((r["set"], r["subtype"]), 0) + 1
    print({f"{k[0]}/{k[1]}": v for k, v in sorted(by.items())}, "own skipped (no description):", n_skip)
    print("e.g.", repr(own[0]["prompt"])[:220], "|", repr(xo[0]["prompt"])[:160])


if __name__ == "__main__":
    main()
