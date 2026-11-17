""" prompts for InternSVG-8B image-to-SVG training data (table 1's 5th block). One prompt per image of the OmniSVG
image-to-SVG training pool ($SIGHTLINE_RUNS/h3a-data/train_pool-{icon,illustration}.jsonl, 50k each; the PNG previews of MMSVG), in the
SArena img2svg format the model was trained and evaluated on:
    <one of SArena's img2svg instruction templates> + "\\n Image: <image>\\n Instruction: " + <MMSVG description of that id>
Templates: the 9 (icon) / 10 (illustration) distinct prefixes of SArena's img2svg test files ($SIGHTLINE_RUNS/isvg-data/i2s_templates.json,
extracted on the an interactive GPU node worker), drawn per prompt with a seeded RNG. Captions: the `description` column of the MMSVG-Icon /
MMSVG-Illustration parquets (model/MMSVG-*), the same short-caption style as SArena's instructions.
Output: $SIGHTLINE_RUNS/isvg-data/i2s/prompts-<class>.jsonl {gidx, id, subtype, image_path, template, caption, prompt}.
  python build_i2s_prompts.py [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import pyarrow.parquet as pq
from sightline.paths import MODELS, RUNS

U = RUNS
M = MODELS
POOL = {"icon": U / "h3a-data/train_pool-icon.jsonl", "illustration": U / "h3a-data/train_pool-illustration.jsonl"}
MMSVG = {"icon": M / "MMSVG-Icon/data", "illustration": M / "MMSVG-Illustration/data"}
TPL = U / "isvg-data/i2s_templates.json"
OUT = U / "isvg-data/i2s"


def captions(sub: str, want: set[str]) -> dict[str, str]:
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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tpl = json.loads(TPL.read_text())
    sep = tpl["sep"]
    OUT.mkdir(parents=True, exist_ok=True)
    for k, sub in enumerate(("icon", "illustration")):
        pool = [json.loads(x) for x in POOL[sub].read_text().splitlines() if x.strip()]
        cap = captions(sub, {r["id"] for r in pool})
        rng = random.Random(a.seed * 10 + k)
        temps = tpl["per_subtype"][sub]
        n = 0
        with open(OUT / f"prompts-{sub}.jsonl", "w") as f:
            for j, r in enumerate(pool):
                c = cap.get(r["id"])
                if not c:
                    continue
                t = rng.choice(temps)
                f.write(json.dumps({"gidx": k * 100000 + j, "id": r["id"], "subtype": sub, "image_path": r["image_path"],
                                    "template": temps.index(t), "caption": c, "prompt": f"{t}{sep}{c}"},
                                   ) + "\n")   # ASCII-escaped: some captions hold U+2028 etc., which splitlines cuts
                n += 1
        print(f"{sub}: {len(pool)} images, {len(cap)} captions found, {n} prompts -> {OUT / f'prompts-{sub}.jsonl'}")


if __name__ == "__main__":
    main()
