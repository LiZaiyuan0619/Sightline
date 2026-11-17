""" cross-benchmark rows for main table v2 (each model also decodes the OTHER family's benchmark prompts).

  omnisvg-sarena: SArena text2svg prompts in OmniSVG's official text-to-svg format (the omnisvg environment; CPU).
                   Same rows as the InternSVG profile: per subset bench = Random(0).sample(150), extra =
                   Random(1).sample(300) of the rest. The SArena instruction wrapper is dropped, the description after
                   "Instruction:" goes through refs/OmniSVG prepare_inputs (build_prompts.build_prompt_ids).
  internsvg-mmsvg: MMSVG-Bench text2svg (150 + 150) and the MMSVG-2M held-out rows (300 + 300) in InternSVG's SFT
                   format: the description is wrapped with one of SArena's own 10 instruction prefixes of that
                   subset (picked by Random(gidx)), as SArena itself does, then the chat template (the dflash environment).
gidx: omnisvg-sarena 20000 + i, internsvg-mmsvg 30000 + i (disjoint from every existing AR cache key).
  python xbench_rows.py omnisvg-sarena  --out $SIGHTLINE_RUNS/tableA/xbench/omnisvg-sarena.jsonl
  python xbench_rows.py internsvg-mmsvg --out $SIGHTLINE_RUNS/tableA/xbench/internsvg-mmsvg.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
SARENA = Path(f"{MODELS}/datasets--InternSVG--SArena/snapshots/"
              "b6b08d734616a8abf67fd221eced1dfa7c0868b9")
T2 = {"icon": "Icon/generation/text2svg.jsonl", "illustration": "illustration/text2svg.jsonl"}
POOL = RUNS / "2026-10-09-prompt-pool"


def sarena_rows():
    """[(set, subtype, sarena id, full query)] in the InternSVG profile's order (bench icon, illus; extra icon, illus)."""
    out = []
    for set_name in ("bench", "extra"):
        for sub in ("icon", "illustration"):
            allr = [json.loads(x) for x in (SARENA / T2[sub]).read_text().splitlines() if x.strip()]
            bench = sorted(random.Random(0).sample(range(len(allr)), 150))
            pick = bench if set_name == "bench" else sorted(
                random.Random(1).sample(sorted(set(range(len(allr))) - set(bench)), 300))
            out += [(f"sarena-{set_name}", sub, int(allr[i]["id"]), allr[i]["conversations"][0]["value"]) for i in pick]
    return out


def omnisvg_sarena(out: Path):
    sys.path.insert(0, str(ROOT / "t6"))
    from build_prompts import build_prompt_ids, load_official_text_only
    official = load_official_text_only()
    with out.open("w") as f:
        for i, (s, sub, sid, q) in enumerate(sarena_rows()):
            desc = q.split("Instruction:", 1)[1].strip()
            f.write(json.dumps({"gidx": 20000 + i, "set": s, "subtype": sub, "id": f"sarena-{sub}-{sid}",
                                "text": desc, "prompt_input_ids": build_prompt_ids(official, desc)}) + "\n")


def mmsvg_rows():
    """[(set, subtype, id, description)]: MMSVG-Bench text2svg (POOL bench-ids) then MMSVG-2M held-out (POOL eval-ids)."""
    out = []
    for l in (POOL / "bench-ids-text2svg.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            out.append(("mmsvg-bench", r["subtype"], r["id"], r["text"]))
    for cls in ("icon", "illustration"):
        for l in (POOL / f"eval-ids/{cls}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                out.append(("mmsvg-extra", cls, r["id"], r["text"]))
    return out


def internsvg_mmsvg(out: Path):
    from transformers import AutoTokenizer
    from internsvg.profile import TARGET, _chat
    tok = AutoTokenizer.from_pretrained(TARGET)
    prefixes = {}
    for sub, rel in T2.items():
        seen = []
        for x in (SARENA / rel).read_text().splitlines():
            if x.strip():
                p = json.loads(x)["conversations"][0]["value"].split("Instruction:", 1)[0]
                if p not in seen:
                    seen.append(p)
        prefixes[sub] = seen
    with out.open("w") as f:
        for i, (s, sub, rid, desc) in enumerate(mmsvg_rows()):
            g = 30000 + i
            q = random.Random(g).choice(prefixes[sub]) + "Instruction: " + desc.strip()
            f.write(json.dumps({"gidx": g, "set": s, "subtype": sub, "id": rid, "text": q,
                                "prompt_input_ids": tok(_chat(q), add_special_tokens=False).input_ids}) + "\n")


def sarena_full_rows():
    """ [(subtype, sarena id, full query)] for the WHOLE SArena text2svg test sets (Icon 6,013 + Illustration 2,001),
    the population of InternSVG's Tables 2 / 3 (FID is sample-size dependent, so table B's SArena columns use all rows)."""
    out = []
    for sub in ("icon", "illustration"):
        for x in (SARENA / T2[sub]).read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                out.append((sub, int(r["id"]), r["conversations"][0]["value"]))
    return out


def omnisvg_sarena_full(out: Path):
    """OmniSVG prompts for every SArena text2svg row; set sarena-full, gidx 40000 + i."""
    sys.path.insert(0, str(ROOT / "t6"))
    from build_prompts import build_prompt_ids, load_official_text_only
    official = load_official_text_only()
    with out.open("w") as f:
        for i, (sub, sid, q) in enumerate(sarena_full_rows()):
            desc = q.split("Instruction:", 1)[1].strip()
            f.write(json.dumps({"gidx": 40000 + i, "set": "sarena-full", "subtype": sub, "id": f"sarena-{sub}-{sid}",
                                "text": desc, "prompt_input_ids": build_prompt_ids(official, desc)}) + "\n")


def internsvg_sarena_full(out: Path):
    """InternSVG's own SFT prompts (= profile.load_rows) for every SArena text2svg row; gidx 50000 + i."""
    from transformers import AutoTokenizer
    from internsvg.profile import TARGET, _chat
    tok = AutoTokenizer.from_pretrained(TARGET)
    with out.open("w") as f:
        for i, (sub, sid, q) in enumerate(sarena_full_rows()):
            f.write(json.dumps({"gidx": 50000 + i, "set": "sarena-full", "subtype": sub, "id": f"sarena-{sub}-{sid}",
                                "text": q, "prompt_input_ids": tok(_chat(q), add_special_tokens=False).input_ids}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["omnisvg-sarena", "internsvg-mmsvg", "omnisvg-sarena-full", "internsvg-sarena-full"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    {"omnisvg-sarena": omnisvg_sarena, "internsvg-mmsvg": internsvg_mmsvg, "omnisvg-sarena-full": omnisvg_sarena_full,
     "internsvg-sarena-full": internsvg_sarena_full}[a.which](a.out)
    n = sum(1 for _ in a.out.open())
    print(f"{a.which}: {n} rows -> {a.out}")


if __name__ == "__main__":
    main()
