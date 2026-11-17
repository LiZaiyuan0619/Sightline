"""Extract the 600 MMSVGBench records and deterministic smoke subsets on CPU."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import pyarrow.parquet as pq
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "model/MMSVGBench"
OUT = RUNS / "2026-10-07-omnisvg-inputs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    ids = set()
    hashes = {}
    for task in ("text2svg", "image2svg"):
        source = DATA / f"data/{task}-00000-of-00001.parquet"
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = pq.read_table(source).to_pylist()
        groups = {"icon": [], "illustration": []}
        for row in rows:
            sample_id = row["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id) or sample_id in ids:
                raise ValueError(f"Invalid or duplicate sample id: {sample_id}")
            ids.add(sample_id)
            assert row["task_type"] == task
            subtype = row["type"]
            assert subtype in groups
            record = {"id": sample_id, "task": task, "subtype": subtype,
                      "text": row["text"], "source_url": row["url"]}
            if task == "image2svg":
                image = row["image"]
                if not image or not image["bytes"]:
                    raise ValueError(f"Missing image: {sample_id}")
                folder = OUT / f"{task}-{subtype}/images"
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{sample_id}.png"
                path.write_bytes(image["bytes"])
                record["image_path"] = str(path)
                record["image_sha256"] = hashlib.sha256(image["bytes"]).hexdigest()
            else:
                assert isinstance(row["text"], str) and row["text"].strip()
            groups[subtype].append(record)
            counts[f"{task}-{subtype}"] += 1
        for subtype, records in groups.items():
            assert len(records) == 150, (task, subtype, len(records))
            folder = OUT / f"{task}-{subtype}"
            folder.mkdir(parents=True, exist_ok=True)
            for filename, subset in [("manifest.jsonl", records), ("smoke5.jsonl", records[:5])]:
                (folder / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
            if task == "text2svg":
                assert all("\n" not in r["text"] for r in records)
                (folder / "prompts.txt").write_text("\n".join(r["text"] for r in records) + "\n")
    (OUT / "summary.json").write_text(json.dumps({"counts": counts, "parquet_sha256": hashes}, indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
