"""H3a S1/S3: image-to-SVG
evaluation rows and the training image pool, on CPU.

Evaluation rows (900, laid out like the text2svg B3 rows so tables line up):
  gidx   0-149  MMSVGBench image2svg icon          (runs/2026-10-07-omnisvg-inputs)
  gidx 150-299  MMSVGBench image2svg illustration
  gidx 300-599  held-out MMSVG-Icon previews        = the text2svg eval pool's items
  gidx 600-899  held-out MMSVG-Illustration previews  (never used for any training)
Training pool: the next rows of the shuffled, bench-filtered universe after the text2svg
extension, rows [150300, 150300 + N) per class (extend_pool.py used [50300, 150300)),
so no training image is an evaluation item or a text2svg prompt of the first 300k streams.

Images are the PNG previews stored in the MMSVG parquets, written as files; the official
preprocessing (white background, background replacement, 448 px) happens at use time.

Usage (the omnisvg environment, dev CPU):
  python build_sets.py [--n-train 15000] [--out <output dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import pyarrow.parquet as pq
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
MODEL = MODELS
POOL = RUNS / "2026-10-09-prompt-pool"
BENCH = RUNS / "2026-10-07-omnisvg-inputs"
SOURCES = {"icon": MODEL / "MMSVG-Icon/data", "illustration": MODEL / "MMSVG-Illustration/data"}
TRAIN_START = 50300 + 100000          # after the text2svg train pool and its 200k extension


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().split("\n") if l.strip()]


def extract(cls: str, wanted: set, out_dir: Path) -> dict:
    """id -> png path for the wanted ids of one class (scans the parquet id columns)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    for shard in sorted(SOURCES[cls].glob("train-*.parquet")):
        pf = pq.ParquetFile(shard)
        for g in range(pf.num_row_groups):
            ids = pf.read_row_group(g, columns=["id"]).column("id").to_pylist()
            hits = [i for i, x in enumerate(ids) if x in wanted and x not in found]
            if not hits:
                continue
            imgs = pf.read_row_group(g, columns=["image"]).column("image")
            for i in hits:
                b = imgs[i].as_py()["bytes"]
                p = out_dir / f"{ids[i]}.png"
                p.write_bytes(b)
                found[ids[i]] = str(p)
        if len(found) == len(wanted):
            break
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"{cls}: {len(missing)} ids not found, e.g. {sorted(missing)[:3]}")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=15000, help="training images per class")
    ap.add_argument("--out", type=Path, default=(RUNS / "h3a-data"))
    args = ap.parse_args()

    rows = []
    for cls in ("icon", "illustration"):
        for r in read_jsonl(BENCH / f"image2svg-{cls}" / "manifest.jsonl"):
            rows.append({"id": r["id"], "set": "bench", "subtype": cls, "image_path": r["image_path"]})
    train_rows = {}
    for cls in ("icon", "illustration"):
        universe = read_jsonl(POOL / f"universe-{cls}.jsonl")
        eval_ids = [r["id"] for r in read_jsonl(POOL / "eval-pool" / f"{cls}.jsonl")]
        assert [r["id"] for r in universe[:300]] == eval_ids, f"{cls}: eval split drift"
        train_slice = [r["id"] for r in universe[TRAIN_START:TRAIN_START + args.n_train]]
        assert len(train_slice) == args.n_train, (cls, len(train_slice))
        assert not set(train_slice) & set(eval_ids)
        paths = extract(cls, set(eval_ids), args.out / "eval-images" / cls)
        for i in eval_ids:
            rows.append({"id": i, "set": "extra", "subtype": cls, "image_path": paths[i]})
        tpaths = extract(cls, set(train_slice), args.out / "train-images" / cls)
        train_rows[cls] = [{"id": i, "subtype": cls, "image_path": tpaths[i]} for i in train_slice]
        print(f"{cls}: eval {len(eval_ids)}, train {len(train_slice)} images extracted", flush=True)
    for g, r in enumerate(rows):
        r["gidx"] = g
    assert len(rows) == 900
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for cls, lst in train_rows.items():
        (args.out / f"train_pool-{cls}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lst))
    (args.out / "manifest.json").write_text(json.dumps({
        "eval_rows": 900, "layout": "bench icon 0-149, bench illustration 150-299, "
                                    "extra icon 300-599, extra illustration 600-899",
        "train_rows_per_class": args.n_train, "train_universe_rows": [TRAIN_START, TRAIN_START + args.n_train],
        "source": "MMSVG parquet PNG previews; bench = MMSVGBench image2svg"}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
