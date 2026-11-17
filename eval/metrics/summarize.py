"""Collect metrics.py outputs and run.py statistics into one Markdown table next to OmniSVG Table 2 (8B)."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PAPER = {  # OmniSVG paper Table 2, OmniSVG-L (8B)
    "text2svg-icon": {"fid": 130.56, "clip": 0.276, "aesthetic": 4.60, "hps": 0.242},
    "text2svg-illustration": {"fid": 138.42, "clip": 0.231, "aesthetic": 4.51, "hps": 0.232},
    "image2svg-icon": {"dino": 0.922, "ssim": 0.893, "lpips": 0.235, "mse": 0.040},
    "image2svg-illustration": {"dino": 0.905, "ssim": 0.907, "lpips": 0.231, "mse": 0.031},
}
TEXT_COLS = ["fid_official", "fid_white", "clip", "aesthetic", "hps"]
IMAGE_COLS = ["dino", "ssim", "lpips", "mse"]


def run_stats(cell, cap=1536):
    rows = [json.loads(l) for l in (cell / "results.jsonl").read_text().splitlines() if l.strip()]
    cands = [c for r in rows for c in r["candidates"]]
    return {"streams": len(cands), "valid": sum(c["valid"] for c in cands) / len(cands),
            "hit_cap": sum((not c["eos_found"]) and c["length"] >= cap for c in cands) / len(cands),
            "len_p50": statistics.median(c["length"] for c in cands),
            "gen_s": statistics.mean(r["generate_seconds"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", required=True, type=Path)
    parser.add_argument("--t1-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    table = defaultdict(dict)
    for f in sorted(args.metrics_dir.glob("*/*.json")):
        s = json.loads(f.read_text())["summary"]
        key = (f.parent.name, Path(s["run"]).name, s["selection"])
        for k, v in s.items():
            if k == "fid":
                k = "fid_white" if f.stem.endswith("fid-white") else "fid_official"
            if k in TEXT_COLS + IMAGE_COLS or k in ("n_scored", "n_missing"):
                table[key][k] = v
    fmt = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) and v < 10 else
                                            f"{v:.2f}" if isinstance(v, float) else str(v))
    lines = ["# OmniSVG reproduction metrics (generated)", "",
             f"Sources: `{args.metrics_dir}` (metrics.py) and `{args.t1_dir}` (run.py). Selection: one stream = candidate 0 (the paper's protocol), "
             "8 streams = the first valid one (the command-line protocol). FID: official = the official reference set on black, white = the reference set on white. The icon FID needs the gated MMSVG-Icon.", ""]
    for task, cols in (("text2svg", TEXT_COLS), ("image2svg", IMAGE_COLS)):
        lines += [f"## {task}", "", "| setting | subset | selection | " + " | ".join(cols) + " | valid streams | at the length cap | median length | mean generate s | n |",
                  "|" + "---|" * (len(cols) + 8)]
        for subset in sorted(PAPER):
            if not subset.startswith(task):
                continue
            ref = PAPER[subset]
            lines.append(f"| **paper, 8B** | {subset.split('-')[1]} | — | " + " | ".join(fmt(ref.get("fid") if c == "fid_official" else ref.get(c)) for c in cols) + " | | | | | |")
            for (mode, sub, sel), vals in sorted(table.items()):
                if sub != subset:
                    continue
                cell = args.t1_dir / mode / sub
                st = run_stats(cell) if (cell / "results.jsonl").exists() else {}
                lines.append(f"| {mode} | {sub.split('-')[1]} | {sel} | " + " | ".join(fmt(vals.get(c)) for c in cols) +
                             f" | {fmt(st.get('valid'))} | {fmt(st.get('hit_cap'))} | {st.get('len_p50', '—')} | {fmt(st.get('gen_s'))}"
                             f" | {vals.get('n_scored', '—')}/{vals.get('n_scored', 0) + vals.get('n_missing', 0)} |")
        lines.append("")
    args.out.write_text("\n".join(lines))
    print(args.out.read_text())


if __name__ == "__main__":
    main()
