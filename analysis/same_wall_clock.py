""" (claim H2, the front-page candidate): the same-wall-clock view. H1 (analysis/preview_curve.py) answers
"how fast does one drawing converge"; H2 answers "at the SAME instant, how much has each method actually delivered" -
the picture a reader gets in one glance: by the time the target model has finished one icon, we have finished several.

Input: the per-row preview records of preview_curve.py (`<dir>/<run>.jsonl`: points = [(wall-clock seconds, pixel IoU
of the rendered prefix against that run's own final image, renders-validly)] at round boundaries). No GPU.

  grid     per method and time t: prompts finished by t, prompts rendering validly, mean progress (finished = 1.0),
           and the per-prompt progress matrix the figure plots. Finished = t >= the row's total decode time.
           Also "drawings completed per prompt-minute" and, for each method pair, the time ratio to reach a progress.
  gallery renders each method's prefix at those instants for a few prompts -> <out>/gallery/<gidx>/<run>-<t>.png,
           so the figure can show real drawings rather than a curve. CPU pool, the svgmetrics environment (cairosvg + the official
           OmniSVG decoder), reusing preview_curve.render_tokens so the rasterisation matches H1 exactly.

  python analysis/same_clock.py grid --src runs/2026-10-29/analysis/preview-t2 \
      --out runs/2026-10-31/analysis/same-clock-t2
  python analysis/same_clock.py gallery --src... --out... --gidx 0,3,7 --workers 12
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from sightline.paths import RUNS as RUNS_DIR

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "src/h2", "src/t7", "src/tol"):
    sys.path.insert(0, str(ROOT / p))

GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]        # seconds
RUNS = ["ar", "ours", "ours-ets"]


def load(src: Path) -> dict:
    out = {}
    for f in sorted(src.glob("*.jsonl")):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if rows:
            out[f.stem] = {r["gidx"]: r for r in rows}
    return out


def at(rec: dict, t: float) -> tuple[float, bool, bool]:
    """(progress, renders validly, finished) of one row at wall clock t. Progress is the last measured point at or
    before t; a row past its total decode time counts as 1.0 (its own final image)."""
    if t >= rec["t_total"]:
        return 1.0, True, True
    prog, valid = 0.0, False
    for tt, iou, ok in rec["points"]:
        if tt > t:
            break
        prog, valid = iou, ok
    return prog, valid, False


def cmd_grid(a):
    data = load(a.src)
    runs = [r for r in RUNS if r in data] + [r for r in data if r not in RUNS]
    common = sorted(set.intersection(*(set(data[r]) for r in runs)))
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"rows": len(common), "runs": runs, "grid": GRID, "by_run": {}}
    for r in runs:
        per_t = {}
        for t in GRID:
            vals = [at(data[r][g], t) for g in common]
            per_t[str(t)] = {"finished": sum(f for _, _, f in vals),
                             "valid": sum(v for _, v, _ in vals),
                             "mean_progress": sum(p for p, _, _ in vals) / len(vals),
                             "per_prompt": [round(p, 4) for p, _, _ in vals]}
        tot = [data[r][g]["t_total"] for g in common]
        res["by_run"][r] = {"per_t": per_t, "t_total_mean": sum(tot) / len(tot),
                            "drawings_per_minute": 60.0 / (sum(tot) / len(tot))}
    (a.out / "same_clock.json").write_text(json.dumps(res, indent=1))

    L = ["# Same wall clock: what each method has delivered at instant t "
         f"(OmniSVG-8B text-to-SVG, {len(common)} MMSVG-Bench prompts, one H200)\n",
         "Progress = pixel IoU of the rendered prefix against that method's own final image; a prompt past its own "
         "decode time counts as finished (1.0). Rows are the preview records of analysis/preview_curve.py.\n",
         "| t (s) | " + " | ".join(f"{r} done | {r} mean progress" for r in runs) + " |",
         "|---|" + "---|" * (2 * len(runs))]
    for t in GRID:
        cells = []
        for r in runs:
            d = res["by_run"][r]["per_t"][str(t)]
            cells += [f"{d['finished']}/{len(common)}", f"{d['mean_progress']:.3f}"]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "| run | mean decode (s) | finished drawings per prompt-minute |", "|---|---|---|"]
    for r in runs:
        L.append(f"| {r} | {res['by_run'][r]['t_total_mean']:.2f} | {res['by_run'][r]['drawings_per_minute']:.1f} |")
    (a.out / "same_clock.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def _job(item):
    import preview_curve as pc
    gidx, run, t, n, subtype, toks, out = item
    img, valid = pc.render_tokens(_W["off"], toks[:n], subtype)
    if img is None:
        return gidx, run, t, "no_svg"
    img.save(out)
    return gidx, run, t, ("ok" if valid else "invalid")


_W: dict = {}


def _init():
    from omnisvg.run import load_official
    off = load_official()
    off.svg_tokenizer = off.SVGTokenizer(off.CONFIG_PATH, model_size="8B")
    _W["off"] = off


def cmd_gallery(a):
    """Render every method's prefix at the grid instants, for the chosen prompts."""
    from multiprocessing import Pool

    import preview_curve as pc
    data = load(a.src)
    runs = [r for r in RUNS if r in data]
    src_rows = {}                                    # the token sequences live in the original result dirs
    for spec in a.runs:
        name, root = spec.split("=", 1)
        src_rows[name] = {r["gidx"]: r for r in pc.load_rows(root, a.mode, a.rows)}
    gidxs = [int(x) for x in a.gidx.split(",")] if a.gidx else sorted(set.intersection(*(set(data[r]) for r in runs)))[:4]
    items = []
    for g in gidxs:
        for r in runs:
            rec, raw = data[r].get(g), src_rows.get(r, {}).get(g)
            if rec is None or raw is None:
                continue
            toks = raw["ar_tokens"] if r.startswith("ar") else raw["spec_tokens"]
            # points[i] and round_times(raw)[i] are the same round (preview_curve.one_run builds them together)
            marks = list(zip(rec["points"], pc.round_times(raw)))
            counts = {t: max([nn for (tt, _, _), (nn, _) in marks if tt <= t] or [0]) for t in GRID}
            d = a.out / "gallery" / f"{g:04d}"
            d.mkdir(parents=True, exist_ok=True)
            for t, n in counts.items():
                items.append((g, r, t, n, rec["subtype"], toks, d / f"{r}-{t}.png"))
    stats: dict = {}
    with Pool(a.workers, initializer=_init) as pool:
        for g, r, t, v in pool.imap_unordered(_job, items, chunksize=4):
            stats[v] = stats.get(v, 0) + 1
    print("[gallery]", stats, "->", a.out / "gallery")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["grid", "gallery"])
    ap.add_argument("--src", type=Path, default=RUNS_DIR / "2026-10-29/analysis/preview-t2")
    ap.add_argument("--out", type=Path, default=RUNS_DIR / "2026-10-31/analysis/same-clock-t2")
    ap.add_argument("--gidx", default=None, help="gallery: comma-separated prompts (default: the first four)")
    ap.add_argument("--runs", nargs="*", default=[], help="gallery: name=<result dir> (as preview_curve render)")
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    {"grid": cmd_grid, "gallery": cmd_gallery}[a.stage](a)


if __name__ == "__main__":
    main()
