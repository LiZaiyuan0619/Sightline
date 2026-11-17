""" what the user actually sees while the model is drawing.

SVG is one of the few generation targets whose PREFIX is renderable, so "how far along is it at time t" is a real
picture, not a proxy. For a decoded run we know the output tokens and the wall clock of the whole decode; the harness
also records the tokens emitted per round, so the time of every token is known to within one round. That gives, for
each method and each prompt:

  t_valid    the first moment the prefix renders as a valid SVG at all;
  t_q(q)     the first moment the rendered prefix is at least q similar to the method's own final image
             (pixel IoU of the non-white mask, which is what "the drawing is nearly there" looks like);
  progress(t) the similarity-to-final as a function of wall-clock time, on a common grid.

Stage 1 (`render`, CPU pool, the svgmetrics environment) renders the prefixes; stage 2 (`report`) writes the curves. Prefixes are
cut at round boundaries, so no partial command is ever rendered.

  python analysis/preview_curve.py render --runs ar=<dir> ours=<dir> --rows <n> --out <dir>
  python analysis/preview_curve.py report --out <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "src/h2", "src/t7", "src/tol"):
    sys.path.insert(0, str(ROOT / p))

GRID = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]      # seconds
QS = (0.5, 0.75, 0.9, 0.95)


def round_times(r: dict) -> list[tuple[int, float]]:
    """(tokens emitted so far, wall clock) at every round boundary; the decode's own total time is the last point."""
    acc = r["acceptance"]
    n = sum(acc)
    if not n:
        return []
    t_tot = r.get("t_spec", r.get("t_total", 0.0))
    out, run = [], 0
    for k, a in enumerate(acc):
        run += a
        out.append((run, t_tot * (k + 1) / len(acc)))       # rounds cost about the same; the last point is exact
    return out


def ar_times(r: dict) -> list[tuple[int, float]]:
    """AR emits one token per forward, so its progress in time is linear in the token index."""
    n = len(r.get("ar_tokens") or [])
    t = r.get("t_ar", 0.0)
    if not n or not t:
        return []
    step = max(1, n // 60)
    return [(i, t * i / n) for i in range(step, n + 1, step)]


def mask(img):
    import numpy as np
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    return (a.sum(-1) < 720)                                # non-white ink


def iou(a, b):
    import numpy as np
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 1.0


def render_tokens(off, tokens, subtype):
    from PIL import Image
    try:
        svg, img, valid, _ = off_decode(off, tokens, subtype)
    except Exception:
        return None, False
    if img is None:
        return Image.new("RGB", (1024, 1024), (255, 255, 255)), False
    return img.convert("RGB"), bool(valid)


def off_decode(off, tokens, subtype):
    from omnisvg.run import decode
    return decode(off, tokens, subtype)


def one_run(off, r, kind):
    """-> {"gidx", "subtype", "t_total", "points": [(t, iou_to_final, valid)]}"""
    pts = ar_times(r) if kind == "ar" else round_times(r)
    toks = (r.get("ar_tokens") if kind == "ar" else r.get("spec_tokens")) or []
    if not pts or not toks:
        return None
    final, _ = render_tokens(off, toks, r["subtype"])
    if final is None:
        return None
    mf = mask(final)
    out = []
    for n, t in pts:
        img, valid = render_tokens(off, toks[:n], r["subtype"])
        out.append((round(t, 4), round(iou(mask(img), mf), 4) if img is not None else 0.0, bool(valid)))
    return {"gidx": r["gidx"], "subtype": r["subtype"], "kind": kind,
            "t_total": pts[-1][1], "n_tokens": len(toks), "points": out}


def load_rows(root, mode, limit):
    rows = [r for f in sorted(glob.glob(f"{root}/shard-*/results.jsonl")) for r in map(json.loads, open(f))
            if r and r["mode"] == mode]
    rows.sort(key=lambda r: r["gidx"])
    return rows[:limit]


def cmd_render(a):
    from omnisvg.run import load_official
    off = load_official()
    off.svg_tokenizer = off.SVGTokenizer(off.CONFIG_PATH, model_size="8B")
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a.runs:
        name, root = spec.split("=", 1)
        kind = "ar" if name.startswith("ar") else "sd"
        f = a.out / f"{name}.jsonl"
        done = {json.loads(x)["gidx"] for x in f.read_text().splitlines()} if f.exists() else set()
        with f.open("a") as fh:
            for r in load_rows(root, a.mode, a.rows):
                if r["gidx"] in done:
                    continue
                rec = one_run(off, r, kind)
                if rec:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(name, r["gidx"], f"{rec['t_total']:.2f}s", len(rec["points"]), "pts", flush=True)
    print("PREVIEW_RENDER_DONE", a.out)


def cmd_report(a):
    import statistics as st
    runs = {}
    for f in sorted(a.out.glob("*.jsonl")):
        runs[f.stem] = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    res = {}
    for name, recs in runs.items():
        by = {}
        for q in QS:
            ts = []
            for rec in recs:
                hit = next((t for t, v, _ in rec["points"] if v >= q), None)
                ts.append(hit if hit is not None else rec["t_total"])
            by[q] = {"mean": st.mean(ts), "median": st.median(ts)}
        tv = []
        for rec in recs:
            hit = next((t for t, _, ok in rec["points"] if ok), None)
            tv.append(hit if hit is not None else rec["t_total"])
        prog = {}
        for g in GRID:
            vs = [max([v for t, v, _ in rec["points"] if t <= g], default=0.0) for rec in recs]
            prog[g] = st.mean(vs) if vs else 0.0
        res[name] = {"n": len(recs), "t_valid_mean": st.mean(tv) if tv else None,
                     "t_valid_median": st.median(tv) if tv else None,
                     "t_total_mean": st.mean(r["t_total"] for r in recs) if recs else None,
                     "quantiles": {str(q): by[q] for q in QS}, "progress": prog}
    (a.out / "report.json").write_text(json.dumps(res, indent=1))
    L = ["# Progressive preview: when does the drawing become watchable, and when is it nearly done", "",
         "Similarity = pixel IoU of the ink mask against that method's own final image. Times are seconds.", "",
         "| run | n | first valid render | 50% of final | 75% | 90% | 95% | full decode |",
         "|---|---|---|---|---|---|---|---|"]
    for n, v in res.items():
        q = v["quantiles"]
        L.append(f"| {n} | {v['n']} | {v['t_valid_mean']:.2f} | " +
                 " | ".join(f"{q[str(x)]['mean']:.2f}" for x in QS) + f" | {v['t_total_mean']:.2f} |")
    L += ["", "Mean similarity to the final image at a fixed wall clock:", "",
          "| t (s) | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for g in GRID:
        L.append(f"| {g} | " + " | ".join(f"{res[n]['progress'][g]:.3f}" for n in res) + " |")
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    print("PREVIEW_REPORT_DONE", a.out / "report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "report"])
    ap.add_argument("--runs", nargs="*", default=[], help="name=<results dir>; names starting with 'ar' use ar_tokens")
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (cmd_render if a.cmd == "render" else cmd_report)(a)


if __name__ == "__main__":
    main()
