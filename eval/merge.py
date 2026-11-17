"""Merge the shards of one B3 evaluation (eval_b3.py --shard i/n) into the
C2 tables.

Reads <dir>/shard-*/results.jsonl and coord_positions.jsonl, rebuilds the
per-stream caliber bundles from the saved records (rec_full / rec_trunc /
rec_loop + onset) and reports, per key mode/set/subtype and pooled per
mode/subtype and per mode:
  - tokens per round, 4 calibers x 2 aggregations (analysis/calibers.py);
  - NFE totals and per-token wall-clock speedup vs same-stack AR (collector excluded);
  - greedy lossless check: streams diverging from AR (first divergence position);
  - H2 evidence on coordinates, split into loop-free positions (pos < loop onset or
    no loop) and loop positions: rejected-coordinate distances and the
    chain-processed probability mass within +-1/2/4 px of the draft / argmax.

Usage (CPU):
  python3 eval_b3_merge.py runs/<date>-b3-eval/<head-tag>
Writes <dir>/summary.json and <dir>/summary.md.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from analysis.calibers import CaliberStream, StreamRecord, caliber_table  # noqa: E402


def record(x):
    return None if x is None else StreamRecord(steps=x[0], emitted=x[1])


def bundle(r: dict) -> CaliberStream:
    return CaliberStream(full=record(r["rec_full"]), eos=r["eos"], onset=r["onset"],
                         trunc=record(r["rec_trunc"]), loop=record(r["rec_loop"]))


def speed(r: dict) -> float:
    return (r["t_ar"] / max(r["n_ar"], 1)) / max(r["t_spec_net"] / max(r["n_spec"], 1), 1e-9)


def coord_stats(rows: list[dict]) -> dict:
    cheb = [c["cheb"] for c in rows if "cheb" in c]
    out = {"n_coord_positions": len(rows),
           "accept_rate": (st.fmean([bool(c["accepted"]) for c in rows]) if rows else None),
           "n_rejected_both_coord": len(cheb),
           "cheb_median": st.median(cheb) if cheb else None,
           "cheb_le1": st.fmean([d <= 1 for d in cheb]) if cheb else None,
           "cheb_le2": st.fmean([d <= 2 for d in cheb]) if cheb else None,
           "cheb_le4": st.fmean([d <= 4 for d in cheb]) if cheb else None}
    for name in ("mass_draft", "mass_argmax"):
        have = [c[name] for c in rows if name in c]
        out[name] = ({f"pm{r}": st.fmean([m[f"mass_pm{r}"] for m in have]) for r in (1, 2, 4)}
                     if have else None)
    return out


def summarize(results: list[dict], coords: list[dict]) -> dict:
    table = caliber_table([bundle(r) for r in results])
    greedy = [r for r in results if r["mode"] == "greedy"]
    healthy = [c for c in coords if c["onset"] is None or c["pos"] < c["onset"]]
    looped = [c for c in coords if c["onset"] is not None and c["pos"] >= c["onset"]]
    return {
        "n": len(results),
        "n_eos": table["n_eos"], "n_loop": table["n_loop"],
        "n_trunc_excluded": table["n_trunc_excluded"],
        "calibers": table["calibers"],
        "nfe_spec_total": sum(r["nfe_spec"] for r in results),
        "nfe_ar_total": sum(r["nfe_ar"] for r in results),
        "speedup_per_token_median": st.median([speed(r) for r in results]) if results else None,
        "speedup_per_token_mean": st.fmean([speed(r) for r in results]) if results else None,
        "collector_share_of_spec_time": (sum(r["t_hook"] for r in results) /
                                         max(sum(r["t_spec"] for r in results), 1e-9)),
        "greedy_divergent_streams": (sum(r["first_div_vs_ar"] is not None for r in greedy)
                                     if greedy else None),
        "greedy_first_div_positions": [r["first_div_vs_ar"] for r in greedy
                                       if r["first_div_vs_ar"] is not None][:50],
        "coord_healthy": coord_stats(healthy),
        "coord_loop": coord_stats(looped),
    }


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    d = Path(sys.argv[1])
    results, coords = [], []
    shards = sorted(d.glob("shard-*"))
    for s in shards:
        for line in (s / "results.jsonl").read_text().split("\n"):
            if line.strip():
                results.append(json.loads(line))
        cp = s / "coord_positions.jsonl"
        if cp.exists():
            for line in cp.read_text().split("\n"):
                if line.strip():
                    coords.append(json.loads(line))
    keys = defaultdict(list)
    ckeys = defaultdict(list)
    for r in results:
        mode, set_name, subtype = r["key"].split("/")
        for k in (r["key"], f"{mode}/all/{subtype}", f"{mode}/all/all"):
            keys[k].append(r)
    for c in coords:
        mode, set_name, subtype = c["key"].split("/")
        for k in (c["key"], f"{mode}/all/{subtype}", f"{mode}/all/all"):
            ckeys[k].append(c)
    summary = {k: summarize(keys[k], ckeys[k]) for k in sorted(keys)}
    summary["_meta"] = {"dir": str(d), "n_shards": len(shards), "n_results": len(results),
                        "n_coord_rows": len(coords)}
    (d / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["| key | n | eos | τ_all | τ_eos | τ_trunc (n) | τ_loop | merged τ_trunc | "
             "NFE spec/AR | speedup (med) | greedy div |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(keys):
        s = summary[k]
        c = s["calibers"]
        lines.append(
            f"| {k} | {s['n']} | {s['n_eos']} | {fmt(c['all']['per_stream'])} | "
            f"{fmt(c['eos']['per_stream'])} | {fmt(c['trunc']['per_stream'])} "
            f"({c['trunc']['n']}) | {fmt(c['loop']['per_stream'])} | "
            f"{fmt(c['trunc']['merged'])} | {s['nfe_spec_total']}/{s['nfe_ar_total']} | "
            f"{fmt(s['speedup_per_token_median'], 2)} | {fmt(s['greedy_divergent_streams'])} |")
    lines += ["", "H2 evidence (healthy positions):", "",
              "| key | coord pos | accept | rejected both-coord | cheb med | ≤1 / ≤2 / ≤4 | "
              "mass@draft ±1/±2/±4 | mass@argmax ±1/±2/±4 |", "|---|---|---|---|---|---|---|---|"]
    for k in sorted(keys):
        h = summary[k]["coord_healthy"]
        md, ma = h["mass_draft"] or {}, h["mass_argmax"] or {}
        lines.append(
            f"| {k} | {h['n_coord_positions']} | {fmt(h['accept_rate'])} | "
            f"{h['n_rejected_both_coord']} | {fmt(h['cheb_median'])} | "
            f"{fmt(h['cheb_le1'], 2)} / {fmt(h['cheb_le2'], 2)} / {fmt(h['cheb_le4'], 2)} | "
            f"{' / '.join(fmt(md.get(f'pm{r}')) for r in (1, 2, 4))} | "
            f"{' / '.join(fmt(ma.get(f'pm{r}')) for r in (1, 2, 4))} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
