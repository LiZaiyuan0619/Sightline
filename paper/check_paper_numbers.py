"""Recompute the headline numbers of the paper from the shipped data, and check them.

No GPU, no model weights, a second of CPU: this script reads data/runs/ and asserts the values
printed in the paper. Each check names where the claim appears, so a reader can go from a
sentence in the paper to the arithmetic behind it.

    python3 paper/check_paper_numbers.py

Exit status 0 means every claim matched. A mismatch prints both values and exits 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from sightline.paths import RUNS, require

OURS = "Ours (lossless)"
checks: list[tuple[bool, str, str]] = []


def check(ok: bool, where: str, detail: str) -> None:
    checks.append((bool(ok), where, detail))


def main_table() -> dict:
    p = require(RUNS / "2026-11-02/figdata/table1.json", "the main results, all 60 settings")
    return json.loads(p.read_text())


def run() -> int:
    d = main_table()
    blocks, methods = d["blocks"], d["meta"]["methods"]

    # --- shape: five model-task pairs x two decodings x six benchmarks, nine decoders ------------
    pairs = {b["block"] for b in blocks}
    modes = {b["mode"] for b in blocks}
    check(len(pairs) == 5, "Section 4.1", f"five model--task pairs: {len(pairs)}")
    check(len(modes) == 2, "Section 4.1", f"greedy decoding and official sampling: {sorted(modes)}")
    check(len(methods) == 9, "Section 4.1", f"eight existing decoders plus ours: {len(methods)}")
    n_cells = sum(len(b["columns"]) for b in blocks)
    check(n_cells == 60, "Section 4.1, Figure 4", f"60 settings in all: {n_cells}")

    # --- the speedup range of the introduction --------------------------------------------------
    ours_speedups = []
    for b in blocks:
        for cell in b["rows"][OURS]:
            if cell and cell.get("speedup"):
                ours_speedups.append(cell["speedup"])
    lo, hi = min(ours_speedups), max(ours_speedups)
    check(len(ours_speedups) == 60 and f"{lo:.2f}" == "4.25", "Introduction, Figures 1 and 4",
          f"slowest setting {lo:.2f}x, paper says 4.25x")
    check(f"{hi:.2f}" == "13.83", "Introduction, Figures 1 and 4",
          f"fastest setting {hi:.2f}x, paper says 13.83x")

    # --- fastest in every setting ---------------------------------------------------------------
    wins, cells = 0, 0
    for b in blocks:
        for i in range(len(b["columns"])):
            ours = b["rows"][OURS][i]
            others = [b["rows"][m][i] for m in methods if m != OURS]
            others = [c["speedup"] for c in others if c and c.get("speedup")]
            if not (ours and ours.get("speedup") and others):
                continue
            cells += 1
            wins += ours["speedup"] > max(others)
    check(wins == cells == 60, "Abstract, Introduction, Section 4.2",
          f"fastest decoder in every setting: {wins} of {cells}")

    # --- the number above each pair in Figure 4a: our mean speedup over the six benchmarks,
    #     divided by that of the fastest other decoder, per model-task pair and decoding ----------
    leads = []
    for b in blocks:
        means = {}
        for m in methods:
            v = [c["speedup"] for c in b["rows"][m] if c and c.get("speedup")]
            if v:
                means[m] = sum(v) / len(v)
        leads.append(means[OURS] / max(v for m, v in means.items() if m != OURS))
    check(f"{min(leads):.2f}" == "1.13" and f"{max(leads):.2f}" == "1.48", "Introduction, Figure 4a",
          f"mean speedup over that of the fastest other decoder: {min(leads):.2f}x to {max(leads):.2f}x")

    # --- tokens per round: ours is the longest in every setting --------------------------------
    tau_wins, tau_cells = 0, 0
    for b in blocks:
        for i in range(len(b["columns"])):
            ours = b["rows"][OURS][i]
            others = [b["rows"][m][i] for m in methods if m != OURS]
            others = [c["tau"] for c in others if c and c.get("tau")]
            if not (ours and ours.get("tau") and others):
                continue
            tau_cells += 1
            tau_wins += ours["tau"] > max(others)
    check(tau_wins == tau_cells == 60, "Section 4.2, Figure 4b",
          f"most tokens per round in every setting: {tau_wins} of {tau_cells}")

    # --- Table 1: the full SArena test sets, lossless and with one shared loop control ---------
    # The ratios divide the seconds as Table 1 prints them (two decimals), as a reader would.
    q = json.loads(require(RUNS / "2026-11-09/table-fill/figdata/table2_v2.json",
                           "the quality table").read_text())["blocks"]
    rows = ("AR", "EAGLE-3", "DFlash", "DDTree", "DARTree")
    fast_lossless = fast_loop = cols = 0
    over_ar, over_ddtree = [], []
    for blk in q.values():
        for s in ("icon", "illustration"):
            sec = {m: blk[m][s]["seconds"] for m in blk}
            cols += 1
            fast_lossless += sec["Ours"] < min(sec[m] for m in rows)
            fast_loop += sec["Ours (fast)"] < min(sec[f"{m} (fast)"] for m in rows)
            over_ar.append(round(sec["AR (fast)"], 2) / round(sec["Ours (fast)"], 2))
            over_ddtree.append(round(sec["DDTree (fast)"], 2) / round(sec["Ours (fast)"], 2))
    check(fast_lossless == cols == 8, "Section 4.2, Table 1",
          f"fastest lossless decoder in every column: {fast_lossless} of {cols}")
    check(fast_loop == cols == 8, "Abstract, Introduction, Table 1",
          f"fastest with one shared loop control in every column: {fast_loop} of {cols}")
    check(f"{min(over_ar):.2f}" == "4.65" and f"{max(over_ar):.2f}" == "7.58", "Introduction, Table 1",
          f"with loop control, over the original model under the same rule: "
          f"{min(over_ar):.2f}x to {max(over_ar):.2f}x")
    check(f"{min(over_ddtree):.2f}" == "1.05" and f"{max(over_ddtree):.2f}" == "1.19", "Section 4.3, Table 1",
          f"with loop control, over DDTree: {min(over_ddtree):.2f}x to {max(over_ddtree):.2f}x")

    # --- report -------------------------------------------------------------------------------
    width = max(len(w) for _, w, _ in checks)
    bad = 0
    for ok, where, detail in checks:
        print(f"  {'OK  ' if ok else 'FAIL'}  {where:<{width}}  {detail}")
        bad += not ok
    print(f"\n{len(checks) - bad} of {len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run())
