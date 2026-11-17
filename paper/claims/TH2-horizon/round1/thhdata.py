"""Data layer for the Proposition 2 (horizon in decisions) figure, round 1 (main line).

ONE dataset for every candidate, the same prompts as Proposition 1's figure: our block-32 head, 300-prompt greedy cache
with the target's entropy, loop-free caliber (277 prompts have a loop-free part), decision point = target entropy > 0.2
nats; a position is scored only if it is not itself a decision point (T2 section B). Built by build_rows.py; pooled
counts equal T2's published table exactly (asserted there and again here).

Per prompt, a cell (class x decisions crossed) counts only with >= 30 scored positions (project rule); the figures
show the median PROMPT, T2's table and the chat quote the pooled position count -- both are given in the README.
"""
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import numpy as np
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = RUNS.parent            # data/, which holds runs/
D = json.loads((HERE / "rows_b32-6ep.json").read_text())
ROWS = [r for r in D["rows"] if r["lim"] > 1]
A = np.array([r["acc"] for r in ROWS], dtype=np.int64)        # prompt x band(3) x class(4) x c(3) x (miss, hit)
BANDS = ("1-8", "9-16", "17-31")
CLS = {"cmd": 0, "coord": 1}
MINN = 30


def cell(cls, band=0, c=0, strict=False):
    """Per-prompt accuracy of one cell; NaN where the prompt has fewer than MINN scored positions."""
    sub = A if not strict else A[[not r["leak"] for r in ROWS]]
    n = sub[:, band, CLS[cls], c, :].sum(-1)
    h = sub[:, band, CLS[cls], c, 1]
    out = np.full(len(n), np.nan)
    m = n >= MINN
    out[m] = h[m] / n[m]
    return out


def pooled(cls, band=0, c=0):
    n = A[:, band, CLS[cls], c, :].sum()
    return float(A[:, band, CLS[cls], c, 1].sum() / n), int(n)


def paired_c2(band=0):
    """Prompts with enough positions in both classes after two or more decisions: (coord, cmd) accuracy pairs."""
    x, y = cell("coord", band, 2), cell("cmd", band, 2)
    m = ~np.isnan(x) & ~np.isnan(y)
    return x[m], y[m]


def lines(cls, band=0):
    """Per-prompt accuracy at c = 0, 1, 2+ for prompts with enough positions at all three."""
    M = np.column_stack([cell(cls, band, c) for c in range(3)])
    return M[~np.isnan(M).any(axis=1)]


def drop(cls, band=0):
    """Per-prompt drop in accuracy from no decision crossed to two or more (percentage points)."""
    a, b = cell(cls, band, 0), cell(cls, band, 2)
    m = ~np.isnan(a) & ~np.isnan(b)
    return 100 * (a[m] - b[m])


def check():
    t2 = json.loads((RUNS / "2026-11-05/theory/decision_horizon_b32-6ep.json").read_text())["B"]["0.2"]
    for bi, bl in enumerate(BANDS):
        for cl in CLS:
            for c in range(3):
                acc, n = pooled(cl, bi, c)
                ref = t2[f"{cl}/{bl}/c{c}"]
                assert n == ref["n"] and abs(acc - ref["acc"]) < 1e-12, (cl, bl, c)
    assert len(ROWS) == 277
    # numbers quoted in the chat and README
    assert abs(np.nanmedian(cell("cmd", 0, 2)) - 0.907) < 0.0015
    assert abs(np.nanmedian(cell("coord", 0, 2)) - 0.407) < 0.0015
    x, y = paired_c2()
    assert len(x) == 194, len(x)
    return True


if __name__ == "__main__":
    check()
    for cl in CLS:
        print(cl, "pooled band 1-8:", [tuple(round(v, 3) if isinstance(v, float) else v for v in pooled(cl, 0, c))
                                       for c in range(3)],
              "| median prompt:", [round(float(np.nanmedian(cell(cl, 0, c))), 3) for c in range(3)],
              "| prompts with all three:", len(lines(cl)), "| median drop c0->c2+ (pp):",
              round(float(np.median(drop(cl))), 1))
    x, y = paired_c2()
    print("paired c2+: prompts", len(x), "| share with cmd > coord:", round(float(np.mean(y > x)), 3))
    for strict in (True,):
        print("strict-healthy median prompt c2+: cmd", round(float(np.nanmedian(cell('cmd', 0, 2, strict))), 3),
              "coord", round(float(np.nanmedian(cell('coord', 0, 2, strict))), 3))
    print("check: ok")
