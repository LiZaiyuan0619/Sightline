"""First experiments composite (main.pdf Figures 5, 7, 8, 9, 10), round 1: render every candidate cell at its print
size (STRICT=0 drafts past the geometry check), then one contact sheet at print scale.

Run:  python3 paper/claims/EX1-grid/round1/make.py [name...]
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as CL  # noqa: E402

STRICT = os.environ.get("STRICT", "1") != "0"
OUT = HERE / "cand"


def render(name):
    fn, kind = CL.CANDIDATES[name]
    w, h = (CL.CELL_W, CL.CELL_H) if kind == "cell" else (CL.TALL_W, CL.TALL_H)
    fig = plt.figure(figsize=(w / 72, h / 72))
    fn(fig, (0.0, 0.0, w, h))
    CL.H.save(fig, str(OUT / name), width=w / 72, strict=STRICT)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    names = sys.argv[1:] or list(CL.CANDIDATES)
    bad = []
    for n in names:
        try:
            render(n)
        except AssertionError as e:
            bad.append(f"{n}: {e}")
            print("FAIL", n, e)
    pngs = [str(OUT / f"{n}.png") for n in CL.CANDIDATES if (OUT / f"{n}.png").exists()]
    CL.H.sheet(pngs, str(HERE / "_sheet.png"), cols=4, title="EX1 round 1 candidates (print scale)")
    print("geometry FAILs:", len(bad))
    for b in bad:
        print("  ", b)
