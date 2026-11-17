"""P1 round 2 --  (B line, C2): round 1's p1b_rays moved onto the house style. Style only: same data
(runs/2026-10-30/analysis/timing_split_all.md via round 1's loader, the identity asserted there), same points, same
iso-speed-up rays (2x... 10x), same axes.
What changes:
  - the rays are reference lines (gid "ref") and stop short of every label they would cross (round 1's PDF audit had
    12 text-stroke FAILs: method names and ray labels drawn across the rays);
  - EAGLE-3's name sits above its point (round 1: left of it, touching DDTree's name -- house.check text-text);
  - the note reads "over the original model" (ENCODING.md SS3; round 1 wrote "over AR"), on two lines;
  - house.apply; house.save (blocks on a geometry FAIL).
Run: python paper/claims/P1-speed-identity/round2/make.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import matplotlib.pyplot as plt
from matplotlib.text import Text
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(RUNS / "2026-11-06/chores/c2"))
import c2lib as C  # noqa: E402

H = C.H
spec = importlib.util.spec_from_file_location("p1r1", HERE.parent / "round1/make.py")
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
H.apply()
XMAX, YMAX = 40.0, 21.5
CLEAR_PT = 1.6                      # gap kept between a ray and any text box


def _cut(x0, y0, x1, y1, boxes):
    """Pieces of the segment (x0,y0)-(x1,y1) outside every box (Liang-Barsky per box), as parameter intervals."""
    keep = [(0.0, 1.0)]
    dx, dy = x1 - x0, y1 - y0
    for bx0, by0, bx1, by1 in boxes:
        lo, hi = 0.0, 1.0
        for p, q in ((-dx, x0 - bx0), (dx, bx1 - x0), (-dy, y0 - by0), (dy, by1 - y0)):
            if abs(p) < 1e-12:
                if q < 0:
                    lo, hi = 1.0, 0.0
                    break
            else:
                t = q / p
                if p < 0:
                    lo = max(lo, t)
                else:
                    hi = min(hi, t)
        if lo >= hi:
            continue
        new = []
        for a, b in keep:
            if hi <= a or lo >= b:
                new.append((a, b))
                continue
            if lo > a:
                new.append((a, lo))
            if hi < b:
                new.append((hi, b))
        keep = new
    return keep


def p1b_rays(rows, ar):
    fig = plt.figure(figsize=(r1.CW, 2.75))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.80])
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    texts, ends = [], {}
    for s in (2, 4, 6, 8, 10):
        xe, ye = 38.5, s * 38.5 / ar
        if ye > 21:
            xe, ye = 21 * ar / s, 21
        ends[s] = (xe, ye)
        texts.append(ax.text(xe, ye, f"{s}×", fontsize=7.0, color="#9A9A9A", ha="left", va="bottom"))
    for m in r1.METHODS:
        r = rows[m]
        ours = m == "Ours"
        ax.scatter(r["ms"], r["tau"], s=46 if ours else 26, color=r1.COLOR[m], edgecolor="white", linewidths=0.5,
                   zorder=4)
        dx, dy, ha, va = {"PLD": (-5, -5, "right", "center"), "REST": (4, 4, "left", "center"),
                          "EAGLE-3": (0, 6, "center", "bottom"), "DFlash": (-5, 2, "right", "center"),
                          "DARTree": (5, 2, "left", "center"), "DDTree": (5, -6, "left", "center"),
                          "Ours": (5, 0, "left", "center")}[m]
        texts.append(ax.annotate(m, (r["ms"], r["tau"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                 fontsize=H.SZ["legend"], color=r1.COLOR["Ours"] if ours else "#2E2E2E",
                                 fontweight="bold" if ours else "normal"))
    texts.append(ax.text(0.98, 0.03, "grey rays: equal speed-up\nover the original model", transform=ax.transAxes,
                         fontsize=H.SZ["note"], color="#8A8A8A", ha="right", va="bottom", linespacing=1.1))
    ax.set_xlabel("ms per round", fontsize=H.SZ["label"], labelpad=1.5)
    ax.set_ylabel("tokens per round τ", fontsize=H.SZ["label"], labelpad=2)
    # the rays, cut where they would cross a text box (measured after the layout is fixed)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    pad = CLEAR_PT * fig.dpi / 72
    boxes = []
    for t in texts:
        bb = Text.get_window_extent(t, rend)
        (bx0, by0), (bx1, by1) = inv.transform([(bb.x0 - pad, bb.y0 - pad), (bb.x1 + pad, bb.y1 + pad)])
        boxes.append((bx0, by0, bx1, by1))
    for s in (2, 4, 6, 8, 10):
        x1 = min(XMAX, YMAX * ar / s)
        y1 = s * x1 / ar
        for a, b in _cut(0.0, 0.0, x1, y1, boxes):
            ln, = ax.plot([a * x1, b * x1], [a * y1, b * y1], color="#DADADA", lw=0.7, zorder=0)
            ln.set_gid("ref")
    H.save(fig, str(HERE / "p1b_rays"), width="col")


def main():
    rows, ar = r1.load()
    p1b_rays(rows, ar)
    print("wrote p1b_rays")


if __name__ == "__main__":
    main()
