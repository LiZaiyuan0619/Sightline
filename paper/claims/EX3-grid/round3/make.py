"""Figure 7 (main.pdf, ), round 3: the 2 x 3 the project decision.

User,  (session 7):
  - cell a of round 1 (the loop-control rose) leaves: its numbers are the merged quality table's DDTree and Ours rows
    with and without loop control, and the same 16 gains are Figure 3f's points; the table now holds all six decoders
    with the same rule, speed and quality (runs/2026-11-09/table-fill);
  - the three frontier cells (round 1's d-f) fold into one: Q1-quality-frontier/round3 f3a_overlay (both axes relative
    to the original model's best of 8, so the three blocks share one axis);
  - one row of three panels per row, no row of two.
  row 1  a why escape then stop (round 1's b)        b target forward passes (round 1's c)   c the frontier, one cell
  row 2  d three VLM judges (round 1's g)            e selection rule (round 1's h)          f window vs concurrency (i)
Cells a, b, d, e, f are round 2's parameterised copies of round 1's code (same data layers, nothing recomputed); c is
f3a's drawing function. Run:  python3 paper/claims/EX3-grid/round3/make.py   (STRICT=0 to draft past the check)
"""
import importlib.util
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CLAIMS = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R2 = _load("ex3r2", HERE.parent / "round2/make.py")       # loads round 1 and its data layers (checks run on import)
QF = _load("q1r3", CLAIMS / "Q1-quality-frontier/round3/make.py")   # runs Q1 round 2's check_numbers
R, H = R2.R, R2.H
H.apply()
STRICT = os.environ.get("STRICT", "1") != "0"
FULL, CW, TOP, LEG = R2.FULL, R2.CW, R.TOP, R.LEG
NAME = "ex3_2x3"


class Rows2:
    """Two rows (top to bottom), the interface round 2's cell functions read: y(k) = bottom, top(k) = top from the
    figure's top edge, h[k] = height, all in points."""

    def __init__(self, r1, r2):
        self.h = {1: r1, 2: r2}
        self.fig_h = TOP + r1 + r2

    def y(self, k):
        return {1: self.h[2], 2: 0.0}[k]

    def top(self, k):
        return {1: TOP, 2: TOP + self.h[1]}[k]


def frontier(fig, rows, k, x0, w):
    """Cell c: the folded frontier (Q1 round 3, f3a), its legend in the cell's legend slot."""
    y0 = rows.y(k)
    bot = H.legend(fig, QF.overlay_items(full_names=True), x_pt=x0 + 14, top_pt=rows.top(k), width_pt=w - 16, fontsize=LEG,
                   gap_pt=5.0, handle_pt=8.0)
    print(f"[c] legend bottom {bot:.1f} pt from the top ({(bot - rows.top(k)) / (LEG * 1.45):.1f} rows)")
    ax = H.axes_pt(fig, x0 + 40.0, y0 + 26.0, w - 46.0, rows.fig_h - bot - 6.0 - (y0 + 26.0))
    return QF.draw_overlay(ax)


def grid(r1=168.0, r2=170.0):
    rows = Rows2(r1, r2)
    fig = plt.figure(figsize=(FULL / 72, rows.fig_h / 72))
    made = {"a": R2.strips(fig, rows, 1, 0.0, CW, plot_x=R2.FL - 2.0),
            "b": R2.compute(fig, rows, 1, CW, CW, plot_x=56.0),
            "c": frontier(fig, rows, 1, 2 * CW, CW),
            "d": R2.judges(fig, rows, 2, 0.0, CW),
            "e": R2.rules(fig, rows, 2, CW, CW),
            "f": R2.serving(fig, rows, 2, 2 * CW, CW)[0]}
    for ax in fig.axes:
        if ax.get_label() == "_house_overlay":
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    fig.canvas.draw()
    for pid, (col, k) in dict(a=(0, 1), b=(1, 1), c=(2, 1), d=(0, 2), e=(1, 2), f=(2, 2)).items():
        H.letter(fig, col * CW + 1.0, rows.top(k), pid)
    R.require_matplotlib_panel_alignment(
        fig, axes=list(made.values()), panel_ids=list(made), row_groups=[["d", "e"]], column_groups=[],
        exemptions=[], json_out=str(HERE / f"{NAME}.alignment.json"), tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        strict=False)
    H.save(fig, str(HERE / NAME), width="full", strict=STRICT)
    plt.close(fig)
    print("wrote", HERE / f"{NAME}.pdf", f"({rows.fig_h:.0f} pt = {rows.fig_h / 72:.2f} in tall)")


def pages(entries, out):
    """Each version at the top of a paper page with a three-line caption and grey two-column text under it (the page
    mock of T1-main-table/round6)."""
    from PIL import Image, ImageDraw, ImageFont
    dpi = 64
    pw, ph, lm, tm = int(8.5 * dpi), int(11 * dpi), int(0.8125 * dpi), int(1.0 * dpi)
    tw, th = int(6.875 * dpi), int(8.875 * dpi)
    font = ImageFont.truetype("DejaVuSans.ttf", 15)
    out_im = Image.new("RGB", (len(entries) * (pw + 16) + 16, ph + 46), "white")
    d0 = ImageDraw.Draw(out_im)
    for i, (n, png) in enumerate(entries):
        page = Image.new("RGB", (pw, ph), "white")
        d = ImageDraw.Draw(page)
        im = Image.open(png).convert("RGB")
        h_in = im.height / im.width * 6.875
        im = im.resize((tw, int(round(h_in * dpi))), Image.LANCZOS)
        page.paste(im, (lm, tm))
        y = tm + im.height + int(0.12 * dpi)
        for k in range(3):
            d.rectangle([lm, y, lm + (tw if k < 2 else int(tw * 0.6)), y + 4], fill="#9A9A9A")
            y += int(0.16 * dpi)
        y += int(0.18 * dpi)
        cw, gap = int(3.25 * dpi), int(0.375 * dpi)
        while y < tm + th - 4:
            for c in range(2):
                x0 = lm + c * (cw + gap)
                d.rectangle([x0, y, x0 + cw, y + 3], fill="#D0D0D0")
            y += int(0.152 * dpi)
        d.rectangle([0, 0, pw - 1, ph - 1], outline="#BBBBBB")
        x = 16 + i * (pw + 16)
        out_im.paste(page, (x, 40))
        d0.text((x, 10), f"{n}  {h_in:.2f} in", fill="black", font=font)
    out_im.save(out)


if __name__ == "__main__":
    grid()
    pages([("round 1 (3 x 3, now in draft)", str(HERE.parent / "round1/ex3_grid.png")),
           ("round 3 (2 x 3)", str(HERE / f"{NAME}.png"))], HERE / "_pages.png")
