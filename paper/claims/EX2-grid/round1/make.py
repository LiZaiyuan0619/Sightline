"""The second experiments composite, round 1: the six cells of cells.py on one 2 x 3 grid (165 x 150 pt cells),
letters level with the top of the axes, at the leftmost text of their cell (the D-group row's placement: the cells'
legends sit in a strip above the axes), nature-figure alignment gate, house checks, 6.875 in; then the figure at the top of a paper page (page.png).

Run:  python3 paper/claims/EX2-grid/round1/make.py   (STRICT=0 to draft past the geometry check)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as K  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402   # stand-in in paper/_house

H, P = K.H, K.P
STRICT = os.environ.get("STRICT", "1") != "0"
TEX = "pdflatex"
V0 = HERE.parents[3] / "draft/v0"
NAME = "ex2_grid"
TOP_PAD = 2.0
LAYOUT = [["a", "b", "c"], ["d", "e", "f"]]
LETTER_ROOM = 0.0           # legends stay where the signed-off cells put them

_orig_legend = P._legend


def _legend_shifted(fig, box, items, left=6.0, **kw):
    return _orig_legend(fig, box, items, left=left + LETTER_ROOM, **kw)


def grid():
    P._legend = _legend_shifted
    W, Hh = 3 * K.BOX_W, 2 * K.BOX_H + TOP_PAD
    fig = plt.figure(figsize=(W / 72, Hh / 72))
    boxes, axes = {}, {}
    for r, ids in enumerate(LAYOUT):
        y0 = K.BOX_H if r == 0 else 0.0
        for c, pid in enumerate(ids):
            boxes[pid] = (c * K.BOX_W, y0, K.BOX_W, K.BOX_H)
            axes[pid] = K.CELLS[pid](fig, boxes[pid])
    P.finish(fig)
    for ax in fig.axes:
        if ax.get_label() == "_house_overlay":
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for pid, (x0, y0, w, h) in boxes.items():
        lefts = []
        for ax in fig.axes:
            for t in [ax.yaxis.label] + [t for t in ax.get_yticklabels() if t.get_visible()]:
                if not t.get_text().strip():
                    continue
                bb = t.get_window_extent(rend)
                xl, yc = bb.x0 * 72 / fig.dpi, (bb.y0 + bb.y1) / 2 * 72 / fig.dpi
                if x0 - 1 <= xl < x0 + w and y0 <= yc <= y0 + h:
                    lefts.append(xl)
        # level with the axes top, at the cell's leftmost text: the D-group row's placement (its cells keep their
        # legends in the strip above the axes, where a row-top letter would sit on them)
        H.letter(fig, max(min(lefts) - 0.3, 0.5), Hh - (y0 + h) + (K.BOX_H - P.AX_TOP), pid)
    order = [p for row in LAYOUT for p in row]
    first = {p: (axes[p][0] if isinstance(axes[p], list) else axes[p]) for p in order}
    require_matplotlib_panel_alignment(
        fig, axes=[first[p] for p in order], panel_ids=order, row_groups=LAYOUT,
        column_groups=[["a", "d"], ["b", "e"], ["c", "f"]], json_out=str(HERE / f"{NAME}.alignment.json"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    H.save(fig, str(HERE / NAME), width="full", strict=STRICT)
    plt.close(fig)
    P._legend = _orig_legend


def page():
    b = Path("/tmp/sd6/ex2page")
    shutil.rmtree(b, ignore_errors=True)
    b.mkdir(parents=True)
    for n in ["preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"]:
        shutil.copy(V0 / n, b / n)
    shutil.copy(HERE / f"{NAME}.pdf", b / "fig.pdf")
    head = (V0 / "main.tex").read_text().split("\\begin{document}")[0]
    fill = "\\ph{" + ("Gray text stands in for the prose around the figure, so that the share of the page it takes "
                      "is visible. It has no content. " * 120) + "}"
    body = (head + "\\begin{document}\n\\begin{figure*}[t]\n\\centering\n\\includegraphics{fig.pdf}\n"
            "\\caption{\\textbf{Where the lead comes from, and the design.} Placeholder caption of about three lines "
            "for a composite of six panels. Placeholder caption of about three lines for a composite of six panels. "
            "Placeholder caption of about three lines for a composite of six panels.}\n\\end{figure*}\n" + fill +
            "\n\\end{document}\n")
    (b / "page.tex").write_text(body)
    env = {"PATH": TEX + ":/usr/bin:/bin"}
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "page"], cwd=b, env=env, capture_output=True)
    subprocess.run(["pdftoppm", "-r", "110", "-f", "2", "-l", "2", "-png", "-singlefile", str(b / "page.pdf"),
                    str(HERE / "page")], check=True, capture_output=True)


if __name__ == "__main__":
    grid()
    page()
    print("wrote", HERE / f"{NAME}.pdf", HERE / "page.png")
