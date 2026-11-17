"""The first experiments composite, round 2: assemble the five cells of cells2.py, put the panel letters at the top of
each row, level with the leftmost text of their cell (guide 5.2), run the nature-figure alignment gate and the house
checks, save at 6.875 in, and set the figure at the top of a paper page (page.png) to see its share of the page.

Run:  python3 paper/claims/EX1-grid/round2/make.py   (STRICT=0 to draft past the geometry check)
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
import cells2 as K  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402   # stand-in in paper/_house

H = K.H
STRICT = os.environ.get("STRICT", "1") != "0"
TEX = "pdflatex"
V0 = HERE.parents[3] / "draft/v0"
NAME = "ex1_grid"


def _left_text(fig, box, rend):
    """x (pt) of the leftmost y-axis title or y tick label inside a box."""
    x0, y0, w, h = box
    xs = []
    for ax in fig.axes:
        texts = [ax.yaxis.label] + [t for t in ax.get_yticklabels() if t.get_visible()]
        for t in texts:
            if not t.get_text().strip():
                continue
            bb = t.get_window_extent(rend)
            xl, yc = bb.x0 * 72 / fig.dpi, (bb.y0 + bb.y1) / 2 * 72 / fig.dpi
            if x0 - 1 <= xl < x0 + w and y0 <= yc <= y0 + h:
                xs.append(xl)
    return min(xs) if xs else x0 + 1.0


def grid():
    fig = plt.figure(figsize=(K.FULL / 72, K.FIG_H / 72))
    made = {pid: K.CELLS[pid](fig, K.BOX[pid]) for pid in "abcde"}
    for ax in fig.axes:                       # the house legend layer: no ticks of its own, ever
        if ax.get_label() == "_house_overlay":
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for pid in "abcde":
        x0, y0, w, h = K.BOX[pid]
        top = K.FIG_H - (y0 + h)
        x = _left_text(fig, K.BOX[pid], rend) if pid != "b" else x0 + 1.0
        H.letter(fig, max(x - 0.3, 0.5), top, pid)
    a1, a2 = made["a"]
    e1, e2 = made["e"]
    axes = [a1, a2, made["c"], made["d"], e1, e2]
    ids = ["a1", "a2", "c", "d", "e1", "e2"]
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids=ids, row_groups=[["d", "e1", "e2"]], column_groups=[["a1", "a2"]],
        exemptions=[{"panels": ["e1", "e2"], "checks": ["panel-width"],
                     "reason": "Figure 10 splits its cell into a greedy and an official-sampling panel; Figure 9's "
                               "single panel is wider by design"},
                    {"panels": ["d", "e1", "e2"], "checks": ["horizontal-gutter"],
                     "reason": "the gap between Figures 9 and 10 holds Figure 10's model names; the gap between "
                               "Figure 10's two panels is a narrow split of one cell"}],
        json_out=str(HERE / f"{NAME}.alignment.json"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    H.save(fig, str(HERE / NAME), width="full", strict=STRICT)
    plt.close(fig)


def page():
    b = Path("/tmp/sd6/ex1page2")
    shutil.rmtree(b, ignore_errors=True)
    b.mkdir(parents=True)
    for n in ["preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"]:
        shutil.copy(V0 / n, b / n)
    shutil.copy(HERE / f"{NAME}.pdf", b / "fig.pdf")
    head = (V0 / "main.tex").read_text().split("\\begin{document}")[0]
    fill = "\\ph{" + ("Gray text stands in for the prose around the figure, so that the share of the page it takes "
                      "is visible. It has no content. " * 120) + "}"
    body = (head + "\\begin{document}\n\\begin{figure*}[t]\n\\centering\n\\includegraphics{fig.pdf}\n"
            "\\caption{\\textbf{Fast and good, general, and why.} Placeholder caption of about three lines for a "
            "composite of five panels. Placeholder caption of about three lines for a composite of five panels. "
            "Placeholder caption of about three lines for a composite of five panels.}\n\\end{figure*}\n" + fill +
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
