"""A PREVIEW of the composite with the cells recommended in round 1 (not aligned or finished yet): Figure 5 as the
left column, Figures 7, 8 / 9, 10 as the 2 x 2 on the right. Shows how the five read together and how much of a page
the composite takes (page_preview.png). The finished composite comes after the user picks the forms.

Run:  python3 paper/claims/EX1-grid/round1/preview.py [a b c d e]   (cell names; default = the recommendation)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as CL  # noqa: E402

H = CL.H
PICK = sys.argv[1:] or ["c5a_tall", "c7b_rows", "c8b_arrows", "c9b_ridges", "c10b_dots"]
FULL = 495.0
TEX = "pdflatex"
V0 = HERE.parents[3] / "draft/v0"


def preview():
    CL.LEG_DX = 12.0
    fig = plt.figure(figsize=(FULL / 72, CL.TALL_H / 72))
    boxes = {"a": (0.0, 0.0, CL.TALL_W, CL.TALL_H),
             "b": (CL.TALL_W, CL.CELL_H, CL.CELL_W, CL.CELL_H), "c": (CL.TALL_W + CL.CELL_W, CL.CELL_H, CL.CELL_W, CL.CELL_H),
             "d": (CL.TALL_W, 0.0, CL.CELL_W, CL.CELL_H), "e": (CL.TALL_W + CL.CELL_W, 0.0, CL.CELL_W, CL.CELL_H)}
    for pid, name in zip("abcde", PICK):
        CL.CANDIDATES[name][0](fig, boxes[pid])
    for pid, (x0, y0, w, h) in boxes.items():
        H.letter(fig, x0 + 1.0, CL.TALL_H - (y0 + h) + 0.5, pid)
    H.save(fig, str(HERE / "preview"), width=FULL / 72, strict=False)
    plt.close(fig)


def page():
    b = Path("/tmp/sd6/ex1page")
    shutil.rmtree(b, ignore_errors=True)
    b.mkdir(parents=True)
    for n in ["preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"]:
        shutil.copy(V0 / n, b / n)
    shutil.copy(HERE / "preview.pdf", b / "preview.pdf")
    head = (V0 / "main.tex").read_text().split("\\begin{document}")[0]
    fill = "\\ph{" + ("Gray text stands in for the prose around the figure, so that the share of the page it takes "
                      "is visible. It has no content. " * 120) + "}"
    body = (head + "\\begin{document}\n\\begin{figure*}[t]\n\\centering\n\\includegraphics{preview.pdf}\n"
            "\\caption{\\textbf{Fast and good, general, and why (preview).} Placeholder caption of about three lines for "
            "a composite of five panels. Placeholder caption of about three lines for a composite of five panels. "
            "Placeholder caption of about three lines for a composite of five panels.}\n\\end{figure*}\n" + fill +
            "\n\\end{document}\n")
    (b / "page.tex").write_text(body)
    env = {"PATH": TEX + ":/usr/bin:/bin"}
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "page"], cwd=b, env=env, capture_output=True)
    subprocess.run(["pdftoppm", "-r", "110", "-f", "2", "-l", "2", "-png", "-singlefile", str(b / "page.pdf"),
                    str(HERE / "page_preview")], check=True, capture_output=True)


if __name__ == "__main__":
    preview()
    page()
    print("wrote", HERE / "preview.png", HERE / "page_preview.png")
