"""Theory grid, round 1 (main line): the theory section's single 2 x 3 figure. Cells in cells.py; README.md says
what each cell is and what changed against the signed-off single figures.

Run:  python3 paper/claims/TH6-theory-grid/round1/make.py   (STRICT=0 to draft past the geometry check)
"""
import os
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

H = K.H
STRICT = os.environ.get("STRICT", "1") != "0"
LAYOUT = [["a", "b", "c"], ["d", "e", "f"]]


def grid(name="theory_grid"):
    fig = plt.figure(figsize=(K.FULL / 72, K.FIG_H / 72))
    mains = {}
    for r, ids in enumerate(LAYOUT):
        y0, h = (K.R2_H, K.R1_H) if r == 0 else (0.0, K.R2_H)
        for c, pid in enumerate(ids):
            mains[pid] = K.CELLS[pid](fig, (c * K.BOX_W, y0, K.BOX_W, h))
    K.finish(fig)
    # letters: top level with the top of their row, left edge at the cell's leftmost y title (guide 5.2)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for r, ids in enumerate(LAYOUT):
        row_top = K.TOP_PAD if r == 0 else K.TOP_PAD + K.R1_H
        for c, pid in enumerate(ids):
            x0 = c * K.BOX_W
            lefts = []
            for t in [a.yaxis.label for a in fig.axes]:
                if not t.get_text().strip():
                    continue
                bb = t.get_window_extent(rend)
                xl = bb.x0 * 72 / fig.dpi
                yc = (bb.y0 + bb.y1) / 2 * 72 / fig.dpi
                in_row = (K.R2_H <= yc) if r == 0 else (yc < K.R2_H)
                if x0 <= xl < x0 + K.BOX_W and in_row:
                    lefts.append(xl)
            H.letter(fig, min(lefts) - 0.3, row_top, pid)
    order = [p for row in LAYOUT for p in row]
    require_matplotlib_panel_alignment(
        fig, axes=[mains[p] for p in order], panel_ids=order,
        row_groups=[LAYOUT[0], LAYOUT[1]], column_groups=[["a", "d"], ["b", "e"], ["c", "f"]],
        json_out=str(HERE / f"{name}.alignment.json"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    H.save(fig, str(HERE / name), width="full", strict=STRICT)
    plt.close(fig)


if __name__ == "__main__":
    grid()
