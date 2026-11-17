"""Design-choice row, round 1: stand-alone candidates for D1 / D2 / D3 at the size of one cell of
the merged double-column row, then the assembled rows. See README.md.

Run:  python3 paper/claims/D-group/round1/make.py [names...]   (no names = everything; STRICT=0 to draft)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dpanels as P  # noqa: E402

H = P.H
STRICT = os.environ.get("STRICT", "1") != "0"

CANDIDATES = {
    "d1_interaction": P.d1_interaction, "d1_mutual": P.d1_mutual, "d1_plane": P.d1_plane,
    "d1_waterfall": P.d1_waterfall,
    "d2_gap": P.d2_gap, "d2_pivot": P.d2_pivot, "d2_doublings": P.d2_doublings, "d2_stack": P.d2_stack,
    "d3_return": P.d3_return, "d3_yield": P.d3_yield, "d3_heat": P.d3_heat,
}


def solo(name):
    fig = plt.figure(figsize=(P.BOX_W / 72, P.BOX_H / 72))
    CANDIDATES[name](fig, (0, 0, P.BOX_W, P.BOX_H))
    P.finish(fig)
    H.save(fig, str(HERE / name), width=P.BOX_W / 72, strict=STRICT)
    plt.close(fig)




# ---------------------------------------------------------------------------------------------- the rows
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402   # stand-in in paper/_house

ROWS = {
    "row_A": ["d1_interaction", "d2_gap", "d3_return"],      # recommended: in every cell the shaded area is the claim
    "row_B": ["d1_plane", "d2_pivot", "d3_heat"],            # more chart families, harder to read
    "row_C": ["d1_waterfall", "d2_stack", "d3_return"],      # closest to the round-1 / round-2 picks
}


def row(name):
    parts = ROWS[name]
    W = 3 * P.BOX_W
    fig = plt.figure(figsize=(W / 72, P.BOX_H / 72))
    mains, exempt = [], []
    for i, n in enumerate(parts):
        axs = CANDIDATES[n]((fig), (i * P.BOX_W, 0, P.BOX_W, P.BOX_H))
        mains.append(axs[-1])
        if len(axs) > 1:
            exempt.append("abc"[i])
    P.finish(fig)
    # letters: left edge level with the panel's leftmost y title (guide SS5.2), top level with the axes top
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for i, n in enumerate(parts):
        x0 = i * P.BOX_W
        lefts = []
        for t in fig.texts + [a.yaxis.label for a in fig.axes]:
            if not t.get_text().strip() or abs(t.get_rotation() % 180 - 90) > 1:
                continue
            bb = t.get_window_extent(rend)
            xl = bb.x0 * 72 / fig.dpi
            if x0 <= xl < x0 + P.BOX_W:
                lefts.append(xl)
        H.letter(fig, min(lefts) - 0.3, P.BOX_H - P.AX_TOP, "abc"[i])
    ex = [{"panels": exempt, "checks": ["row"],
           "reason": "two stacked heatmaps fill this cell; together they span the row's top and bottom edges"}] if exempt else []
    require_matplotlib_panel_alignment(fig, axes=mains, panel_ids=["a", "b", "c"], row_groups=[["a", "b", "c"]],
                                       exemptions=ex, json_out=str(HERE / f"{name}.alignment.json"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    H.save(fig, str(HERE / name), width="full", strict=STRICT)
    plt.close(fig)


if __name__ == "__main__":
    names = sys.argv[1:] or (list(CANDIDATES) + list(ROWS))
    for n in names:
        row(n) if n in ROWS else solo(n)
