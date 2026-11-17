"""Proposition 1 figure, round 4 (main line): p3a_joint + the two window lines.

User, prefers p3a_joint over p3c_pair if it does not lose information. The one thing p3a lacked against p3c is
the window: p3c's panel a says "93% of the tree's tokens sit in runs of 32 tokens or more" and its dashed lines tie
both panels to 16 and 32. Here the two windows are drawn through the scatter and the top marginal (named once, in the
free strip above the x2 line), so the reader sees the tree's prompts sitting far to the right of both; the 93% figure
moves to the caption. The top marginal drops its two median dashes: the grey one sat 1 pt from the window-16 line and
read as a double line, and the white diamonds already mark the median prompts (the right marginal keeps its medians,
they carry x1.34 and x1.81). Everything else is round 3's p3a (same data: block-32 head, 277 prompts, loop-free, greedy).
Run:  python3 paper/claims/TH1-run-length/round4/make.py   (STRICT=0 to draft)
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
import make as R3  # noqa: E402  (round 3, imported; its files are not touched)

H, SZ, MUTED = R3.H, R3.SZ, R3.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"


def p4a_joint():
    fig = R3.p3a_joint()
    ax, axt, axr = [a for a in fig.axes if a.get_label() != "_house_overlay"][:3]
    meds = {R3.MED[k][0] for k in (1, 16)}
    for ln in list(axt.lines):                       # the two vertical median dashes of the top marginal
        xd = ln.get_xdata()
        if len(xd) == 2 and xd[0] == xd[1] and xd[0] in meds:
            ln.remove()
    assert len(axt.lines) == 2, len(axt.lines)       # what is left: the two density outlines
    for x in (16, 32):
        for a in (ax, axt):
            ln = a.axvline(x, color=H.REF, lw=0.75, ls=(0, (3, 2)), zorder=0.5)
            ln.set_gid("ref")
    # the x2 note moves under its line, left of window 16 (the window lines now cross the top-left corner)
    for t in list(ax.texts):
        if t.get_text().startswith("the most a doubling"):
            t.remove()
    ax.text(1.15, 1.975, "the most a\ndoubling gives", ha="left", va="top", fontsize=SZ["tick"] - 0.9,
            color=MUTED, linespacing=1.15)
    # window names in the free strip above the x2 line: "window" left of 16, each value right of its line
    ax.text(16 / 1.35, 2.03, "window", ha="right", va="bottom", fontsize=7.2, color=MUTED)
    for x in (16, 32):
        ax.text(x * 1.08, 2.03, f"{x}", ha="left", va="bottom", fontsize=7.2, color=MUTED)
    return fig


if __name__ == "__main__":
    fig = p4a_joint()
    H.save(fig, str(HERE / "p4a_joint"), width="col", strict=STRICT)
    plt.close(fig)
