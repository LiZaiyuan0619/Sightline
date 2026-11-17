"""ACL-sized matplotlib helpers, taken from the DiscoRD and PACT final figures.

Generic versions of what the two papers' figure scripts did by hand: the ACL
column widths, Times type, the type sizes the user settled on (2026-10-02,
made the default on 2026-10-06), panel letters, saving at print size with a
width check, an assertion that no point falls outside hand-set limits, and the
two-row x axis with group lines. Everything here is a default, not a rule
(guide section 0.2): override any size or colour where a figure needs it.

    import sys; sys.path.insert(0, "<...>/docs/paper-figures/tools")
    import figstyle as fs
    fs.apply_style()
    fig = plt.figure(figsize=(fs.COL_W_IN, 2.4))
    ...
    fs.letter_at(fig, 0, 0, "a")
    fs.save(fig, "out/fig_x", width="col")
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

# --- page geometry (measured from acl.sty) ---------------------------------
COL_W_PT = 219.086          # \columnwidth, single column
FULL_W_PT = 455.244         # \textwidth, double column
COL_W_IN = COL_W_PT / 72.27
FULL_W_IN = FULL_W_PT / 72.27

# --- type sizes, in points, as printed ---------------------------------------
SIZE = {
    "tick": 7.0,      # tick labels, method names, group names
    "label": 8.0,     # axis titles
    "note": 7.5,      # legends, notes, direct labels; enlarge up to 9 where there is room
    "note_max": 9.0,
    "letter": 12.0,   # bold panel letters, the same in every figure of a paper
    "min": 7.0,       # nothing smaller
}

INK = "#272727"
REF = "#767676"       # reference, zero and diagonal lines
GROUP_LINE = "#AEB5B5"


def apply_style(tick: float = SIZE["tick"], label: float = SIZE["label"],
                linewidth: float = 0.7) -> None:
    """Times (the ACL body font), STIX maths, editable text in PDF and SVG,
    no top or right spines, frameless legends."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman",
                       "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": tick,
        "axes.labelsize": label,
        "axes.titlesize": label,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "legend.fontsize": SIZE["note"],
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": linewidth,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": linewidth,
        "ytick.major.width": linewidth,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "lines.linewidth": 1.2,
        "savefig.dpi": 600,
    })


def letter_at(fig, x_pt: float, top_pt: float, s: str,
              fs: float = SIZE["letter"]) -> None:
    """Panel letter at the panel's left edge, its top level with the top of the
    plot (user, 2026-09-26): it doubles as the divider between neighbouring
    panels and never takes a strip of its own above them. Points are measured
    from the figure's left and top edges."""
    w_pt, h_pt = fig.get_figwidth() * 72, fig.get_figheight() * 72
    fig.text(x_pt / w_pt, 1 - top_pt / h_pt, s, fontsize=fs,
             fontweight="bold", ha="left", va="top", color=INK)


def assert_inside(ax, xs, ys) -> None:
    """Call after setting limits by hand: matplotlib silently drops points
    outside them, and the picture does not show that anything is missing
    (PACT 09-24, 09-30)."""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    assert (xs >= x0).all() and (xs <= x1).all(), ("x outside limits", xs.min(), xs.max(), x0, x1)
    assert (ys >= y0).all() and (ys <= y1).all(), ("y outside limits", ys.min(), ys.max(), y0, y1)


def two_row_xaxis(ax, positions, tick_labels, groups, line_down_pt: float = 13.0,
                  name_down_pt: float = 14.0, pad: float = 0.38,
                  fs_group: float | None = None) -> None:
    """Two-row x axis (user, 09-23): row 1 the real tick values, row 2 the group
    names, each under a thin line spanning its group's ticks. ``groups`` is a
    list of ``(name, [positions])``. The line and the names move down by a fixed
    number of points, so they stay put when the plot gets shorter."""
    fs_group = fs_group or SIZE["tick"] + 0.5
    ax.set_xticks(positions, tick_labels)
    ax.tick_params(axis="x", length=0, pad=1.5)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    down = ScaledTranslation(0, -line_down_pt / 72, ax.figure.dpi_scale_trans)
    for name, gpos in groups:
        lo, hi = min(gpos) - pad, max(gpos) + pad
        ax.plot([lo, hi], [0, 0], transform=tr + down, color=GROUP_LINE,
                lw=0.65, clip_on=False)
        ax.annotate(name, xy=((lo + hi) / 2, 0), xycoords=tr,
                    xytext=(0, -name_down_pt), textcoords="offset points",
                    ha="center", va="top", fontsize=fs_group, color=INK,
                    annotation_clip=False)


_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]")
# An optional external figure-audit toolkit (glyph sizes, label collisions). Point
# $FIGURE_AUDIT_DIR at it to have its checks run after every save; without it the
# save path is unchanged and nothing is skipped in the figure itself.
_QA = Path(os.environ.get("FIGURE_AUDIT_DIR", "/nonexistent"))


def save(fig, out_base, width="col", formats=("pdf", "png"), dpi: int = 600,
         audit: bool = True) -> list[str]:
    """Save at print size and check the PDF width, so LaTeX inserts it at scale
    1 and the type sizes hold. Do not pass bbox_inches='tight': trimming changes
    the width. ``width`` is "col", "full" or a width in inches. If $FIGURE_AUDIT_DIR
    points at the optional audit toolkit, its glyph-size and collision checks run on
    the PDF; their findings print and never raise (some are known false alarms,
    such as numbers on one line being read as one string)."""
    target = {"col": COL_W_IN, "full": FULL_W_IN}.get(width, width)
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in formats:
        p = f"{base}.{ext}"
        fig.savefig(p, dpi=dpi, facecolor="white")
        saved.append(p)
    pdf = f"{base}.pdf"
    if os.path.exists(pdf):
        m = _MEDIABOX.search(Path(pdf).read_bytes())
        if m:
            w = (float(m.group(3)) - float(m.group(1))) / 72
            ok = abs(w - target) <= 0.02
            print(f"{base.name}: width {w:.3f} in, target {target:.3f} in "
                  f"({'OK' if ok else 'LaTeX will rescale; fix figsize'})")
        if audit and _QA.exists():
            for script, args in [("audit_pdf_text.py", ["--min-pt", str(SIZE["min"] - 0.1)]),
                                 ("audit_figure_collisions.py", [])]:
                r = subprocess.run([sys.executable, str(_QA / script), pdf, *args],
                                   capture_output=True, text=True)
                lines = r.stdout.splitlines()
                print(f"  {script}:", next((l for l in lines if l.startswith("verdict:")), "?"))
                for l in [l for l in lines if l.startswith(("[FAIL]", "[WARN]"))][:8]:
                    print("    ", l)
    return saved
