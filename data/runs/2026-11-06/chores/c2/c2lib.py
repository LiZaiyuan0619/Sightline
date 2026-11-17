"""C2 (B line): the two pieces every migrated early figure shares. Style only -- forms and data untouched.

legend_rect(fig, rect, items)  the old figures place a legend row inside a rectangle given in figure fractions and
                               spread the entries over its width; this lays the same row out with house.legend
                               (handles and gaps in points, Ours bold in its colour), vertically centred in the old
                               rectangle, entries spread over the same width (gap capped like the old helper).
redirect(module, out_dir, rename) makes an imported finalized-round module save through house.save (strict: a
                               geometry FAIL raises) into the NEW round directory, under the new name; nothing is
                               written into the finalized directory.
"""
from __future__ import annotations

import sys
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402


def _natural_width(fig, items, fontsize, pad_pt=3.0):
    ov = H._overlay(fig)
    rend = fig.canvas.get_renderer()
    tot = 0.0
    for it in items:
        t = ov.text(0, 0, it["label"], fontsize=fontsize, fontweight="bold" if it["bold"] else "normal")
        tot += t.get_window_extent(rend).width * 72 / fig.dpi
        t.remove()
        tot += (11.0 if it["kind"] in ("line", "band") else 7.0) + pad_pt
    return tot


def legend_rect(fig, rect, items, spread=True, gap_cap_frac=None, fontsize=None, align="left", min_gap=6.0):
    """rect = [x, y, w, h] in figure fractions (the old legend axes). Returns the legend's bottom (pt from top)."""
    W, Ht = fig.get_figwidth() * 72, fig.get_figheight() * 72
    fsz = fontsize or H.SZ["legend"]
    row = fsz * 1.45
    x_pt, w_pt = rect[0] * W, rect[2] * W
    top_pt = (1 - rect[1] - rect[3] / 2) * Ht - row / 2
    gap = 9.0
    if spread and len(items) > 1:
        gap = (w_pt - 2.0 - _natural_width(fig, items, fsz)) / (len(items) - 1)
        if gap_cap_frac is not None:
            gap = min(gap, gap_cap_frac * w_pt)
        gap = max(gap, min_gap)
    return H.legend(fig, items, x_pt=x_pt, top_pt=top_pt, width_pt=w_pt, fontsize=fsz, gap_pt=gap, align=align)


def patch_items(pairs):
    """[(colour, label)] as the old legend_row took them -> house legend items (filled squares)."""
    return [H.item(lab, col) for col, lab in pairs]


class _FS:
    """figstyle with a different save; everything else is the real module (house keeps the real one)."""

    def __init__(self, real, save):
        self._real, self.save = real, save

    def __getattr__(self, k):
        return getattr(self._real, k)


def redirect(module, out_dir: Path, rename: dict | None = None, strict: bool = True):
    """Make `module`'s own `fs.save(...)` calls go through house.save into out_dir (rename: old -> new stems).
    Only that module's global `fs` is swapped for a proxy; figstyle itself (which house.save calls) is untouched."""
    rename = rename or {}
    real = H.fs

    def save(fig, out_base, width=None, **kw):
        stem = Path(out_base).name
        stem = rename.get(stem, stem)
        w = {3.25: "col", 6.875: "full"}.get(round(float(width), 3), width) if width is not None else "col"
        return H.save(fig, str(out_dir / stem), width=w, strict=strict, **kw)

    module.fs = _FS(real, save)
    return save
