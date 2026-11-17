"""M1 round 3 --  (B line): B8 + C2 in one pass, same two figures as the round-2 pick (m1e_bars + m1h_text).

B8 (X4): the InternSVG-8B text-to-SVG block now reads TABLE 1's heads (tableA/isvg-mix, the continued training);
round 2 read the heads before it (profiles_all.B["isvg"] -> isvg-own / isvg-x). Data: runs/2026-11-06/chores/b8/
(fix_isvg.py: unpatched recompute == the profile exactly; patched rows == table 1's; pooled tokens per round == table 1's).
  m1e_bars  InternSVG text bars: structure wrong 2.4% -> 2.3%, numbers 6.4% -> 5.8% (both still print as 2 / 6);
            the other four blocks are unchanged.
  m1h_text the example is re-picked by the rule (runs/2026-11-03/figdata_b1_m1_example.py, verbatim) from the
            table-1 head: gidx 75 (round 2: gidx 0, which no longer passes the rule on the new head).
C2 (style only): house.apply; the legend row by house.legend; the token-by-token source text marked gid "run"
(house.check: pieces of one string abut on purpose); every file through house.save (blocks on a geometry FAIL).
Run: python paper/claims/M1-token-classes/round3/make.py
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(RUNS / "2026-11-06/chores/c2"))
import c2lib as C  # noqa: E402

H = C.H
B8 = RUNS / "2026-11-06/chores/b8/figdata"
spec = importlib.util.spec_from_file_location("m1r2", HERE.parent / "round2/make.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
r1 = r2.r1
H.apply()

# B8: swap in the InternSVG text profile on table 1's heads (same dict object round 2's group_acc reads)
ISVG = r1.BLOCKS[2]
assert ISVG.startswith("InternSVG-8B") and "text" in ISVG, ISVG
_old = r1.PROF[ISVG]
r1.PROF[ISVG] = json.loads((B8 / "profile_isvg.json").read_text())
assert r2.PROF is r1.PROF


def legend_row(fig, rect, pairs):
    C.legend_rect(fig, rect, C.patch_items(pairs), gap_cap_frac=0.05)


r1.m2.legend_row = legend_row
C.redirect(r2, HERE)


def _layout(width_in, toks, s, start_of, stop, fsz):
    """Line-break the token pieces at this width (same rule as round 2); -> [(line, x0, w, piece, boundary, stop)]
    with x in axes fractions. Measured on a scratch figure, so the real one can be built at its final height."""
    f = plt.figure(figsize=(width_in, 2.0))
    ax = f.add_axes([0.015, 0.1, 0.97, 0.8])
    rend = f.canvas.get_renderer()
    aw = ax.get_window_extent(rend).width
    out, x, line = [], 0.0, 0
    for i, piece in enumerate(toks[:s]):
        boundary = i in start_of and i > 1
        if boundary:
            x += 0.016
        t = ax.text(x, 0.5, piece, fontsize=fsz, family="monospace")
        w = t.get_window_extent(rend).width / aw
        t.remove()
        if x + w > 0.998:
            x, line, piece = 0.0, line + 1, piece.lstrip()
            t = ax.text(x, 0.5, piece, fontsize=fsz, family="monospace")
            w = t.get_window_extent(rend).width / aw
            t.remove()
        out.append((line, x, w, piece, boundary, i in stop))
        x += w
    plt.close(f)
    return out


def m1h_text():
    """Round 2's m1h_text on the re-picked example; token pieces carry gid 'run'; round boundaries are filled bars
    (no stroke: the PDF audit merges a line of text into one string and calls any stroke inside it a crossing);
    the canvas height follows the number of text lines (house rule size from the content)."""
    ex = json.loads((B8 / "m1_example_isvg.json").read_text())
    toks, rounds, cap = ex["tokens"], ex["rounds"], ex["cap"]
    assert ex["gidx"] == 75 and rounds == [32, 5, 7, 6, 4], (ex["gidx"], rounds)
    start_of, stop = {}, set()
    s = 1                                          # token 0 comes from the prefill
    for j, e in enumerate(rounds):
        start_of[s] = j
        if e < cap:
            stop.add(s + e - 1)
        s += e
    fsz, pitch, top, bot, leg_h, gap = 7.4, 21.0, 8.5, 8.5, 17.5, 4.0        # points (round 2's spacing)
    lay = _layout(r2.CW, toks, s, start_of, stop, fsz)
    n_lines = lay[-1][0] + 1
    text_h = top + (n_lines - 1) * pitch + bot
    Ht = 2.0 + leg_h + gap + text_h + 2.0
    fig = plt.figure(figsize=(r2.CW, Ht / 72))
    Wp = r2.CW * 72
    ax = H.axes_pt(fig, 0.015 * Wp, 2.0 + leg_h + gap, 0.97 * Wp, text_h)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, text_h)
    aw_pt = 0.97 * Wp
    n_num, stops_txt = 0, []
    for line, x, w, piece, boundary, is_stop in lay:
        y = text_h - top - line * pitch
        if boundary:
            bw = 0.9 / aw_pt                          # a 0.9 pt bar, filled, no stroke
            ax.add_patch(Rectangle((x - 0.008 - bw / 2, y - 6.4), bw, 12.8, facecolor="#9A9A9A", linewidth=0))
        ax.text(x, y, piece, fontsize=fsz, family="monospace", ha="left", va="center", color="#2E2E2E", gid="run")
        if is_stop:
            n_num += bool(re.search(r"[0-9]", piece))
            stops_txt.append(piece)
            ax.add_patch(FancyBboxPatch((x - 0.002, y - 7.2), w + 0.004, 14.4, boxstyle="round,pad=0.004",
                                        facecolor="#F1C27D", edgecolor="none", linewidth=0, zorder=0, gid="bg"))
    print(f"m1h: {len(rounds)} rounds, {len(stop)} stopped by a wrong draft token, {n_num} of them on a number: "
          f"{stops_txt}; {n_lines} lines, canvas {Ht / 72:.2f} in (round 2: 1.62 in, 4 lines)")
    lg = H.axes_pt(fig, 0.015 * Wp, 2.0, 0.97 * Wp, leg_h)
    lg.axis("off")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    lg.add_patch(FancyBboxPatch((0.0, 0.22), 0.045, 0.56, boxstyle="round,pad=0.01", facecolor="#F1C27D",
                                edgecolor="none", linewidth=0))
    lg.text(0.06, 0.5, "round stopped here (draft wrong)", fontsize=H.SZ["note"], va="center")
    lg.add_patch(Rectangle((0.66 - 0.002, 0.15), 0.004, 0.70, facecolor="#9A9A9A", linewidth=0))
    lg.text(0.68, 0.5, "next round", fontsize=H.SZ["note"], va="center")
    H.save(fig, str(HERE / "m1h_text"), width="col")
    return n_num, len(stop)


def main():
    for b in r1.BLOCKS:
        g = r2.group_acc(b)
        print(f"{b:28s} structure {g['structure']:.4f}  numbers {g['numbers']:.4f}")
    new = r2.group_acc(ISVG)
    r1.PROF[ISVG] = _old
    old = r2.group_acc(ISVG)
    r1.PROF[ISVG] = json.loads((B8 / "profile_isvg.json").read_text())
    print(f"InternSVG text, draft wrong: structure {100 * (1 - old['structure']):.2f}% -> {100 * (1 - new['structure']):.2f}%,"
          f" numbers {100 * (1 - old['numbers']):.2f}% -> {100 * (1 - new['numbers']):.2f}%")
    r2.m1e_bars()
    m1h_text()
    print("wrote m1e_bars, m1h_text")


if __name__ == "__main__":
    main()
