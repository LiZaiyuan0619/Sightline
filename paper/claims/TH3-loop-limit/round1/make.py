"""Proposition 3 (loop limit) evidence figure, round 1 (main line). Single-column candidates.

Claim (one sentence): inside a loop a long-window decoder already writes many tokens per round, so stopping the loop
saves it fewer rounds than it saves a short window or the original model -- each cell's gain follows from the loop
share it removes, and our lead survives in every cell.

Every candidate reads the SAME data (th3data.py): table 2's SArena full test sets, 4 blocks x icon / illustration,
window 16 (the strongest published decoder, a tree) and window 32 (ours), lossless vs loop control, same job per pair.
Encoding as the table-1 figure: filled triangle = SArena-Icon, hollow = SArena-Illustration; window 16 in the strongest
baseline's teal, ours in our blue, the original model grey.
Run:  python3 paper/claims/TH3-loop-limit/round1/make.py [names...]   (STRICT=0 to draft)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import th3data as T  # noqa: E402

T.check()
H.apply()
SZ, C, INK, MUTED, M = H.SZ, H.C, H.INK, H.MUTED, H.METHOD
STRICT = os.environ.get("STRICT", "1") != "0"
W_PT = H.COL_W * 72
LEG_FS = 7.8
COL = {"DDTree": M["DDTree"], "Ours": C["ours"], "AR": C["ar"]}
NAME = {"AR": "original model", "DDTree": "window 16", "Ours": "window 32 (ours)"}
MARK = {"icon": dict(marker="^", fill=True), "illustration": dict(marker="^", fill=False)}


def _fig(h_pt):
    return plt.figure(figsize=(W_PT / 72, h_pt / 72))


def _times(v, _p=None):
    return "×1" if abs(v - 1) < 1e-9 else f"×{v:g}"


def _pt(ax, x, y, meth, sub, ms=6.0, z=5):
    m = MARK[sub]
    ax.plot([x], [y], ls="none", marker=m["marker"], ms=ms, mew=1.2, mec=COL[meth],
            mfc=COL[meth] if m["fill"] else "white", zorder=z)


def _legend_items(with_ar=True, kind="line"):
    items = []
    if with_ar:
        items.append(H.item("original model", COL["AR"], kind="line", lw=1.5, ls=(0, (4, 2))))
    items += [H.item("window 16", COL["DDTree"], kind=kind, lw=1.8),
              H.item("window 32 (ours)", COL["Ours"], kind=kind, lw=2.0, bold=True, text_color=C["ours"]),
              H.item("SArena-Icon", INK, kind="marker", marker="^", ms=5.0),
              H.item("SArena-Illus.", INK, kind="marker", marker="^", hollow=True, ms=5.0)]
    return items


# =============================================================================================== candidates
Y1A = 2.0


def r1a_curves():
    """The law and the cells: speed-up from stopping the loop against the share of tokens it removes. One curve per
    decoder, 1 + phi / (r (1 - phi)) with its own r = tokens per round inside loops / outside (the original model
    writes one token a round everywhere, r = 1); the 16 measured cells sit on their decoder's curve."""
    fig = _fig(196)
    bot = H.legend(fig, _legend_items(), x_pt=44, top_pt=2, width_pt=W_PT - 46, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, 30, W_PT - 44 - 8, 196 - 30 - bot - 4)
    for meth, r, ls, lw in (("AR", 1.0, (0, (4, 2)), 1.5), ("DDTree", T.R_MED["DDTree"], "-", 1.8),
                            ("Ours", T.R_MED["Ours"], "-", 2.2)):
        phi = np.linspace(0, min(0.56, (Y1A - 1) * r / (1 + (Y1A - 1) * r)), 200)
        ax.plot(phi, T.speedup(phi, r), color=COL[meth], lw=lw, ls=ls, zorder=2,
                path_effects=[pe.withStroke(linewidth=lw + 1.8, foreground="white")])
    P = {(x["block"], x["subset"], x["method"]): x for x in T.ROWS}
    for (blk, sub, meth), d in P.items():                       # same cell, same loops removed: window 16 -> 32
        if meth != "DDTree":
            continue
        o = P[(blk, sub, "Ours")]
        ax.plot([d["phi"], o["phi"]], [d["measured"], o["measured"]], color="#AEB6C1", lw=0.8, zorder=3)
    for x in T.ROWS:
        _pt(ax, x["phi"], x["measured"], x["method"], x["subset"])
    ax.set_xlim(0, 0.56)
    ax.set_ylim(0.98, Y1A)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:.0f}%"))
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: {1.0: "×1", 2.0: "×2"}.get(v, f"×{v:g}")))
    ax.set_xlabel("output tokens the loop control removes", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("speed-up from stopping loops", fontsize=SZ["label"], labelpad=2)
    return fig


def r1b_pred_meas():
    """Prediction against measurement, cell by cell: the run-length accounting predicts each cell's speed-up from the
    loop share it removes; the diagonal is a perfect prediction, the band +-10%."""
    fig = _fig(208)
    items = [H.item("window 16", COL["DDTree"], kind="marker", marker="o", ms=5.0),
             H.item("window 32 (ours)", COL["Ours"], kind="marker", marker="o", ms=5.0, bold=True, text_color=C["ours"]),
             H.item("SArena-Icon", INK, kind="marker", marker="^", ms=5.0),
             H.item("SArena-Illus.", INK, kind="marker", marker="^", hollow=True, ms=5.0)]
    bot = H.legend(fig, items, x_pt=44, top_pt=2, width_pt=W_PT - 46, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, 30, W_PT - 44 - 16, 208 - 30 - bot - 4)
    lo, hi = 1.0, 1.75
    g = np.linspace(lo, hi, 50)
    band = ax.fill_between(g, g / 1.1, g * 1.1, color="#EEF1F5", lw=0, zorder=0)
    band.set_gid("bg")
    ln, = ax.plot([lo, hi], [lo, hi], color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref")
    for x in T.ROWS:
        _pt(ax, x["predicted"], x["measured"], x["method"], x["subset"])
    err = sorted(abs(x["predicted"] / x["measured"] - 1) for x in T.ROWS)
    ax.text(1.73, 1.02, f"median error {100 * err[len(err) // 2]:.0f}%\nband: ±10%", ha="right", va="bottom",
            fontsize=SZ["tick"] - 0.6, color=MUTED, linespacing=1.15)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(FuncFormatter(_times))
    ax.set_xticks([1.0, 1.25, 1.5, 1.75])
    ax.set_yticks([1.0, 1.25, 1.5, 1.75])
    ax.set_xlabel("predicted speed-up from stopping loops", fontsize=SZ["label"] - 0.3, labelpad=1.5)
    ax.set_ylabel("measured speed-up", fontsize=SZ["label"] - 0.3, labelpad=2)
    return fig


def r1c_lead():
    """Our lead over window 16, per cell, before and after both decoders stop loops: it shrinks, because the long
    window had already compressed the loops, and stays above x1 in every cell."""
    fig = _fig(186)
    items = [H.item("SArena-Icon", C["ours"], kind="marker", marker="^", ms=5.0),
             H.item("SArena-Illus.", C["ours"], kind="marker", marker="^", hollow=True, ms=5.0)]
    bot = H.legend(fig, items, x_pt=40, top_pt=2, width_pt=W_PT - 42, fontsize=LEG_FS, gap_pt=10)
    ax = H.axes_pt(fig, 40, 30, W_PT - 40 - 60, 186 - 30 - bot - 4)
    ln = ax.axhline(1.0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref")
    ax.text(1.06, 1.006, "tie", ha="left", va="bottom", fontsize=SZ["tick"] - 0.8, color=MUTED)
    for l in T.LEADS:
        y = [l["lossless"], l["loop_control"]]
        ax.plot([0, 1], y, color=C["ours"], lw=1.0, alpha=0.55, zorder=2)
        for xi, yi in zip((0, 1), y):
            _pt(ax, xi, yi, "Ours", l["subset"], ms=5.4, z=4)
    m0 = float(np.median([l["lossless"] for l in T.LEADS]))
    m1 = float(np.median([l["loop_control"] for l in T.LEADS]))
    ax.plot([0, 1], [m0, m1], color=C["ours"], lw=2.4, zorder=3,
            path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])
    mp = float(np.median([l["predicted"] for l in T.LEADS]))
    tk, = ax.plot([0.9, 1.1], [mp, mp], color=INK, lw=1.4, zorder=6)
    ax.text(1.08, m1 + 0.004, f"median ×{m1:.2f}", ha="left", va="bottom", fontsize=SZ["tick"] - 0.4,
            color=C["ours"], fontweight="bold")
    ax.text(1.14, mp - 0.006, f"law: ×{mp:.2f}", ha="left", va="top", fontsize=SZ["tick"] - 0.6, color=INK)
    ax.text(-0.08, m0, f"×{m0:.2f}", ha="right", va="center", fontsize=SZ["tick"] - 0.4, color=C["ours"],
            fontweight="bold")
    ax.set_xlim(-0.35, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["lossless", "both stop loops"])
    ax.set_ylim(0.97, 1.45)
    ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "×1" if abs(v - 1) < 1e-9 else f"×{v:.1f}"))
    ax.set_ylabel("our lead over window 16", fontsize=SZ["label"], labelpad=2)
    return fig


def r1d_pairs():
    """Cell by cell, the speed-up from stopping loops for window 16 and window 32 (bars, measured) with the
    prediction (black tick): the same loops removed buy the long window less, in every cell."""
    fig = _fig(186)
    items = [H.item("window 16", COL["DDTree"]), H.item("window 32 (ours)", COL["Ours"], bold=True, text_color=C["ours"]),
             H.item("predicted", INK, kind="line", lw=1.4)]
    bot = H.legend(fig, items, x_pt=44, top_pt=2, width_pt=W_PT - 46, fontsize=LEG_FS, gap_pt=9)
    ax = H.axes_pt(fig, 44, 44, W_PT - 44 - 4, 186 - 44 - bot - 4)
    order = sorted({(x["block"], x["subset"]) for x in T.ROWS}, key=lambda t: (t[0], t[1]))
    xs, labs = [], []
    for i, (blk, sub) in enumerate(order):
        for dx, meth in ((-0.19, "DDTree"), (0.19, "Ours")):
            x = [r for r in T.ROWS if r["block"] == blk and r["subset"] == sub and r["method"] == meth][0]
            ax.bar(i + dx, x["measured"] - 1, bottom=1, width=0.36, color=COL[meth], lw=0, zorder=2)
            ln, = ax.plot([i + dx - 0.16, i + dx + 0.16], [x["predicted"]] * 2, color=INK, lw=1.4, zorder=3)
        xs.append(i)
        labs.append("icon" if sub == "icon" else "illus.")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=SZ["tick"] - 1.0)
    blocks = [b for b, _ in order[::2]]
    for j, blk in enumerate(blocks):
        a, b = 2 * j, 2 * j + 1
        ax.plot([a - 0.4, b + 0.4], [-0.115, -0.115], transform=ax.get_xaxis_transform(), clip_on=False,
                color=H.GROUP_LINE, lw=0.8)
        ax.text((a + b) / 2, -0.135, T.short(blk).replace(" ", "\n"), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=SZ["tick"] - 1.0, linespacing=1.05)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(1.0, 1.75)
    ax.set_yticks([1.0, 1.25, 1.5, 1.75])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: "×1" if abs(v - 1) < 1e-9 else f"×{v:g}"))
    ax.set_ylabel("speed-up from stopping loops", fontsize=SZ["label"] - 0.3, labelpad=2)
    return fig


def r1e_collapse():
    """One law for every decoder: stopping loops saves exactly the verify rounds spent inside them, so the speed-up is
    1 / (1 - psi) with psi = the share of rounds inside loops. Measured speed-ups of all 16 cells against their own psi
    (from the tokens removed and the decoder's r); a thin line joins the two decoders of one cell, which removed the
    same loops: the long window had spent fewer rounds on them, so it slides down the same curve."""
    fig = _fig(196)
    items = [H.item("window 16", COL["DDTree"], kind="marker", marker="o", ms=5.0),
             H.item("window 32 (ours)", COL["Ours"], kind="marker", marker="o", ms=5.0, bold=True, text_color=C["ours"]),
             H.item("SArena-Icon", INK, kind="marker", marker="^", ms=5.0),
             H.item("SArena-Illus.", INK, kind="marker", marker="^", hollow=True, ms=5.0)]
    bot = H.legend(fig, items, x_pt=44, top_pt=2, width_pt=W_PT - 46, fontsize=LEG_FS, gap_pt=7)
    ax = H.axes_pt(fig, 44, 30, W_PT - 44 - 8, 196 - 30 - bot - 4)
    g = np.linspace(0, 0.45, 200)
    ln, = ax.plot(g, 1 / (1 - g), color=INK, lw=1.3, zorder=2)
    ln.set_gid("ref")
    ax.text(0.445, 1 / (1 - 0.445) + 0.035, "1 / (1 − ψ)", ha="right", va="bottom", fontsize=SZ["tick"] - 0.4,
            color=INK)
    P = {(x["block"], x["subset"], x["method"]): x for x in T.ROWS}
    for (blk, sub, meth), d in P.items():
        if meth != "DDTree":
            continue
        o = P[(blk, sub, "Ours")]
        ax.plot([T.psi(d["phi"], d["r"]), T.psi(o["phi"], o["r"])], [d["measured"], o["measured"]],
                color="#AEB6C1", lw=0.8, zorder=3)
    for x in T.ROWS:
        _pt(ax, T.psi(x["phi"], x["r"]), x["measured"], x["method"], x["subset"])
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0.98, 1.85)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:.0f}%"))
    ax.set_yticks([1.0, 1.25, 1.5, 1.75])
    ax.yaxis.set_major_formatter(FuncFormatter(_times))
    ax.set_xlabel("verify rounds spent inside loops, ψ", fontsize=SZ["label"], labelpad=1.5)
    ax.set_ylabel("speed-up from stopping loops", fontsize=SZ["label"], labelpad=2)
    return fig


CANDIDATES = {"r1a_curves": r1a_curves, "r1b_pred_meas": r1b_pred_meas, "r1c_lead": r1c_lead, "r1d_pairs": r1d_pairs,
              "r1e_collapse": r1e_collapse}


def main(names):
    outs = []
    for n in names or list(CANDIDATES):
        fig = CANDIDATES[n]()
        H.save(fig, str(HERE / n), width="col", strict=STRICT)
        plt.close(fig)
        outs.append(str(HERE / f"{n}.png"))
    if not names:
        H.sheet(outs, str(HERE / "_sheet.png"), cols=2,
                title="TH3 loop limit, round 1: one dataset (SArena full sets, 16 cells, lossless vs loop control)")


if __name__ == "__main__":
    main(sys.argv[1:])
