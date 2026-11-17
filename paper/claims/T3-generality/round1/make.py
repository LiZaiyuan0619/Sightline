"""Table 3 as a figure -- two candidates to set beside the compact table (make_table.py).  night.

  t3a_ticks   table 1's tick strip (round 4, r4d): one column per model x subset, one tick per method, the light blue
              band = our lead over the best baseline; greedy on top, official sampling below; double column
  t3b_ratio   everything as a fraction of ours: one column per model x subset, a dot per baseline (filled greedy,
              hollow official sampling); ours is the line at 1; one row, double column

Run: python3 paper/claims/T3-generality/round1/make.py   (STRICT=0 for drafts)
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import t3data as T  # noqa: E402

H.apply()
SZ, C, M, INK, MUTED = H.SZ, H.C, H.METHOD, H.INK, H.MUTED
STRICT = os.environ.get("STRICT", "1") != "0"
W = H.FULL_W * 72
SUBLAB = {"icon": "icon", "illustration": "illus.", "real": "real", "synthetic": "synth."}
FAMLAB = {"SVGen-3B": "SVGen-3B", "SVGen-7B": "SVGen-7B", "HiVG-3B": "HiVG-3B", "OmniLottie-3B": "OmniLottie-3B",
          "StarVector-1B": "StarVector-1B", "StarVector-8B": "StarVector-8B"}
BASES = [m for m in T.METHODS if m != "Ours"]


def finish(fig):
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if "\n" in t.get_text() and getattr(t, "_linespacing", 1.2) < 1.12:
            t.set_linespacing(1.12)


def columns():
    xs, labs, groups = [], [], []
    x = 0.0
    for f in T.FAMILIES:
        g = []
        for s in T.SUBSETS[f]:
            xs.append((f, s, x))
            labs.append(SUBLAB[s])
            g.append(x)
            x += 1.0
        groups.append((FAMLAB[f], g))
        x += 0.55
    return xs, labs, groups


def t3a_ticks():
    cols, labs, groups = columns()
    items = [H.item(m, M[m]) for m in T.METHODS] + [H.item("our lead over the best baseline", C["ours_light"], kind="band")]
    leg = H.legend_height(items, H.FULL_W, x_pt=40, width_pt=W - 44, fontsize=SZ["legend"], gap_pt=10)
    ah, gap, bottom, lead_h = 92.0, 22.0, 34.0, 13.0
    Ht = leg + 4 + 2 * (ah + lead_h) + gap + bottom
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    H.legend(fig, items, x_pt=40, top_pt=0, width_pt=W - 44, fontsize=SZ["legend"], gap_pt=10)
    for k, mode in enumerate(T.MODES):
        y0 = bottom + (0 if k == 1 else ah + lead_h + gap)
        ax = H.axes_pt(fig, 40, y0, W - 40 - 4, ah)
        vals = []
        for f, s, x in cols:
            v = {m: T.spd(f, m, mode, s) for m in T.METHODS}
            vals += list(v.values())
            b = max(v[m] for m in BASES)
            if v["Ours"] > b:
                band = ax.add_patch(plt.Rectangle((x - 0.36, b), 0.72, v["Ours"] - b, facecolor=C["ours_light"],
                                                  edgecolor="none", zorder=1))
                band.set_gid("bg")
            for m in T.METHODS:
                ours = m == "Ours"
                ax.plot([x - 0.36, x + 0.36], [v[m], v[m]], color=M[m], lw=2.6 if ours else 1.6, zorder=3 if ours else 2,
                        solid_capstyle="butt")
        # per-family lead (min - max over its two subsets), over the axes
        for name, g in groups:
            f = [ff for ff in T.FAMILIES if FAMLAB[ff] == name][0]
            leads = [T.spd(f, "Ours", mode, s) / T.best_base(f, mode, s)[0] for s in T.SUBSETS[f]]
            txt = "×" + (f"{min(leads):.2f}–{max(leads):.2f}" if abs(max(leads) - min(leads)) >= 0.005 else f"{leads[0]:.2f}")
            ax.text(np.mean(g), 1.0 + 2 / ah, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=SZ["legend"] - 0.3, color=C["ours"])
        ax.axhline(1.0, color=H.REF, lw=0.8, ls=(0, (3, 2)), zorder=0).set_gid("ref")
        ax.set_xlim(-0.6, cols[-1][2] + 0.6)
        top = 17.0 if mode == "greedy" else 9.5
        ax.set_ylim(0, top)
        H.fs.assert_inside(ax, [c[2] for c in cols], vals)
        ax.set_ylabel("speed-up,\n" + ("greedy" if mode == "greedy" else "official sampling"), fontsize=SZ["label"],
                      labelpad=2)
        if k == 1:
            H.fs.two_row_xaxis(ax, [c[2] for c in cols], labs, groups, line_down_pt=11.0, name_down_pt=12.0, pad=0.42,
                               fs_group=SZ["tick"])
        else:
            ax.set_xticks([])
    finish(fig)
    H.save(fig, str(HERE / "t3a_ticks"), width="full", strict=STRICT)
    plt.close(fig)


def t3b_ratio():
    cols, labs, groups = columns()
    items = [H.item(m, M[m], kind="marker", ms=4.6) for m in BASES] + \
            [H.item("Ours = 1", C["ours"], kind="line", lw=1.8, bold=True, text_color=C["ours"]),
             H.item("greedy", INK, kind="marker", ms=4.2), H.item("official sampling", INK, kind="marker", hollow=True,
                                                                ms=4.2, mew=0.9)]
    leg = H.legend_height(items, H.FULL_W, x_pt=40, width_pt=W - 44, fontsize=SZ["legend"], gap_pt=9)
    ah, bottom = 104.0, 34.0
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    H.legend(fig, items, x_pt=40, top_pt=0, width_pt=W - 44, fontsize=SZ["legend"], gap_pt=9)
    ax = H.axes_pt(fig, 46, bottom, W - 46 - 4, ah)
    ax.axhline(1.0, color=C["ours"], lw=1.8, zorder=2)
    vals = []
    for f, s, x in cols:
        for k, mode in enumerate(T.MODES):
            o = T.spd(f, "Ours", mode, s)
            for m in BASES:
                r = T.spd(f, m, mode, s) / o
                vals.append(r)
                ax.plot([x + (-0.14 if k == 0 else 0.14)], [r], "o", ms=4.6 if k == 0 else 4.2,
                        mfc=M[m] if k == 0 else "white", mec=M[m], mew=0.5 if k == 0 else 1.0, zorder=4)
    ax.set_xlim(-0.6, cols[-1][2] + 0.6)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:g}")
    ax.set_ylabel("speed-up,\nshare of ours", fontsize=SZ["label"], labelpad=2)
    H.fs.two_row_xaxis(ax, [c[2] for c in cols], labs, groups, line_down_pt=11.0, name_down_pt=12.0, pad=0.42,
                       fs_group=SZ["tick"])
    H.fs.assert_inside(ax, [c[2] for c in cols], vals)
    finish(fig)
    H.save(fig, str(HERE / "t3b_ratio"), width="full", strict=STRICT)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ c radar
def t3c_radar():
    """Single column. Radius = speed-up as a share of ours, so ours is the unit circle and every baseline sits inside
    it; 24 spokes = 6 models x (2 subsets x greedy / official sampling), one sector per model. The light blue ring
    between DDTree (the runner-up in every cell) and our circle is our lead; under each model, our own speed-ups."""
    Wc = H.COL_W * 72
    items = [H.item(m, M[m], kind="marker", ms=4.0) for m in BASES] + \
            [H.item("Ours = 1", C["ours"], kind="line", lw=1.9, bold=True, text_color=C["ours"]),
             H.item("our lead", C["ours_light"], kind="band"),
             H.item("greedy", INK, kind="marker", ms=3.8), H.item("official sampling", INK, kind="marker", hollow=True,
                                                                ms=3.8, mew=0.9)]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=Wc - 5, fontsize=7.6, gap_pt=6)
    D, side, vpad = 150.0, None, 27.0
    Ht = leg + 2 + vpad + D + vpad
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=Wc - 5, fontsize=7.6, gap_pt=6)
    W_, Hf = fig.get_figwidth() * 72, fig.get_figheight() * 72
    ax = fig.add_axes([((Wc - D) / 2) / W_, vpad / Hf, D / W_, D / Hf], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    sector, step = 60.0, 12.0
    th, meta = [], []
    for i, f in enumerate(T.FAMILIES):
        spokes = [(mo, s) for mo in T.MODES for s in T.SUBSETS[f]]
        for j, (mo, s) in enumerate(spokes):
            th.append(np.deg2rad(i * sector + (j - 1.5) * step))
            meta.append((f, mo, s))
    th = np.array(th)
    R = {m: np.array([T.spd(f, m, mo, s) / T.spd(f, "Ours", mo, s) for f, mo, s in meta]) for m in BASES}
    for i in range(len(T.FAMILIES)):
        idx = np.arange(4 * i, 4 * i + 4)
        tt = np.linspace(th[idx[0]] - np.deg2rad(step / 2), th[idx[-1]] + np.deg2rad(step / 2), 80)
        base = np.interp(tt, th[idx], R["DDTree"][idx])
        ax.fill_between(tt, np.minimum(base, 1.0), 1.0, color=C["ours_light"], lw=0, zorder=1).set_gid("bg")
        ax.plot(tt, base, color=M["DDTree"], lw=1.1, zorder=3)
    for r in (0.25, 0.5, 0.75):
        ax.plot(np.linspace(0, 2 * np.pi, 361), [r] * 361, color="#E4E4E4", lw=0.6, zorder=0)[0].set_gid("ref")
    ax.plot(np.linspace(0, 2 * np.pi, 361), [1.0] * 361, color=C["ours"], lw=1.9, zorder=5)
    for m in BASES:
        for k in range(len(th)):
            hollow = meta[k][1] == "sample"
            ax.plot([th[k]], [R[m][k]], "o", ms=3.2 if not hollow else 3.0, mfc="white" if hollow else M[m], mec=M[m],
                    mew=0.4 if not hollow else 0.8, zorder=4)
    ax.set_ylim(0, 1.04)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    for r, lab, ang in ((0.5, "50%", 150), (0.75, "75%", 90)):
        ax.text(np.deg2rad(ang), r, lab, ha="center", va="center", fontsize=7.0, color="#9A9A9A",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.15), zorder=6)
    for i, f in enumerate(T.FAMILIES):
        idx = range(4 * i, 4 * i + 4)
        ours = [T.spd(meta[k][0], "Ours", meta[k][1], meta[k][2]) for k in idx]
        ang = i * sector
        ha = "center" if ang in (0, 180) else ("left" if ang < 180 else "right")
        rr = 1.13 if ang in (0, 180) else 1.07
        ax.text(np.deg2rad(ang), rr, f"{f}\n{min(ours):.1f}–{max(ours):.1f}×", ha=ha,
                va="bottom" if ang == 0 else ("top" if ang == 180 else "center"), fontsize=7.4, color=INK)
    finish(fig)
    H.save(fig, str(HERE / "t3c_radar"), width="col", strict=STRICT)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ d arrows
def t3d_arrows():
    """Single column. One row per model x subset; on a log speed-up axis, an arrow from DDTree (runner-up everywhere)
    to ours, greedy above official sampling; the other baselines are ticks on the same row."""
    Wc = H.COL_W * 72
    items = [H.item(m, M[m], kind="patch") for m in BASES] + \
            [H.item("Ours", C["ours"], kind="marker", ms=4.4), H.item("greedy", INK, kind="line", lw=1.3),
             H.item("official sampling", INK, kind="line", lw=1.1, ls=(0, (2.5, 1.5)))]
    leg = H.legend_height(items, H.COL_W, x_pt=3, width_pt=Wc - 5, fontsize=7.6, gap_pt=6)
    rows = [(f, s) for f in T.FAMILIES for s in T.SUBSETS[f]]
    rh, bottom = 12.0, 28.0
    ah = rh * len(rows) + 4
    Ht = leg + 4 + ah + bottom
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    H.legend(fig, items, x_pt=3, top_pt=0, width_pt=Wc - 5, fontsize=7.6, gap_pt=6)
    ax = H.axes_pt(fig, 92, bottom, Wc - 92 - 6, ah)
    ax.set_xscale("log")
    from matplotlib.patches import FancyArrowPatch
    allx = []
    for i, (f, s) in enumerate(rows):
        for mo, dy, ls in (("greedy", -0.17, "-"), ("sample", 0.17, (0, (2.5, 1.5)))):
            y = i + dy
            for m in BASES:
                v = T.spd(f, m, mo, s)
                allx.append(v)
                ax.plot([v, v], [y - 0.13, y + 0.13], color=M[m], lw=1.6, zorder=2, solid_capstyle="butt")
            b, o = T.spd(f, "DDTree", mo, s), T.spd(f, "Ours", mo, s)
            allx.append(o)
            ax.add_patch(FancyArrowPatch((b, y), (o, y), arrowstyle="-|>,head_length=3.2,head_width=1.8",
                                         mutation_scale=1, color=C["ours"], lw=1.0, linestyle=ls, shrinkA=0, shrinkB=1.5,
                                         zorder=3))
            ax.plot([o], [y], "o", ms=3.6, mfc=C["ours"] if mo == "greedy" else "white", mec=C["ours"],
                    mew=0.5 if mo == "greedy" else 0.9, zorder=4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [f"{f}, {SUBLAB[s]}" for f, s in rows], fontsize=7.0)
    ax.tick_params(axis="y", length=0, pad=2)
    for i in range(1, len(T.FAMILIES)):
        ax.axhline(2 * i - 0.5, color="#E6E6E6", lw=0.6, zorder=0).set_gid("ref")
    ax.set_xlim(1.2, 18)
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([1.5, 2, 3, 5, 8, 12]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(lambda v, _p: f"{v:g}×")
    ax.set_xlabel("speed-up over the original model", fontsize=SZ["label"], labelpad=1.5)
    H.fs.assert_inside(ax, allx, [0])
    finish(fig)
    H.save(fig, str(HERE / "t3d_arrows"), width="col", strict=STRICT)
    plt.close(fig)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["t3a_ticks", "t3b_ratio", "t3c_radar", "t3d_arrows"]:
        globals()[n]()
