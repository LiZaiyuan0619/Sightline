"""The six cells of the theory section's single figure (2 x 3, double column), round 1 (main line).

User, the theory figures become one 2 x 3 figure, cells in the order the text cites them; Proposition 2 changes
from the per-prompt slopes to violins with one dot per prompt (round 1's q1c); the ceiling-and-losses panel moves here
from the ablation group. Each cell draws itself into a box given in points (the D-group pattern), so every cell has
the same plot width inside its row and the two rows share their columns (nature-figure alignment gate).

Data: every cell loads the data layer of its signed-off figure, unchanged, and those modules assert their published
numbers on import.
  a  Proposition 1, per prompt                      TH1-run-length round 3 helpers (round 2's th2data), as p4a_joint
  b per-prompt miss rate, vector graphics vs NL    TH4-nl-control round 1 (n2_pair panel a)
  c  Proposition 2, class x decisions crossed       TH2-horizon round 1 thhdata (q1c_swarm)
  d accuracy with no decision crossed, vs NL       TH4-nl-control round 1 (n2_pair panel b)
  e the ceiling and the three losses               TH5-ceiling-losses round 1 load (c1_models)
  f  Proposition 3, law curves and 16 cells         TH3-loop-limit round 1 th3data (r1a_curves)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from matplotlib.transforms import blended_transform_factory
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
CLAIMS = HERE.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
import house as H  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P1 = _load("th1r3", CLAIMS / "TH1-run-length/round3/make.py")          # imports th2data, runs its check
NL = _load("th4r1", CLAIMS / "TH4-nl-control/round1/make.py")          # asserts against nl_vs_svg.json
P2 = _load("th2r1", CLAIMS / "TH2-horizon/round1/make.py")             # imports thhdata, runs its check
CL = _load("th5r1", CLAIMS / "TH5-ceiling-losses/round1/make.py")      # asserts the published ceiling levels
P3 = _load("th3r1", CLAIMS / "TH3-loop-limit/round1/make.py")          # imports th3data, runs its check

# --- type sizes for a cell one third of the page wide (floor 7 pt, guide 4.1) --------------------------------------
H.apply()
TICK, LAB, NOTE, KEY = 7.5, 8.0, 7.2, 7.4
plt.rcParams.update({"xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.labelsize": LAB})
C, M, INK, MUTED = H.C, H.METHOD, H.INK, H.MUTED

# --- geometry, points (double column 6.875 in = 495 pt) ---------------------------------------------------------
FULL = 495.0
BOX_W = FULL / 3
AX_L, AX_R = 40.0, 5.0                   # same in every cell: one plot width per column
R1_B, R1_AX, R1_T = 33.0, 100.0, 30.0    # row a-c: a two-row x axis under b, the run-length marginal over a
R2_B, R2_AX, R2_T = 27.0, 104.0, 8.0     # row d-f
R1_H, R2_H = R1_B + R1_AX + R1_T, R2_B + R2_AX + R2_T
TOP_PAD = 3.0                            # clear of the canvas edge (house check: 1 pt)
FIG_H = R1_H + R2_H + TOP_PAD


def _ax(fig, box, row):
    x0, y0, w, _h = box
    b, h = (R1_B, R1_AX) if row == 1 else (R2_B, R2_AX)
    return H.axes_pt(fig, x0 + AX_L, y0 + b, w - AX_L - AX_R, h)


def _free(*artists):
    for a in artists:
        a.set_gid("free")
    return artists[0] if len(artists) == 1 else artists


def _ref_h(ax, y):
    ln = ax.axhline(y, color=H.REF, lw=0.75, ls=(0, (3, 2)), zorder=1)
    ln.set_gid("ref")
    return ln


# ===================================================================================================== a  Proposition 1
def cell_a(fig, box):
    """Per prompt, mean run length against the gain from doubling the window 16 -> 32; chains hollow grey, trees
    filled blue, one shared trend band; the run-length densities on top; windows 16 and 32 through both."""
    x0, y0, w, _h = box
    ax = _ax(fig, box, 1)
    axt = H.axes_pt(fig, x0 + AX_L, y0 + R1_B + R1_AX + 2, w - AX_L - AX_R, R1_T - 4)
    X, Y, MED = P1.X, P1.Y, P1.MED
    P1._shared_band(ax)
    sc1 = ax.scatter(X[1], Y[1], s=4.0, facecolors="none", edgecolors=P1.GREY, linewidths=0.5, alpha=0.75, zorder=3)
    sc2 = ax.scatter(X[16], Y[16], s=4.0, color=P1.BLUE, alpha=0.55, lw=0, zorder=4)
    for sc in (sc1, sc2):
        sc.set_rasterized(True)
    for k in (1, 16):
        ax.plot([MED[k][0]], [MED[k][1]], marker="D", ms=4.8, mfc="white", mec=P1.BLUE if k == 16 else "#6E6E6E",
                mew=1.3, zorder=6)
    _ref_h(ax, 2.0)
    _ref_h(ax, 1.0)
    ax.text(P1.XLIM[1] * 0.9, 1.022, "no gain", ha="right", va="bottom", fontsize=NOTE, color=MUTED)
    for x in (16, 32):
        for a in (ax, axt):
            ln = a.axvline(x, color=H.REF, lw=0.7, ls=(0, (3, 2)), zorder=0.5)
            ln.set_gid("ref")
    ax.text(16 / 1.3, 2.035, "window", ha="right", va="bottom", fontsize=NOTE, color=MUTED)
    for x in (16, 32):
        ax.text(x * 1.07, 2.035, f"{x}", ha="left", va="bottom", fontsize=NOTE, color=MUTED)
    P1._logx(ax)
    P1._gain_axis(ax)
    ax.set_ylim(P1.YLIM[0], 2.2)          # room for the window names over the x2 line, below the marginal
    ax.set_xlabel("mean run length of a prompt (tokens)", labelpad=1.5)
    ax.set_ylabel("gain, window 16 → 32", labelpad=2)
    P1._top_marginal(axt, labels=True)
    meds = {MED[k][0] for k in (1, 16)}
    for ln in list(axt.lines):                       # the median dashes go (p4a): the diamonds mark the medians
        xd = ln.get_xdata()
        if len(xd) == 2 and xd[0] == xd[1] and xd[0] in meds:
            ln.remove()
    assert len([ln for ln in axt.lines if ln.get_gid() != "ref"]) == 2
    for t in axt.texts:
        t.set_fontsize(KEY)
    axt.set_ylim(0, 1.62)                 # the two names stay inside the marginal's own box
    return ax


# ===================================================================================================== b clumping, per prompt
FLOOR = 0.003


def cell_b(fig, box):
    """Per prompt miss rate; each pair of corpora shares its average miss rate (black bar = pooled rate)."""
    x0, y0, w, _h = box
    ax = _ax(fig, box, 1)
    D, ORDER, KIND = NL.D, NL.ORDER, NL.KIND
    rng = np.random.default_rng(1107)
    XPOS = [0.0, 1.2, 2.5, 3.7]           # wide enough that no two corpus names run together
    for i, c in zip(XPOS, ORDER):
        lam = D[c]["per"]["lam"]
        nz = lam[lam > 0]
        v = np.log10(np.maximum(lam, FLOOR))
        if len(nz) >= 10:
            g = np.linspace(np.log10(max(nz.min(), FLOOR)), np.log10(nz.max()), 200)
            d = gaussian_kde(np.log10(nz), bw_method=0.3)(g)
            d = d / d.max() * 0.38
            f = ax.fill_betweenx(10 ** g, i - d, i + d, color=NL.FILL[KIND[c]], lw=0, alpha=0.85, zorder=1)
            f.set_gid("bg")
        sc = ax.scatter(i + rng.uniform(-0.11, 0.11, len(v)), 10 ** v, s=3.0, color=NL.COL[KIND[c]], alpha=0.55,
                        lw=0, zorder=3)
        sc.set_rasterized(True)
        ax.plot([i], [np.median(lam)], marker="D", ms=4.6, mfc="white", mec=NL.COL[KIND[c]], mew=1.2, zorder=6)
        ax.plot([i - 0.3, i + 0.3], [D[c]["pooled"]["lam"]] * 2, color=INK, lw=1.4, zorder=5)
        nzero = int((lam == 0).sum())
        if nzero:
            _free(ax.text(i, FLOOR * 0.74, f"{nzero} with\nno miss", ha="center", va="top", fontsize=7.0,
                          color=MUTED, linespacing=1.05))
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.30, 0.7)
    ax.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:g}%"))
    ax.set_xlim(-0.5, 4.2)
    ax.set_xticks(XPOS)
    ax.set_xticklabels([NL.SHORT[c] for c in ORDER])
    ax.tick_params(axis="x", length=0, pad=2)
    for a, b in ((XPOS[0], XPOS[1]), (XPOS[2], XPOS[3])):
        ax.plot([a - 0.4, b + 0.4], [-0.135, -0.135], color=H.GROUP_LINE, lw=0.8, clip_on=False,
                transform=ax.get_xaxis_transform(), zorder=1)
        ax.text((a + b) / 2, -0.155, "same miss rate", ha="center", va="top", fontsize=NOTE, color=MUTED,
                transform=ax.get_xaxis_transform())
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("miss rate per prompt", labelpad=2)
    # key for the two marks, in the band over the plot
    tr = blended_transform_factory(ax.transAxes, ax.transAxes)
    yk = 1 + 9.0 / R1_AX
    _free(*ax.plot([0.02, 0.10], [yk, yk], color=INK, lw=1.4, transform=tr, clip_on=False))
    _free(ax.text(0.13, yk, "pooled rate", ha="left", va="center", fontsize=KEY, transform=tr))
    _free(*ax.plot([0.56], [yk], marker="D", ms=4.6, mfc="white", mec=MUTED, mew=1.2, transform=tr, clip_on=False))
    _free(ax.text(0.61, yk, "median prompt", ha="left", va="center", fontsize=KEY, transform=tr))
    return ax


# ===================================================================================================== c  Proposition 2
def cell_c(fig, box):
    """Violin + one dot per prompt for each class and number of decisions crossed; median bar, value on top."""
    ax = _ax(fig, box, 1)
    T = P2.T
    rng = np.random.default_rng(1106)
    pos, xs, labs, spans = 0.0, [], [], {}
    for cl in ("cmd", "coord"):
        start = pos
        for c in range(3):
            v = 100 * T.cell(cl, 0, c)
            v = v[~np.isnan(v)]
            vp = ax.violinplot([v], positions=[pos], widths=0.8, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(P2.LIGHT[cl])
                body.set_edgecolor("none")
                body.set_alpha(1.0)
                body.set_gid("bg")
            sc = ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)), v, s=1.6, color=P2.COL[cl], alpha=0.45, lw=0,
                            zorder=3)
            sc.set_rasterized(True)
            med = float(np.median(v))
            ax.plot([pos - 0.32, pos + 0.32], [med, med], color=P2.COL[cl], lw=1.7, zorder=4, solid_capstyle="butt",
                    path_effects=[pe.withStroke(linewidth=2.9, foreground="white")])
            _free(ax.text(pos, 102.5, f"{med:.0f}%", ha="center", va="bottom", fontsize=NOTE, color=P2.COL[cl],
                          fontweight="bold" if c == 2 else "normal"))
            xs.append(pos)
            labs.append(["0", "1", "2+"][c])
            pos += 1.0
        spans[cl] = (start, pos - 1.0)
        pos += 0.55
    ax.set_xlim(-0.6, pos - 0.95)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_bounds(0, 100)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlabel("decisions crossed in the block", labelpad=2)
    ax.set_ylabel("first guess right (%)", labelpad=2)
    # the two classes named over their groups, in the band over the plot, with a thin line in the class colour
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for cl, (a, b) in spans.items():
        y = 1 + 7.0 / R1_AX
        ax.plot([a - 0.36, b + 0.36], [y, y], color=P2.COL[cl], lw=0.9, transform=tr, clip_on=False)
        ax.text((a + b) / 2, y + 2.0 / R1_AX, P2.NAME[cl], ha="center", va="bottom", fontsize=KEY, color=P2.COL[cl],
                transform=tr)
    return ax


# ===================================================================================================== d distance, vs NL
def cell_d(fig, box):
    """Accuracy at offset j with no decision crossed (the distance term), vector graphics against natural language."""
    ax = _ax(fig, box, 2)
    D, ORDER, KIND, PAIR, JMAX = NL.D, NL.ORDER, NL.KIND, NL.PAIR, NL.JMAX
    js = np.arange(1, JMAX + 1)
    for c in ORDER:
        g, lo, hi = (D[c]["dec"][k][1:JMAX + 1] for k in ("g", "g_lo", "g_hi"))
        f = ax.fill_between(js, 100 * lo, 100 * hi, color=NL.FILL[KIND[c]], lw=0, alpha=0.6, zorder=1)
        f.set_gid("bg")
        ax.plot(js, 100 * g, color=NL.COL[KIND[c]], lw=1.6 if KIND[c] == "svg" else 1.35, ls=NL.LS[PAIR[c]],
                marker=NL.MARK[PAIR[c]], ms=2.5, mew=0, zorder=3 if KIND[c] == "svg" else 2)
    ax.set_xlim(0.4, 19.6)
    ax.set_xticks([1, 4, 8, 12, 15])
    ax.set_ylim(72, 101)
    ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.set_xlabel("offset inside the block (tokens)", labelpad=1.5)
    ax.set_ylabel("accuracy, no decision\ncrossed (%)", labelpad=2, linespacing=1.12)
    ends = {c: 100 * D[c]["dec"]["g"][JMAX] for c in ORDER}
    s_lo, s_hi = sorted((ends["svg16"], ends["svg32"]))
    labels = [((s_lo + s_hi) / 2, f"{s_lo:.0f}–{s_hi:.0f}%" if s_hi - s_lo >= 0.5 else f"{s_hi:.0f}%", NL.BLUE,
               "bold")]
    labels += [(ends[c], f"{ends[c]:.0f}%", NL.GREY, "normal") for c in ("humaneval", "gsm8k")]
    for y, txt, col, fw in labels:
        _free(ax.text(15.6, y, txt, ha="left", va="center", fontsize=NOTE, color=col, fontweight=fw))
    for n, (txt, col, fw) in enumerate([("vector graphics", NL.BLUE, "bold"), ("natural language", NL.GREY, "normal")]):
        y = 80.0 - 4.2 * n
        _free(*ax.plot([1.0, 2.9], [y, y], color=col, lw=1.6 if col == NL.BLUE else 1.35, zorder=4))
        _free(ax.text(3.5, y, txt, ha="left", va="center", fontsize=KEY, color=col, fontweight=fw))
    return ax


# ===================================================================================================== e ceiling and losses
SHADE = {64: C["ours_mid"], 32: C["ours"], 16: M["DDTree"]}     # the block-16 head is the published drafters' head


def cell_e(fig, box):
    """Tokens per round under four increasingly real models, one line per head; the chain on the same head as a
    control; open circles = measured."""
    ax = _ax(fig, box, 2)
    D, HEADS = CL.D, CL.HEADS
    xs = np.arange(4)
    ch = D[CL.CHAIN_HEAD]["chain"]
    ax.plot(xs, ch[:4], color=CL.GREY, lw=1.3, ls=(0, (3, 1.6)), marker="o", ms=2.6, mew=0, zorder=2)
    for h, _name, blk in HEADS:
        row = D[h]["tree"]
        ax.plot(xs, row[:4], color=SHADE[blk], lw=1.9 if blk == 32 else 1.5, marker="o", ms=3.2, mew=0,
                zorder=4 if blk == 32 else 3)
        ax.plot([3], [row[4]], marker="o", ms=6.2, mfc="none", mec=SHADE[blk], mew=1.0, zorder=5)
    keys = [(SHADE[64], "blocks of 64", "normal", "-"), (SHADE[32], "blocks of 32", "bold", "-"),
            (SHADE[16], "blocks of 16", "normal", "-"), (CL.GREY, "chain, same head", "normal", (0, (3, 1.6)))]
    for n, (col, txt, fw, ls) in enumerate(keys):
        y = 95 * (0.79 ** n)
        _free(*ax.plot([1.02, 1.36], [y, y], color=col, lw=1.9 if fw == "bold" else 1.5, ls=ls, zorder=6))
        _free(ax.text(1.46, y, txt, ha="left", va="center", fontsize=KEY, color=col, fontweight=fw))
    CL._tau_axis(ax)
    ax.set_xlim(-0.3, 3.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CL.LEVEL_NAME, fontsize=7.2)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("tokens per round", labelpad=2)
    return ax


# ===================================================================================================== f  Proposition 3
def cell_f(fig, box):
    """The law 1 + phi / (r (1 - phi)) for each decoder and the 16 measured cells; same cell joined 16 -> 32."""
    ax = _ax(fig, box, 2)
    T, COL = P3.T, P3.COL
    ymax = 2.0
    for meth, r, ls, lw in (("AR", 1.0, (0, (4, 2)), 1.4), ("DDTree", T.R_MED["DDTree"], "-", 1.6),
                            ("Ours", T.R_MED["Ours"], "-", 2.0)):
        phi = np.linspace(0, min(0.56, (ymax - 1) * r / (1 + (ymax - 1) * r)), 200)
        ax.plot(phi, T.speedup(phi, r), color=COL[meth], lw=lw, ls=ls, zorder=2,
                path_effects=[pe.withStroke(linewidth=lw + 1.6, foreground="white")])
    P = {(x["block"], x["subset"], x["method"]): x for x in T.ROWS}
    for (blk, sub, meth), d in P.items():
        if meth != "DDTree":
            continue
        o = P[(blk, sub, "Ours")]
        ax.plot([d["phi"], o["phi"]], [d["measured"], o["measured"]], color="#AEB6C1", lw=0.7, zorder=3)
    for x in T.ROWS:
        m = P3.MARK[x["subset"]]
        ax.plot([x["phi"]], [x["measured"]], ls="none", marker=m["marker"], ms=4.6, mew=1.0, mec=COL[x["method"]],
                mfc=COL[x["method"]] if m["fill"] else "white", zorder=5)
    ax.set_xlim(0, 0.56)
    ax.set_ylim(0.98, ymax)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{100 * v:.0f}%"))
    ax.set_yticks([1.0, 1.25, 1.5, 1.75, 2.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: {1.0: "×1", 2.0: "×2"}.get(v, f"×{v:g}")))
    ax.set_xlabel("tokens removed by loop control", labelpad=1.5)
    ax.set_ylabel("speed-up from loop control", labelpad=2)
    # key in the empty upper left (every curve and cell is below x1.4 for loops under 25%)
    rows = [("line", COL["AR"], "original model", "normal", (0, (4, 2)), 1.4),
            ("line", COL["DDTree"], "window 16", "normal", "-", 1.6),
            ("line", COL["Ours"], "window 32 (ours)", "bold", "-", 2.0),
            ("mark", INK, "SArena-Icon", "normal", True, 0),
            ("mark", INK, "SArena-Illus.", "normal", False, 0)]
    for n, (kind, col, txt, fw, style, lw) in enumerate(rows):
        y = 1.93 - 0.095 * n
        if kind == "line":
            _free(*ax.plot([0.012, 0.052], [y, y], color=col, lw=lw, ls=style, zorder=6))
        else:
            _free(*ax.plot([0.032], [y], ls="none", marker="^", ms=4.4, mew=1.0, mec=col,
                           mfc=col if style else "white", zorder=6))
        _free(ax.text(0.064, y, txt, ha="left", va="center", fontsize=KEY,
                      color=C["ours"] if fw == "bold" else INK, fontweight=fw))
    return ax


CELLS = {"a": cell_a, "b": cell_b, "c": cell_c, "d": cell_d, "e": cell_e, "f": cell_f}


def finish(fig):
    """Every multi-line string at line spacing >= 1.05 (house check reads tighter as touching)."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if "\n" in t.get_text() and getattr(t, "_linespacing", 1.2) < 1.05:
            t.set_linespacing(1.05)
