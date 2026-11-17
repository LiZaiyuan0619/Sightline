"""MF method figure, round 1: candidates. Data layer in mfdata.py; README.md says what each one is.

Run:  python3 paper/claims/MF-method/round1/make.py [names...]   (no names = everything; STRICT=0 to draft)
Three properties of SVG decoding, one component each; every piece drawn here is a real run (mfdata.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle
from PIL import Image
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = RUNS.parent            # data/, which holds runs/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import mfdata as D  # noqa: E402

STRICT = os.environ.get("STRICT", "1") != "0"
H.apply()
INK, MUTED, OURS, AR, LOOP = H.INK, H.MUTED, H.C["ours"], H.C["ar"], H.C["loop"]
LOOP_L, OURS_L = H.C["loop_light"], H.C["ours_light"]
FRAME, PANEL_BG = "#C9C9C9", "#F5F6F7"
W_FULL, W_COL = H.FULL_W * 72, H.COL_W * 72
SZ_PROP, SZ_COMP, SZ_NOTE = 8.5, 9.0, 8.0


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

# the three properties and the component each one buys (ENCODING.md: no internal names on the figure)
PROPS = ["most next tokens are predictable", "outputs fall into cycles", "every prefix renders and scores"]
COMPS = ["block draft, verified as a tree", "loop control", "best of N by rendered score"]

_img = {}


def img(path, px=260):
    if path not in _img:
        im = Image.open(path).convert("RGB").resize((px, px), Image.LANCZOS)
        _img[path] = np.asarray(im)
    return _img[path]


def cell(fig, x, y, w, h, bg=True):
    ax = H.axes_pt(fig, x, y, w, h)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if bg:
        ax.set_facecolor(PANEL_BG)
        ax.patch.set_gid("bg")
    else:
        ax.patch.set_alpha(0)
    return ax


LETTER_W = 13.0


def head(fig, x, y_top, w, prop, comp, align="left"):
    """Two lines over a cell: what the data looks like (grey), then what we do about it (blue, bold)."""
    ov = H._overlay(fig)
    xa = {"left": x, "center": x + w / 2, "right": x + w}[align]
    ha = {"left": "left", "center": "center", "right": "right"}[align]
    ov.text(xa, y_top, prop, ha=ha, va="top", fontsize=SZ_PROP, color=MUTED)
    ov.text(xa, y_top - SZ_PROP - 3.0, comp, ha=ha, va="top", fontsize=SZ_COMP, color=OURS, fontweight="bold")
    return y_top - SZ_PROP - 3.0 - SZ_COMP - 2.0


def note(fig, x, y, s, ha="left", va="top", color=MUTED, fs=SZ_NOTE, bold=False, **kw):
    return H._overlay(fig).text(x, y, s, ha=ha, va=va, fontsize=fs, color=color,
                                fontweight="bold" if bold else "normal", **kw)


def text_w(fig, s, fs, **kw):
    """Width of a string in points, measured on the figure (so bands can be sized to their text)."""
    t = H._overlay(fig).text(0, 0, s, fontsize=fs, **kw)
    w = t.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    t.remove()
    return w


def arrow(fig, x0, y0, x1, y1, color=None, lw=1.1, hw=4.0):
    ov = H._overlay(fig)
    a = FancyArrow(x0, y0, x1 - x0, y1 - y0, width=0.6, head_width=hw, head_length=5.0, length_includes_head=True,
                   color=color or "#9AA0A6", lw=0)
    a.set_gid("bg")
    ov.add_patch(a)


def box(fig, x, y, w, h, color="#DDE2E6", lw=0.8, fc="white", r=2.5):
    ov = H._overlay(fig)
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", linewidth=lw, edgecolor=color,
                       facecolor=fc)
    p.set_gid("bg")
    ov.add_patch(p)


# --- the three pieces, each drawn into a given rectangle -----------------------------------------------------
def draw_tree(fig, x, y, w, h, label_fs=SZ_NOTE, dense=True):
    """The real 127-node draft tree of one round: depth on x, the accepted path straight along the middle."""
    t = D.tree()
    f = t["facts"]
    ax = cell(fig, x, y, w, h, bg=False)
    ys = [p[1] for p in t["nodes"]]
    ax.set_xlim(0.2, f["max_depth"] + 0.8)
    ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
    segs_off = [(e[0], e[1]) for e in t["edges"] if not e[2]]
    for (x0, y0), (x1, y1) in segs_off:
        ax.plot([x0, x1], [y0, y1], color="#D5DBDF", lw=0.35, zorder=1, solid_capstyle="round")
    for (x0, y0), (x1, y1), ok in t["edges"]:
        if ok:
            ax.plot([x0, x1], [y0, y1], color=OURS, lw=1.9, zorder=3, solid_capstyle="round")
    off = [(d, yy) for d, yy, a, _ in t["nodes"] if not a]
    ax.plot([p[0] for p in off], [p[1] for p in off], ls="none", marker="o", ms=1.5, mfc="#B9C2C8", mec="none",
            zorder=2)
    acc = [(d, yy) for d, yy, a, _ in t["nodes"] if a]
    ax.plot([p[0] for p in acc], [p[1] for p in acc], ls="none", marker="o", ms=2.8, mfc=OURS, mec="none", zorder=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # every position where the accepted token was not the draft's first guess (a chain would stop at the first)
    for d_off, _rank in f["off_chain"]:
        ax.plot([d_off], [0], marker="o", ms=5.0, mfc="none", mec=LOOP, mew=1.1, zorder=5)
    return ax, f


def draw_loop(fig, x, y, w, h, fs=7.6, side=True):
    """The real cycle: three identical copies, then the token the escape bans and where the stop lands.
    The amber band is sized to the measured string; the font shrinks until the row fits the given width."""
    e = D.loop()
    P = e["period"]
    cyc = "".join(e["cycle"])
    ov = H._overlay(fig)
    side_lab = [f"cycle of {P} tokens", "escape, then stop"]
    side_w = (max(text_w(fig, s_, SZ_NOTE) for s_ in side_lab) + 9) if side else 0.0
    while fs > 7.0 and text_w(fig, cyc, fs, family="monospace") + 6 + side_w > w:
        fs -= 0.2
    if side and text_w(fig, cyc, fs, family="monospace") + 6 + side_w > w:
        side, side_w = False, 0.0          # no room beside the strip: the numbers go in the note line instead
    bw = text_w(fig, cyc, fs, family="monospace") + 5.0
    row_h = min(fs * 1.62, (h - 2) / 4)
    y0 = y + (h + 4 * row_h) / 2 - row_h / 2
    for k in range(3):
        ov.add_patch(Rectangle((x, y0 - k * row_h - row_h * 0.42), bw, row_h * 0.84, facecolor=LOOP_L,
                               edgecolor="none", linewidth=0, gid="bg"))
        ov.text(x + 2.5, y0 - k * row_h, cyc, ha="left", va="center", fontsize=fs, color=INK, family="monospace")
    yk = y0 - 3 * row_h
    ov.add_patch(Rectangle((x, yk - row_h * 0.42), bw, row_h * 0.84, facecolor="white", edgecolor=LOOP,
                           linewidth=0.8, linestyle=(0, (2, 1.6)), gid="bg"))
    ov.text(x + 2.5, yk, cyc, ha="left", va="center", fontsize=fs, color="#BFC4C7", family="monospace")
    if side:
        note(fig, x + bw + 8, y0 - row_h, side_lab[0], va="center", color=LOOP)
        note(fig, x + bw + 8, yk, side_lab[1], va="center", color=OURS, bold=True)
    e = dict(e)
    e["bottom"] = yk - row_h * 0.42
    e["side"] = side
    return e


def draw_cands(fig, x, y, w, h, cols=4, fs=7.2, show_scores=True):
    """The 8 candidates of one prompt with their rendered score; the kept one is framed."""
    c = D.candidates()
    rows = (8 + cols - 1) // cols
    lab = fs * 1.45 if show_scores else 2.0
    g = 3.0
    s = min((w - (cols - 1) * g) / cols, (h - (rows - 1) * (lab + 1.5)) / rows)
    ov = H._overlay(fig)
    for k, cd in enumerate(c["cands"]):
        r, col = divmod(k, cols)
        cx = x + col * (s + g)
        cy = y + h - s - r * (s + lab + 1.5)
        ax = H.axes_pt(fig, cx, cy, s, s)
        ax.imshow(img(cd["png"]), interpolation="antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(1.3 if cd["chosen"] else 0.4)
            sp.set_edgecolor(OURS if cd["chosen"] else FRAME)
        if show_scores:
            ov.text(cx + s / 2, cy - 1.5, f"{cd['clip']:.2f}" if cd["valid"] else "–", ha="center", va="top",
                    fontsize=fs, color=OURS if cd["chosen"] else MUTED,
                    fontweight="bold" if cd["chosen"] else "normal")
    return c, s


def save(fig, name, width):
    H.save(fig, str(HERE / name), width=width, strict=STRICT)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
def mfa_three():
    """One row of three cells: property (grey) over component (blue), each with its real artefact underneath."""
    g = 16.0
    w = (W_FULL - 2 * g - 2 - LETTER_W) / 3
    Ht = 136.0
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    xs = [1 + LETTER_W + j * (w + g) for j in range(3)]
    tops = [head(fig, xs[j], Ht - 3, w, PROPS[j], COMPS[j]) for j in range(3)]
    y_note, body_y = 20.0, 34.0
    body_h = tops[0] - body_y - 2
    ax, f = draw_tree(fig, xs[0], body_y, w, body_h)
    note(fig, xs[0], y_note, f"{f['accepted']} tokens accepted in one pass of the model", color=INK)
    note(fig, xs[0], y_note - 10.0, f"a chain of first guesses stops at {f['chain_stops']} (circles)")
    e = draw_loop(fig, xs[1], body_y, w, body_h)
    note(fig, xs[1], y_note, f"{e['saved']:,} of {e['n_tokens']:,} tokens cut here", color=INK)
    note(fig, xs[1], y_note - 10.0, f"a {e['period']}-token cycle, three copies"
         if e["side"] else f"escape, then stop, on a {e['period']}-token cycle")
    c, _ = draw_cands(fig, xs[2], body_y, w, body_h)
    note(fig, xs[2], y_note, f"8 candidates in {c['t_sd8']:.1f} s, best rendered score kept", color=INK)
    note(fig, xs[2], y_note - 10.0, "one sample of the original model, 11.6 s")
    for j in range(3):
        H.letter(fig, xs[j] - LETTER_W, 3.0, "abc"[j])
    save(fig, "mfa_three", "full")


def mfb_pipeline():
    """One decoding round left to right (top band), the loop rule and the best-of-N branch under it."""
    Ht = 200.0
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    ov = H._overlay(fig)
    y_mid = Ht - 50.0                       # top band: Ht-96.. Ht-4
    x = 20.0
    bw = 104.0
    box(fig, x, y_mid - 16, bw, 32)
    note(fig, x + bw / 2, y_mid + 20, "draft head, one pass", ha="center", va="bottom", color=OURS, bold=True,
         fs=SZ_COMP)
    for k in range(32):
        ov.add_patch(Rectangle((x + 5 + k * (bw - 10) / 32, y_mid - 3), (bw - 10) / 32 - 0.6, 11,
                               facecolor=OURS_L, edgecolor="none", linewidth=0, gid="bg"))
    note(fig, x + bw / 2, y_mid - 20, "32 positions", ha="center", va="top", color=MUTED)
    arrow(fig, x + bw + 4, y_mid, x + bw + 18, y_mid)
    x2 = x + bw + 21
    tw_ = 150.0
    note(fig, x2 + tw_ / 2, y_mid + 20, "one draft tree, 127 nodes", ha="center", va="bottom", color=OURS,
         bold=True, fs=SZ_COMP)
    ax, f = draw_tree(fig, x2, y_mid - 17, tw_, 34)
    note(fig, x2 + tw_ / 2, y_mid - 20, "top 16 guesses per position", ha="center", va="top", color=MUTED)
    arrow(fig, x2 + tw_ + 4, y_mid, x2 + tw_ + 18, y_mid)
    x3 = x2 + tw_ + 21
    vw = 96.0
    box(fig, x3, y_mid - 16, vw, 32, color=AR, fc="#F2F2F2")
    note(fig, x3 + vw / 2, y_mid + 20, "original model", ha="center", va="bottom", color=INK, fs=SZ_COMP)
    note(fig, x3 + vw / 2, y_mid + 9, "one forward pass", ha="center", color=INK)
    note(fig, x3 + vw / 2, y_mid - 2, f"{f['accepted']} tokens kept", ha="center", color=OURS, bold=True)
    arrow(fig, x3 + vw + 4, y_mid, x3 + vw + 18, y_mid)
    x4 = x3 + vw + 21
    note(fig, x4, y_mid + 20, "output", ha="left", va="bottom", color=INK, fs=SZ_COMP)
    note(fig, x4, y_mid + 6, "one round of\nthe decoder", ha="left", color=MUTED)
    # lower band: 4.. Ht-110
    y2 = 46.0
    note(fig, x, y2 + 42, PROPS[1], ha="left", va="bottom", color=MUTED, fs=SZ_PROP)
    note(fig, x, y2 + 32, COMPS[1], ha="left", va="bottom", color=OURS, bold=True, fs=SZ_COMP)
    e = draw_loop(fig, x, y2 - 16, 232.0, 46, fs=7.2)
    note(fig, x, y2 - 20, f"{e['saved']:,} of {e['n_tokens']:,} tokens cut here", color=INK)
    xr = 296.0
    note(fig, xr, y2 + 42, PROPS[2], ha="left", va="bottom", color=MUTED, fs=SZ_PROP)
    note(fig, xr, y2 + 32, COMPS[2], ha="left", va="bottom", color=OURS, bold=True, fs=SZ_COMP)
    c, _ = draw_cands(fig, xr, y2 - 4, W_FULL - xr - 2, 32.0, cols=8, fs=7.0)
    note(fig, xr, y2 - 20, f"8 candidates in {c['t_sd8']:.1f} s,  one original sample 11.6 s", color=INK)
    arrow(fig, 8.0, y_mid - 30, 8.0, y2 + 52)
    save(fig, "mfb_pipeline", "full")


def mfc_single():
    """Single column: the three properties stacked, each with its real artefact and one line of numbers."""
    Ht = 282.0
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    x = 1.0
    w = W_COL - 2
    row = (Ht - 12) / 3
    for j in range(3):
        y_top = Ht - 3 - j * row
        yt = head(fig, x, y_top, w, PROPS[j], COMPS[j]) - 10.0
        yb = y_top - row + 8
        hb = yt - yb - 2
        if j == 0:
            ax, f = draw_tree(fig, x, yb, w, hb)
            note(fig, x, yt + 10, f"{f['accepted']} tokens in one pass, a chain stops at {f['chain_stops']}",
                 color=INK)
        elif j == 1:
            e = draw_loop(fig, x, yb, w, hb, fs=7.0)
            note(fig, x, yt + 10, f"{e['saved']:,} of {e['n_tokens']:,} tokens cut here", color=INK)
        else:
            c, _ = draw_cands(fig, x, yb, w, hb, cols=8, fs=7.0)
            note(fig, x, yt + 10, f"8 candidates in {c['t_sd8']:.1f} s, one sample 11.6 s", color=INK)
    save(fig, "mfc_single", "col")


def mfd_hero():
    """Asymmetric: the draft tree as the hero on the left, the other two components stacked on the right."""
    Ht = 182.0
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    gx = 20.0
    xl = 1 + LETTER_W
    wl = W_FULL * 0.44
    xr = xl + wl + gx + LETTER_W
    wr = W_FULL - xr - 2
    yt = head(fig, xl, Ht - 3, wl, PROPS[0], COMPS[0])
    ax, f = draw_tree(fig, xl, 30.0, wl, yt - 32)
    note(fig, xl, 25.0, f"{f['accepted']} tokens accepted in one pass of the model", color=INK)
    note(fig, xl, 15.0, f"a chain of first guesses stops at {f['chain_stops']};  the "
                        f"{ordinal(f['off_chain'][0][1])} guess is accepted at position {f['off_chain'][0][0]}")
    yt1 = head(fig, xr, Ht - 3, wr, PROPS[1], COMPS[1])
    e = draw_loop(fig, xr, yt1 - 50, wr, 48, fs=7.0)
    note(fig, xr, yt1 - 56, f"{e['saved']:,} of {e['n_tokens']:,} tokens cut here", color=INK)
    yt2 = head(fig, xr, yt1 - 72, wr, PROPS[2], COMPS[2])
    c, _ = draw_cands(fig, xr, yt2 - 42, wr, 38.0, cols=8, fs=7.0)
    note(fig, xr, max(yt2 - 48, 11.0), f"8 candidates in {c['t_sd8']:.1f} s,  one original sample 11.6 s",
         color=INK)
    H.letter(fig, xl - LETTER_W, 3.0, "a")
    H.letter(fig, xr - LETTER_W, 3.0, "b")
    save(fig, "mfd_hero", "full")


CANDIDATES = {"mfa_three": mfa_three, "mfb_pipeline": mfb_pipeline, "mfc_single": mfc_single, "mfd_hero": mfd_hero}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CANDIDATES)
    for n in names:
        CANDIDATES[n]()
    pngs = [str(HERE / f"{n}.png") for n in CANDIDATES if (HERE / f"{n}.png").exists()]
    H.sheet(pngs, str(HERE / "_sheet.png"), cols=1, title="MF method figure, round 1 (11-05)")
