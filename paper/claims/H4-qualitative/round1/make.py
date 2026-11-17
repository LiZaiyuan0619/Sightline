"""H4 qualitative comparison, round 1: candidates. Data layer in h4data.py; README.md says what each one is.

Run:  python3 paper/claims/H4-qualitative/round1/make.py [names...]   (no names = everything; STRICT=0 to draft)
All renders come from the data tables (official renderer, white background); nothing is rendered here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = RUNS.parent            # data/, which holds runs/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "_house"))   # shared figure style
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))                 # the package root
sys.path.insert(0, str(HERE))
import house as H  # noqa: E402
import h4data as D  # noqa: E402

STRICT = os.environ.get("STRICT", "1") != "0"
H.apply()
INK, MUTED, OURS, AR = H.INK, H.MUTED, H.C["ours"], H.C["ar"]
FRAME = "#C9C9C9"
W_FULL, W_COL = H.FULL_W * 72, H.COL_W * 72
ROW_LABEL = {"input": ["input"], "ar1s": ["original model", "+ loop control", "1 sample"],
             "ar8s": ["original model", "+ loop control", "best of 8"], "sd8": ["Ours", "best of 8"]}
TIME_FS = 8.0

_img = {}


def img(path, px=420):
    if path not in _img:
        im = Image.open(path)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, "white")
            bg.alpha_composite(im)
            im = bg
        _img[path] = np.asarray(im.convert("RGB").resize((px, px), Image.LANCZOS))
    return _img[path]


def tile(fig, x, y, s, path, kind="plain"):
    """One square render at (x, y) pt from the figure's bottom-left; Ours gets the blue frame."""
    ax = H.axes_pt(fig, x, y, s, s)
    ax.imshow(img(path), interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.4 if kind == "ours" else 0.5)
        sp.set_edgecolor(OURS if kind == "ours" else FRAME)
    return ax


def text_box(fig, x, y, s, text, fontsize=7.5):
    """A prompt in the cell where an image-to-SVG figure shows its input: light box, italic, wrapped to fit."""
    ax = H.axes_pt(fig, x, y, s, s)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F4F4F4")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_edgecolor(FRAME)
    lines = wrap(fig, f"“{text}”", s - 7, fontsize, style="italic")
    ov = H._overlay(fig)
    ov.text(x + s / 2, y + s / 2, "\n".join(lines), ha="center", va="center", fontsize=fontsize, style="italic",
            color=INK, linespacing=1.15)
    return ax


def wrap(fig, text, width_pt, fontsize, style="normal", weight="normal"):
    ov = H._overlay(fig)
    rend = fig.canvas.get_renderer()
    out, cur = [], ""
    for w in text.split():
        cand = (cur + " " + w).strip()
        t = ov.text(0, 0, cand, fontsize=fontsize, style=style, fontweight=weight)
        wid = t.get_window_extent(rend).width * 72 / fig.dpi
        t.remove()
        if wid <= width_pt or not cur:
            cur = cand
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    return out


def row_label(fig, x, yc, key, fontsize=None):
    fontsize = fontsize or H.SZ["legend"]
    ours = key == "sd8"
    H._overlay(fig).text(x, yc, "\n".join(ROW_LABEL[key]), ha="left", va="center", fontsize=fontsize,
                         color=OURS if ours else INK, fontweight="bold" if ours else "normal", linespacing=1.15)


def time_label(fig, xc, y_top, key, t):
    ours = key == "sd8"
    H._overlay(fig).text(xc, y_top - 1.0, f"{t:.1f} s", ha="center", va="top", fontsize=TIME_FS,
                         color=OURS if ours else MUTED, fontweight="bold" if ours else "normal")


T_SCALE = 32.0      # seconds that fill the bar slot under a tile (the slowest shown output is 30.8 s)


def time_bar(fig, x, y_top, s, key, t, fontsize=TIME_FS):
    """Under a tile: a thin bar whose length is the seconds on a common scale, the number right-aligned after it."""
    ours = key == "sd8"
    ov = H._overlay(fig)
    lab = f"{t:.1f} s"
    num_w = 21.0 if fontsize >= 8 else 19.0
    slot = s - num_w - 2
    L = max(1.2, min(t / T_SCALE, 1.0) * slot)
    yc = y_top - 5.2
    ov.add_patch(matplotlib.patches.Rectangle((x, yc - 1.6), L, 3.2, facecolor=OURS if ours else "#B5B5B5",
                                              edgecolor="none"))
    ov.text(x + s, yc, lab, ha="right", va="center", fontsize=fontsize, color=OURS if ours else MUTED,
            fontweight="bold" if ours else "normal")


def group_header(fig, x0, x1, y, name, color=INK, bold=False, fontsize=None, line=True):
    """Group name centred over [x0, x1] with the thin group line under it (guide SS3.1, two-row axis style)."""
    ov = H._overlay(fig)
    fontsize = fontsize or H.SZ["header"]
    ov.text((x0 + x1) / 2, y + 2.2, name, ha="center", va="bottom", fontsize=fontsize, color=color,
            fontweight="bold" if bold else "normal")
    if line:
        ln, = ov.plot([x0, x1], [y, y], color=H.GROUP_LINE, lw=0.8, solid_capstyle="butt")
        ln.set_gid("ref")


def save(fig, name, width):
    H.save(fig, str(HERE / name), width=width, strict=STRICT)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
def h4a_grid(rule="first", name="h4a_grid", bars=False, wide=False):
    """Image-to-SVG, icons + illustrations: input / 1 sample / best of 8 / Ours best of 8, seconds under each.
    bars: the seconds also as a bar on a common scale; wide: 4 + 4 columns (picks_wide) instead of 3 + 3."""
    if wide:
        ex = [D.example(r, "i2") for r in D.picks_wide("i2", rule)]
    else:
        ex = D.all_picks(rule)["i2"]
    n = len(ex)
    half = n // 2
    L, g, gg = 62.0, 4.0, 10.0
    s = (W_FULL - L - 2 - (n - 2) * g - gg) / n
    head, lab, gap = 15.0, (12.0 if bars else 11.0), 4.0
    Ht = head + s + 5 + 3 * (s + lab + gap) - gap + 3
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    xs, x = [], L
    for j in range(n):
        xs.append(x)
        x += s + (gg if j == half - 1 else g)
    group_header(fig, xs[0], xs[half - 1] + s, Ht - head + 2, "icons")
    group_header(fig, xs[half], xs[n - 1] + s, Ht - head + 2, "illustrations")
    top = Ht - head
    for k, key in enumerate(["input", "ar1s", "ar8s", "sd8"]):
        y = top - s
        row_label(fig, 2, y + s / 2, key)
        for j, e in enumerate(ex):
            if key == "input":
                tile(fig, xs[j], y, s, e["input"])
            else:
                tile(fig, xs[j], y, s, e["rows"][key]["png"], "ours" if key == "sd8" else "plain")
                if bars:
                    time_bar(fig, xs[j], y, s, key, e["rows"][key]["t"], fontsize=TIME_FS if s > 60 else 7.5)
                else:
                    time_label(fig, xs[j] + s / 2, y, key, e["rows"][key]["t"])
        top = y - (5 if key == "input" else lab + gap)
    save(fig, name, "full")


def h4e_single(rule="first", name="h4e_single"):
    """Single column: rows = 2 image-to-SVG + 1 text-to-SVG prompts, columns = input / 1 sample / best of 8 / Ours."""
    P = D.all_picks(rule)
    ex = [P["i2"][0], P["i2"][3], P["t2"][0]]
    g = 3.0
    s = (W_COL - 2 - 3 * g) / 4
    lab, gap = 11.0, 4.0
    heads = [["input"], ["original model", "1 sample"], ["original model", "best of 8"], ["Ours", "best of 8"]]
    head = 25.0
    Ht = head + 3 * (s + lab + gap) - gap + 1
    fig = plt.figure(figsize=(H.COL_W, Ht / 72))
    ov = H._overlay(fig)
    xs = [1 + j * (s + g) for j in range(4)]
    for j, h in enumerate(heads):
        ours = j == 3
        ov.text(xs[j] + s / 2, Ht - head + 3, "\n".join(h), ha="center", va="bottom", fontsize=8.0,
                color=OURS if ours else INK, fontweight="bold" if ours else "normal", linespacing=1.1)
    top = Ht - head
    for e in ex:
        y = top - s
        if e["task"] == "i2":
            tile(fig, xs[0], y, s, e["input"])
        else:
            text_box(fig, xs[0], y, s, e["caption"], fontsize=7.2)
        for j, key in enumerate(["ar1s", "ar8s", "sd8"], start=1):
            tile(fig, xs[j], y, s, e["rows"][key]["png"], "ours" if key == "sd8" else "plain")
            time_label(fig, xs[j] + s / 2, y, key, e["rows"][key]["t"])
        top = y - lab - gap
    save(fig, name, "col")


def h4d_mixed(rule="first", name="h4d_mixed"):
    """Three settings side by side (2 prompts each): OmniSVG-8B image-to-SVG, OmniSVG-8B text-to-SVG, InternSVG-8B
    text-to-SVG. First row = the input image or the prompt."""
    P = D.all_picks(rule)
    blocks = [("i2", P["i2"][0], P["i2"][3]), ("t2", *P["t2"]), ("isvg", *P["isvg"])]
    L, g, gg = 62.0, 4.0, 10.0
    s = (W_FULL - L - 2 - 3 * g - 2 * gg) / 6
    head, lab, gap = 26.0, 11.0, 4.0
    Ht = head + s + 5 + 3 * (s + lab + gap) - gap + 1
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    ov = H._overlay(fig)
    xs, x = [], L
    for j in range(6):
        xs.append(x)
        x += s + (gg if j in (1, 3) else g)
    for b, (task, *_) in enumerate(blocks):
        model, kind = D.TASK_NAME[task]
        x0, x1 = xs[2 * b], xs[2 * b + 1] + s
        group_header(fig, x0, x1, Ht - head + 13, model)
        ov.text((x0 + x1) / 2, Ht - head + 3, kind, ha="center", va="bottom", fontsize=H.SZ["header"], color=INK)
    top = Ht - head
    for k, key in enumerate(["input", "ar1s", "ar8s", "sd8"]):
        y = top - s
        row_label(fig, 2, y + s / 2, "input" if key == "input" else key)
        if key == "input":
            pass
        for b, (task, *exs) in enumerate(blocks):
            for i, e in enumerate(exs):
                xj = xs[2 * b + i]
                if key == "input":
                    if task == "i2":
                        tile(fig, xj, y, s, e["input"])
                    else:
                        text_box(fig, xj, y, s, e["caption"])
                else:
                    tile(fig, xj, y, s, e["rows"][key]["png"], "ours" if key == "sd8" else "plain")
                    time_label(fig, xj + s / 2, y, key, e["rows"][key]["t"])
        top = y - (5 if key == "input" else lab + gap)
    # the first-row label covers both kinds of input
    save(fig, name, "full")


def h4b_timeline(rule="first", name="h4b_timeline"):
    """Each prompt is one row on a shared seconds axis; each output sits where it finished. Left = input / prompt."""
    P = D.all_picks(rule)
    ex = [P["i2"][0], P["i2"][1], P["i2"][3], P["i2"][4]]       # 2 icons + 2 illustrations, image-to-SVG
    ref, th = 46.0, 40.0
    x0, x1 = 62.0, W_FULL - 24
    tmax = 32.0
    leg_items = [H.item("original model + loop control, 1 sample", AR, kind="marker", marker="s", ms=5),
                 H.item("original model + loop control, best of 8", AR, kind="marker", marker="D", ms=4.6),
                 H.item("Ours, best of 8", OURS, kind="marker", marker="o", ms=5.5)]
    leg_h = H.legend_height(leg_items, H.FULL_W, x_pt=x0 - 6, width_pt=x1 - x0 + 10)
    row_h, axis_h = ref + 10, 27.0
    Ht = leg_h + 4 + len(ex) * row_h + axis_h
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    H.legend(fig, leg_items, x_pt=x0 - 6, top_pt=0, width_pt=x1 - x0 + 10)
    ax = H.axes_pt(fig, x0, axis_h, x1 - x0, len(ex) * row_h)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, len(ex) * row_h)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, 33, 4))
    ax.set_xlabel("seconds per prompt")
    X = lambda t: x0 + t / tmax * (x1 - x0)                                        # noqa: E731
    mk = {"ar1s": ("s", 5.0, AR), "ar8s": ("D", 4.6, AR), "sd8": ("o", 5.5, OURS)}
    for i, e in enumerate(ex):
        yb = (len(ex) - 1 - i) * row_h + 6          # track height inside the axes (data units = pt)
        ytr = yb
        ln, = ax.plot([0, tmax], [ytr, ytr], color="#DADADA", lw=0.8, zorder=1)
        ln.set_gid("ref")
        y_fig = axis_h + yb
        if e["task"] == "i2":
            tile(fig, 4, y_fig + 2, ref, e["input"])
        else:
            text_box(fig, 4, y_fig + 2, ref, e["caption"], fontsize=7.0)
        # thumbnails above the track, nudged right when two finish close together, with a leader to the exact time
        order = sorted(D.ROWS, key=lambda k: e["rows"][k]["t"])
        last_right = -1e9
        for k in order:
            t = e["rows"][k]["t"]
            xc = X(t)
            left = max(xc - th / 2, last_right + 3)
            if left + th > W_FULL - 2:
                left = W_FULL - 2 - th
            tile(fig, left, y_fig + 7, th, e["rows"][k]["png"], "ours" if k == "sd8" else "plain")
            last_right = left + th
            m, ms, c = mk[k]
            ax.plot([t], [ytr], marker=m, ms=ms, color=c, mec=c, ls="none", zorder=3)
            if abs(left + th / 2 - xc) > 1.5:
                ld, = ax.plot([t, (left + th / 2 - x0) / (x1 - x0) * tmax], [ytr + 2.5, ytr + 6.5],
                              color=c, lw=0.6, zorder=2)
                ld.set_gid("ref")
    save(fig, name, "full")


def h4c_cands(name="h4c_cands"):
    """The mechanism on one prompt: the original model's one sample next to all 8 of our candidates with their
    rendered CLIP score; the kept one is framed. (Second rendered prompt of is not used, see README.)"""
    c = D.candidates_t2()[0]
    g, gsep = 3.0, 12.0
    s = (W_FULL - 2 - 8 * g - gsep) / 9
    prompt_h, head, lab = 13.0, 13.0, 11.0
    Ht = prompt_h + head + s + lab + 1
    fig = plt.figure(figsize=(H.FULL_W, Ht / 72))
    ov = H._overlay(fig)
    ov.text(1, Ht - 1, f"“{c['caption']}”", ha="left", va="top", fontsize=H.SZ["legend"], style="italic")
    y = Ht - prompt_h - head - s
    xs = [1.0]
    x = 1.0 + s + gsep
    for k in range(8):
        xs.append(x)
        x += s + g
    group_header(fig, xs[0], xs[0] + s, y + s + 1.5, "original model", fontsize=8.0)
    tile(fig, xs[0], y, s, c["ar1s"]["png"])
    ov.text(xs[0] + s / 2, y - 1, f"{c['ar1s']['t']:.1f} s", ha="center", va="top", fontsize=TIME_FS, color=MUTED)
    group_header(fig, xs[1], xs[8] + s, y + s + 1.5, f"Ours, 8 candidates in {c['t_sd8']:.1f} s, rendered CLIP score",
                 color=OURS, bold=True, fontsize=8.0)
    for k, cd in enumerate(c["cands"]):
        tile(fig, xs[k + 1], y, s, cd["png"], "ours" if cd["chosen"] else "plain")
        lab_s = f"{cd['clip']:.3f}" if cd["valid"] else "invalid"
        ov.text(xs[k + 1] + s / 2, y - 1, lab_s, ha="center", va="top", fontsize=TIME_FS,
                color=OURS if cd["chosen"] else MUTED, fontweight="bold" if cd["chosen"] else "normal")
    save(fig, name, "full")


CANDIDATES = {
    "h4a_grid": lambda: h4a_grid("first", "h4a_grid"),
    "h4a_grid_any": lambda: h4a_grid("any", "h4a_grid_any"),
    "h4b_timeline": lambda: h4b_timeline("first"),
    "h4c_cands": h4c_cands,
    "h4d_mixed": lambda: h4d_mixed("first"),
    "h4e_single": lambda: h4e_single("first"),
    "h4f_grid_bars": lambda: h4a_grid("first", "h4f_grid_bars", bars=True),
    "h4g_grid8_bars": lambda: h4a_grid("first", "h4g_grid8_bars", bars=True, wide=True),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CANDIDATES)
    for n in names:
        CANDIDATES[n]()
    pngs = [str(HERE / f"{n}.png") for n in CANDIDATES if (HERE / f"{n}.png").exists()]
    H.sheet(pngs, str(HERE / "_sheet.png"), cols=1, title="H4 qualitative, round 1 (11-05)")
