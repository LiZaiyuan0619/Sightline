"""The figure-text pass (11-10, user-approved): the words printed in a finished figure, made to follow the wording guide.

  multiples after the number   x1.27 -> 1.27x, x2.6-3.9 -> 2.6-3.9x (docs/writing/01-wording.md section 1: "1.19x")
  no "baseline"                "lead over the best baseline" -> "our lead" (section 2: the fastest published decoder)
  American spelling            judgements -> judgments (section 1)

Used by a NEW round of a figure: the round imports the previous round's code, calls redirect_save(house, HERE), and
draws again. Just before house.save the texts are rewritten (tick labels through a fixed formatter, so a redraw keeps
them) and the file goes into the new round's directory; house.check then runs on the rewritten figure as usual.
A "x" not followed by a digit is a unit sign ("seconds, x original's best of 8") and stays.
"""
import re
from pathlib import Path

from matplotlib.text import Text

MULT = re.compile(r"×(\d+(?:\.\d+)?)(?:([–-])(\d+(?:\.\d+)?))?")
WORDS = [("lead over the best baseline", "our lead"), ("judgements", "judgments"), ("Judgements", "Judgments")]


def fix_string(s):
    out = MULT.sub(lambda m: m.group(1) + ((m.group(2) + m.group(3)) if m.group(2) else "") + "×", s)
    for a, b in WORDS:
        out = out.replace(a, b)
    return out


def fix_figure(fig):
    """Rewrite every text of the figure; returns [(old, new)]."""
    fig.canvas.draw()
    changed = []
    for ax in fig.axes:                  # the legend overlay is axis("off"), but the draw above builds its tick labels,
        if ax.get_label() == "_house_overlay":     # which house.check would then read as text off the canvas
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    for ax in fig.axes:
        for get_ticks, get_labels, set_ticks in ((ax.get_xticks, ax.get_xticklabels, ax.set_xticks),
                                                 (ax.get_yticks, ax.get_yticklabels, ax.set_yticks)):
            labels = get_labels()
            texts = [t.get_text() for t in labels]
            if not any(MULT.search(t) for t in texts):
                continue
            ticks = list(get_ticks())
            assert len(ticks) == len(texts), (ax, ticks, texts)
            new = [fix_string(t) for t in texts]
            props = dict(fontsize=labels[0].get_fontsize(), color=labels[0].get_color())
            set_ticks(ticks, new, **props)
            changed += [(o, n) for o, n in zip(texts, new) if o != n]
    for t in fig.findobj(Text):
        s = t.get_text()
        n = fix_string(s)
        if n != s:
            t.set_text(n)
            changed.append((s, n))
    return changed


def redirect_save(house, out_dir, log):
    """Wrap house.save: fix the texts, then save under out_dir with the same file name. Returns the original."""
    orig = house.save

    def save(fig, out_base, **kw):
        log.extend(fix_figure(fig))
        return orig(fig, str(Path(out_dir) / Path(out_base).name), **kw)

    house.save = save
    return orig
