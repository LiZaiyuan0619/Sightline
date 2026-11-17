"""Data layer of hero round 4 (main-line session 8): main.pdf Figure 4's numbers, and Figure 5d's for the wall.

Every candidate of this round carries the main result (user the hero takes Figure 4's data and Figure 4 leaves
the main text once a candidate is chosen; candidate C also takes Figure 5d's).
  speed-up, tokens per round   runs/2026-11-02/figdata/table1.json   (the source of Figure 4 and appendix Table A1;
                               all tokens, over the original model timed in the same job)
  rounds per window            runs/2026-11-06/chores/b8/figdata/m2_healthy.json  (the source of Figure 5d; greedy and
                               official sampling, loop-free rounds, table 1's heads)
Coverage: PLD, REST, EAGLE-3, DFlash, DARTree, DDTree and ours in all 60 settings; SpS only on OmniSVG-8B text-to-SVG
(12 settings, it needs a smaller model of the same family); ViSpec only on image-to-SVG (24 settings, it is an
image-conditioned drafter). Nothing else is left out.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import json
import statistics as st
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[4]
T1 = json.loads((RUNS / "2026-11-02/figdata/table1.json").read_text())["blocks"]
M2 = json.loads((RUNS / "2026-11-06/chores/b8/figdata/m2_healthy.json").read_text())["blocks"]

PAIRS = ["OmniSVG-8B, text-to-SVG", "OmniSVG-4B, text-to-SVG", "InternSVG-8B, text-to-SVG",
         "OmniSVG-8B, image-to-SVG", "InternSVG-8B, image-to-SVG"]
PAIR_LINES = {p: p.replace(", ", "\n") for p in PAIRS}
MODES = ["greedy", "sample"]
MODE_NAME = {"greedy": "greedy", "sample": "official sampling"}
BENCH = ["MMSVG-Icon", "MMSVG-Illus.", "MMSVG-2M-Icon", "MMSVG-2M-Illus.", "SArena-Icon", "SArena-Illus."]
# family order of table 1 and Figure 4 (training-free, layer-serial, block / tree, ours)
DECODERS = ["PLD", "REST", "SpS", "EAGLE-3", "ViSpec", "DFlash", "DARTree", "DDTree", "Ours"]
PUBLISHED = DECODERS[:-1]
KEY = {m: m for m in DECODERS} | {"Ours": "Ours (lossless)"}

_BLK = {(b["block"], b["mode"]): b for b in T1}
assert sorted(_BLK) == sorted((p, m) for p in PAIRS for m in MODES)
for b in T1:
    assert [c[0].rstrip("†") for c in b["columns"]] == BENCH, b["columns"]


def cells(pair, mode, m, what="speedup"):
    """Six values (one per benchmark, BENCH order), None where the decoder was not run."""
    return [c[what] if c else None for c in _BLK[(pair, mode)]["rows"][KEY[m]]]


def settings():
    """All 60 settings in a fixed order: pair, then decoding, then benchmark."""
    return [(p, md, j) for p in PAIRS for md in MODES for j in range(6)]


def value(p, md, j, m, what="speedup"):
    return cells(p, md, m, what)[j]


def window(p, md, m):
    r = M2[p][md][m]
    return dict(cap=r["cap"], fill=r["healthy"]["limit_share"], tau=r["healthy"]["tau"],
                hist={int(k): v for k, v in r["healthy"]["hist"].items()})


WIN_DECODERS = ["PLD", "REST", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]
CAP = {m: window(PAIRS[0], "greedy", m)["cap"] for m in WIN_DECODERS}


def pooled_hist(m, mode="greedy"):
    """Share of loop-free rounds that collect k tokens, k = 1..32, averaged over the five pairs (Figure 5d's pooling)."""
    out = [0.0] * 33
    for p in PAIRS:
        h = window(p, mode, m)["hist"]
        for k in range(1, 33):
            out[k] += h.get(k, 0.0) / len(PAIRS)
    return out


# --- checks against numbers printed in main.pdf and the logs -----------------------------------------------------
_S = settings()
assert len(_S) == 60
_lead = []
for p, md, j in _S:
    o = value(p, md, j, "Ours")
    best = max(value(p, md, j, m) for m in PUBLISHED if value(p, md, j, m) is not None)
    assert o > best, (p, md, j)                      # ours fastest in all 60 (Figure 4, story v2)
    _lead.append(o / best)
assert (round(min(_lead), 2), round(max(_lead), 2)) == (1.06, 1.63), (min(_lead), max(_lead))
_ours = [value(p, md, j, "Ours") for p, md, j in _S]
assert (round(min(_ours), 2), round(max(_ours), 2)) == (4.25, 13.83)
assert sum(value(p, md, j, "SpS") is not None for p, md, j in _S) == 12
assert sum(value(p, md, j, "ViSpec") is not None for p, md, j in _S) == 24


def pair_lead(p, md):
    """Figure 4's number above each pair: our mean over the six benchmarks / the fastest published decoder's mean."""
    om = st.mean(cells(p, md, "Ours"))
    return om / max(st.mean(cells(p, md, m)) for m in PUBLISHED if None not in cells(p, md, m))


assert [f"{pair_lead(p, 'greedy'):.2f}" for p in PAIRS] == ["1.27", "1.21", "1.48", "1.19", "1.22"]
assert [f"{pair_lead(p, 'sample'):.2f}" for p in PAIRS] == ["1.20", "1.14", "1.17", "1.20", "1.13"]
assert CAP == {"PLD": 11, "REST": 17, "EAGLE-3": 13, "DFlash": 16, "DARTree": 17, "DDTree": 16, "Ours": 32}, CAP
_fill = {m: [window(p, md, m)["fill"] for p in PAIRS for md in MODES] for m in WIN_DECODERS}
assert (round(100 * min(_fill["Ours"])), round(100 * max(_fill["Ours"]))) == (4, 14)
_tri = _fill["EAGLE-3"] + _fill["DARTree"] + _fill["DDTree"]
assert (round(100 * min(_tri)), round(100 * max(_tri))) == (17, 53)
