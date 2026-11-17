"""Data for the first experiments composite (main.pdf Figures 5, 7, 8, 9, 10), round 1 (night, main line).

Every loader reuses the signed-off figure's own data layer, unchanged, and asserts numbers that are already printed
(guide SS8: reproduce a printed number before drawing). Nothing here recomputes a result.

  landscape   main.pdf Figure 5   T2-landscape round 1 load: runs/2026-11-03/table2_refrows.json + table 2a (SArena-Icon)
  generality main.pdf Figure 7   T3-generality round 1 t3data: runs/2026-11-04/ext_v4.json
  classes     main.pdf Figure 8   M1 round 3 (round 1 profiles + the 4B recompute + B8's InternSVG text profile)
  window      main.pdf Figure 9   runs/2026-11-06/chores/b8/figdata/m2_healthy.json (loop-free rounds, table-1 rows, B8 fixed;
                                   main.pdf still prints round 2's ridges, made before the B8 fix)
  loops       main.pdf Figure 10  M5 round 1 load: runs/2026-10-30/analysis/loop_share.md
"""
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import numpy as np
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
CLAIMS = HERE.parents[1]
ROOT = CLAIMS.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Figure 5: speed and quality of SVG generators (SArena-Icon, text-to-SVG) -------------------------------------
_T2 = _load("t2r1", CLAIMS / "T2-landscape/round1/make.py")
OTHERS, OURS = _T2.load()                      # asserts Ours s = 1.01 and InternSVG AR x8 FID = 8.75
CAT_OF = _T2.CAT_OF                            # optimisation / LLM, reported / LLM, measured

# --- Figure 7: six model families ------------------------------------------------------------------------------
T3 = _load("t3data", CLAIMS / "T3-generality/round1/t3data.py")   # asserts 23 of 24, 3.41 / 3.35, 15.97, DDTree 2nd

# --- Figure 8: token classes ----------------------------------------------------------------------------------
_M1 = _load("m1r3", CLAIMS / "M1-token-classes/round3/make.py")    # swaps in B8's InternSVG text profile
M1R1, M1R2 = _M1.r1, _M1.r2
BLOCKS = list(M1R1.BLOCKS)
SHORT = {BLOCKS[0]: "OmniSVG\n8B, text", BLOCKS[1]: "OmniSVG\n4B, text", BLOCKS[2]: "InternSVG\n8B, text",
         BLOCKS[3]: "OmniSVG\n8B, image", BLOCKS[4]: "InternSVG\n8B, image"}


def classes(blk, mode="greedy", m="Ours"):
    """Per token class of one block: (class, group, share of the output, share of the draft's misses, accuracy)."""
    d = M1R1.PROF[blk][mode][m]["classes"]
    return [(k, M1R1.GROUP[k], v["share"], v["rej_share"], v["acc"]) for k, v in d.items()]


def numbers_share(blk, mode="greedy", m="Ours"):
    out = M1R1.group_shares(blk, mode, m, "share")["numbers"]
    miss = M1R1.group_shares(blk, mode, m, "rej_share")["numbers"]
    return out, miss


def wrong_rate(blk, mode="greedy", m="Ours"):
    g = M1R2.group_acc(blk, mode, m)
    return {k: 1 - v for k, v in g.items()}


# printed: m1e_bars round 3 tops 1.2 / 8, 1.2 / 9, 2 / 6, 1.4 / 10, 5 / 11; B8 README: InternSVG text 2.26% / 5.78%
_w = [wrong_rate(b) for b in BLOCKS]
assert [round(100 * w["structure"], 1) for w in _w][:2] == [1.2, 1.2], _w
assert round(100 * _w[2]["structure"], 2) == 2.26 and round(100 * _w[2]["numbers"], 2) == 5.78, _w[2]
assert [round(100 * w["numbers"]) for w in _w] == [8, 9, 6, 10, 11], [100 * w["numbers"] for w in _w]

# --- Figure 9: rounds that fill the window (loop-free rounds) -------------------------------------------------------
_M2 = json.loads((RUNS / "2026-11-06/chores/b8/figdata/m2_healthy.json").read_text())["blocks"]
WIN_METHODS = ["PLD", "REST", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]
MODES = ["greedy", "sample"]


def window(blk, mode, m):
    r = _M2[blk][mode][m]
    return dict(cap=r["cap"], fill=r["healthy"]["limit_share"], hist={int(k): v for k, v in r["healthy"]["hist"].items()},
                rounds=r["healthy"]["rounds"])


CAP = {m: window(BLOCKS[0], "greedy", m)["cap"] for m in WIN_METHODS}
assert CAP == {"PLD": 11, "REST": 17, "EAGLE-3": 13, "DFlash": 16, "DARTree": 17, "DDTree": 16, "Ours": 32}, CAP
_fill = {m: [window(b, md, m)["fill"] for b in BLOCKS for md in MODES] for m in WIN_METHODS}
# printed (M2 round 3 README, B8): ours 4-14%, the three tree / layer-serial baselines %
assert (round(100 * min(_fill["Ours"])), round(100 * max(_fill["Ours"]))) == (4, 14), _fill["Ours"]
_tri = _fill["EAGLE-3"] + _fill["DARTree"] + _fill["DDTree"]
assert (round(100 * min(_tri)), round(100 * max(_tri))) == (17, 53), (min(_tri), max(_tri))
assert all(window(b, md, "Ours")["fill"] < min(window(b, md, m)["fill"] for m in ("EAGLE-3", "DARTree", "DDTree"))
           for b in BLOCKS for md in MODES)

# --- Figure 10: loops ---------------------------------------------------------------------------------------------
_M5 = _load("m5r1", CLAIMS / "M5-loops-common/round1/make.py")
LOOP_ROWS = _M5.load()
LOOP_TARGETS = [t for t in _M5.TARGETS if "Lottie" not in t]


def loops(target, mode):
    """Shares of outputs: finished, hit the length cap outside a loop, ended inside a loop (row-weighted, as M5)."""
    eos = _M5.agg(LOOP_ROWS, target, mode, "eos")
    per = _M5.agg(LOOP_ROWS, target, mode, "periodic")
    cap = _M5.agg(LOOP_ROWS, target, mode, "capped") - per
    return dict(finished=eos, cap=cap, loop=per)


# printed (main.pdf Figure 10 right column): 26/24, 32/27, 38/15, 29/27, 25/11, 34/3, 47/25
assert [(round(100 * loops(t, "greedy")["loop"]), round(100 * loops(t, "sample")["loop"])) for t in LOOP_TARGETS] == \
    [(26, 24), (32, 27), (38, 15), (29, 27), (25, 11), (34, 3), (47, 25)]

if __name__ == "__main__":
    for b in BLOCKS:
        o, m = numbers_share(b)
        w = wrong_rate(b)
        print(f"{b:28s} numbers {100 * o:4.1f}% of output, {100 * m:4.1f}% of misses; wrong: structure "
              f"{100 * w['structure']:4.2f}%, numbers {100 * w['numbers']:4.2f}%")
    for m in WIN_METHODS:
        print(f"{m:8s} window {CAP[m]:2d}: fills {100 * min(_fill[m]):4.1f}-{100 * max(_fill[m]):4.1f}%")
