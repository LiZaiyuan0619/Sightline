"""Table 3 (generality) data: six SVG / Lottie models x six decoding methods x greedy / official sampling x two
subsets. Source runs/2026-11-04/ext_v4.json (harvested by runs/2026-11-04/ext_tables_v4.py; StarVector
sampling at its authors' evaluation settings, T=0.2 for 1B and T=0.7 for 8B; the T=1.0 block is appendix only).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import json
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[4]
_D = json.loads((RUNS / "2026-11-04/ext_v4.json").read_text())
FAMILIES = ["SVGen-3B", "SVGen-7B", "HiVG-3B", "OmniLottie-3B", "StarVector-1B", "StarVector-8B"]
METHODS = ["PLD", "DFlash", "EAGLE-3", "DDTree", "DARTree", "Ours"]
MODES = ["greedy", "sample"]
SUBSETS = {f: (["real", "synthetic"] if f == "OmniLottie-3B" else ["icon", "illustration"]) for f in FAMILIES}
TASK = {"SVGen-3B": "text-to-SVG", "SVGen-7B": "text-to-SVG", "HiVG-3B": "text-to-SVG",
        "OmniLottie-3B": "text-to-Lottie", "StarVector-1B": "image-to-SVG", "StarVector-8B": "image-to-SVG"}
BACKBONE = {"SVGen-3B": "Llama-3.2-3B", "SVGen-7B": "Qwen2.5-Coder-7B", "HiVG-3B": "Qwen2.5-VL-3B",
            "OmniLottie-3B": "Qwen2.5-VL-3B", "StarVector-1B": "StarCoderBase-1B", "StarVector-8B": "StarCoder2-7B"}
TOKENS = {"SVGen-3B": "SVG code", "SVGen-7B": "SVG code", "HiVG-3B": "compressed SVG tokens",
          "OmniLottie-3B": "Lottie JSON", "StarVector-1B": "SVG code", "StarVector-8B": "SVG code"}


def key(m, mode, sub):
    if mode == "sample" and sub in ("icon", "illustration", "real", "synthetic"):
        return f"{m}|{mode}|{sub}"
    return f"{m}|{mode}|{sub}"


def spd(f, m, mode, sub):
    return _D[f][key(m, mode, sub)]["speedup"]


def tau(f, m, mode, sub):
    return _D[f][key(m, mode, sub)]["tau"]


def cells():
    return [(f, mode, sub) for f in FAMILIES for mode in MODES for sub in SUBSETS[f]]


def best_base(f, mode, sub):
    return max((spd(f, m, mode, sub), m) for m in METHODS if m != "Ours")


# published: 23 of 24 cells ours first; the exception StarVector-8B sample icon, DDTree 3.41x vs ours 3.35x
_first = [spd(f, "Ours", mo, s) > best_base(f, mo, s)[0] for f, mo, s in cells()]
assert len(_first) == 24 and sum(_first) == 23, sum(_first)
assert (round(spd("StarVector-8B", "DDTree", "sample", "icon"), 2), round(spd("StarVector-8B", "Ours", "sample", "icon"), 2)) == (3.41, 3.35)
assert round(spd("HiVG-3B", "Ours", "greedy", "illustration"), 2) == 15.97
assert all(best_base(f, mo, s)[1] == "DDTree" for f, mo, s in cells())        # DDTree is the runner-up everywhere

if __name__ == "__main__":
    for f, mo, s in cells():
        b, bm = best_base(f, mo, s)
        print(f"{f:14s} {mo:6s} {s:12s} ours {spd(f, 'Ours', mo, s):5.2f} best {bm} {b:5.2f} lead {spd(f, 'Ours', mo, s) / b:.2f}")
