"""Q3 data: which rule should pick one of eight candidates? Quality of the picked candidate minus a random pick, paired
per prompt, 95% bootstrap CI.

  OmniSVG-8B text-to-SVG    900 prompts x 8 strict-SD samples   runs/2026-10-15-harvest/s2/consensus_t2_report.txt
  OmniSVG-8B image-to-SVG   900 prompts x 8 strict-SD samples   runs/2026-10-15-harvest/s2/consensus_i2_report.txt
  InternSVG-8B text-to-SVG  300 prompts x 8 AR samples, T = 0.3 / 0.6 / 1.0   runs/2026-10-16-harvest/isvg-select-T*/report.md

The render score picks by CLIP (text) or DINO (image), so that metric is not independent for that rule; the rules are
compared on the other metrics (aesthetic, HPS; SSIM, LPIPS, MSE). LPIPS and MSE are lower-is-better: their sign is
flipped here so that + always means a better pick.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import re
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[4]
H_ = RUNS / "2026-10-15-harvest/s2"
I_ = RUNS / "2026-10-16-harvest"
RULES = ["max likelihood", "min likelihood", "text consensus", "render consensus", "render score"]
LOWER_BETTER = {"lpips", "mse"}
SEL = {"OmniSVG-8B, text-to-SVG": "clip", "OmniSVG-8B, image-to-SVG": "dino", "InternSVG-8B, text-to-SVG": "clip"}


def _txt(path):
    out = {}
    num = r"([+-]?\d+\.\d+)"
    for line in path.read_text().splitlines():
        m = re.match(r"\s+(max likelihood|min likelihood|text consensus|render consensus|render score \(\w+\)|max \w+ \(render\))\s+(.*)", line)
        if not m:
            continue
        rule = m.group(1)
        rule = "render score" if rule.startswith(("render score", "max clip", "max dino")) else rule
        for part in m.group(2).split("|"):
            mm = re.match(rf"\s*(\w+) {num} \({num} \[{num}, {num}\]\)", part)
            if mm:
                met = mm.group(1)
                out.setdefault(rule, {})[met] = tuple(float(mm.group(k)) for k in (3, 4, 5))
    return out


def _md(path):
    out = {}
    num = r"([+-]?\d+\.\d+)"
    name = {"max likelihood": "max likelihood", "min likelihood": "min likelihood", "text consensus (MBR)": "text consensus",
            "render score (max CLIP)": "render score"}
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|" + rf"\s*{num} \[{num}, {num}\]\s*\|" * 3, line)
        if m and m.group(1) in name:
            g = [float(x) for x in m.groups()[1:]]
            out[name[m.group(1)]] = {"clip": tuple(g[0:3]), "aesthetic": tuple(g[3:6]), "hps": tuple(g[6:9])}
    return out


D = {"OmniSVG-8B, text-to-SVG": _txt(H_ / "consensus_t2_report.txt"),
     "OmniSVG-8B, image-to-SVG": _txt(H_ / "consensus_i2_report.txt")}
for T in ("0.3", "0.6", "1.0"):
    D[f"InternSVG-8B, text-to-SVG, T={T}"] = _md(I_ / f"isvg-select-T{T}" / "report.md")
# min likelihood for the OmniSVG blocks lives in the likelihood reports
for blk, f in (("OmniSVG-8B, text-to-SVG", "lik_t2_report.txt"), ("OmniSVG-8B, image-to-SVG", "lik_i2_report.txt")):
    D[blk]["min likelihood"] = _txt(H_ / f)["min likelihood"]


def delta(blk, rule, met):
    """(delta, lo, hi) with + = better."""
    v = D[blk].get(rule, {}).get(met)
    if v is None:
        return None
    if met in LOWER_BETTER:
        return (-v[0], -v[2], -v[1])
    return v


def independent(blk):
    sel = "dino" if "image" in blk else "clip"
    mets = list(next(iter(D[blk].values())).keys())
    return [m for m in mets if m != sel]


# published (STATUS SS1.4): OmniSVG text max likelihood -0.0064 / -0.046 / -0.0028; render +0.0287 / +0.022 / +0.0066;
# text consensus +0.0022 / +0.057 / +0.0020; image render +0.048 / +0.036 / -0.086 / -0.027; InternSVG T0.6 max lik -0.0174
t2 = "OmniSVG-8B, text-to-SVG"
assert [round(D[t2]["max likelihood"][m][0], 4) for m in ("clip", "hps")] == [-0.0064, -0.0028]
assert round(D[t2]["render score"]["clip"][0], 4) == 0.0287 and round(D[t2]["text consensus"]["aesthetic"][0], 3) == 0.057
assert round(D["OmniSVG-8B, image-to-SVG"]["render score"]["lpips"][0], 4) == -0.0862
assert round(D["InternSVG-8B, text-to-SVG, T=0.6"]["max likelihood"]["clip"][0], 4) == -0.0174
assert set(D["InternSVG-8B, text-to-SVG, T=0.6"]) == {"max likelihood", "min likelihood", "text consensus", "render score"}

if __name__ == "__main__":
    for blk in D:
        print(blk, "independent:", independent(blk))
        for r in RULES:
            if r in D[blk]:
                print("   ", f"{r:16s}", {m: round(delta(blk, r, m)[0], 4) for m in D[blk][r]})
