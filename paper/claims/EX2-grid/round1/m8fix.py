"""Figure 6 (two sources of the lead) with the InternSVG-8B text-to-SVG block on table 1's heads.

M8 reads runs/2026-11-02/caliber/healthy_vs_all.md, which takes its rows from analysis/profiles_all.B; for the
InternSVG text block that dict points DFlash / DDTree / EAGLE-3 / Ours at the heads before the continued training
(B8's source_audit, ). B8 fixed M1 and M2 only. The same split (a round is loop-free if it starts before the row's
loop onset) is stored, with round counts, in B8's m2_healthy.json, so tokens per round inside loops follows exactly:
  tau_loop = (rounds_all * tau_all - rounds_healthy * tau_healthy) / (rounds_all - rounds_healthy).
Checks (asserts): from the unpatched file and from the four untouched blocks, the derived loop-free / loop tokens per round of DDTree and
ours reproduce every value printed in healthy_vs_all.md (2 decimals).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import json
import re
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
NEW = json.loads((RUNS / "2026-11-06/chores/b8/figdata/m2_healthy.json").read_text())["blocks"]
OLD = json.loads((RUNS / "2026-11-06/chores/b8/figdata/_unpatched/m2_healthy.json").read_text())["blocks"]
NAMES = {"OmniSVG-8B, text-to-SVG": "OmniSVG-8B text", "OmniSVG-4B, text-to-SVG": "OmniSVG-4B text",
         "InternSVG-8B, text-to-SVG": "InternSVG-8B text", "OmniSVG-8B, image-to-SVG": "OmniSVG-8B image",
         "InternSVG-8B, image-to-SVG": "InternSVG-8B image"}
METH = ("DDTree", "Ours")


def split(entry):
    h, a = entry["healthy"], entry["all"]
    loop_r = a["rounds"] - h["rounds"]
    loop_t = (a["rounds"] * a["tau"] - h["rounds"] * h["tau"]) / loop_r
    return dict(healthy=h["tau"], loop=loop_t, all=a["tau"])


def printed():
    out = {}
    for line in (RUNS / "2026-11-02/caliber/healthy_vs_all.md").read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 11 and c[2] in METH and c[6][:1].isdigit():
            out[(c[0], c[1], c[2])] = dict(all=float(c[6]), healthy=float(c[7]), loop=float(c[8]))
    return out


PRINTED = printed()
for blk, b in OLD.items():
    for mode in ("greedy", "sample"):
        for m in METH:
            got, want = split(b[mode][m]), PRINTED[(NAMES[blk], mode, m)]
            for k in ("all", "healthy", "loop"):
                assert abs(round(got[k], 2) - want[k]) < 0.0051, (blk, mode, m, k, got[k], want[k])
for blk in NEW:
    if blk != "InternSVG-8B, text-to-SVG":
        assert NEW[blk] == OLD[blk], blk                      # B8 changed the InternSVG text block only

TAU = {(NAMES[blk], mode, m): split(NEW[blk][mode][m]) for blk in NEW for mode in ("greedy", "sample") for m in METH}
TAU_OLD = {(NAMES[blk], mode, m): split(OLD[blk][mode][m]) for blk in OLD for mode in ("greedy", "sample") for m in METH}

if __name__ == "__main__":
    print("every printed DDTree / ours value of healthy_vs_all.md reproduced from the unpatched file")
    for mode in ("greedy", "sample"):
        k = ("InternSVG-8B text", mode)
        for seg in ("healthy", "loop"):
            o = TAU_OLD[k + ("Ours",)][seg] / TAU_OLD[k + ("DDTree",)][seg]
            n = TAU[k + ("Ours",)][seg] / TAU[k + ("DDTree",)][seg]
            print(f"InternSVG-8B text {mode:6s} {seg:7s}: ours / DDTree {o:.3f} (old heads) -> {n:.3f} (table 1's heads)")
