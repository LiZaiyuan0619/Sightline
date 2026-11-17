"""Q2 data: three VLM judges' pairwise preferences (runs/2026-10-18/judge-{t2front,isvgfront}/report-<judge>.md;
summary runs/2026-10-30/judge_summary.md) + seconds per prompt from the Q1 data pack (runs/2026-11-04/figdata_b3/
q1_frontier.json, same 300 prompts, same runs).

pref(A) = mean over prompts of (1 A wins, 0.5 tie, 0 B wins), both orders averaged; wins / ties / losses counted per
order (600 judgements per pair per judge); 95% bootstrap CI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
from sightline.paths import RUNS as RUNS_DIR
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
JUDGES = ["claude-opus-5-5", "gpt-6.1-sol", "gemini-3.8-flash"]
RUNS = {"OmniSVG-8B, text-to-SVG": "judge-t2front", "InternSVG-8B, text-to-SVG": "judge-isvgfront"}
_ROW = re.compile(r"^\|\s*(\S+):(\S+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*\[([\d.]+),\s*([\d.]+)\]\s*\|\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)")


def _parse(path):
    out = {}
    for line in path.read_text().splitlines():
        m = _ROW.match(line)
        if m:
            a, b = m.group(1), m.group(2)
            out[(a, b)] = dict(n=int(m.group(3)), pref=float(m.group(4)), lo=float(m.group(5)), hi=float(m.group(6)),
                               win=int(m.group(7)), tie=int(m.group(8)), loss=int(m.group(9)))
    return out


P = {blk: {j: _parse(RUNS_DIR / "2026-10-18" / run / f"report-{j}.md") for j in JUDGES} for blk, run in RUNS.items()}
_q1 = json.loads((RUNS_DIR / "2026-11-04/figdata_b3/q1_frontier.json").read_text())["rows"]
SECONDS = {(r["block"], r["arm"]): r["seconds"] for r in _q1}


def pref(blk, a, b, judge=None):
    if judge:
        return P[blk][judge][(a, b)]["pref"]
    return sum(P[blk][j][(a, b)]["pref"] for j in JUDGES) / len(JUDGES)


def counts(blk, a, b):
    """wins / ties / losses pooled over the three judges (each 600 judgements)."""
    return tuple(sum(P[blk][j][(a, b)][k] for j in JUDGES) for k in ("win", "tie", "loss"))


# the published means (runs/2026-10-30/judge_summary.md; STATUS SS1.5)
assert round(pref("OmniSVG-8B, text-to-SVG", "sd8", "ar1s"), 3) == 0.691
assert round(pref("OmniSVG-8B, text-to-SVG", "sd8", "ar8s"), 3) == 0.491
assert round(pref("InternSVG-8B, text-to-SVG", "sd8", "ar1s"), 3) == 0.671
assert round(pref("InternSVG-8B, text-to-SVG", "sd8", "ar8s"), 3) == 0.506
assert P["OmniSVG-8B, text-to-SVG"]["claude-opus-5-5"][("sd8", "ar1s")]["win"] == 393
assert all(sum(counts(b, *k)) == 3 * 600 for b in RUNS for k in [("sd8", "ar1s"), ("sd8", "ar8s"), ("sd1s", "ar1s")])
assert round(SECONDS[("OmniSVG-8B, text-to-SVG", "sd8")], 2) == 4.71 and round(SECONDS[("OmniSVG-8B, text-to-SVG", "ar8s")], 2) == 18.23

if __name__ == "__main__":
    for b in RUNS:
        for k in sorted(P[b]["claude-opus-5-5"]):
            print(b[:12], k, [round(pref(b, *k, j), 3) for j in JUDGES], round(pref(b, *k), 3), counts(b, *k))
