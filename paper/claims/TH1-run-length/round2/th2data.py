"""Data layer for the Proposition 1 figure, round 2 (main line).

ONE dataset for every candidate (user, round 1's candidates mixed heads and slices and could not be compared):
our block-32 head (b32-6ep), its 300-prompt greedy cache, loop-free caliber (before the loop onset, as T1 and table 1's
mechanism figures), offset-1 labels at 1 candidate per position (chain) and 16 (tree). Built by build_rows.py.

Vocabulary used on the figures: a *run* is the stretch from one miss to the next (the stream cut after every miss);
a window-W decoder spends ceil(n / W) rounds on a run of n tokens, so tokens per round = total tokens / total rounds;
with an unlimited window every run takes one round (the most a round can write, the "unlimited-window" value).
That is Lemma 1 per stream; the pooled values equal T1's published identity exactly (asserted).

T1's waterfall prints a "ceiling" 1 / miss rate (96.3 at 16 candidates), which ignores the open last run of every
prompt; the exact unlimited-window value under the same accounting is total tokens / runs (82.9). Round 2 uses the
exact one everywhere and says so in the README.
"""
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import numpy as np
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = RUNS.parent            # data/, which holds runs/
HEAD = "b32-6ep"
KS = (1, 16)
WINDOWS = (16, 32, 64)
D = json.loads((HERE / f"rows_{HEAD}.json").read_text())
ROWS = [r for r in D["rows"] if all(r["k"][str(k)]["L"] > 0 for k in KS)]


def runs(r, k):
    v = r["k"][str(k)]
    return v["closed"] + v["open"]


def tau(r, k, W=None):
    v = r["k"][str(k)]
    if W is None:
        return v["L"] / v["pieces"]
    return v["L"] / v["rounds"][str(W)]


def subset(strict=False):
    return [r for r in ROWS if not (strict and r["leak"])]


def per_prompt(strict=False):
    out = {}
    for k in KS:
        R = subset(strict)
        out[k] = dict(
            unlimited=np.array([tau(r, k) for r in R]),
            tau={W: np.array([tau(r, k, W) for r in R]) for W in WINDOWS},
            share={W: np.array([tau(r, k, W) / tau(r, k) for r in R]) for W in WINDOWS},
            gain={(16, 32): np.array([tau(r, k, 32) / tau(r, k, 16) for r in R]),
                  (32, 64): np.array([tau(r, k, 64) / tau(r, k, 32) for r in R])},
            gidx=np.array([r["gidx"] for r in R]))
    return out


def pooled(strict=False):
    out = {}
    for k in KS:
        R = subset(strict)
        L = sum(r["k"][str(k)]["L"] for r in R)
        P = sum(r["k"][str(k)]["pieces"] for r in R)
        out[k] = {"unlimited": L / P}
        out[k].update({W: L / sum(r["k"][str(k)]["rounds"][str(W)] for r in R) for W in WINDOWS})
    return out


def token_ccdf(k, strict=False):
    """Share of all tokens that sit in runs at least x tokens long, as a step function over x."""
    n = np.sort(np.array([x for r in subset(strict) for x in runs(r, k)]))
    w = n / n.sum()
    tail = np.cumsum(w[::-1])[::-1]              # tail[i] = share of tokens in runs >= n[i]
    xs, ys = [1.0], [1.0]
    last = None
    for x, t in zip(n, tail):
        if x != last:
            xs.append(float(x))
            ys.append(float(t))
            last = x
    return np.array(xs), np.array(ys), int(len(n))


def share_tokens_at_least(k, x, strict=False):
    n = np.array([y for r in subset(strict) for y in runs(r, k)])
    return float(n[n >= x].sum() / n.sum())


def check():
    t1 = json.loads((ROOT / f"runs/2026-11-05/theory/stream_gaps_{HEAD}.json").read_text())
    P = pooled()
    for k in KS:
        for W in WINDOWS:
            ref = t1["A"][f"healthy/j1/k{k}"]["identity"][str(W)]
            assert abs(P[k][W] - ref) < 1e-9, (k, W, P[k][W], ref)
    # 23 of the 300 prompts loop from their first token (ref_onset = 0): no loop-free position, as in T1
    assert len(D["rows"]) == 300 and len(ROWS) == 277, len(ROWS)
    assert sum(r["leak"] for r in ROWS) == 49
    # the numbers quoted in the README and the chat
    assert abs(P[16]["unlimited"] - 82.91) < 0.01 and abs(P[1]["unlimited"] - 13.70) < 0.01
    assert abs(share_tokens_at_least(16, 32) - 0.93) < 0.005 and abs(share_tokens_at_least(1, 32) - 0.59) < 0.005
    pp = per_prompt()
    assert abs(np.median(pp[16]["share"][16]) - 0.172) < 0.0015
    assert abs(np.median(pp[16]["gain"][(16, 32)]) - 1.815) < 0.0015
    assert abs(np.median(pp[1]["gain"][(16, 32)]) - 1.344) < 0.0015
    return True


if __name__ == "__main__":
    check()
    for strict in (False, True):
        P, pp = pooled(strict), per_prompt(strict)
        print("strict" if strict else "healthy", len(subset(strict)), "prompts")
        for k in KS:
            print(f"  k={k:>2} pooled", {kk: round(v, 2) for kk, v in P[k].items()},
                  "| median share", {W: round(float(np.median(pp[k]['share'][W])), 3) for W in WINDOWS},
                  "| median gain", {f"{a}->{b}": round(float(np.median(v)), 3) for (a, b), v in pp[k]["gain"].items()},
                  "| tokens in runs >=16/32/64:",
                  [round(share_tokens_at_least(k, x, strict), 3) for x in (16, 32, 64)])
    print("check: ok")
