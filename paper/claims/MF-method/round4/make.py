"""Method figure (main.pdf Figure 2), round 4 (main-line session 8): round 3's v1 and v2 merged.

User on round 3: no loop-control or best-of-8 strips (too thin to add anything, and they pull attention from the
trunk); v1's real tree is preferred, since it is drawn from a real round, reads as empirical and shows the tree
narrow at first and wide later; v2's marks are clearer. Merge them: v1's real tree and windows, plus v2's marks:
  - the two accepted second candidates (positions 11 and 15) ringed on the path,
  - the two rejected first candidates (their siblings, one row under the path) in dark grey,
  (round 3 v2's faint dots for positions are left out: the light band already shows the room left, and the
  dots sat under the "21 accepted" note.)
The two marks are named in a legend in the empty top-left corner (positions 1-6 are a single file), so no leader
lines cross the tree. Data and checks as round 3 (../round3/make.py,../round1/mfdata.py).
  m4a_merged      the merged figure
  m4b_merged_tall the same with more height for the tree (nodes spaced wider)
Run: python3 paper/claims/MF-method/round4/make.py
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mf3", HERE.parent / "round3/make.py")
M3 = importlib.util.module_from_spec(spec)
sys.modules["mf3"] = M3
spec.loader.exec_module(M3)
H, MD, T, F = M3.H, M3.MD, M3.T, M3.F
INK, MUTED, OURS, NODE, FS, W = M3.INK, M3.MUTED, M3.OURS, M3.NODE, M3.FS, M3.W
ACC, BONUS = M3.ACC, M3.BONUS
REJ = "#6F6F6F"
M3.HERE = HERE                     # save here, not in round 3


def first_candidates():
    """Layout (depth, y) of the rejected first candidates: the rank-0 siblings of the accepted second candidates."""
    t = json.loads(MD.TREE.read_text())
    nodes = t["rounds"][MD.ROUND]["nodes"]
    lay = {n["i"]: y for n, (d, y, a, rk) in zip(nodes, T["nodes"])}
    out = []
    for d, rank in F["off_chain"]:
        acc = next(n for n in nodes if n["depth"] == d and n["accepted"])
        assert acc["rank"] == rank - 1 == 1
        first = next(n for n in nodes if n["depth"] == d and n["parent"] == acc["parent"] and n["rank"] == 0)
        out.append((d, lay[first["i"]]))
    assert out == [(11, -1.0), (15, -1.0)], out
    return out


def merged(name, Ht, tree_top_pad=12.0):
    fig = M3.fig_pt(Ht)
    ax = M3.draw_tree(fig, 6, 22, W - 10, Ht - 22 - tree_top_pad)
    # v2's marks on the real tree
    fc = first_candidates()
    ax.scatter([d for d, _ in fc], [y for _, y in fc], s=14, color=REJ, edgecolors="none", zorder=6)
    second = [d for d, _ in F["off_chain"]]
    ax.scatter(second, [0] * len(second), s=40, facecolors="none", edgecolors=OURS, linewidths=0.9, zorder=6)
    M3.tree_labels(fig, ax)
    items = [H.item("second candidate, accepted", OURS, kind="marker", marker="o", hollow=True, ms=5.0, mew=0.9,
                    bold=False, text_color=OURS),
             H.item("first candidate, rejected", REJ, kind="marker", marker="o", ms=3.6)]
    for k, it in enumerate(items):
        H.legend(fig, [it], x_pt=8, top_pt=tree_top_pad + 2 + k * 10.0, width_pt=120, fontsize=FS["small"])
    M3.save(fig, name)


if __name__ == "__main__":
    merged("m4a_merged", 112.0)
    merged("m4b_merged_tall", 132.0)
