"""Tidy data pack for figure batch 4 -- D1 (recipe ladder), D2 (why block 32), D3 (tree width and budget).

Same contract as figdata_b3.py: the figure session draws, it does not re-derive numbers (guide SS7.5 / SS8); every
block asserts a number that is already published, so a wrong read fails here rather than on the canvas.

Caliber (decision): mechanism and ablation figures use the HEALTHY segment (rounds that start before the row's
loop onset). Every arm below is split the same way as runs/2026-11-02/caliber/healthy_vs_all.py does it, from the
same result rows; tau_all is kept beside it so the two calibers can be shown together.

  d1_ladder   block 16 / 32 x chain / tree, one 2x2 per block, from the runs table 1 itself reports where possible.
              The block-32 chain arm is a decode-time variant of the SAME head as ours (asserted), so the four arms
              differ in exactly two binary factors.
  d2_block    block 16 / 32 / 64 at the same recipe and epoch budget (r1027/b64-eval), plus the draft-depth sweep
              (2 / 5 / 10 layers, runs/2026-10-31/cap_scale.md).
  d3_tree     tree width k at budget 127 and the budget curve at k=16 (runs/2026-10-29/width.md), where the budget's
              nodes go and what they buy (runs/2026-10-31/analysis/tree_budget/tree_budget.json), and nodes spent
              per accepted token for every method (runs/2026-10-30/analysis/profile_*.json).

Writes runs/2026-11-04/figdata_b4/{d1_ladder,d2_block,d3_tree}.json.
Run: python3 runs/2026-11-04/figdata_b4.py
"""
import glob
import json
import re
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[2]
U = RUNS
OUT = Path(__file__).resolve().parent / "figdata_b4"
OUT.mkdir(exist_ok=True)

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figdata_b3 import md_tables  # noqa: E402


# ------------------------------------------------------------------------------------------------ raw rows
def rows(root, design, mode="greedy", sets=("bench",)):
    out = []
    for f in sorted(glob.glob(str(U / root / design / "shard-*/results.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["mode"] == mode and (sets is None or r.get("set") in sets):
                out.append(r)
    return out


def head_of(root, design):
    """The drafter checkpoint a run used -- the ladder's factors are only clean if two arms share it."""
    p = sorted(glob.glob(str(U / root / design / "shard-*/protocol.json")))[0]
    return json.loads(Path(p).read_text())["drafter"]


def split_tau(rs):
    """-> (tau_all, tau_healthy, tau_loop, loop_token_share, rows, loop_rows). A round is loop-free if it starts
    before the row's loop onset (runs/2026-11-02/caliber/healthy_vs_all.py)."""
    r_cnt = [0, 0, 0]      # all, loop-free, loop: rounds
    e_cnt = [0, 0, 0]      # all, loop-free, loop: emitted tokens
    n_loop = 0
    for r in rs:
        on = r.get("onset")
        n_loop += on is not None
        pos = 0
        for a in r["acceptance"]:
            k = 1 if (on is None or pos < on) else 2
            for j in (0, k):
                r_cnt[j] += 1
                e_cnt[j] += a
            pos += a
    tau = [e_cnt[j] / r_cnt[j] if r_cnt[j] else None for j in range(3)]
    return dict(tau_all=tau[0], tau_healthy=tau[1], tau_loop=tau[2],
                loop_token_share=e_cnt[2] / e_cnt[0] if e_cnt[0] else 0.0,
                rows=len(rs), loop_rows=n_loop, rounds=r_cnt[0])


def speed(rs):
    """Wall clock per token against the same rows' AR, and ms per verification round."""
    t = sum(r["t_spec"] for r in rs)
    n = sum(r["n_spec"] for r in rs)
    t_ar = sum(r["t_ar"] for r in rs)
    n_ar = sum(r["n_ar"] for r in rs)
    rounds = sum(len(r["acceptance"]) for r in rs)
    return dict(speedup=(t_ar / n_ar) / (t / n), ms_round=1000 * t / rounds,
                ar_tok_s=n_ar / t_ar, ar_source=rs[0].get("ar_source"))


def arm(root, design, mode="greedy", sets=("bench",)):
    rs = rows(root, design, mode, sets)
    assert rs, (root, design, mode)
    return dict(root=root, design=design, mode=mode, head=head_of(root, design).split("/")[-3],
                ckpt=head_of(root, design).split("/")[-1], **split_tau(rs), **speed(rs))


# ------------------------------------------------------------------------------------------------ D1
# The four recipe steps. b16 chain is DFlash's released recipe, b16 + tree is DDTree's, b32 + tree is ours;
# b32 chain is ours' own head decoded as a chain, so the 2 x 2 changes exactly two things.
D1_SRC = {
    "OmniSVG-8B, text-to-SVG": {
        ("16", "chain"): ("tableA/t2-own/b16", "D0-chain"),
        ("16", "tree"): ("tableA/t2-own/b16", "D1-top16-N127"),
        ("32", "chain"): ("r1022/budget/ours", "D0-chain"),
        ("32", "tree"): ("tableA/t2-own/ours", "D1-top16-N127"),
    },
    # InternSVG: table 1's own b16 rows read AR from a cache (no inline timing), so the ladder uses the matching
    # timed re-runs of the same two heads; the published ladder was built the same way.
    "InternSVG-8B, text-to-SVG": {
        ("16", "chain"): ("isvg-eval/b16-300k-2ep-time", "D0-chain"),
        ("16", "tree"): ("isvg-eval/b16-300k-2ep-time", "D1-top16-N127"),
        ("32", "chain"): ("isvg-eval/main-step37500-time", "D0-chain"),
        ("32", "tree"): ("tableA/isvg-own/ours", "D1-top16-N127"),
    },
    "OmniSVG-8B, image-to-SVG": {
        ("16", "chain"): ("tableA/i2-own-b127/b16", "D0-chain"),
        ("16", "tree"): ("tableA/i2-own-b127/b16", "D1-top16-N127"),
        ("32", "chain"): ("h3a-eval/ft-b32-m3warm-100k-3ep/step18750", "D0-chain"),
        ("32", "tree"): ("tableA/i2-own-b127/ours", "D1-top16-N127"),
    },
}


def d1_ladder():
    out = {}
    for block, arms in D1_SRC.items():
        cells = {}
        for (blk, draft), (root, design) in arms.items():
            cells[f"{blk}-{draft}"] = arm(root, design)
        # the two arms per block length must share a head, and so must the two tree / chain arms per length
        assert cells["16-chain"]["head"] == cells["16-tree"]["head"], block
        assert cells["32-chain"]["head"] == cells["32-tree"]["head"], (block, cells["32-chain"]["head"],
                                                                      cells["32-tree"]["head"])
        assert cells["32-chain"]["ckpt"] == cells["32-tree"]["ckpt"], block
        out[block] = cells
    # Node hygiene. The four arms did not all run on the same node (their AR runs at tok/s), so the raw
    # speed-up column is NOT comparable across arms. Two node-free readings are, and both are stored:
    #   tokens per round            tokens accepted per verification round;
    #   tok_per_s      tokens per round / (ms per round) -- the decoder's own token rate, free of the target's node speed.
    # The published ladder (docs/paper/10-20-main-table.md: tree x1.16 / 1.29 / 1.16, block 32 x1.23 / 1.22 / 1.15,
    # both x1.48 / 1.71 / 1.42) is reproduced by tok_per_s on the two text blocks; the image block's 32-chain run
    # landed on a slow node (31.6 ms per round against 23.9 for a BIGGER tree), so its cost side is flagged and the
    # figure uses tokens per round there. This is the "re-rank the ladder to the standard caliber" item open since.
    for block, c in out.items():
        for k, a in c.items():
            a["tok_per_s"] = 1000 * a["tau_all"] / a["ms_round"]
            a["tok_per_s_healthy"] = 1000 * a["tau_healthy"] / a["ms_round"]
    # Only the OmniSVG text ladder was published from one node throughout; the doc itself flags InternSVG's
    # 32-chain cell as "the same-node ratio from another machine", so it is not asserted against.
    want = {"OmniSVG-8B, text-to-SVG": (1.16, 1.23, 1.48)}
    for block, (tree, blk, both) in want.items():
        c = out[block]
        got = (c["16-tree"]["tok_per_s"] / c["16-chain"]["tok_per_s"],
               c["32-chain"]["tok_per_s"] / c["16-chain"]["tok_per_s"],
               c["32-tree"]["tok_per_s"] / c["16-chain"]["tok_per_s"])
        for g, w, name in zip(got, (tree, blk, both), ("tree", "block 32", "both")):
            assert abs(g - w) < 0.07, (block, name, round(g, 3), w)
    i2 = out["OmniSVG-8B, image-to-SVG"]
    assert i2["32-chain"]["ms_round"] > i2["32-tree"]["ms_round"], "the image 32-chain node flag no longer applies"
    i2["32-chain"]["cost_contaminated"] = True
    # both factors help the loop-free content in every block, and together they beat either alone
    for block, c in out.items():
        base = c["16-chain"]["tau_healthy"]
        g = {k: c[k]["tau_healthy"] / base for k in c}
        assert g["16-tree"] > 1.05 and g["32-chain"] > 1.05, (block, g)
        assert g["32-tree"] > max(g["16-tree"], g["32-chain"]) + 0.1, (block, g)
    return dict(meta={
        "what": "the recipe as a 2 x 2: draft block length 16 / 32, chain / top-16 tree at budget 127",
        "caliber": "300 MMSVG-Bench prompts, greedy; tau_all and tau_healthy both reported (healthy = rounds that "
                   "start before the row's loop onset, as in runs/2026-11-02/caliber/healthy_vs_all.py)",
        "speedup": "wall clock per token against the same rows' AR; 16-chain is DFlash's released recipe, "
                   "16-tree is DDTree's, 32-tree is ours, 32-chain is our own head decoded as a chain",
        "published": "docs/paper/10-20-main-table.md (gains asserted here)"}, blocks=out)


# ------------------------------------------------------------------------------------------------ D2
D2_SRC = {("16", "6ep", "tree"): ("r1027/b64-eval/b16-6ep", "D1-top16-N127"),
          ("32", "3ep", "chain"): ("r1027/b64-eval/b32-3ep", "D0-chain"),
          ("32", "3ep", "tree"): ("r1027/b64-eval/b32-3ep", "D1-top16-N127"),
          ("32", "3ep", "tree255"): ("r1027/b64-eval/b32-3ep", "D1-top16-N255"),
          ("32", "6ep", "tree"): ("r1027/b64-eval/b32-6ep", "D1-top16-N127"),
          ("64", "3ep", "chain"): ("r1027/b64-eval/b64-3ep", "D0-chain"),
          ("64", "3ep", "tree"): ("r1027/b64-eval/b64-3ep", "D1-top16-N127"),
          ("64", "3ep", "tree255"): ("r1027/b64-eval/b64-3ep", "D1-top16-N255")}


def d2_block():
    cells = {}
    for (blk, ep, design), (root, d) in D2_SRC.items():
        for mode in ("greedy", "sample"):
            try:
                cells[f"{blk}|{ep}|{design}|{mode}"] = arm(root, d, mode)
            except AssertionError:
                pass
    g32, g64 = cells["32|3ep|tree|greedy"], cells["64|3ep|tree|greedy"]
    # the rewritten conclusion (inventory SS4.3): at the same recipe and epochs, doubling the block buys all-token
    # tokens per round and wall clock, but NOT on loop-free content -- loop-free tokens per round is 12.36 at 32 and 12.24 at 64.
    assert (round(g32["tau_healthy"], 2), round(g64["tau_healthy"], 2)) == (12.36, 12.24), \
        (g32["tau_healthy"], g64["tau_healthy"])
    assert g64["tau_all"] / g32["tau_all"] > 1.04 and g64["tau_loop"] / g32["tau_loop"] > 1.25
    assert g64["speedup"] / g32["speedup"] > 1.15
    # the draft-depth sweep, as published
    depth = []
    for head, hdr, rs in md_tables(RUNS / "2026-10-31/cap_scale.md"):
        if head and head.startswith("Draft depth"):
            for r in rs:
                depth.append(dict(head=r[0], design=r[1], mode=r[2], rows=int(r[3]), tau=float(r[4]),
                                  tau_healthy=float(r[5]), speedup=float(r[6].rstrip("x×")),
                                  ms_round=float(r[7])))
    assert len(depth) == 12 and {d["head"] for d in depth} == {"depth 2", "depth 5 (ours)", "depth 10"}, depth[:2]
    return dict(meta={
        "what": "why the draft block stops at 32: block 16 / 32 / 64 at the same recipe, and draft heads 2 / 5 / 10 "
                "layers deep at the same training budget",
        "caliber": "300 MMSVG-Bench prompts; tau_healthy is the figure caliber (11-02), tau_all is table 1's",
        "block": "runs under $SIGHTLINE_RUNS/r1027/b64-eval (block 64 trained on the same 300k x 3 epochs as block 32's 3ep arm)",
        "depth": "runs/2026-10-31/cap_scale.md, one epoch each, block 32"}, cells=cells, depth=depth)


# ------------------------------------------------------------------------------------------------ D3
def d3_tree():
    width, budget = [], []
    for head, hdr, rs in md_tables(RUNS / "2026-10-29/width.md"):
        if not head or head.startswith("Tree width"):
            continue
        for r in rs:
            rec = dict(block=head, design=r[0], k=int(r[1]), nodes=int(r[2]), rows=int(r[3]), tau=float(r[4]),
                       ms_round=float(r[5]), speedup=float(r[6].rstrip("x×")))
            if rec["nodes"] == 127:
                width.append(rec)                              # the width sweep is read at the standard budget
            if rec["k"] == 16:
                budget.append(rec)                             # the budget curve is read at the chosen width
    ks = {r["k"] for r in width}
    assert ks == {2, 4, 8, 16, 32}, ks
    t2 = {r["k"]: r["tau"] for r in width if r["block"].startswith("OmniSVG-8B text")}
    assert (t2[2], t2[16], t2[32]) == (14.25, 15.95, 15.80), t2        # D3 as published: the peak is k = 16
    ns = sorted(r["nodes"] for r in budget if r["block"].startswith("OmniSVG-8B image"))
    assert ns == [15, 31, 63, 127, 255], ns                            # D4: only the image block has the curve

    # spend / accepted are counts keyed "depth,rank" (rank 8 pools ranks >= 8); fold them into the two margins
    raw = json.loads((RUNS / "2026-10-31/analysis/tree_budget/tree_budget.json").read_text())
    spend = {}
    for who, d in raw.items():
        m = {}
        for field in ("spend", "accepted"):
            by_rank, by_depth, tot = {}, {}, sum(d[field].values())
            for key, v in d[field].items():
                dep, rank = key.split(",")
                by_rank[rank] = by_rank.get(rank, 0) + v
                by_depth[dep] = by_depth.get(dep, 0) + v
            m[field] = dict(rank={k: v / tot for k, v in by_rank.items()},
                            depth={k: v / tot for k, v in by_depth.items()}, total=tot)
        m.update({k: d[k] for k in ("tau", "nodes_per_round", "accepted_per_round", "rounds", "emitted")})
        spend[who] = m
    # as published (tree_budget.md): rank 0 takes 69.3% of ours' nodes and brings 95.7% of the accepted tokens
    assert abs(spend["ours"]["spend"]["rank"]["0"] - 0.693) < 0.002, spend["ours"]["spend"]["rank"]["0"]
    assert abs(spend["ours"]["accepted"]["rank"]["0"] - 0.957) < 0.002, spend["ours"]["accepted"]["rank"]["0"]

    nodes_per_token = {}
    for tag, f in (("OmniSVG-8B, text-to-SVG", "profile_t2.json"), ("OmniSVG-4B, text-to-SVG", "profile_4b.json"),
                   ("InternSVG-8B, text-to-SVG", "profile_isvg.json"),
                   ("OmniSVG-8B, image-to-SVG", "profile_i2.json"),
                   ("InternSVG-8B, image-to-SVG", "profile_i2s.json")):
        p = RUNS / "2026-10-30/analysis" / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for mode, per_method in d.items():
            for meth, v in per_method.items():
                if isinstance(v, dict) and v.get("nodes_per_token"):
                    nodes_per_token[f"{tag}|{mode}|{meth}"] = v["nodes_per_token"]
    assert nodes_per_token, "no nodes_per_token in the profiles"
    return dict(meta={
        "what": "the tree: how wide (k), how big (node budget N), where the budget goes, and what a node buys",
        "caliber": "300 MMSVG-Bench prompts, greedy, same-job AR (runs/2026-10-29/width.md); the budget split is "
                   "healthy positions only (runs/2026-10-31/analysis/tree_budget)",
        "nodes_per_token": "runs/2026-10-30/analysis/profile_*.json, table 1's own rows"},
        width=width, budget=budget, spend=spend, nodes_per_token=nodes_per_token)


if __name__ == "__main__":
    for name, fn in (("d1_ladder", d1_ladder), ("d2_block", d2_block), ("d3_tree", d3_tree)):
        try:
            d = fn()
        except Exception as e:
            print(f"{name}: FAILED {type(e).__name__} {e}")
            continue
        (OUT / f"{name}.json").write_text(json.dumps(d, indent=1))
        print(f"{name}: ok -> {OUT / (name + '.json')}")
