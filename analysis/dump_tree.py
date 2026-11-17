""" dump real draft trees for the tree-visualisation figure.

Decodes a few rows normally (sightline.tree_spec.tree_spec_generate, unchanged) with build_tree wrapped so every round's
candidate tree is recorded; afterwards the accepted path is recovered by walking the tree along the tokens the round
actually emitted (per_block carries them), and every node is tagged with its SVG token class. One JSON per row:

  rounds: [{start, nodes: [{i, parent, depth, rank, token, text, cls, accepted}], n_accepted, emitted, bonus}]

so a figure can draw the tree, highlight the accepted path and colour the rejection point by token class.

  python analysis/dump_tree.py --head <dir> --gidx 0 --rounds 6 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "src/h2", "src/t7"):
    sys.path.insert(0, str(ROOT / p))


def cls_of(t: int) -> str:
    from sightline.svgtok.types import COORD_END, COORD_START
    if COORD_START <= t <= COORD_END:
        return "coord"
    if 196_900 <= t <= 196_999:
        return "special"
    if t >= 192_072:
        return "color"
    return "cmd"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path,
                    default=Path(f"{RUNS}/b3-train/"
                                 "omnisvg-dflash-main300k-b32-6ep-1012/heads/step150000"))
    ap.add_argument("--design", default="D1-top16-N127")
    ap.add_argument("--gidx", default="0,150")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from sightline import design as sim
    from dflash.model import DFlashDraftModel
    from eval_b3 import EOS, MODEL, chain_params, load_rows
    from sightline import tree_spec as TS
    from sightline.tree_spec import TreeDrafter, tree_spec_generate
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.enable_cudnn_sdp(False)
    target = AutoModelForCausalLM.from_pretrained(MODEL / "OmniSVG1.1_8B-qwen2-text", dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").cuda().eval()
    head = DFlashDraftModel.from_pretrained(a.head, dtype=torch.bfloat16).cuda().eval()
    design = sim.pick_designs(a.design)[0]
    want = {int(x) for x in a.gidx.split(",")}
    rows = [r for r in load_rows(["bench"]) if r["gidx"] in want]
    a.out.mkdir(parents=True, exist_ok=True)

    trees: list = []
    orig = TS.build_tree

    def spy(*args, **kw):
        nodes = orig(*args, **kw)
        trees.append(nodes)
        return nodes

    TS.build_tree = spy
    try:
        for r in rows:
            trees.clear()
            res = tree_spec_generate(target, TreeDrafter(head, target, int(head.block_size)),
                                     r["prompt_input_ids"], a.max_new, [EOS],
                                     chain_params("greedy", r["subtype"]), design,
                                     block_size=int(head.block_size))
            recs = []
            for k, blk in enumerate(res.per_block[:a.rounds]):
                nodes = trees[k]
                kids: dict = {}
                for i, n in enumerate(nodes):
                    kids.setdefault(n.parent, []).append(i)
                path, cur = [], -1
                for t in blk["emitted"][:blk["n_accepted"]]:
                    nxt = next((c for c in kids.get(cur, []) if nodes[c].token == t), None)
                    if nxt is None:
                        break
                    path.append(nxt)
                    cur = nxt
                acc = set(path)
                recs.append({
                    "start": blk["start"], "n_accepted": blk["n_accepted"], "emitted": blk["emitted"],
                    "bonus": blk["emitted"][blk["n_accepted"]] if len(blk["emitted"]) > blk["n_accepted"] else None,
                    "nodes": [{"i": i, "parent": n.parent, "depth": n.depth, "rank": n.rank, "token": int(n.token),
                               "cls": cls_of(int(n.token)), "accepted": i in acc} for i, n in enumerate(nodes)]})
            f = a.out / f"tree-gidx{r['gidx']}.json"
            f.write_text(json.dumps({"gidx": r["gidx"], "subtype": r["subtype"], "design": a.design,
                                     "block_size": int(head.block_size), "rounds": recs}, indent=1))
            print("wrote", f, "rounds", len(recs), "accepted", [x["n_accepted"] for x in recs], flush=True)
    finally:
        TS.build_tree = orig
    print("DUMP_TREE_DONE", a.out)


if __name__ == "__main__":
    main()
