""" (theory probe, user-approved): decode a DFlash head with a draft window W shorter than its trained block.

Wraps eval_tree.py unchanged (same rows, seeds, tree designs, record schema); only the draft window differs:
  --window W --window-mode mask    feed W-1 mask tokens (what an inference engine does when the window is lowered: P5's knob)
  --window W --window-mode slice   draft the head's full block, keep only the first W-1 positions (only the verify depth changes)
With W = the head's own block both modes are eval_tree.py itself (the probe's smoke checks this against the D2 rows).
Each design's protocol.json gets "window" / "window_mode" / "head_block" added after the run.
  python eval_window.py --window 16 --window-mode slice <eval_tree.py arguments>
  ... --xrows <jsonl> --profile internsvg...   (cross-benchmark rows go through eval_tree_x.py)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from eval import eval_text as ET
from sightline.tree_spec import TreeDrafter, tree_spec_generate            # noqa: E402


class SliceDrafter(TreeDrafter):
    """Drafts the head's full block every round and returns only the first verify_size - 1 positions."""

    def __init__(self, head, target, full_block: int):
        super().__init__(head, target, full_block)
        self.full_block = full_block

    def draft_logprobs(self, start: int, block_first: int, verify_size: int) -> torch.Tensor:
        return super().draft_logprobs(start, block_first, self.full_block)[:verify_size - 1]


def pop_opt(argv: list[str], name: str) -> str:
    i = argv.index(name)
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


def main():
    argv = sys.argv[1:]
    W = int(pop_opt(argv, "--window"))
    mode = pop_opt(argv, "--window-mode")
    assert mode in ("mask", "slice"), mode
    out = Path(argv[argv.index("--out") + 1])
    seen = {}

    def run_design(target, head, d, prompt, params, seed, max_new, block_size, coord_tol=0, copy_long=0):
        assert d.name not in ET.PLDCFG and d.name not in ET.E3CFG, "the window override is for DFlash heads only"
        assert 1 < W <= block_size, (W, block_size)               # block_size = the head's trained block
        seen["head_block"] = block_size
        drafter = TreeDrafter(head, target, W) if mode == "mask" else SliceDrafter(head, target, block_size)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = tree_spec_generate(target, drafter, prompt, max_new, [ET.EOS], params, d,
                                 generator=gen, block_size=W, coord_tol=coord_tol, copy_long=copy_long)
        torch.cuda.synchronize()
        assert max(res.acceptance) <= W, (max(res.acceptance), W)
        return res, time.perf_counter() - t0

    ET.run_design = run_design                       # eval_tree.main resolves the name at call time
    sys.argv = [sys.argv[0]] + argv
    if "--xrows" in argv:                            # cross-benchmark rows (InternSVG etc.) via eval_tree_x,
        import eval_tree_x                           # which patches the row loader and then calls eval_tree.main
        eval_tree_x.main()
    else:
        ET.main()
    for p in out.glob("*/shard-*/protocol.json"):
        proto = json.loads(p.read_text())
        proto.update({"window": W, "window_mode": mode, "head_block": seen.get("head_block"),
                      "wrapper": "eval_window.py"})
        p.write_text(json.dumps(proto, indent=1))
    print(f"EVAL_WINDOW_DONE window={W} mode={mode} head_block={seen.get('head_block')}", flush=True)


if __name__ == "__main__":
    main()
