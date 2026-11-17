""" train_dflash_fam.py for the Domino draft (train_domino.py) -- the head the DARTree baseline
needs on each extension family. Same single job as its DFlash sibling: take the target-feature layers from the draft
config instead of train_domino's hard-coded LAYERS = [1, 7, 13, 19, 25], which is right only for 28-layer targets
(HiVG-3B and OmniLottie-3B have 36, so their configs say [1, 9, 17, 25, 33]). train_domino.py is not modified (queued
jobs import it); all its flags pass through. Text-style targets only (--task t2svg); StarVector's VLM targets go
through train_sv_domino.py.
  torchrun... train_domino_fam.py --task t2svg --draft-cfg runs/2026-11-02/domino-cfgs/<fam>-domino-draft-b16.json...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script


def main():
    import train_domino as T
    argv = sys.argv[1:]
    cfg = Path(argv[argv.index("--draft-cfg") + 1])
    layers = json.loads(cfg.read_text())["dflash_config"]["target_layer_ids"]
    T.LAYERS[:] = layers
    print(f"[fam] target-feature layers from {cfg.name}: {T.LAYERS}", flush=True)
    T.main()


if __name__ == "__main__":
    main()
