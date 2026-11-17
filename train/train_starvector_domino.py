""" train_sv.py for the Domino draft (train_domino.py) -- the head the DARTree baseline needs
on the two StarVector targets. train_domino.py has the same module-level LAYERS / Features / load_samples hooks as
train_dflash.py, so this is train_sv.py with the import swapped; both trainers stay unmodified (queued jobs import
them). Features is imported from train_sv so the frozen StarVector forward (official processor after _on_white,
adapter(encoder) in place of the PAD positions) exists in exactly one place.
  python train_sv_domino.py --which 1b --task i2svg --data <train.jsonl> --mask-id 4 \
      --pad-id 49152 --draft-cfg runs/2026-11-02/domino-cfgs/starvector-1b-domino-draft-b16.json --max-length 4096 \
      --epochs 2 --global-batch 16 --micro-batch 2 --save-per-epoch --out <dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script

import train_sv as SV  # noqa: E402


def main():
    argv = sys.argv
    i = argv.index("--which")
    SV.WHICH.append(argv[i + 1])
    del argv[i:i + 2]
    cfg = Path(argv[argv.index("--draft-cfg") + 1])
    layers = json.loads(cfg.read_text())["dflash_config"]["target_layer_ids"]
    SV.LAYERS[:] = layers
    import train_domino as TD
    TD.LAYERS[:] = layers
    TD.Features, TD.load_samples = SV.Features, SV.load_samples
    print(f"[starvec-domino] {SV.WHICH[0]}: target-feature layers from {cfg.name}: {layers}", flush=True)
    TD.main()


if __name__ == "__main__":
    main()
