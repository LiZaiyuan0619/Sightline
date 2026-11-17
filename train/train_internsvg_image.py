""" DFlash-family draft training on the InternSVG-8B image-to-SVG target (table 1's 5th block). Runs the OmniSVG
image-to-SVG trainers unchanged with --task i2svg after swapping the trainer module's Features / load_samples for the
InternVL versions below (both are module globals looked up at call time; the trainer files are not edited):
    dflash -> train_dflash.py (ours, block 32; DFlash / DDTree head, block 16)
    domino -> train_domino.py (Domino head of DARTree)
EAGLE-3: train_eagle3_i2s.py (its target forward is inline, so that trainer is a copy).
  - load_samples: train.jsonl of build_i2s_trainset.py ({input_ids, loss_mask, image_path}; prompt ids as
    vLLM expanded them = vlm_i2s.prompt_ids_of), cut to --max-length, read line by line;
  - Features: the frozen InternVL forward of vlm_i2s.py (one 448 tile, 1-D positions) -> hidden states of
    the trainer's LAYERS ([1, 7, 13, 19, 25]: the 28-layer LM, as the InternSVG text heads we warm-start from).
Usage (multi-GPU: torchrun through src/a batch GPU job/py_job_dist.sh, as the OmniSVG image heads; PYTHONPATH=refs/SpecForge):
  python train_i2s.py dflash --task i2svg --data <train.jsonl> --init <text head> \
      --mask-id 151662 --draft-cfg internsvg-8b-dflash-draft-b32.json --max-length 4096 --epochs 3 \
      --global-batch 16 --micro-batch 2 --save-per-epoch --out <dir>
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from targets import internsvg as V  # noqa: E402

TRAINERS = {"dflash": "train_dflash", "domino": "train_domino"}
LAYERS: list[int] = []


def load_samples(task: str, data: Path, max_length: int) -> list[dict]:
    assert task == "i2svg", task
    out = []
    with Path(data).open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append({"input_ids": r["input_ids"][:max_length], "loss_mask": r["loss_mask"][:max_length],
                            "image_path": r["image_path"]})
    return out


class Features:
    """frozen InternVL forward -> concatenated hidden states of the trainer's LAYERS"""

    def __init__(self, task: str, target: Path | None = None):
        assert task == "i2svg", task
        self.proc = V.load_processor()
        self.target, note = V.load_target()
        print("[target]", note, "layers", LAYERS, flush=True)

    def __call__(self, s: dict):
        return V.target_features(self.target, self.proc, s["input_ids"], s["image_path"], LAYERS)


def main():
    which = sys.argv.pop(1)
    mod = importlib.import_module(TRAINERS[which])
    LAYERS[:] = mod.LAYERS
    assert LAYERS == [1, 7, 13, 19, 25], LAYERS
    mod.Features, mod.load_samples = Features, load_samples
    mod.main()


if __name__ == "__main__":
    main()
