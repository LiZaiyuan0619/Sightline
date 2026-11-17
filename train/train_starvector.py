""" (C2): DFlash draft training on the StarVector targets (target.py) = train_dflash.py run
unchanged with --task i2svg after swapping its module-level Features / load_samples (as train_i2s.py) and
setting its LAYERS from the draft config (as train_dflash_fam.py; StarVector-1B has 24 layers -> [1, 6, 11, 16,
21], 8B 32 -> [1, 8, 15, 22, 29]):
  - load_samples: train.jsonl of build_sv_trainset.py ({input_ids, loss_mask, image_path}), cut to --max-length;
  - Features: the frozen StarVector forward with the image prefix (official processor after _on_white, adapter(encoder))
    in place of the PAD positions, hidden states of LAYERS; the OnlineDFlashModel shares the target's lm_head (tied to
    its input embeddings) and embeddings through the target's attributes.
  python train_sv.py --which 1b --task i2svg --data <train.jsonl> --mask-id 4 --pad-id 49152 \
      --draft-cfg starvector-1b-dflash-draft-b32.json --max-length 4096 --epochs 2 --global-batch 16 \
      --micro-batch 2 --save-per-epoch --out <dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from starvec import target as T  # noqa: E402

LAYERS: list[int] = []
WHICH: list[str] = []


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
    def __init__(self, task: str, target: Path | None = None):
        import torch
        assert task == "i2svg", task
        self.target = T.load(WHICH[0], torch.bfloat16)
        self.proc = T.processor(WHICH[0])
        print("[target] StarVector", WHICH[0], "layers", LAYERS, flush=True)

    def __call__(self, s: dict):
        import torch
        from PIL import Image
        with torch.inference_mode():
            pre = self.target.image_prefix(self.proc(Image.open(s["image_path"])).cuda())
            ids = torch.tensor([s["input_ids"]], device="cuda")
            out = self.target(ids, prefix_embeds=pre, output_hidden_states=True, logits_to_keep=1)
            return torch.cat([out.hidden_states[l + 1][0] for l in LAYERS], dim=-1)


def main():
    argv = sys.argv
    i = argv.index("--which")
    WHICH.append(argv[i + 1])
    del argv[i:i + 2]
    cfg = Path(argv[argv.index("--draft-cfg") + 1])
    LAYERS[:] = json.loads(cfg.read_text())["dflash_config"]["target_layer_ids"]
    import train_dflash as TD
    TD.LAYERS[:] = LAYERS
    TD.Features, TD.load_samples = Features, load_samples
    print(f"[starvec] {WHICH[0]}: target-feature layers from {cfg.name}: {LAYERS}", flush=True)
    TD.main()


if __name__ == "__main__":
    main()
