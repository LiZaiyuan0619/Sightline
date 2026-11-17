""" baselines/train_eagle3.py for the extension families (the EAGLE-3 baseline row of the generality table).

train_eagle3.py hard-codes AUX = [1, 13, 24], SpecForge's default [1, L/2 - 1, L - 4] for the 28-layer OmniSVG-8B /
InternSVG-8B targets. HiVG-3B and OmniLottie-3B have 36 layers, so that triple is simply wrong for them (it would feed
the draft three early-to-middle layers and never the one SpecForge means by "L - 4"). This wrapper reads the target's
own layer count and applies the same formula; everything else -- the online objective, the d2t / t2d vocabulary built
from the data, the optimizer, the multi-GPU scheme -- is train_eagle3.main unchanged (that file is not edited:
queued jobs import it). Text-style targets only; StarVector's VLM targets need a feature extractor of their own.

Measured  (runs/2026-11-02/e3_vocab_scan.md): the four families use 1,752-6,954 distinct loss-bearing tokens, so
--draft-vocab 16000 covers 1.00000 of each (OmniSVG needed 64k, InternSVG 16k).

  the glance environment/bin/torchrun... train_eagle3_fam.py --target <model dir> --data <train.jsonl> \
      --draft-vocab 16000 --epochs 2 --out <dir>
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script


def aux_layers(n_layers: int) -> list[int]:
    """SpecForge's default triple: the second layer, the middle, and four from the top."""
    return [1, n_layers // 2 - 1, n_layers - 4]


def main():
    import train_eagle3 as T
    argv = sys.argv[1:]
    target = Path(argv[argv.index("--target") + 1])
    n = json.loads((target / "config.json").read_text())["num_hidden_layers"]
    T.AUX[:] = aux_layers(n)
    print(f"[fam] {target.name}: {n} target layers -> EAGLE-3 aux layers {T.AUX}", flush=True)
    T.main()


if __name__ == "__main__":
    main()
