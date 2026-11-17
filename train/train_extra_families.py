""" train_dflash.py for the third-family targets, with the target-feature layers taken from the draft
config (dflash_config.target_layer_ids). train_dflash.py hard-codes LAYERS = [1, 7, 13, 19, 25] (right for the 28-layer
OmniSVG-8B / InternSVG-8B / SVGen targets), while the harness reads the layers from the saved head config; the
OmniSVG-4B heads (36 layers, config [1, 9, 17, 25, 33]) were therefore trained on [1, 7, 13, 19, 25] (found chain tokens per round 3.88 with the config's layers vs 8.74 with the trained ones). Here the two always agree. train_dflash.py is
not modified (running jobs import it); all its flags pass through.
  torchrun... train_dflash_fam.py --task t2svg --draft-cfg <fam>-dflash-draft-b32.json... (same flags)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script


def main():
    import train_dflash as T
    argv = sys.argv[1:]
    cfg = Path(argv[argv.index("--draft-cfg") + 1])
    layers = json.loads(cfg.read_text())["dflash_config"]["target_layer_ids"]
    T.LAYERS[:] = layers
    print(f"[fam] target-feature layers from {cfg.name}: {T.LAYERS}", flush=True)
    T.main()


if __name__ == "__main__":
    main()
