""" (C2): target.py (transformers 5.15) vs the official dump of dump_official.py
(the starvector environment, transformers 4.49): pixel values, prefix embeddings, and the greedy continuation of our harness AR loop
(vlm.vlm_ar_generate, EOS stop) against the official ids (stopped at "</svg>"), compared over the official length.
  python check_target.py --dump <which>-<dtype>.pt [--dtype float16] [--also-fp32]
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.verify import ChainParams  # noqa: E402
from starvec import target as T  # noqa: E402
from vlm import vlm_ar_generate  # noqa: E402


def run(dump, dtype, label):
    from PIL import Image
    which = dump["meta"]["which"]
    m = T.load(which, dtype)
    proc = T.processor(which)
    pids = T.prompt_ids(which)
    res = []
    for r in dump["rows"]:
        pv = proc(Image.open(r["image_path"]))
        pdiff = (pv - r["pixel_values"].float()).abs().max().item()
        inp = T.build_inputs(m, proc, Image.open(r["image_path"]), pids)
        e = torch.cat([inp.prefix.float().cpu(), m.get_input_embeddings()(torch.tensor([pids[T.Q[which]:]], device="cuda")).float().cpu()], 1)
        ediff = (e - r["prefix_embeds"]).abs().max().item()
        n_off = len(r["official_ids"])
        ours = vlm_ar_generate(m, inp, n_off + 8, [T.EOS], ChainParams(greedy=True))
        o = r["official_ids"]
        div = next((j for j in range(min(len(o), len(ours.tokens))) if o[j] != ours.tokens[j]), None)
        res.append({"gidx": r["gidx"], "pixel_maxdiff": round(pdiff, 6), "prefix_maxdiff": round(ediff, 4),
                    "n_official": n_off, "n_ours": len(ours.tokens), "first_div": div,
                    "same_over_official": div is None and len(ours.tokens) >= len(o)})
        print(label, res[-1], flush=True)
    del m
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--also-fp32", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    dump = torch.load(a.dump, weights_only=False)
    run(dump, getattr(torch, a.dtype), a.dtype)
    if a.also_fp32:
        run(dump, torch.float32, "float32")


if __name__ == "__main__":
    main()
