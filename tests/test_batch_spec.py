""" GPU checks for sightline/batch_spec.py (batched chain SD of N sequences sharing a prompt).

  1. --dtype float32: every sequence of the batched decoder == the single-sequence AR (harness/ar.ar_generate) with the
     same ChainParams and generator seed, token for token (exactness);
  2. --dtype bfloat16: agreement with the single-sequence chain decoder (harness/spec.spec_generate) per sequence, and
     the per-phase time split (draft / verify / walk / cache).
OmniSVG text target + the new block-32 T2SVG head; bench rows; sample mode with the official per-subtype chain (+ the
loop stop rule when --stop), seeds 42 + gidx + offset as the multi-sample runs.

  python sightline/test_batch_spec.py --dtype float32 --gidx 0,150 --n 4 --max-new 300
  python sightline/test_batch_spec.py --dtype bfloat16 --gidx 0,1,150,151 --n 8 --max-new 1536 --stop
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from eval_b3 import EOS, MODEL, chain_params, load_rows      # noqa: E402
from sightline.ar import ar_generate                            # noqa: E402
from sightline.batch_spec import BatchChainSD                   # noqa: E402
from sightline.spec import DFlashDrafter, spec_generate         # noqa: E402
from sightline.paths import RUNS

HEAD = (RUNS / "b3-train/omnisvg-dflash-main300k-b32-6ep-1012/heads/step150000")


def params(mode, subtype, stop):
    p = chain_params(mode, subtype)
    if stop:
        p.loop_ban, p.loop_repeats, p.loop_min_period, p.loop_action = 128, 3, 8, "eos"
    return p


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gidx", default="0,150")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--mode", default="sample")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--vs-spec", action="store_true", help="fp32: compare with the single-sequence chain SD (acceptance)")
    a = ap.parse_args()
    from dflash.model import DFlashDraftModel
    from transformers import AutoModelForCausalLM
    dt = getattr(torch, a.dtype)
    if dt == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
    target = AutoModelForCausalLM.from_pretrained(MODEL / "OmniSVG1.1_8B-qwen2-text", dtype=dt,
                                                  attn_implementation="sdpa").cuda().eval()
    head = DFlashDraftModel.from_pretrained(HEAD, dtype=dt).cuda().eval()
    keep = {int(x) for x in a.gidx.split(",")}
    rows = [r for r in load_rows(["bench", "extra"]) if r["gidx"] in keep]
    offs = [1000 * k for k in range(a.n)]
    n_same = n_tot = n_acc_same = n_acc_tot = 0
    for r in rows:
        ps = [params(a.mode, r["subtype"], a.stop) for _ in offs]
        seeds = [42 + r["gidx"] + o for o in offs]
        dec = BatchChainSD(target, head, [EOS], a.max_new)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        dec.start(r["prompt_input_ids"], ps, [gen(s) for s in seeds])
        dec.run()
        torch.cuda.synchronize(); t_b = time.perf_counter() - t0
        outs = [s.generated(dec.n_prompt) for s in dec.seqs]
        refs, accs, t_ref = [], [], 0.0
        for p, s in zip(ps, seeds):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if dt == torch.float32 and not a.vs_spec:
                ref, acc = ar_generate(target, r["prompt_input_ids"], a.max_new, [EOS], p, generator=gen(s)).tokens, None
            else:
                dr = DFlashDrafter(head, target, int(head.block_size))
                res = spec_generate(target, dr, r["prompt_input_ids"], a.max_new, [EOS], p, generator=gen(s),
                                    block_size=int(head.block_size))
                ref, acc = res.tokens, res.acceptance
            torch.cuda.synchronize(); t_ref += time.perf_counter() - t0
            refs.append(ref)
            accs.append(acc)
        for i, (o, f) in enumerate(zip(outs, refs)):
            first = next((j for j in range(min(len(o), len(f))) if o[j] != f[j]), None if len(o) == len(f) else min(len(o), len(f)))
            n_tot += 1
            n_same += o == f
            if accs[i] is not None and o == f:
                same_acc = dec.seqs[i].acceptance == accs[i]
                n_acc_same += same_acc
                n_acc_tot += 1
                if not same_acc:
                    k = next((j for j in range(min(len(dec.seqs[i].acceptance), len(accs[i])))
                              if dec.seqs[i].acceptance[j] != accs[i][j]), None)
                    print(f"   acceptance differs at round {k} of {len(accs[i])} (emitted before: "
                          f"{sum(accs[i][:k]) if k is not None else '-'})", flush=True)
            print(f"gidx {r['gidx']} cand {i}: batched {len(o)} tok ({dec.seqs[i].stop_reason}), "
                  f"{'AR' if dt == torch.float32 else 'single SD'} {len(f)} tok, identical {o == f}, first diff {first}",
                  flush=True)
        print(f"gidx {r['gidx']}: batched {t_b:.2f}s ({dec.rounds} rounds, {dec.nfe} target forwards; "
              + ", ".join(f"{k} {v:.2f}s" for k, v in dec.t.items()) + f") vs {len(refs)} sequential refs {t_ref:.2f}s",
              flush=True)
    print(f"IDENTICAL {n_same}/{n_tot} ({a.dtype}, mode {a.mode}, stop {a.stop}); same per-round acceptance as the "
          f"single-sequence chain decoder on identical outputs: {n_acc_same}/{n_acc_tot}")


if __name__ == "__main__":
    main()
