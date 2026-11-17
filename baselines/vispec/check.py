""" training / inference consistency of the ViSpec draft (smoke rule: the drafter must compute what the trainer
trained). On real image-to-SVG training samples (teacher-forced target forward with the image, as train_vispec.Target):
  - training path: ViSpecDraft.forward_seq over all pairs -> top-1 of the target lm_head at every pair;
  - inference path: ViSpecDrafter fed the same target states round by round (prefill = the prompt, then chunks of
    --chunk committed tokens): the depth-1 rank-0 node of each round = the top-1 at pair start - 1.
Reports the agreement (bf16, different kernel shapes: expect ~100%) and the image span handling.
  python baselines/check_vispec.py --kind omnisvg --head <dir> --data <i2s data> [--n 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
import train_vispec as T                                  # noqa: E402
from baselines.vispec.model import load_head                        # noqa: E402
from baselines.vispec.spec import ViSpecDrafter                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["omnisvg", "internsvg"], required=True)
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--fp32", action="store_true", help="head, states and lm_head in fp32: the two paths must agree ~exactly")
    args = ap.parse_args()
    max_len = args.max_length or (2048 if args.kind == "omnisvg" else 4096)
    samples = T.load_image_samples(args.kind, args.data, max_len)
    samples = [samples[0], samples[len(samples) // 2], samples[-1]][:args.n]
    tgt = T.Target(args.kind, 2)
    dt = torch.float32 if args.fp32 else torch.bfloat16
    head = load_head(args.head, dtype=dt)
    lm_head = tgt.lm_head()
    if args.fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        w = lm_head.weight.detach().float()
        lm_head = torch.nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=torch.float32)
        lm_head.weight.data.copy_(w)
    res = []
    for s in samples:
        f, logits, emb, ids = tgt(s)
        f, emb = f.to(dt), emb.to(dt)
        n = f.shape[0]
        P = int((torch.from_numpy(s["loss_mask"]) == 0).sum())       # prompt length (loss mask is 0 on the prompt)
        span = T.image_span(ids, tgt.image_token)
        with torch.inference_mode():
            out = head.forward_seq(emb[1:], f[:-1], span)
            top_train = lm_head(out).argmax(-1).tolist()
            d = ViSpecDrafter(head, tgt.model, topk=4, depth=3, budget=12, image_token=tgt.image_token)
            d.lm_head = lm_head
            if d._hook is not None:
                d._hook.remove()
                d._hook = None
            d._cap = emb[None, :P]
            d.prefill_done([f[None, :P]])
            start, agree, tot, first_dep = P, 0, 0, []
            committed = ids.tolist()
            while start < n - 1:
                nodes = d.propose_tree(committed[:start + 1], start, 3)
                top = next(nd.token for nd in nodes if nd.depth == 1 and nd.rank == 0)
                agree += int(top == top_train[start - 1])
                tot += 1
                c = min(args.chunk, n - 1 - start)
                d.update_hidden_kept([f[None, start:start + c]], list(range(c)))
                start += c
        r = {"n": n, "prompt": P, "span": list(span) if span else None, "rounds": tot, "top1_agree": agree / max(tot, 1),
             "train_acc_vs_target": sum(int(top_train[t] == int(logits[t + 1].argmax())) for t in range(P - 1, n - 2))
             / max(1, n - 1 - P)}
        res.append(r)
        print(json.dumps(r), flush=True)
    print("CHECK_VISPEC", json.dumps({"mean_agree": sum(r["top1_agree"] for r in res) / len(res)}))


if __name__ == "__main__":
    main()
