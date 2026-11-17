""" EAGLE-3 baseline drafters for the SVG text targets (OmniSVG-8B text2svg export, InternSVG-8B text export), our
own loop around SpecForge's EAGLE-3 objective (mirrors train_dflash.py: same data, global batch, multi-GPU scheme).

From refs/SpecForge (PYTHONPATH, the glance environment):
  - draft:      specforge.modeling.draft.llama3_eagle.LlamaForCausalLMEagle3 from SpecForge's Qwen2.5-7B EAGLE-3 config
                (one decoder layer, fc over the three captured layers), with the target's vocab / special ids; the token
                embedding is copied from the target and frozen (SpecForge's load_embedding + freeze_embedding);
  - vocab:      d2t / t2d from the loss-bearing token frequencies of the training set (process_token_dict_to_mappings).
                OmniSVG outputs use 58,488 distinct tokens (top 32k cover 97.9%, 48k 99.99%, 64k 100%), so its draft
                vocabulary is 64k instead of the usual 16k / 32k; InternSVG uses 1,680 (16k);
  - objective:  OnlineEagle3Model, TTT length 7, loss = sum_i 0.8^i ploss_i (Eagle3TrainStrategy's ploss_decay);
  - optimizer:  BF16Optimizer (fp32 masters, AdamW, warmup + cosine); SpecForge's recipe lr 5e-5 / clip 0.5 at batch 1 per
                GPU, here lr 1e-4 for the global batch 16 of our DFlash heads (linear scaling).
Target features (online, frozen HF bf16 forward): the outputs of layers [1, 13, 24] (SpecForge's default [1, L/2 - 1, L - 4]
for 28 layers; HF hidden_states[l + 1]), concatenated (3 x 3584), and the full-vocab logits as the teacher. As in SpecForge's
online path, input ids and logits are shifted left by one (the draft at position t reads f_t and x_{t+1} and is trained
toward the target's distribution of x_{t+2}); the last position carries no loss. Micro-batch 1 (no padding).

  the glance environment/bin/torchrun --standalone --nproc_per_node 4 baselines/train_eagle3.py --target <text model> \
      --data <train.jsonl> --draft-vocab 64000 --epochs 2 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFS / "SpecForge"))
BASE_CFG = REFS / "SpecForge/configs/qwen2.5-7b-eagle3.json"
AUX = [1, 13, 24]


def load_samples(data: Path, max_length: int) -> list[dict]:
    out = []
    with data.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append({"input_ids": r["input_ids"][:max_length], "loss_mask": r["loss_mask"][:max_length]})
    return out


def process_token_dict_to_mappings(token_dict: Counter, draft_vocab_size: int, target_vocab_size: int):
    """Same as specforge.data.preprocessing.process_token_dict_to_mappings (that module needs pydantic, which the glance environment
    lacks): pad with zero-frequency ids below draft_vocab_size, keep the top-N by frequency, sort; d2t[i] = used[i] - i."""
    token_dict = Counter(token_dict)
    if len(token_dict) < draft_vocab_size:
        for token in set(range(draft_vocab_size)) - set(token_dict.keys()):
            token_dict[token] = 0
            if len(token_dict) >= draft_vocab_size:
                break
    used = sorted(key for key, _ in token_dict.most_common(draft_vocab_size))
    used_set = set(used)
    d2t = torch.tensor([used[i] - i for i in range(len(used))])
    t2d = torch.tensor([i in used_set for i in range(target_vocab_size)])
    return d2t, t2d


def vocab_mapping(samples: list[dict], target_vocab: int, draft_vocab: int):
    cnt: Counter = Counter()
    for s in samples:
        cnt.update(t for t, m in zip(s["input_ids"], s["loss_mask"]) if m)
    d2t, t2d = process_token_dict_to_mappings(cnt, draft_vocab, target_vocab)
    tot = sum(cnt.values())
    cov = sum(f for _, f in cnt.most_common(draft_vocab)) / max(tot, 1)
    return d2t, t2d, {"distinct": len(cnt), "loss_tokens": tot, "coverage": cov}


def dist_setup(backend: str) -> tuple[int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
        dist.init_process_group(backend=backend)
    return world, rank


def all_reduce_(t: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(t)
    return t


def save_head(draft, cfg: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(out)                       # weights incl. the d2t / t2d buffers
    c = json.loads((out / "config.json").read_text())
    c.update({k: cfg[k] for k in ("architectures", "draft_vocab_size", "eagle_config")})
    (out / "config.json").write_text(json.dumps(c, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draft-vocab", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.015)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--ttt", type=int, default=7)
    ap.add_argument("--ploss-decay", type=float, default=0.8)
    ap.add_argument("--attn", default="flex_attention")
    ap.add_argument("--save-per-epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dist-backend", default="nccl")
    args = ap.parse_args()
    world, rank = dist_setup(args.dist_backend)
    assert args.global_batch % world == 0, (args.global_batch, world)
    accum = args.global_batch // world                               # micro-batch 1
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    say = print if rank == 0 else (lambda *a, **k: None)

    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
    from specforge.optimizer import BF16Optimizer
    from specforge.utils import padding
    from transformers import AutoModelForCausalLM, LlamaConfig

    samples = load_samples(args.data, args.max_length)
    steps_per_epoch = math.ceil(len(samples) / args.global_batch)
    total = args.max_steps or int(round(steps_per_epoch * args.epochs))
    say(f"[data] {len(samples)} samples; {steps_per_epoch} steps/epoch; total {total} steps "
        f"(global batch {args.global_batch} = {world} ranks x 1 x {accum})", flush=True)

    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").cuda().eval()
    target.requires_grad_(False)
    tc = target.config
    raw_tc = json.loads((args.target / "config.json").read_text())   # transformers 5 moves rope_theta into rope_parameters
    cfg = json.loads(BASE_CFG.read_text())
    cfg.update({"vocab_size": tc.vocab_size, "draft_vocab_size": args.draft_vocab, "bos_token_id": tc.bos_token_id,
                "eos_token_id": tc.eos_token_id, "hidden_size": tc.hidden_size,
                "intermediate_size": tc.intermediate_size, "num_attention_heads": tc.num_attention_heads,
                "num_key_value_heads": tc.num_key_value_heads, "rope_theta": raw_tc["rope_theta"],
                "max_position_embeddings": tc.max_position_embeddings, "rms_norm_eps": tc.rms_norm_eps,
                "eagle_config": {"eagle_aux_hidden_state_layer_ids": AUX}})
    d2t, t2d, cov = vocab_mapping(samples, tc.vocab_size, args.draft_vocab)
    say(f"[vocab] {cov['distinct']} distinct loss tokens; top {args.draft_vocab} cover {cov['coverage']:.5f}", flush=True)
    draft = LlamaForCausalLMEagle3(LlamaConfig.from_dict(cfg), attention_backend=args.attn).to(torch.bfloat16).cuda()
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(target.get_input_embeddings().weight)
        draft.t2d.copy_(t2d)
        draft.d2t.copy_(d2t)
    draft.freeze_embedding()
    model = OnlineEagle3Model(draft_model=draft, length=args.ttt, attention_backend=args.attn).cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    say(f"[model] trainable {sum(p.numel() for p in trainable) / 1e9:.3f}B params", flush=True)
    if world > 1:
        import torch.distributed as dist
        with torch.no_grad():
            for p in trainable:
                dist.broadcast(p.data, src=0)
        torch.manual_seed(args.seed + rank)
    opt = BF16Optimizer(model, lr=args.lr, max_grad_norm=args.max_grad_norm,
                        warmup_ratio=args.warmup_ratio, lr_scheduler="cosine", total_steps=total)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                        "n_samples": len(samples), "world_size": world, "aux": AUX,
                                                        "steps_per_epoch": steps_per_epoch, "total_steps": total,
                                                        "vocab": cov}, indent=1))
    curve = (args.out / "train_curve.jsonl").open("a") if rank == 0 else None
    order: list[int] = []
    epoch = 0
    t_log = time.perf_counter()
    acc_sum = torch.zeros(args.ttt, dtype=torch.float64)
    loss_sum, n_mb = 0.0, 0
    for step in range(1, total + 1):
        glob = []
        for _ in range(args.global_batch):
            if not order:
                order = list(range(len(samples)))
                random.Random(args.seed + epoch).shuffle(order)
                epoch += 1
            glob.append(order.pop())
        for i in glob[rank::world]:
            s = samples[i]
            ids = torch.tensor([s["input_ids"]], device="cuda")
            lm = torch.tensor([s["loss_mask"]], device="cuda")
            lm[0, -1] = 0
            with torch.inference_mode():
                out = target(ids, output_hidden_states=True)
                aux = torch.cat([out.hidden_states[l + 1] for l in AUX], dim=-1)
                logits = out.logits
            aux, logits = aux.clone(), logits.clone()                  # leave inference mode for autograd
            plosses, _, acces, *_ = model(input_ids=padding(ids, left=False), attention_mask=torch.ones_like(ids),
                                          target=padding(logits, left=False), loss_mask=lm[..., None],
                                          hidden_states=aux)
            loss = sum(args.ploss_decay ** k * plosses[k] for k in range(len(plosses)))
            (loss / accum).backward()
            loss_sum += float(loss.detach())
            acc_sum += torch.tensor([float(a) for a in acces], dtype=torch.float64)
            n_mb += 1
            del out, aux, logits
        if world > 1:
            with torch.no_grad():
                grads = [p.grad for p in trainable if p.grad is not None]
                flat = torch.cat([g.reshape(-1).float() for g in grads])
                all_reduce_(flat, world)
                flat.div_(world)
                off = 0
                for g in grads:
                    g.copy_(flat[off:off + g.numel()].view_as(g))
                    off += g.numel()
                del flat
        opt.step()
        if step % args.log_every == 0 or step == total:
            dt = (time.perf_counter() - t_log) / args.log_every
            tot = all_reduce_(torch.tensor([loss_sum, n_mb, *acc_sum.tolist()], device="cuda", dtype=torch.float64),
                              world)
            n = max(float(tot[1]), 1.0)
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": float(tot[0]) / n,
                   "acc": [round(float(a) / n, 4) for a in tot[2:]], "lr": opt.scheduler.get_last_lr()[0],
                   "grad_norm": float(opt.last_grad_norm), "s_per_step": round(dt, 3)}
            if curve is not None:
                curve.write(json.dumps(rec) + "\n")
                curve.flush()
            say(f"step {step}: {rec}", flush=True)
            t_log = time.perf_counter()
            acc_sum.zero_()
            loss_sum, n_mb = 0.0, 0
        if args.save_per_epoch and step % steps_per_epoch == 0 and step < total:
            if rank == 0:
                save_head(model.draft_model, cfg, args.out / "heads" / f"step{step}")
            say(f"[save] step {step}", flush=True)
            if world > 1:
                torch.distributed.barrier()
    if rank == 0:
        save_head(model.draft_model, cfg, args.out / "heads" / f"step{total}")
        (args.out / "TRAIN_RC").write_text("0\n")
    say(f"TRAIN_DONE {args.out}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
