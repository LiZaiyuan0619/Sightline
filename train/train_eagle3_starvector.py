""" the EAGLE-3 baseline head on the two StarVector targets (generality table).

baselines/train_eagle3.py is text-only (one HF `AutoModelForCausalLM`, no image prefix) and baselines/train_eagle3_i2.py's
image branch is bound to OmniSVG's VLM (`vlm.py`, pixel values + M-RoPE) and additionally *requires* a
text-to-SVG head to warm-start from -- StarVector has no text-to-SVG mode at all. So this is a third entry point, the
way train_sv.py is for DFlash. Neither existing trainer is modified (queued jobs import them).

What is reused, so the vocabulary and bookkeeping exist once: `vocab_mapping` / `process_token_dict_to_mappings`
(SpecForge's d2t / t2d), `dist_setup`, `all_reduce_` and `save_head` from baselines/train_eagle3.py, and `load_samples`
from train_sv.py (train.jsonl of build_sv_trainset.py: input_ids, loss_mask, image_path).

What is StarVector's:
  - target:   target.py (frozen, bf16). The first Q positions of every sample are PAD placeholders that
              `forward(prefix_embeds=...)` replaces with adapter(image_encoder(pixels)), exactly as at inference.
  - features: hidden states of AUX = [1, L/2-1, L-4] on the target's real depth (1B gpt_bigcode 24 -> [1, 11, 20];
              8B starcoder2 32 -> [1, 15, 28]), concatenated, plus the full-vocab logits as the teacher.
  - draft:    SpecForge's EAGLE-3 config with the geometry of this family's own DFlash draft config
              (starvector-<w>-dflash-draft-b16.json), so all three trained baselines of this family share
              one draft-side setting. That matters here because StarVector-1B is gpt_bigcode: it has no rope_theta and
              no intermediate_size to copy, and it uses learned absolute positions while every draft in this project
              is a RoPE decoder reading the target's hidden states (the DFlash / Domino heads already do this and
              reach 4-8x on these targets).

  the glance environment/bin/torchrun --standalone --nproc_per_node 4 train_eagle3_sv.py --which 1b \
      --data <train.jsonl> --draft-vocab 16000 --epochs 2 --global-batch 16 --max-length 4096 --save-per-epoch --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "e3", ROOT / "h3a", ROOT / "starvec", REFS / "SpecForge"):
    sys.path.insert(0, str(p))

import train_eagle3 as TE  # noqa: E402
import train_sv as SV  # noqa: E402
from starvec import target as T  # noqa: E402

BASE_CFG = REFS / "SpecForge/configs/qwen2.5-7b-eagle3.json"
DRAFT_CFG = {w: ROOT / f"starvector-{w}-dflash-draft-b16.json" for w in ("1b", "8b")}


def aux_layers(n_layers: int) -> list[int]:
    """SpecForge's default triple, on this target's real depth."""
    return [1, n_layers // 2 - 1, n_layers - 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["1b", "8b"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draft-vocab", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.015)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--ttt", type=int, default=7)
    ap.add_argument("--ploss-decay", type=float, default=0.8)
    ap.add_argument("--attn", default="flex_attention")
    ap.add_argument("--save-per-epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dist-backend", default="nccl")
    args = ap.parse_args()
    world, rank = TE.dist_setup(args.dist_backend)
    assert args.global_batch % world == 0, (args.global_batch, world)
    accum = args.global_batch // world                               # micro-batch 1, as baselines/train_eagle3.py
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    say = print if rank == 0 else (lambda *a, **k: None)

    from PIL import Image
    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
    from specforge.optimizer import BF16Optimizer
    from specforge.utils import padding
    from transformers import LlamaConfig

    samples = SV.load_samples("i2svg", args.data, args.max_length)
    steps_per_epoch = math.ceil(len(samples) / args.global_batch)
    total = args.max_steps or int(round(steps_per_epoch * args.epochs))
    say(f"[data] {len(samples)} samples; {steps_per_epoch} steps/epoch; total {total} steps "
        f"(global batch {args.global_batch} = {world} ranks x 1 x {accum})", flush=True)

    target = T.load(args.which, torch.bfloat16)
    proc = T.processor(args.which)
    target.requires_grad_(False)
    tc = target.config
    aux = aux_layers(tc.num_hidden_layers)
    say(f"[target] StarVector-{args.which} ({tc.model_type}, {tc.num_hidden_layers} layers) -> aux layers {aux}",
        flush=True)

    dcfg = json.loads(DRAFT_CFG[args.which].read_text())              # this family's own draft geometry
    cfg = json.loads(BASE_CFG.read_text())
    cfg.update({"vocab_size": tc.vocab_size, "draft_vocab_size": args.draft_vocab,
                "bos_token_id": dcfg["bos_token_id"], "eos_token_id": dcfg["eos_token_id"],
                "hidden_size": tc.hidden_size, "intermediate_size": dcfg["intermediate_size"],
                "num_attention_heads": dcfg["num_attention_heads"],
                "num_key_value_heads": dcfg["num_key_value_heads"], "rope_theta": dcfg["rope_theta"],
                "max_position_embeddings": dcfg["max_position_embeddings"], "rms_norm_eps": dcfg["rms_norm_eps"],
                "eagle_config": {"eagle_aux_hidden_state_layer_ids": aux}})
    assert cfg["hidden_size"] == dcfg["hidden_size"], (cfg["hidden_size"], dcfg["hidden_size"])
    d2t, t2d, cov = TE.vocab_mapping(samples, tc.vocab_size, args.draft_vocab)
    say(f"[vocab] {cov['distinct']} distinct loss tokens; top {args.draft_vocab} cover {cov['coverage']:.5f}",
        flush=True)
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
                                                        "n_samples": len(samples), "world_size": world, "aux": aux,
                                                        "steps_per_epoch": steps_per_epoch, "total_steps": total,
                                                        "coverage": cov}, indent=1))
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
                pre = target.image_prefix(proc(Image.open(s["image_path"])).cuda())
                out = target(ids, prefix_embeds=pre, output_hidden_states=True)
                feats = torch.cat([out.hidden_states[l + 1] for l in aux], dim=-1)
                logits = out.logits
            feats, logits = feats.clone(), logits.clone()             # leave inference mode for autograd
            plosses, _, acces, *_ = model(input_ids=padding(ids, left=False), attention_mask=torch.ones_like(ids),
                                          target=padding(logits, left=False), loss_mask=lm[..., None],
                                          hidden_states=feats)
            loss = sum(args.ploss_decay ** k * plosses[k] for k in range(len(plosses)))
            (loss / accum).backward()
            loss_sum += float(loss.detach())
            acc_sum += torch.tensor([float(a) for a in acces], dtype=torch.float64)
            n_mb += 1
            del out, feats, logits
        if world > 1:
            with torch.no_grad():
                grads = [p.grad for p in trainable if p.grad is not None]
                flat = torch.cat([g.reshape(-1).float() for g in grads])
                TE.all_reduce_(flat, world)
                flat.div_(world)
                off = 0
                for g in grads:
                    g.copy_(flat[off:off + g.numel()].view_as(g))
                    off += g.numel()
                del flat
        opt.step()
        if step % args.log_every == 0 or step == total:
            dt = (time.perf_counter() - t_log) / args.log_every
            tot = TE.all_reduce_(torch.tensor([loss_sum, n_mb, *acc_sum.tolist()], device="cuda",
                                              dtype=torch.float64), world)
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
                TE.save_head(model.draft_model, cfg, args.out / "heads" / f"step{step}")
            say(f"[save] step {step}", flush=True)
            if world > 1:
                torch.distributed.barrier()
    if rank == 0:
        TE.save_head(model.draft_model, cfg, args.out / "heads" / f"step{total}")
        (args.out / "TRAIN_RC").write_text("0\n")
    say(f"TRAIN_DONE {args.out}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
