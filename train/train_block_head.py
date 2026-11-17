"""H3a S4: our own DFlash training loop for VLM targets, around SpecForge's own objective.

SpecForge's online capture is text-only, so the target forward is ours (HF, frozen, bf16); all
else is imported from refs/SpecForge (put it on PYTHONPATH, the glance environment):
  - draft:      specforge.modeling.draft.dflash.DFlashDraftModel (from the JSON config, or
                from_pretrained(<exported head>) for a warm start; save_pretrained = the export
                format the harness loads);
  - objective:  OnlineDFlashModel (random anchors, in-block masks, loss decay gamma), sharing the
                target's frozen lm_head and embedding;
  - optimizer:  specforge.optimizer.BF16Optimizer (fp32 masters, AdamW, warmup + cosine, clip);
  - batching:   SpecForge's semantics: every micro-batch backpropagates the loss NUMERATOR, and
                at the optimizer boundary the gradients are divided by the summed denominator
                (controller._normalize_gradients), i.e. the token-weighted mean over the global
                batch, independent of how it is split.
Target features: the hidden states of layers [1, 7, 13, 19, 25] (+1 for the embedding output,
dflash.extract_context_feature), concatenated -> (L, 17920). Text2svg: the exported text model;
image2svg: the OmniSVG VLM with pixel values and M-RoPE positions from get_rope_index over the
whole teacher-forced sequence (vlm.py).

Data:
  --task t2svg  --data <SpecForge prompts jsonl {input_ids, loss_mask}> (the B3 train sets)
  --task i2svg  --data <dir of gen_i2svg.py units>  (C1 loop rule applied here)

Multi-GPU: run under torchrun (src/a batch GPU job/py_job_dist.sh). Every rank walks the same global sample order; each
step's global batch is exactly the single-GPU one, split by rank (samples r, r + W,...), so with W = 1 nothing changes.
Gradients and the loss denominator are summed over ranks (fp32 all-reduce) before the division, which keeps the
token-weighted mean over the global batch. Rank 0 logs and saves. --dist-backend gloo lets a 1-GPU smoke run 2 ranks.

Usage (GPU, the glance environment, PYTHONPATH=refs/SpecForge):
  python train_dflash.py --task t2svg --data <train.jsonl> --epochs 2 --out <dir>
  python train_dflash.py --task i2svg --data <gen dir> --init <head> --epochs 2 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from sightline.paths import MODELS, REFS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
sys.path.insert(0, str(REFS / "SpecForge"))
MODEL = MODELS
LAYERS = [1, 7, 13, 19, 25]
MASK = 196900
DRAFT_CFG = ROOT / "configs/heads/omnisvg-8b-dflash-draft.json"


def load_samples(task: str, data: Path, max_length: int) -> list[dict]:
    out = []
    if task == "t2svg":
        for line in data.read_text().split("\n"):
            if line.strip():
                r = json.loads(line)
                out.append({"input_ids": r["input_ids"][:max_length],
                            "loss_mask": r["loss_mask"][:max_length]})
        return out
    from build_trainset import kept_gen                          # C1 loop rule
    for f in sorted(data.glob("*/unit-*/results.jsonl")):
        if not (f.parent / "DONE").exists():
            continue
        for line in f.read_text().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            c = r["candidates"][0]
            gen, _, _ = kept_gen(c["token_ids"], c["eos_found"])
            if MASK in gen or len(gen) < 2:
                continue
            ids = (r["prompt_input_ids"] + gen)[:max_length]
            lm = ([0] * len(r["prompt_input_ids"]) + [1] * len(gen))[:max_length]
            out.append({"input_ids": ids, "loss_mask": lm, "image_path": r["image_path"]})
    return out


class Features:
    """Frozen target forward -> concatenated hidden states of LAYERS."""

    def __init__(self, task: str, target: Path | None = None):
        self.task = task
        if task == "t2svg":
            from transformers import AutoModelForCausalLM
            self.target = AutoModelForCausalLM.from_pretrained(
                target or MODEL / "OmniSVG1.1_8B-qwen2-text", torch_dtype=torch.bfloat16,
                attn_implementation="sdpa").cuda().eval()
        else:
            import vlm
            self.vlm = vlm
            self.official = vlm.load_official()
            self.target, note = vlm.load_target()
            print("[target]", note, flush=True)
        self.target.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, s: dict) -> torch.Tensor:
        ids = torch.tensor([s["input_ids"]], device="cuda")
        if self.task == "t2svg":
            out = self.target(ids, output_hidden_states=True, logits_to_keep=1)
        else:
            from qwen_vl_utils import process_vision_info
            img, _ = self.official.preprocess_image_for_svg(s["image_path"], replace_background=True)
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}]}]
            image_inputs, _ = process_vision_info(msgs)
            pv = self.official.processor.image_processor(images=image_inputs, return_tensors="pt")
            thw = pv["image_grid_thw"].cuda()
            mm = (ids == self.vlm.IMAGE_TOKEN_ID).long()
            self.target.model.rope_deltas = None
            pos, _ = self.target.model.get_rope_index(ids, mm, thw, None, None)
            out = self.target(ids, pixel_values=pv["pixel_values"].cuda().to(torch.bfloat16),
                              image_grid_thw=thw, position_ids=pos, output_hidden_states=True,
                              logits_to_keep=1)
        return torch.cat([out.hidden_states[l + 1][0] for l in LAYERS], dim=-1)


def collate(samples, feats, pad_id: int):
    L = max(len(s["input_ids"]) for s in samples)
    B = len(samples)
    H = None
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lm = torch.zeros((B, L), dtype=torch.long)
    hs_list = []
    for i, s in enumerate(samples):
        n = len(s["input_ids"])
        ids[i, :n] = torch.tensor(s["input_ids"])
        lm[i, :n] = torch.tensor(s["loss_mask"])
        h = feats(s)
        H = h.shape[-1]
        hs_list.append(h)
    hs = torch.zeros((B, L, H), dtype=hs_list[0].dtype, device="cuda")
    for i, h in enumerate(hs_list):
        hs[i, :h.shape[0]] = h
    return ids, hs, lm


def save_head(draft, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(out)
    cfg = json.loads((out / "config.json").read_text())
    cfg["architectures"] = ["DFlashDraftModel"]
    cfg["auto_map"] = {"AutoModel": "dflash.DFlashDraftModel"}       # as `specforge export --to hf`
    (out / "config.json").write_text(json.dumps(cfg, indent=2))


def dist_setup(backend: str) -> tuple[int, int]:
    import os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["t2svg", "i2svg"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None, help="exported head to warm-start from")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="run only this many steps of the --epochs schedule (the SpecForge main "
                         "run's epoch-1/2 heads sat at 1/6, 2/6 of a 6-epoch cosine)")
    ap.add_argument("--global-batch", type=int, default=12)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.04)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--num-anchors", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=7.0)
    ap.add_argument("--attn", default="flex_attention")
    ap.add_argument("--save-per-epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dist-backend", default="nccl")
    ap.add_argument("--target", type=Path, default=None,
                    help="t2svg target (default OmniSVG1.1_8B-qwen2-text); 10-15: model/InternSVG-8B-qwen2-text")
    ap.add_argument("--mask-id", type=int, default=MASK, help="DFlash mask token (InternSVG: 151662 <|fim_pad|>)")
    ap.add_argument("--draft-cfg", type=Path, default=DRAFT_CFG, help="draft config for a from-scratch head")
    ap.add_argument("--pad-id", type=int, default=151643)
    args = ap.parse_args()
    world, rank = dist_setup(args.dist_backend)
    assert args.global_batch % (world * args.micro_batch) == 0, (args.global_batch, world, args.micro_batch)
    accum = args.global_batch // (world * args.micro_batch)          # micro-batches per rank per step
    random.seed(args.seed)
    torch.manual_seed(args.seed)                                      # same draft init on every rank
    say = print if rank == 0 else (lambda *a, **k: None)

    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from specforge.optimizer import BF16Optimizer
    from transformers import Qwen3Config

    samples = load_samples(args.task, args.data, args.max_length)
    steps_per_epoch = math.ceil(len(samples) / args.global_batch)
    total = args.max_steps or int(round(steps_per_epoch * args.epochs))
    say(f"[data] {len(samples)} samples; {steps_per_epoch} steps/epoch; total {total} steps "
        f"(global batch {args.global_batch} = {world} ranks x {args.micro_batch} x {accum})", flush=True)

    feats = Features(args.task, args.target)
    if args.init is not None:
        draft = DFlashDraftModel.from_pretrained(args.init, torch_dtype=torch.bfloat16)
        say(f"[draft] warm start from {args.init}", flush=True)
        if args.mask_id != MASK:
            # warm start across targets (e.g. the OmniSVG head on InternSVG): the saved config must carry the
            # new target's mask / special ids, which the harness reads (dflash_config.mask_token_id); no weights change
            over = json.loads(args.draft_cfg.read_text())
            dc = dict(getattr(draft.config, "dflash_config", {}) or {})
            dc["mask_token_id"] = args.mask_id
            draft.config.dflash_config = dc
            draft.mask_token_id = args.mask_id
            for k in ("bos_token_id", "eos_token_id", "vocab_size"):
                setattr(draft.config, k, over[k])
            say(f"[draft] config ids set for the new target: mask {args.mask_id}, "
                f"bos/eos {over['bos_token_id']}/{over['eos_token_id']}, vocab {over['vocab_size']}", flush=True)
    else:
        cfg = Qwen3Config.from_dict(json.loads(args.draft_cfg.read_text()))
        draft = DFlashDraftModel(cfg).to(torch.bfloat16)
        say("[draft] from scratch", flush=True)
    draft.config._attn_implementation = args.attn
    model = OnlineDFlashModel(
        draft_model=draft, target_lm_head=feats.target.lm_head,
        target_embed_tokens=feats.target.get_input_embeddings(), mask_token_id=args.mask_id,
        block_size=int(draft.block_size), attention_backend=args.attn,
        num_anchors=args.num_anchors, loss_decay_gamma=args.gamma, objective_chunk_blocks=128,
    ).cuda().to(torch.bfloat16)
    for name, p in model.named_parameters():
        if not name.startswith("draft_model."):
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    say(f"[model] trainable {sum(p.numel() for p in trainable) / 1e9:.3f}B params", flush=True)
    if world > 1:                                                     # belt and braces: identical start on all ranks
        import torch.distributed as dist
        with torch.no_grad():
            for p in trainable:
                dist.broadcast(p.data, src=0)
        torch.manual_seed(args.seed + rank)                           # anchors etc. differ per rank (W = 1: RNG untouched)
    opt = BF16Optimizer(model, lr=args.lr, max_grad_norm=args.max_grad_norm,
                        warmup_ratio=args.warmup_ratio, lr_scheduler="cosine", total_steps=total)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                        "n_samples": len(samples), "world_size": world,
                                                        "steps_per_epoch": steps_per_epoch,
                                                        "total_steps": total}, indent=1))
    curve = (args.out / "train_curve.jsonl").open("a") if rank == 0 else None
    order: list[int] = []
    epoch = 0
    t_log = time.perf_counter()
    num_acc = den_acc = cor_acc = 0.0
    run_steps = min(total, args.stop_after or total)
    for step in range(1, run_steps + 1):
        den_total = torch.zeros((), device="cuda")
        glob = []                                                     # this step's global batch, same on all ranks
        for _ in range(args.global_batch):
            if not order:
                order = list(range(len(samples)))
                random.Random(args.seed + epoch).shuffle(order)
                epoch += 1
            glob.append(order.pop())
        mine = glob[rank::world]
        for a in range(accum):
            batch = [samples[i] for i in mine[a * args.micro_batch:(a + 1) * args.micro_batch]]
            ids, hs, lm = collate(batch, feats, pad_id=args.pad_id)
            loss, acc, metrics = model(input_ids=ids.cuda(), hidden_states=hs, loss_mask=lm.cuda(),
                                       collect_detailed_metrics=False)
            num, den = metrics["loss_terms"]
            num.reshape(()).backward()
            den_total += den.detach().reshape(())
            num_acc += float(num.detach())
            den_acc += float(den.detach())
            cor_acc += float(acc.detach())
        with torch.no_grad():
            if world > 1:
                grads = [p.grad for p in trainable if p.grad is not None]
                flat = torch.cat([g.reshape(-1).float() for g in grads])
                all_reduce_(flat, world)
                off = 0
                for g in grads:
                    g.copy_(flat[off:off + g.numel()].view_as(g))
                    off += g.numel()
                del flat
                all_reduce_(den_total, world)
            for p in trainable:
                if p.grad is not None:
                    p.grad.div_(den_total)
        opt.step()
        if step % args.log_every == 0 or step == run_steps:
            dt = (time.perf_counter() - t_log) / args.log_every
            if world > 1:
                tot = all_reduce_(torch.tensor([num_acc, den_acc, cor_acc], device="cuda", dtype=torch.float64), world)
                num_acc, den_acc, cor_acc = (float(x) for x in tot)
            rec = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                   "loss": num_acc / max(den_acc, 1e-9), "acc_mean": cor_acc / (args.log_every * accum * world),
                   "lr": opt.scheduler.get_last_lr()[0], "grad_norm": float(opt.last_grad_norm),
                   "s_per_step": round(dt, 3)}
            if curve is not None:
                curve.write(json.dumps(rec) + "\n")
                curve.flush()
            say(f"step {step}: {rec}", flush=True)
            t_log = time.perf_counter()
            num_acc = den_acc = cor_acc = 0.0
        if args.save_per_epoch and step % steps_per_epoch == 0 and step < run_steps:
            if rank == 0:
                save_head(model.draft_model, args.out / "heads" / f"step{step}")
            say(f"[save] step {step}", flush=True)
            if world > 1:
                torch.distributed.barrier()
    if rank == 0:
        save_head(model.draft_model, args.out / "heads" / f"step{run_steps}")
        (args.out / "TRAIN_RC").write_text("0\n")
    say(f"TRAIN_DONE {args.out}", flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
