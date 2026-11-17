""" ViSpec baseline drafts for table 1's image-to-SVG blocks (OmniSVG-8B, InternSVG-8B), the two-stage recipe of
refs/ViSpec (README "Model Training"), trained online on our target-generated data (the data of every other head):

  stage 1 (refs/ViSpec/vispec/train/main.py, plain EAGLE on text data): text-to-SVG train set, text model target;
          loss = SmoothL1(draft hidden, target h_{t+1}) + 0.1 * (CE(target p, draft p) + 0.1 * rank loss over the
          target's top-10) (main.py compute_loss; v_w 1, p_w 0.1);
  stage 2 (refs/ViSpec/vispec/train/main_mtp.py --use-ours --num-q 2 --mtp-steps 1): image-to-SVG train set, VLM target,
          init = the stage-1 head (vision adaptor fresh, img_fc = identity); loss over the step-0 pass and one MTP pass
          (the draft's own outputs, shifted by one, replace h_t): 10 * L1(draft p, target p) + 0.1 * rank loss.
Both: uniform noise on the input hidden states ((U - 0.5) * 0.2 * 512 / n, ViSpec's AddUniformNoise), AdamW (betas 0.9 /
0.95, torch default weight decay), linear warmup over 5% of the steps (official: 1 of 20 epochs) then linear decay,
gradient value clipping 0.5, fp32 master weights with a bf16 compute copy (the official runs accelerate bf16).
Learning rates: the official 3e-5 (stage 1) / 3e-6 (stage 2). the first attempt used x2 (linear scaling for our
global batch 16, as EAGLE-3) and stage 1 diverged once the warmup finished - loss spikes to 1e3 from rows whose target
hidden states are huge, accuracy 0.37 -> 0.14 (curve kept in $SIGHTLINE_RUNS/vispec-train-failed-1029/omnisvg). Gradients are now
norm-clipped to 1.0 before ViSpec's value clip of 0.5, and non-finite steps are skipped.

Budget parity with the block's EAGLE-3 head (user "fair version"): OmniSVG stage 1 = main300k x 6 epochs,
InternSVG stage 1 = b32-300k x 2 + mix600k x 1 (the two EAGLE-3 runs, here one schedule); stage 2 = the image-to-SVG
100k x 3 epochs of every image head; max length as the block's EAGLE-3 head (2048 OmniSVG, 4096 InternSVG).

EAGLE pairs (as inference, baselines/vispec_spec.py): pair t = (h_t, x_{t+1}), t = 0.. n-2; its output is trained toward
h_{t+1} and the target's distribution of x_{t+2} (logits[t + 1]); loss where x_{t+2} is a response token. The image span
= the pairs whose x_{t+1} is an image token; their input embedding is the vision feature the target merged into its input
(captured with a pre-hook on the target's language model).

  the glance environment/bin/torchrun --standalone --nproc_per_node 4 baselines/train_vispec.py --kind omnisvg --stage 1 \
      --data <train.jsonl>:6 --out <dir> [--max-steps N]
  ... --stage 2 --data <i2s data>:3 --init <stage-1 head> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from baselines.vispec.model import ViSpecDraft, default_config, load_head, save_head        # noqa: E402
from sightline.paths import MODELS

M = MODELS
TEXT_TARGET = {"omnisvg": M / "OmniSVG1.1_8B-qwen2-text", "internsvg": M / "InternSVG-8B-qwen2-text"}


# ---------------------------------------------------------------------------------------------------- data
def compact(r: dict) -> dict:
    """int32 / uint8 arrays instead of Python int lists (900k InternSVG samples would take ~40 GB per rank as lists)"""
    out = {"input_ids": np.asarray(r["input_ids"], dtype=np.int32), "loss_mask": np.asarray(r["loss_mask"], dtype=np.uint8)}
    if "image_path" in r:
        out["image_path"] = r["image_path"]
    return out


def load_text_samples(data: Path, max_length: int) -> list[dict]:
    out = []
    with data.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append(compact({"input_ids": r["input_ids"][:max_length], "loss_mask": r["loss_mask"][:max_length]}))
    return out


def load_image_samples(kind: str, data: Path, max_length: int) -> list[dict]:
    if kind == "omnisvg":
        sys.path.insert(0, str(ROOT / "h3a"))
        import train_dflash as TD
        return [compact(r) for r in TD.load_samples("i2svg", data, max_length)]
    from internsvg import train_i2s
    return [compact(r) for r in train_i2s.load_samples("i2svg", data, max_length)]


def schedule(datasets, global_batch, seed):
    """datasets: [(samples, epochs)] -> (items [(ds, i)], epoch boundaries in steps)."""
    items, bounds = [], []
    for d, (samples, epochs) in enumerate(datasets):
        n_ep = math.ceil(epochs)
        for e in range(n_ep):
            order = list(range(len(samples)))
            random.Random(seed + 1000 * d + e).shuffle(order)
            take = len(order) if e + 1 <= epochs else int(round(len(order) * (epochs - e)))
            items += [(d, i) for i in order[:take]]
            bounds.append(math.ceil(len(items) / global_batch))
    return items, bounds


# ---------------------------------------------------------------------------------------------------- target
class Target:
    """Frozen target forward -> (h = hidden_states[-1] (n, H), logits (n, V), emb (n, H) merged input embeddings)."""

    def __init__(self, kind: str, stage: int):
        self.kind, self.stage = kind, stage
        self._emb = None
        if stage == 1:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(TEXT_TARGET[kind], torch_dtype=torch.bfloat16,
                                                              attn_implementation="sdpa").cuda().eval()
            self.note = f"text target {TEXT_TARGET[kind].name}"
            self.image_token = None
        elif kind == "omnisvg":
            sys.path.insert(0, str(ROOT / "h3a"))
            import vlm
            self.vlm = vlm
            self.official = vlm.load_official()
            self.model, self.note = vlm.load_target()
            self.image_token = vlm.IMAGE_TOKEN_ID
        else:
            from targets import internsvg as V
            self.V = V
            self.proc = V.load_processor()
            self.model, self.note = V.load_target()
            self.image_token = V.IMAGE_TOKEN_ID
        self.model.requires_grad_(False)
        if stage == 2:
            lm = self.model.model.language_model
            lm.register_forward_pre_hook(self._grab, with_kwargs=True)

    def _grab(self, module, args, kwargs):
        self._emb = kwargs.get("inputs_embeds")

    def lm_head(self):
        return self.model.get_output_embeddings()

    def embed(self):
        return self.model.get_input_embeddings()

    def __call__(self, s: dict):
        with torch.inference_mode():
            f, logits, emb, ids = self._forward(s)
        return f.clone(), logits.clone(), emb.clone(), ids.clone()        # outside inference mode: autograd-usable

    def _forward(self, s: dict):
        ids = torch.from_numpy(s["input_ids"]).long().cuda()[None]
        self._emb = None
        if self.stage == 1:
            out = self.model(ids, output_hidden_states=True)
            emb = self.embed()(ids)
        elif self.kind == "omnisvg":                                       # = baselines/train_eagle3_i2.py
            from qwen_vl_utils import process_vision_info
            img, _ = self.official.preprocess_image_for_svg(s["image_path"], replace_background=True)
            image_inputs, _ = process_vision_info([{"role": "user", "content": [{"type": "image", "image": img}]}])
            pv = self.official.processor.image_processor(images=image_inputs, return_tensors="pt")
            thw = pv["image_grid_thw"].cuda()
            mm = (ids == self.image_token).long()
            self.model.model.rope_deltas = None
            pos, _ = self.model.model.get_rope_index(ids, mm, thw, None, None)
            out = self.model(ids, pixel_values=pv["pixel_values"].cuda().to(torch.bfloat16), image_grid_thw=thw,
                             position_ids=pos, output_hidden_states=True)
            emb = self._emb
        else:                                                              # = train_eagle3_i2s.py
            out = self.model(ids, pixel_values=self.V.pixel_values(self.proc, s["image_path"]),
                             output_hidden_states=True)
            emb = self._emb
        assert emb is not None and emb.shape[:2] == ids.shape, "inputs_embeds not captured"
        return out.hidden_states[-1][0], out.logits[0], emb[0], ids[0]


def image_span(ids: torch.Tensor, image_token):
    """pairs t with x_{t+1} an image token -> [a, b) (contiguous), or None"""
    if image_token is None:
        return None
    w = torch.nonzero(ids == image_token).flatten()
    if w.numel() == 0:
        return None
    first, last = int(w[0]), int(w[-1])
    assert last - first + 1 == w.numel(), "image tokens not contiguous"
    return first - 1, last


# ---------------------------------------------------------------------------------------------------- losses
def rank_loss(logits_sel: torch.Tensor, tp: torch.Tensor, k: int = 10):
    """ViSpec / EAGLE ranking term: Plackett-Luce log-likelihood of the target's top-k order under the draft logits."""
    _, top = torch.topk(tp, k=k, dim=-1)
    s = logits_sel.gather(-1, top).float()
    rev = torch.flip(s, dims=[-1])
    den = torch.flip(torch.logcumsumexp(rev, dim=-1), dims=[-1])
    return -(s - den).sum(-1).mean()


# ---------------------------------------------------------------------------------------------------- optimizer
class MasterAdamW:
    """bf16 compute params, fp32 master copies; AdamW + linear warmup / decay + value clipping (ViSpec's settings)."""

    def __init__(self, params, lr, total, warmup_ratio, clip_value, max_grad_norm=1.0,
                 betas=(0.9, 0.95), weight_decay=0.01):
        self.params = [p for p in params if p.requires_grad]
        self.master = [p.detach().float().clone().requires_grad_(True) for p in self.params]
        self.opt = torch.optim.AdamW(self.master, lr=lr, betas=betas, weight_decay=weight_decay)
        self.base, self.total = lr, total
        self.warm = max(1, int(round(total * warmup_ratio)))
        self.clip, self.max_norm = clip_value, max_grad_norm
        self.n, self.n_skipped, self.last_norm = 0, 0, 0.0

    def lr_at(self, n):
        if n < self.warm:
            return self.base * n / self.warm
        return self.base * max(0.0, (self.total - n) / max(1, self.total - self.warm))

    def step(self):
        """ norm-clip first, then ViSpec's value clip. The  OmniSVG stage-1 run diverged with value clipping
        alone: a few rows have huge target hidden states (Qwen2 massive activations), their SmoothL1 term reached 1e3,
        and Adam's moments never recovered (accuracy 0.37 -> 0.14). A non-finite step is skipped instead of poisoning
        the moments."""
        gs = [p.grad.float() for p in self.params if p.grad is not None]
        total = torch.sqrt(sum((g * g).sum() for g in gs)) if gs else torch.zeros((), device="cuda")
        self.last_norm = float(total)
        if not torch.isfinite(total):
            self.n_skipped += 1
            for p in self.params:
                p.grad = None
            return self.opt.param_groups[0]["lr"]
        scale = float(min(1.0, self.max_norm / (self.last_norm + 1e-6)))
        for p, m in zip(self.params, self.master):
            m.grad = None if p.grad is None else (p.grad.float() * scale).clamp_(-self.clip, self.clip)
        self.n += 1
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(self.n)
        self.opt.step()
        with torch.no_grad():
            for p, m in zip(self.params, self.master):
                p.copy_(m.to(p.dtype))
                p.grad = None
        return self.opt.param_groups[0]["lr"]


# ---------------------------------------------------------------------------------------------------- main
def dist_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
        dist.init_process_group(backend="nccl")
    return world, rank


def all_reduce_(t, world):
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["omnisvg", "internsvg"], required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--data", required=True, help="path[:epochs][,path[:epochs]] (default 1 epoch each)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None, help="stage 2: the stage-1 head")
    ap.add_argument("--global-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default = the official 3e-5 (stage 1) / 3e-6 (stage 2)")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--clip-value", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="10-29: norm clip before ViSpec's value clip")
    ap.add_argument("--max-length", type=int, default=None, help="default 2048 (omnisvg) / 4096 (internsvg)")
    ap.add_argument("--noise-std", type=float, default=0.2)
    ap.add_argument("--mtp-steps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-save-epochs", action="store_true")
    ap.add_argument("--feat-norm", action="store_true",
                    help="10-31 (stage 1): per-channel feature scale sigma (RMS over the response tokens of the first "
                         "--feat-norm-rows rows); the head works on h / sigma and the regression is SmoothL1(o/sigma, "
                         "h/sigma) (vispec_model.py docstring)")
    ap.add_argument("--feat-norm-rows", type=int, default=64)
    ap.add_argument("--v-weight", type=float, default=1.0, help="stage-1 weight of the feature regression (official 1)")
    ap.add_argument("--out-norm", action="store_true",
                    help="10-31 (stage 1): RMSNorm on the head's output, initialised with the target's final-norm "
                         "weights (vispec_model.py docstring; the table's ViSpec†)")
    args = ap.parse_args()
    lr = args.lr or (3e-5 if args.stage == 1 else 3e-6)
    max_len = args.max_length or (2048 if args.kind == "omnisvg" else 4096)
    world, rank = dist_setup()
    assert args.global_batch % world == 0
    accum = args.global_batch // world
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    say = print if rank == 0 else (lambda *a, **k: None)

    datasets = []
    for spec in args.data.split(","):
        p, _, ep = spec.partition(":")
        samples = (load_text_samples(Path(p), max_len) if args.stage == 1
                   else load_image_samples(args.kind, Path(p), max_len))
        datasets.append((samples, float(ep or 1)))
        say(f"[data] {p}: {len(samples)} samples x {float(ep or 1)} epochs", flush=True)
    items, bounds = schedule(datasets, args.global_batch, args.seed)
    total = args.max_steps or math.ceil(len(items) / args.global_batch)
    say(f"[data] {len(items)} sample passes -> {total} steps (global batch {args.global_batch} = {world} x 1 x {accum}); "
        f"epoch ends at steps {bounds}", flush=True)

    tgt = Target(args.kind, args.stage)
    say(f"[target] {tgt.note}", flush=True)
    head, embed = tgt.lm_head(), tgt.embed()
    tcfg = tgt.model.config
    tcfg = getattr(tcfg, "text_config", None) or tcfg
    raw = json.loads((TEXT_TARGET[args.kind] / "config.json").read_text())
    if args.stage == 1:
        cfg = default_config({"hidden_size": tcfg.hidden_size, "num_attention_heads": tcfg.num_attention_heads,
                              "intermediate_size": tcfg.intermediate_size, "rms_norm_eps": tcfg.rms_norm_eps,
                              "rope_theta": raw["rope_theta"], "vocab_size": head.weight.shape[0]})
        if args.feat_norm:
            cfg["feat_norm"] = True
        if args.out_norm:
            cfg["out_norm"] = True
        draft = ViSpecDraft(cfg).to(torch.bfloat16).cuda()
        if args.out_norm:                                   # start from the target's own final norm
            with torch.no_grad():
                draft.out_norm.weight.copy_(tgt.model.model.norm.weight.to(draft.out_norm.weight.dtype))
            say(f"[out-norm] initialised from the target's final norm (mean {float(draft.out_norm.weight.float().mean()):.3f}, "
                f"max {float(draft.out_norm.weight.float().max()):.3f})", flush=True)
        if args.feat_norm:                                  # sigma = per-channel RMS of h over response tokens
            ss, cnt = torch.zeros(tcfg.hidden_size, dtype=torch.float64, device="cuda"), 0
            for s in datasets[0][0][:args.feat_norm_rows]:
                f, _, _, _ = tgt(s)
                sel = torch.from_numpy(s["loss_mask"]).cuda()[:f.shape[0]] > 0
                ss += f[sel].double().pow(2).sum(0)
                cnt += int(sel.sum())
                del f
            sigma = (ss / max(cnt, 1)).sqrt().clamp_min(1e-2).float()
            if world > 1:
                import torch.distributed as dist
                dist.broadcast(sigma, src=0)
            draft.feat_scale.copy_(sigma.to(draft.feat_scale.dtype))
            say(f"[feat-norm] sigma from {cnt} tokens: median {float(sigma.median()):.2f}, max {float(sigma.max()):.1f} "
                f"(channel {int(sigma.argmax())}), min {float(sigma.min()):.3f}", flush=True)
    else:
        assert args.init is not None, "stage 2 starts from the stage-1 head (--init)"
        draft = load_head(args.init, use_img=True)
        draft.cfg["use_img"] = True
        with torch.no_grad():                                             # fresh ViSpec modules (official init)
            fresh = ViSpecDraft({**draft.cfg, "use_img": True})
            draft.imadpt.load_state_dict(fresh.imadpt.state_dict())
            draft.reset_img_fc()
        draft.train()
    assert head.weight.shape[0] == draft.cfg["vocab_size"], (head.weight.shape, draft.cfg["vocab_size"])
    draft.requires_grad_(True)
    if args.stage == 1:
        draft.imadpt.requires_grad_(False)
        draft.img_fc.requires_grad_(False)
    trainable = [p for p in draft.parameters() if p.requires_grad]
    say(f"[draft] trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params; use_img {draft.use_img}; lr {lr}",
        flush=True)
    if world > 1:
        import torch.distributed as dist
        with torch.no_grad():
            for p in trainable:
                dist.broadcast(p.data, src=0)
        torch.manual_seed(args.seed + rank)
    opt = MasterAdamW(trainable, lr, total, args.warmup_ratio, args.clip_value, args.max_grad_norm)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()}, "lr_used": lr,
                                                        "max_length_used": max_len, "world_size": world,
                                                        "total_steps": total, "epoch_bounds": bounds,
                                                        "n_samples": [len(s) for s, _ in datasets],
                                                        "draft_config": draft.cfg}, indent=1))
    curve = (args.out / "train_curve.jsonl").open("a") if rank == 0 else None
    t_log = time.perf_counter()
    stats = torch.zeros(5, dtype=torch.float64)            # loss, acc pass 1 (correct), acc pass 2, count, n_mb
    for step in range(1, total + 1):
        glob = items[(step - 1) * args.global_batch: step * args.global_batch]
        for d, i in glob[rank::world]:
            s = datasets[d][0][i]
            f, logits, emb_full, ids = tgt(s)
            n = f.shape[0]
            lm = torch.from_numpy(s["loss_mask"]).cuda()
            # pairs t = 0.. n-2: emb of x_{t+1}, h_t; targets h_{t+1}, p(x_{t+2}) = logits[t+1]; loss iff x_{t+2} response
            emb_in, hid = emb_full[1:], f[:-1]
            m = torch.zeros(n - 1, dtype=torch.bool, device="cuda")
            m[:n - 2] = lm[2:] > 0
            if int(m.sum()) == 0:
                continue
            noise = (torch.rand_like(hid, dtype=torch.float32) - 0.5) * args.noise_std * 512 / (n - 1)
            hid_n = (hid.float() + noise).to(hid.dtype)
            span = image_span(ids, tgt.image_token) if args.stage == 2 else None
            tl = logits[1:][m].float()
            tp = torch.softmax(tl, dim=-1)
            out1 = draft.forward_seq(emb_in, hid_n, span)
            if args.stage == 1:
                o = out1[m]
                sc = draft.feat_scale.float() if draft.feat_norm else 1.0          # regress on h / sigma
                vloss = F.smooth_l1_loss(o.float() / sc, f[1:][m].float() / sc, reduction="none").mean(-1).mean()
                dl = head(o).float()
                ploss = -(tp * torch.log_softmax(dl, dim=-1)).sum(-1).mean() + 0.1 * rank_loss(dl, tp)
                loss = args.v_weight * vloss + 0.1 * ploss
                acc2 = 0.0
            else:
                outs = [out1]
                for _ in range(args.mtp_steps):
                    hid2 = torch.cat((hid_n[:1], outs[-1][:-1].to(hid_n.dtype)), dim=0)
                    outs.append(draft.forward_seq(emb_in, hid2, span))
                losses, dls = [], []
                for o in outs:
                    dl = head(o[m]).float()
                    pl = (torch.softmax(dl, dim=-1) - tp).abs().sum(-1).mean()
                    losses.append(10 * pl + 0.1 * rank_loss(dl, tp))
                    dls.append(dl)
                loss = sum(losses) / len(losses)                   # = the official mean over the concatenated passes
                dl = dls[0]
                acc2 = float((dls[-1].argmax(-1) == tl.argmax(-1)).sum())
            (loss / accum).backward()
            stats += torch.tensor([float(loss.detach()), float((dl.argmax(-1) == tl.argmax(-1)).sum()), acc2,
                                   float(m.sum()), 1.0], dtype=torch.float64)
            del f, logits, emb_full, tp, tl, out1, loss
        if world > 1:
            with torch.no_grad():
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable]
                flat = torch.cat([g.reshape(-1).float() for g in grads])
                all_reduce_(flat, world)
                flat.div_(world)
                off = 0
                for p, g in zip(trainable, grads):
                    p.grad = flat[off:off + g.numel()].view_as(g).to(p.dtype)
                    off += g.numel()
                del flat
        cur_lr = opt.step()
        if step % args.log_every == 0 or step == total:
            dt = (time.perf_counter() - t_log) / args.log_every
            tot = all_reduce_(stats.cuda(), world).cpu()
            rec = {"step": step, "loss": float(tot[0] / max(tot[4], 1)), "acc": round(float(tot[1] / max(tot[3], 1)), 4),
                   "acc_mtp": round(float(tot[2] / max(tot[3], 1)), 4), "lr": cur_lr, "s_per_step": round(dt, 3),
                   "grad_norm": round(opt.last_norm, 3), "skipped": opt.n_skipped}
            if curve is not None:
                curve.write(json.dumps(rec) + "\n")
                curve.flush()
            say(f"step {step}: {rec}", flush=True)
            t_log = time.perf_counter()
            stats.zero_()
        if not args.no_save_epochs and step in bounds and step < total:
            if rank == 0:
                save_head(draft, args.out / "heads" / f"step{step}")
            say(f"[save] step {step}", flush=True)
            if world > 1:
                torch.distributed.barrier()
    if rank == 0:
        save_head(draft, args.out / "heads" / f"step{total}")
        (args.out / "TRAIN_RC").write_text("0\n")
    say(f"TRAIN_DONE {args.out}", flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
