"""H2 step 1: cache the draft head's block distributions along the
target's realized continuations, for exact offline simulation of candidate trees.

Why the simulation is exact, greedy AND sampling: the harness verifies by sample-and-compare,
which draws exactly one target token per emitted token from the seeded generator, so the
emitted sequence does not depend on the drafter (B3: main100k and main gave identical sampling
outputs on 900/900 rows). A candidate tree therefore accepts the longest prefix of that realized
continuation it contains, and the bonus token is the next realized token. The draft
distribution at a block start depends only on the committed prefix, never on the tree, so one
cache serves every design.

Cached per (mode, row), for every block start t = 0.. n_gen-2 (block = [tokens[t], mask x k],
k = 15, fewer in the last 15 positions of a capped stream, exactly as the harness does):
  n_off       (T,)            offsets drafted at this block start
  top_ids/lp  (T, 15, 32)     draft top-32 ids / log-probs (fp32 log-softmax, stored fp16)
  real_rank   (T, 15)         rank of the realized token tokens[t+j] in the draft (-1 = none)
  real_lp     (T, 15)         its log-prob
  coord_best  (T, 15)         the draft's best coordinate token
  coord_lse   (T, 15)         log of the draft mass on coordinates
  nb_lp       (T, 15, 49)     log-probs on the 7x7 grid around coord_best, row-major (dy, dx)
                              from (-3, -3); NaN off the 200x200 grid
  xy_x, xy_y  (T, 15, 8)      top-8 x / y of the coordinate marginals
  xy_lp       (T, 15, 8, 8)   joint log-probs on that grid, [y index, x index]
  copy_ids    (T, 16)         last 16 distinct coordinate tokens of tokens[:t+1] (-1 pad)
  copy_lp     (T, 15, 16)     their log-probs
  type_lse    (T, 15, 6)      log mass per class: coord, cmd, color, arc, eos, other
plus the row metadata and the reference run's per-block acceptance (for the chain replay
check in simulate.py).

Draft path: the same calls as harness DFlashDrafter.propose (official dflash model code). The
draft KV cache holds the context features of positions [0, s); each step appends one position
(the whole prompt for t = 0), runs the block, and crops back to s. Context features come from
one teacher-forced target forward over prompt + continuation; in decoding they come from the
verification forwards, equal up to bf16 kernel-shape noise (the replay check measures it).

Usage (GPU, the dflash environment):
  python cache_draft.py --head <head dir> --out <dir> \
      [--ref <B3 eval dir with shard-*/results.jsonl>] [--modes greedy,sample] \
      [--sets bench,extra] [--shard i/n] [--limit N] [--gidx 0,5,7]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.svgtok.types import (ARC_END, ARC_START, CMD_CLOSE, CMD_MOVE, COLOR_END,  # noqa: E402
                          COLOR_START, COORD_START, EOS, GRID)
from sightline.paths import MODELS, RUNS

MODEL = MODELS
REF_DEFAULT = Path(f"{RUNS}/b3-eval/"
                   "omnisvg-dflash-main100k-1010/step16666")
MAX_NEW = 1536
TOPK, NB_R, XY_K, N_COPY = 32, 3, 8, 16
NB_W = 2 * NB_R + 1
N_COORD = GRID * GRID
TYPE_NAMES = ["coord", "cmd", "color", "arc", "eos", "other"]


def load_ref(ref_dir: Path) -> dict:
    """(mode, gidx) -> B3 results record (spec_tokens, acceptance, onset, eos,...)."""
    out = {}
    for f in sorted(ref_dir.glob("shard-*/results.jsonl")):
        for line in f.read_text().split("\n"):
            if line.strip():
                r = json.loads(line)
                out[(r["mode"], r["gidx"])] = r
    return out


class Stats:
    """Per-row GPU buffers, filled one block start at a time, moved to CPU once."""

    def __init__(self, T: int, K: int, dev):
        f16, i32 = torch.float16, torch.int32
        nan = float("nan")
        self.n_off = torch.zeros(T, dtype=torch.int8, device=dev)
        self.top_ids = torch.full((T, K, TOPK), -1, dtype=i32, device=dev)
        self.top_lp = torch.full((T, K, TOPK), nan, dtype=f16, device=dev)
        self.real_rank = torch.full((T, K), -1, dtype=i32, device=dev)
        self.real_lp = torch.full((T, K), nan, dtype=f16, device=dev)
        self.coord_best = torch.full((T, K), -1, dtype=i32, device=dev)
        self.coord_lse = torch.full((T, K), nan, dtype=f16, device=dev)
        self.nb_lp = torch.full((T, K, NB_W * NB_W), nan, dtype=f16, device=dev)
        self.xy_x = torch.full((T, K, XY_K), -1, dtype=torch.int16, device=dev)
        self.xy_y = torch.full((T, K, XY_K), -1, dtype=torch.int16, device=dev)
        self.xy_lp = torch.full((T, K, XY_K, XY_K), nan, dtype=f16, device=dev)
        self.copy_ids = torch.full((T, N_COPY), -1, dtype=i32, device=dev)
        self.copy_lp = torch.full((T, K, N_COPY), nan, dtype=f16, device=dev)
        self.type_lse = torch.full((T, K, len(TYPE_NAMES)), nan, dtype=f16, device=dev)

    def numpy(self) -> dict:
        return {k: v.cpu().numpy() for k, v in vars(self).items()}


def _nb_offsets(dev):
    d = torch.arange(-NB_R, NB_R + 1, device=dev)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    return dy.reshape(-1), dx.reshape(-1)


@torch.inference_mode()
def fill_block(st: Stats, t: int, lp: torch.Tensor, realized: torch.Tensor,
               copy_ids: torch.Tensor, nb_dydx):
    """lp: (k, V) fp32 draft log-probs for offsets 1..k; realized: (m,) realized tokens
    tokens[t+1.. t+m], m <= k; copy_ids: (N_COPY,) with -1 pad."""
    k = lp.shape[0]
    st.n_off[t] = k
    top = lp.topk(TOPK, dim=-1)
    st.top_ids[t, :k] = top.indices.to(torch.int32)
    st.top_lp[t, :k] = top.values.half()
    m = realized.shape[0]
    if m:
        rl = lp[:m].gather(1, realized[:, None])                  # (m, 1)
        st.real_lp[t, :m] = rl[:, 0].half()
        st.real_rank[t, :m] = (lp[:m] > rl).sum(1).to(torch.int32)
    if lp.shape[1] < COORD_START + N_COORD:
        # not the OmniSVG vocabulary (e.g. InternSVG-8B, 152,126 ids): only the generic fields (top-k, realized,
        # copy); the coordinate / token-type fields keep their NaN / -1 defaults (top-k tree designs do not read them).
        st.copy_ids[t] = copy_ids
        valid = copy_ids >= 0
        cl = lp[:, copy_ids.clamp(min=0, max=lp.shape[1] - 1)]
        st.copy_lp[t, :k] = torch.where(valid[None, :], cl, torch.full_like(cl, float("nan"))).half()
        return
    c = lp[:, COORD_START:COORD_START + N_COORD]                  # (k, 40000), idx = y*200 + x
    coord_lse = torch.logsumexp(c, -1)
    st.coord_lse[t, :k] = coord_lse.half()
    best = c.argmax(-1)
    st.coord_best[t, :k] = (best + COORD_START).to(torch.int32)
    bx, by = best % GRID, best // GRID
    dy, dx = nb_dydx
    xx = bx[:, None] + dx[None, :]
    yy = by[:, None] + dy[None, :]
    ok = (xx >= 0) & (xx < GRID) & (yy >= 0) & (yy < GRID)
    idx = (yy.clamp(0, GRID - 1) * GRID + xx.clamp(0, GRID - 1))
    nb = c.gather(1, idx)
    st.nb_lp[t, :k] = torch.where(ok, nb, torch.full_like(nb, float("nan"))).half()
    c2 = c.view(k, GRID, GRID)                                     # [y, x]
    xt = torch.logsumexp(c2, dim=1).topk(XY_K, -1).indices         # x marginal (sum over y)
    yt = torch.logsumexp(c2, dim=2).topk(XY_K, -1).indices         # y marginal (sum over x)
    st.xy_x[t, :k] = xt.to(torch.int16)
    st.xy_y[t, :k] = yt.to(torch.int16)
    ar = torch.arange(k, device=lp.device)
    st.xy_lp[t, :k] = c2[ar[:, None, None], yt[:, :, None], xt[:, None, :]].half()
    st.copy_ids[t] = copy_ids
    valid = copy_ids >= 0
    cl = lp[:, copy_ids.clamp(min=0)]
    st.copy_lp[t, :k] = torch.where(valid[None, :], cl, torch.full_like(cl, float("nan"))).half()
    known = torch.stack([
        coord_lse,
        torch.logsumexp(lp[:, CMD_MOVE:CMD_CLOSE + 1], -1),
        torch.logsumexp(lp[:, COLOR_START:COLOR_END + 1], -1),
        torch.logsumexp(lp[:, ARC_START:ARC_END + 1], -1),
        lp[:, EOS],
    ], dim=-1)                                                     # (k, 5)
    other = torch.log1p(-torch.exp(torch.logsumexp(known, -1)).clamp(max=1 - 1e-7))
    st.type_lse[t, :k] = torch.cat([known, other[:, None]], -1).half()


@torch.inference_mode()
def cache_row(target, head, prompt: list[int], tokens: list[int], max_new: int = MAX_NEW,
              block_size: int = 16) -> dict:
    """Draft block distributions at every block start along prompt + tokens."""
    from dflash.model import (_crop_to, _make_cache, _output_head, _raw_input_embeddings,
                              extract_context_feature)
    dev = next(head.parameters()).device
    n_p, n_gen = len(prompt), len(tokens)
    ids = torch.tensor([list(prompt) + list(tokens)], device=dev)
    out = target(ids, output_hidden_states=True, logits_to_keep=1)
    feats = extract_context_feature(out.hidden_states, head.target_layer_ids)   # (1, L, n*H)
    del out
    T = max(n_gen - 1, 0)                 # the last token starts no block
    K = block_size - 1
    st = Stats(T, K, dev)
    cache = _make_cache(head.config)
    lm_head = _output_head(target)
    nb_dydx = _nb_offsets(dev)
    tok_dev = torch.tensor(list(tokens), device=dev)
    recent: list[int] = []                # distinct coordinate tokens, most recent last
    copy_buf = torch.full((N_COPY,), -1, dtype=torch.long)
    for t in range(T):
        s = n_p + t                       # global position of tokens[t] = committed[-1]
        tok = tokens[t]
        if COORD_START <= tok < COORD_START + N_COORD:
            if tok in recent:
                recent.remove(tok)
            recent.append(tok)
            del recent[:-N_COPY]
            copy_buf.fill_(-1)
            copy_buf[:len(recent)] = torch.tensor(recent[::-1])          # most recent first
        vs = min(block_size, max_new - t)  # harness: min(block, max_len - len(committed) + 1)
        k = vs - 1
        ctx = feats[:, :s] if t == 0 else feats[:, s - 1:s]
        block_ids = torch.full((1, vs), head.mask_token_id, dtype=torch.long, device=dev)
        block_ids[0, 0] = tok
        pos = torch.arange(s - ctx.shape[1], s + vs, device=dev).unsqueeze(0)
        noise = _raw_input_embeddings(target, block_ids, 1.0)
        h = head(target_hidden=ctx, noise_embedding=noise, position_ids=pos,
                 past_key_values=cache, use_cache=True)[:, 1 - vs:, :]
        _crop_to(cache, s)
        if k <= 0:
            continue
        lp = torch.log_softmax(head.compute_logits(h, lm_head)[0].float(), dim=-1)
        realized = tok_dev[t + 1:t + 1 + k]
        fill_block(st, t, lp, realized, copy_buf.to(dev), nb_dydx)
    return st.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=REF_DEFAULT,
                    help="B3 eval dir whose spec_tokens are the realized continuations")
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--sets", default="bench,extra")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gidx", default=None, help="comma-separated row subset (tests)")
    ap.add_argument("--target", type=Path, default=MODEL / "OmniSVG1.1_8B-qwen2-text")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    sys.path.insert(0, str(ROOT / "t7"))
    from eval_b3 import load_rows                                  # same rows, same gidx
    rows = [r for r in load_rows(args.sets.split(",")) if r["gidx"] % shard_n == shard_i]
    if args.gidx:
        keep = {int(x) for x in args.gidx.split(",")}
        rows = [r for r in rows if r["gidx"] in keep]
    if args.limit:
        rows = rows[:args.limit]
    ref = load_ref(args.ref)

    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    head = DFlashDraftModel.from_pretrained(args.head, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"head {args.head}: block={head.block_size} mask={head.mask_token_id} "
          f"layers={list(head.target_layer_ids)}; rows={len(rows)} modes={args.modes}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"protocol-shard-{shard_i}-of-{shard_n}.json").write_text(json.dumps({
        "head": str(args.head), "target": str(args.target), "ref": str(args.ref),
        "modes": args.modes, "sets": args.sets, "shard": args.shard, "n_rows": len(rows),
        "max_new": MAX_NEW, "topk": TOPK, "nb_radius": NB_R, "xy_k": XY_K, "n_copy": N_COPY,
        "type_names": TYPE_NAMES, "env": "the dflash environment torch2.13/transformers5.15"}, indent=2))

    t_all = time.perf_counter()
    n_done = n_blocks = 0
    for mode in args.modes.split(","):
        (args.out / mode).mkdir(exist_ok=True)
        for r in rows:
            f = args.out / mode / f"{r['gidx']:04d}.npz"
            if f.exists():
                continue
            rec = ref.get((mode, r["gidx"]))
            if rec is None:
                print(f"[skip] {mode} gidx {r['gidx']}: not in ref", flush=True)
                continue
            tokens = rec["spec_tokens"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            arrays = cache_row(target, head, r["prompt_input_ids"], tokens,
                               block_size=int(head.block_size))
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            meta = dict(gidx=r["gidx"], mode=mode, subtype=r["subtype"], set=r["set"],
                        n_prompt=len(r["prompt_input_ids"]),
                        tokens=np.asarray(tokens, dtype=np.int32),
                        ref_acceptance=np.asarray(rec["acceptance"], dtype=np.int16),
                        ref_onset=-1 if rec["onset"] is None else rec["onset"],
                        ref_eos=bool(rec["eos"]), seconds=dt)
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, **arrays, **meta)
            os.replace(tmp, f)
            n_done += 1
            n_blocks += len(tokens) - 1
            print(f"[done] {mode} gidx {r['gidx']} ({r['subtype']}): {len(tokens)} tok, "
                  f"{dt:.1f}s ({dt / max(len(tokens) - 1, 1) * 1e3:.2f} ms/block)", flush=True)
    wall = time.perf_counter() - t_all
    print(f"CACHE_DONE rows={n_done} blocks={n_blocks} wall={wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
