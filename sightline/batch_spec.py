""" batched chain speculative decoding of N sequences that share one prompt (best-of-N under strict SD).

Every sequence s has its own committed tokens, ChainParams and torch.Generator. One round = ONE drafter forward and ONE
target forward over all active sequences (a padded batch), then each sequence's strict walk exactly as
harness/verify.verify_block (one draw per emitted token from the target's own processed distribution at the AR history;
accept while the draw equals the draft; the first mismatch is emitted as the bonus). Each sequence's output law is
therefore its AR law, and with the same generator its tokens equal harness/spec.spec_generate's (up to bf16 near-ties
between batch shapes; the fp32 audit checks token identity with AR).

Layout (per batch row b, right-padded):
  target KV:   committed[:-1] in entries [0, tlen_b); the round appends the block at [T, T + vs) for every row,
               then each row keeps [0, tlen_b) + [T, T + produced_b) (gathered to the front, so the cache stays compact);
  drafter KV:  the absorbed context in [0, dlen_b), dlen_b = start_b - ctx_b; target_hidden (B, C, F) holds the
               ctx_b not-yet-absorbed context features (right-padded); the drafter keeps [0, dlen_b) + [D, D + ctx_b);
  masks:       explicit 4D masks hide every padded entry; positions are explicit (start_b + offsets), so holes never
               reach RoPE.
The prefill is shared (one batch-1 forward, KV replicated). Sequences leave the batch when they stop (EOS / cap) or are
pruned (prune). The block size is fixed (no shorter block near the cap: emitted tokens are cut at the cap instead,
which can change round counts there but not outputs).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from sightline.spec import _chain_draw
from sightline.verify import ChainParams


@dataclass
class BSeq:
    sid: int
    committed: list[int]
    params: ChainParams
    gen: torch.Generator | None
    acceptance: list[int] = field(default_factory=list)
    stop_reason: str | None = None

    def generated(self, n_prompt: int) -> list[int]:
        return self.committed[n_prompt:]


def _gather_layers(cache, idx: torch.Tensor):
    """keys/values (B, H, T, d) -> (B, H, L', d) with idx (B, L') per row."""
    for layer in cache.layers:
        if getattr(layer, "keys", None) is None or layer.keys is None or layer.keys.numel() == 0:
            continue
        H, d = layer.keys.shape[1], layer.keys.shape[3]
        ix = idx[:, None, :, None].expand(-1, H, -1, d)
        layer.keys = layer.keys.gather(2, ix)
        layer.values = layer.values.gather(2, ix)


def _select_rows(cache, rows: torch.Tensor):
    for layer in cache.layers:
        if getattr(layer, "keys", None) is None or layer.keys is None or layer.keys.numel() == 0:
            continue
        layer.keys = layer.keys.index_select(0, rows)
        layer.values = layer.values.index_select(0, rows)


def _keep_index(prefix: list[int], tail_start: int, tail: list[int], dev) -> torch.Tensor:
    """Row b keeps [0, prefix[b]) + [tail_start, tail_start + tail[b]); padded with 0 to the longest row."""
    L = max(p + t for p, t in zip(prefix, tail))
    idx = torch.zeros((len(prefix), max(L, 1)), dtype=torch.long)
    for b, (p, t) in enumerate(zip(prefix, tail)):
        idx[b, :p] = torch.arange(p)
        idx[b, p:p + t] = torch.arange(tail_start, tail_start + t)
    return idx.to(dev)


def loop_fire_block(seq: list[int], n0: int, n_pos: int, max_period: int, repeats: int, min_period: int) -> list[int]:
    """process.loop_fire(seq[:n0 + j]) for j = 0..n_pos-1 at once (numpy over a tail window, all periods together):
    the smallest P in [min_period, max_period] such that the history ends with `repeats` copies of a P-token cycle;
    0 if none."""
    import numpy as np
    if max_period <= 0:
        return [0] * n_pos
    W = repeats * max_period + n_pos                       # every needed comparison lies in this tail
    lo = max(0, n0 + n_pos - 1 - W)
    w = np.asarray(seq[lo:n0 + n_pos - 1], dtype=np.int64)  # histories end at most at n0 + n_pos - 1
    m = len(w)
    Ps = np.arange(min_period, min(max_period, m - 1) + 1)
    if m < 2 or len(Ps) == 0:
        return [0] * n_pos
    idx = np.arange(m)
    src = idx[None, :] - Ps[:, None]                        # (nP, m): position compared with
    eq = (src >= 0) & (w[None, :] == w[np.clip(src, 0, None)])
    last_false = np.maximum.accumulate(np.where(~eq, idx[None, :], -1), axis=1)
    run = idx[None, :] - last_false                         # consecutive True ending at i, per period
    n = n0 + np.arange(n_pos)                               # true history length per position
    e = n - lo                                              # history end inside the window
    ok_e = e >= 1
    r_at = run[:, np.clip(e - 1, 0, None)]                  # (nP, n_pos)
    fire = (r_at >= (repeats - 1) * Ps[:, None]) & (repeats * Ps[:, None] <= n[None, :]) & ok_e[None, :]
    any_fire = fire.any(axis=0)
    first = fire.argmax(axis=0)
    return [int(Ps[f]) if a else 0 for f, a in zip(first, any_fire)]


def block_process(logits: torch.Tensor, committed: list[list[int]], drafts: list[list[int]], p: ChainParams,
                  eos_id: int) -> torch.Tensor:
    """The processing chain of process.apply_chain for every (row b, block position j) of one round, assuming the
    first j drafts of row b are accepted (the only case in which position j is drawn). logits: (B, vs, V) raw.
    Returns fp32 (B, vs, V): processed logits (greedy) or probabilities (sampling). Element-wise steps are the same
    fp32 operations as apply_chain; the row-wise ones (top-k, sort / cumsum, softmax) run per row."""
    from sightline.process import NEG_INF
    B, vs, V = logits.shape
    x = logits.float()
    dev = x.device
    if p.repetition_penalty != 1.0:
        pres = torch.zeros((B, V), dtype=torch.bool, device=dev)
        for b in range(B):
            pres[b, torch.unique(torch.as_tensor(committed[b], dtype=torch.long, device=dev))] = True
        dl = torch.as_tensor(drafts, dtype=torch.long, device=dev)       # (B, K)
        ar = torch.arange(B, device=dev)
        pen = p.repetition_penalty
        for j in range(vs):
            col = x[:, j]
            x[:, j] = torch.where(pres, torch.where(col > 0, col / pen, col * pen), col)
            if j < vs - 1:
                pres[ar, dl[:, j]] = True
    if getattr(p, "loop_ban", 0):
        assert p.loop_action == "eos", "batched decoder: only the stop rule (loop_action eos) is supported"
        for b in range(B):
            fires = loop_fire_block(committed[b] + drafts[b], len(committed[b]), vs, p.loop_ban,
                                    p.loop_repeats, p.loop_min_period)
            for j, P in enumerate(fires):
                if P:
                    x[b, j] = NEG_INF
                    x[b, j, eos_id] = 0.0
    if p.greedy:
        return x
    x = x.reshape(B * vs, V)
    if p.temperature != 1.0:
        x = x / p.temperature
    if 0 < p.top_k < V:
        thr = torch.topk(x, p.top_k, dim=-1).values[..., -1, None]
        x = torch.where(x < thr, NEG_INF, x)
    if 0.0 < p.top_p < 1.0:
        sx, si = torch.sort(x, descending=False, dim=-1)
        cum = sx.softmax(dim=-1).cumsum(dim=-1)
        rem = cum <= (1.0 - p.top_p)
        rem[..., -1:] = False
        sx = torch.where(rem, NEG_INF, sx)
        x = torch.empty_like(x).scatter_(-1, si, sx)
    return torch.softmax(x, dim=-1).reshape(B, vs, V)


class BatchChainSD:
    def __init__(self, target, head, eos_ids: list[int], max_new_tokens: int, block_size: int | None = None,
                 cudnn_sdp: bool = False, fast_walk: bool = True, rng_mode: str = "sequential",
                 ar_mode: bool = False):
        # the cuDNN SDPA backend rebuilds a graph for every new attention shape (~30 ms each); every round of a
        # decoder has new shapes, so it is off by default here
        torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp)
        self.target, self.head = target, head
        self.fast_walk = fast_walk       # batched processing chain (block_process); False = per-position _chain_draw
        # rng_mode "sequential": one multinomial call per emitted token, the same generator use as spec_generate (same
        # tokens for the same seed); "block": one call per row per round for all vs positions, then the walk uses the
        # draw of position j only when drafts[:j] were accepted -- each position's draw comes from its own processed
        # distribution, so every sequence is still an exact sample of the target, but seeds no longer reproduce the
        # single-sequence tokens. "block" is the latency mode (vs - 1 fewer host syncs per row and round).
        assert rng_mode in ("sequential", "block")
        self.rng_mode = rng_mode
        self.ar_mode = ar_mode           # batched plain AR (no drafter, one token per row and round): the b-N AR baseline
        self.eos_id = eos_ids[0]
        self.eos = set(eos_ids)
        self.max_new = max_new_tokens
        self.vs = int(block_size or head.block_size)
        self.layer_ids = head.target_layer_ids
        self.mask_id = head.mask_token_id
        self.dev = next(target.parameters()).device
        self.dtype = next(target.parameters()).dtype
        self.nfe = 0                     # batched target forwards (prefill included)
        self.rounds = 0
        self.t = {"draft": 0.0, "verify": 0.0, "walk": 0.0, "cache": 0.0}

    def _sync_time(self) -> float:
        torch.cuda.synchronize()
        return time.perf_counter()

    @torch.inference_mode()
    def start(self, prompt_ids: list[int], params: list[ChainParams], gens: list[torch.Generator | None],
              prefill_kwargs: dict | None = None, positions=None):
        """prefill_kwargs / positions: the VLM adapter of vlm.py (I2SVGInputs: M-RoPE prefill positions, pixel
        values; positions(pos (B, n)) -> (3, B, n)); default = 1-D positions (text targets). The drafter stays 1-D."""
        from dflash.model import _make_cache, extract_context_feature
        from transformers import DynamicCache
        n = len(prompt_ids)
        self.n_prompt, self.max_len = n, n + self.max_new
        B = len(params)
        cache = DynamicCache()
        self.positions = positions or (lambda pos: pos)
        kw = prefill_kwargs if prefill_kwargs is not None else {"position_ids": torch.arange(n, device=self.dev)[None]}
        out = self.target(torch.tensor([prompt_ids], device=self.dev), past_key_values=cache, use_cache=True,
                          output_hidden_states=True, **kw)
        self.nfe += 1
        for layer in cache.layers:
            layer.keys = layer.keys.expand(B, -1, -1, -1).contiguous()
            layer.values = layer.values.expand(B, -1, -1, -1).contiguous()
        self.tcache = cache
        self.th = extract_context_feature(out.hidden_states, self.layer_ids).expand(B, -1, -1).contiguous()
        self.ctx = [n] * B
        self.dcache = _make_cache(self.head.config)
        self.dlen = [0] * B
        self.tlen = [n] * B
        last = out.logits[0, -1]
        self.seqs = []
        for i, (p, g) in enumerate(zip(params, gens)):
            first = int(_chain_draw(last, list(prompt_ids), p, g))
            s = BSeq(i, list(prompt_ids) + [first], p, g)        # acceptance as spec_generate: per round only
            if first in self.eos:
                s.stop_reason, s.acceptance = "eos", [1]
            self.seqs.append(s)
        self.rows = list(range(B))       # batch row r <-> self.seqs[self.rows[r]]
        self._drop_finished()

    # ------------------------------------------------------------------ batch bookkeeping
    def _keep_rows(self, keep: list[int]):
        if len(keep) == len(self.rows):
            return
        sel = torch.tensor(keep, dtype=torch.long, device=self.dev)
        _select_rows(self.tcache, sel)
        _select_rows(self.dcache, sel)
        self.th = self.th.index_select(0, sel)
        self.rows = [self.rows[r] for r in keep]
        self.ctx = [self.ctx[r] for r in keep]
        self.dlen = [self.dlen[r] for r in keep]
        self.tlen = [self.tlen[r] for r in keep]

    def _drop_finished(self):
        self._keep_rows([r for r, i in enumerate(self.rows) if self.seqs[i].stop_reason is None])

    def prune(self, keep_sids: set[int]):
        """Stop every active sequence whose sid is not in keep_sids."""
        for i in self.rows:
            if i not in keep_sids:
                self.seqs[i].stop_reason = "pruned"
        self._drop_finished()

    @property
    def active(self) -> list[int]:
        return list(self.rows)

    # ------------------------------------------------------------------ one round
    @torch.inference_mode()
    def _step_ar(self):
        """One batched AR step: every active row feeds its root (committed[-1]) and draws the next token."""
        B, dev = len(self.rows), self.dev
        seqs = [self.seqs[i] for i in self.rows]
        t0 = self._sync_time()
        T = self.tcache.get_seq_length()
        ids = torch.tensor([[s.committed[-1]] for s in seqs], device=dev)
        pos = torch.tensor([[len(s.committed) - 1] for s in seqs], device=dev)
        neg = torch.finfo(self.dtype).min
        mask = torch.full((B, 1, 1, T + 1), neg, dtype=self.dtype)
        for b in range(B):
            mask[b, 0, 0, :self.tlen[b]] = 0
            mask[b, 0, 0, T] = 0
        out = self.target(ids, position_ids=self.positions(pos), past_key_values=self.tcache, use_cache=True,
                          attention_mask=mask.to(dev))
        self.nfe += 1
        t1 = self._sync_time()
        p0 = seqs[0].params
        proc = block_process(out.logits, [x.committed for x in seqs], [[] for _ in seqs], p0, self.eos_id)
        if p0.greedy:
            ys = proc[:, 0].argmax(-1).tolist()
        else:
            ys = [int(torch.multinomial(proc[b, 0].reshape(1, -1), 1, generator=s.gen)) for b, s in enumerate(seqs)]
        for s, y in zip(seqs, ys):
            s.committed.append(y)
            s.acceptance.append(1)
            if y in self.eos:
                s.stop_reason = "eos"
            elif len(s.committed) >= self.max_len:
                s.stop_reason = "max_new_tokens"
        t2 = self._sync_time()
        _gather_layers(self.tcache, _keep_index(self.tlen, T, [1] * B, dev))
        self.tlen = [t + 1 for t in self.tlen]
        self.rounds += 1
        self._drop_finished()
        t3 = self._sync_time()
        for k, a, z in (("verify", t0, t1), ("walk", t1, t2), ("cache", t2, t3)):
            self.t[k] += z - a

    @torch.inference_mode()
    def step(self):
        from dflash.model import _output_head, _raw_input_embeddings, extract_context_feature
        B = len(self.rows)
        if B == 0:
            return
        if self.ar_mode:
            return self._step_ar()
        vs, K, dev = self.vs, self.vs - 1, self.dev
        seqs = [self.seqs[i] for i in self.rows]
        starts = [len(s.committed) - 1 for s in seqs]
        t0 = self._sync_time()
        # ---- drafter: [absorbed ctx (cache) | new ctx (th) | block]
        C = self.th.shape[1]
        D = self.dcache.get_seq_length() if self.dcache.layers and getattr(self.dcache.layers[0], "keys", None) is not None else 0
        block_ids = torch.full((B, vs), self.mask_id, dtype=torch.long, device=dev)
        block_ids[:, 0] = torch.tensor([s.committed[-1] for s in seqs], device=dev)
        noise = _raw_input_embeddings(self.target, block_ids, 1.0)
        pos = torch.empty((B, C + vs), dtype=torch.long)
        dmask = torch.zeros((B, 1, vs, D + C + vs), dtype=torch.bool)
        for b in range(B):
            c, st = self.ctx[b], starts[b]
            pos[b, :C] = st - 1
            pos[b, :c] = torch.arange(st - c, st)
            pos[b, C:] = torch.arange(st, st + vs)
            dmask[b, 0, :, :self.dlen[b]] = True
            dmask[b, 0, :, D:D + c] = True
            dmask[b, 0, :, D + C:] = True
        h = self.head(target_hidden=self.th, noise_embedding=noise, position_ids=pos.to(dev),
                      attention_mask=dmask.to(dev), past_key_values=self.dcache, use_cache=True)[:, 1 - vs:, :]
        drafts = self.head.compute_logits(h, _output_head(self.target)).argmax(-1)          # (B, K)
        _gather_layers(self.dcache, _keep_index(self.dlen, D, self.ctx, dev))
        self.dlen = [d + c for d, c in zip(self.dlen, self.ctx)]                             # == starts
        t1 = self._sync_time()
        # ---- target verify: [committed[:-1] (cache) | root + K drafts]
        T = self.tcache.get_seq_length()
        block = torch.cat([block_ids[:, :1], drafts], dim=1)
        positions = torch.tensor(starts, device=dev)[:, None] + torch.arange(vs, device=dev)[None]
        neg = torch.finfo(self.dtype).min
        tmask = torch.full((B, 1, vs, T + vs), neg, dtype=self.dtype)
        causal = torch.triu(torch.full((vs, vs), neg, dtype=self.dtype), diagonal=1)
        for b in range(B):
            tmask[b, 0, :, :self.tlen[b]] = 0
            tmask[b, 0, :, T:] = causal
        out = self.target(block, position_ids=self.positions(positions), past_key_values=self.tcache, use_cache=True,
                          attention_mask=tmask.to(dev), output_hidden_states=True)
        self.nfe += 1
        logits = out.logits
        t2 = self._sync_time()
        # ---- per-sequence strict walk (= verify_block without its diagnostics)
        dl = drafts.tolist()
        produced = []
        if self.fast_walk:
            p0 = seqs[0].params
            assert all(x.params == p0 for x in seqs), "fast walk: one ChainParams for the batch"
            proc = block_process(logits, [x.committed for x in seqs], dl, p0, self.eos_id)
            am = proc.argmax(-1).tolist() if p0.greedy else None
            if am is None and self.rng_mode == "block":
                am = [torch.multinomial(proc[b], 1, generator=seqs[b].gen)[:, 0].tolist() for b in range(B)]
        for b, s in enumerate(seqs):
            history = list(s.committed)
            emitted = []
            full = True
            for j in range(1, vs + 1):
                if self.fast_walk:
                    y = am[b][j - 1] if am is not None else int(torch.multinomial(
                        proc[b, j - 1].reshape(1, -1), 1, generator=s.gen))
                else:
                    y = int(_chain_draw(logits[b, j - 1], history, s.params, s.gen))
                emitted.append(y)
                if j == vs:                      # the bonus position (all K drafts accepted)
                    break
                if y != dl[b][j - 1]:
                    full = False
                    break
                history.append(y)
            stop = False
            for i, tok in enumerate(emitted):
                if tok in self.eos:
                    emitted = emitted[:i + 1]
                    stop = True
                    break
            if len(s.committed) + len(emitted) > self.max_len:
                emitted = emitted[:self.max_len - len(s.committed)]
            s.committed += emitted
            s.acceptance.append(len(emitted))
            if stop:
                s.stop_reason = "eos"
            elif len(s.committed) >= self.max_len:
                s.stop_reason = "max_new_tokens"
            produced.append(len(emitted))
        t3 = self._sync_time()
        # ---- caches: keep root + accepted drafts of every row; their target features become the new context
        _gather_layers(self.tcache, _keep_index(self.tlen, T, produced, dev))
        self.tlen = [t + p for t, p in zip(self.tlen, produced)]
        feats = extract_context_feature(out.hidden_states, self.layer_ids)                  # (B, vs, F)
        Cn = max(produced)
        th = torch.zeros((B, Cn, feats.shape[-1]), dtype=feats.dtype, device=dev)
        for b, p in enumerate(produced):
            th[b, :p] = feats[b, :p]
        self.th, self.ctx = th, produced
        self.rounds += 1
        self._drop_finished()
        t4 = self._sync_time()
        for k, a, z in (("draft", t0, t1), ("verify", t1, t2), ("walk", t2, t3), ("cache", t3, t4)):
            self.t[k] += z - a

    def run(self, max_rounds: int | None = None):
        n = 0
        while self.rows and (max_rounds is None or n < max_rounds):
            self.step()
            n += 1
