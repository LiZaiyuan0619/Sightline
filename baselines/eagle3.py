""" EAGLE-3 drafting for the strict tree verifier (sightline/tree_spec.py), the EAGLE-3 baseline rows of the main table.

The drafter is SpecForge's LlamaForCausalLMEagle3 trained by baselines/train_eagle3.py; its one decoder layer is run here by
our own forward (same weights, RoPE and attention arithmetic as SpecForge's training layer) so that the KV cache and the tree
mask are ours. Conventions (= SpecForge's online training, see train_eagle3.py):
  context pair at position t:  hidden = fc(cat(target layers [1, 13, 24] at t)), token = x_{t+1}  ->  logits for x_{t+2};
                               RoPE position t; causal over the absorbed context (TTT step 0).
  tree query of a node v at depth d (v's token is a candidate for x_{m+d}, m = index of the last committed token):
                               hidden = output of the query that proposed v (the last context pair for d = 1),
                               token = v's token  ->  logits for v's children; RoPE position (m - 1) + d (TTT step d);
                               attends to the whole absorbed context and to the queries of v's ancestors and itself
                               (TTT: step-0 keys at positions <= t plus the same position's keys of steps 1..d).
Tree growth is EAGLE-2's (dynamic tree): depth 1 = the top-k of the last context pair; at every further depth the top-k
frontier nodes by cumulative log-prob are expanded with their top-k children; finally the best `budget` nodes by cumulative
log-prob over everything proposed form the (prefix-closed) tree. topk = 1 gives the chain.
Verification and the walk are tree_spec_generate's (strict sample-and-compare, any tree is lossless).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
import torch.nn.functional as F
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REFS / "SpecForge"))
from sightline.tree_utils import TreeNode                                     # noqa: E402

AUX = [1, 13, 24]


def load_eagle3(path, dtype=torch.bfloat16, device="cuda"):
    from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
    m = LlamaForCausalLMEagle3.from_pretrained(path, torch_dtype=dtype).to(device).eval()
    m.aux_layers = list((getattr(m.config, "eagle_config", None) or {}).get("eagle_aux_hidden_state_layer_ids", AUX))
    # from_pretrained (transformers 5) leaves SpecForge's non-persistent RoPE buffers (inv_freq, cos/sin cache) as
    # uninitialized memory, which broke every EAGLE-3 decode of 10-18/19 (tokens per round 1.0-2.0; teacher-forced acc 0.0-0.45 instead
    # of 0.95). Rebuild them as train_eagle3.py's constructor does: computed in fp32; the forward casts the cache to the
    # query dtype, i.e. bf16 queries see exactly the training-time bf16 cache.
    attn = m.midlayer.self_attn
    attn._init_rope()
    attn.rotary_emb.to(device)
    inv = attn.rotary_emb.inv_freq.float()
    ref = 1.0 / (attn.rotary_emb.base ** (torch.arange(0, 2 * inv.numel(), 2, device=inv.device).float() / (2 * inv.numel())))
    assert torch.allclose(inv, ref, rtol=1e-6, atol=0), "EAGLE-3 RoPE buffers not rebuilt"
    return m


def is_eagle3_head(path) -> bool:
    import json
    cfg = json.loads((Path(path) / "config.json").read_text())
    return "LlamaForCausalLMEagle3" in (cfg.get("architectures") or [])


class Eagle3Drafter:
    """One sequence. State: the draft KV of the absorbed context pairs (positions [0, absorbed)) and the target features
    of the positions not yet absorbed (pending, positions [absorbed, absorbed + P))."""
    needs_hidden = True

    def __init__(self, model, target, topk: int = 10, depth: int = 8, budget: int = 63):
        self.m, self.target = model, target
        self.topk, self.depth, self.budget = topk, depth, budget
        layer = model.midlayer
        self.attn = layer.self_attn
        self.nh, self.nkv, self.hd = self.attn.num_heads, self.attn.num_key_value_heads, self.attn.head_dim
        self.dev = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.d2t = model.d2t.to(self.dev)
        self.K = self.V = None                       # (1, nkv, absorbed, hd), RoPE applied
        self.absorbed = 0
        self.pending = None                          # (1, P, 3H) target features
        self.last_h = None

    # ---------------------------------------------------------------- target features
    def _feats(self, hidden_states):
        return torch.cat([hidden_states[l + 1] for l in self.m.aux_layers], dim=-1)

    def prefill_done(self, hidden_states):
        self.pending = self._feats(hidden_states)
        self.K = self.V = None
        self.absorbed = 0

    def update_hidden_kept(self, hidden_states, keep: list[int]):
        self.pending = self._feats(hidden_states)[:, keep, :]

    # ---------------------------------------------------------------- the draft layer
    def _rope(self, x, pos):
        from specforge.modeling.draft.llama3_eagle import rotate_half
        cos, sin = self.attn.rotary_emb(x, seq_len=int(pos.max()) + 1)
        cos = cos.squeeze(1).squeeze(0).to(x.device)[pos].unsqueeze(1)          # (1, 1, q, hd)
        sin = sin.squeeze(1).squeeze(0).to(x.device)[pos].unsqueeze(1)
        return (x * cos) + (rotate_half(x) * sin)

    def _layer(self, emb, hid, pos, mask, extra_kv=None):
        """emb, hid: (1, q, H); pos: (1, q) RoPE positions; mask: additive (1, 1, q, S) over [context | extra | self].
        Returns (out (1, q, H), k (1, nkv, q, hd), v)."""
        layer = self.m.midlayer
        h_in = hid
        x = torch.cat((layer.input_layernorm(emb), layer.hidden_norm(hid)), dim=-1)
        q = self.attn.q_proj(x).view(1, -1, self.nh, self.hd).transpose(1, 2)
        k = self.attn.k_proj(x).view(1, -1, self.nkv, self.hd).transpose(1, 2)
        v = self.attn.v_proj(x).view(1, -1, self.nkv, self.hd).transpose(1, 2)
        q, k = self._rope(q, pos), self._rope(k, pos)
        ks = [t for t in (self.K, extra_kv[0] if extra_kv else None, k) if t is not None]
        vs = [t for t in (self.V, extra_kv[1] if extra_kv else None, v) if t is not None]
        Kall, Vall = torch.cat(ks, dim=2), torch.cat(vs, dim=2)
        rep = self.nh // self.nkv
        Kall = Kall[:, :, None].expand(1, self.nkv, rep, *Kall.shape[2:]).reshape(1, self.nh, *Kall.shape[2:])
        Vall = Vall[:, :, None].expand(1, self.nkv, rep, *Vall.shape[2:]).reshape(1, self.nh, *Vall.shape[2:])
        o = F.scaled_dot_product_attention(q, Kall, Vall, attn_mask=mask)
        o = self.attn.o_proj(o.transpose(1, 2).reshape(1, -1, self.nh * self.hd))
        h = h_in + o
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, k, v

    def _logp(self, h):
        """Draft log-probs over the draft vocab and the matching target ids: (n, Vd), (Vd,)."""
        logits = self.m.compute_logits(h).float()
        return torch.log_softmax(logits, dim=-1)

    # ---------------------------------------------------------------- one round
    @torch.inference_mode()
    def propose_tree(self, committed: list[int], start: int, max_nodes_depth: int) -> list[TreeNode]:
        """committed[-1] is the root (position start); returns prefix-closed nodes in packed order (parents first).
        max_nodes_depth caps the depth (tokens left before max_len)."""
        dev, dt = self.dev, self.dtype
        # absorb the pending context pairs: positions [absorbed, start), tokens committed[absorbed + 1.. start]
        a, P = self.absorbed, self.pending.shape[1]
        assert a + P == start, (a, P, start)
        toks = torch.tensor([committed[a + 1:start + 1]], device=dev)
        emb = self.m.embed_input_ids(toks).to(dt)
        hid = self.m.project_hidden_states(self.pending.to(dt))
        pos = torch.arange(a, start, device=dev)[None]
        neg = torch.finfo(dt).min
        mask = torch.cat((torch.zeros((P, a), dtype=dt), torch.triu(torch.full((P, P), neg, dtype=dt), diagonal=1)),
                         dim=1).to(dev)[None, None]
        h, k, v = self._layer(emb, hid, pos, mask)
        self.K = k if self.K is None else torch.cat((self.K, k), dim=2)
        self.V = v if self.V is None else torch.cat((self.V, v), dim=2)
        self.absorbed, self.pending = start, None
        h_last = h[:, -1:, :]                                          # proposes depth-1 tokens
        D = max(1, min(self.depth, max_nodes_depth))
        k1 = self.topk
        # ---- EAGLE-2 growth. node record: (cum logp, token, depth, parent node index, rank)
        lp = self._logp(h_last)[0, 0]                                  # (Vd,)
        top = torch.topk(lp, k1)
        vals, tids = top.values.tolist(), (top.indices + self.d2t[top.indices]).tolist()   # one host sync
        cand = [(vals[r], tids[r], 1, -1, r) for r in range(k1)]
        nodes_all = list(cand)                                         # every proposed node
        frontier = list(range(len(cand)))                              # indices into nodes_all (expanded next)
        par_h = {i: h_last for i in frontier}                          # hidden input of the query of node i
        q_kv = {}                                                      # node index -> (k, v) of its query
        for d in range(2, D + 1):
            if not frontier:
                break
            Fn = len(frontier)
            emb = self.m.embed_input_ids(torch.tensor([[nodes_all[i][1] for i in frontier]], device=dev)).to(dt)
            hid = torch.cat([par_h[i] for i in frontier], dim=1)
            pos = torch.full((1, Fn), start - 1 + (d - 1), device=dev)
            # extra keys: the queries of every ancestor of each frontier node (depths 1.. d - 2 have queries)
            anc_ids = sorted({j for i in frontier for j in self._ancestors(nodes_all, i) if j in q_kv})
            ek = torch.cat([q_kv[j][0] for j in anc_ids], dim=2) if anc_ids else None
            ev = torch.cat([q_kv[j][1] for j in anc_ids], dim=2) if anc_ids else None
            E = len(anc_ids)
            tail = torch.full((Fn, E + Fn), neg, dtype=dt)              # [ancestor queries | frontier], built on the host
            col = {j: c for c, j in enumerate(anc_ids)}
            for r, i in enumerate(frontier):
                for j in self._ancestors(nodes_all, i):
                    if j in col:
                        tail[r, col[j]] = 0
                tail[r, E + r] = 0
            mask = torch.cat((torch.zeros((Fn, self.absorbed), dtype=dt), tail), dim=1).to(dev)[None, None]
            h, kq, vq = self._layer(emb, hid, pos, mask, (ek, ev) if anc_ids else None)
            for r, i in enumerate(frontier):
                q_kv[i] = (kq[:, :, r:r + 1], vq[:, :, r:r + 1])
            lp = self._logp(h)[0]                                      # (Fn, Vd)
            top = torch.topk(lp, k1, dim=-1)
            vals, tids = top.values.tolist(), (top.indices + self.d2t[top.indices]).tolist()
            new = []
            for r, i in enumerate(frontier):
                base = nodes_all[i][0]
                for rk in range(k1):
                    nodes_all.append((base + vals[r][rk], tids[r][rk], d, i, rk))
                    new.append(len(nodes_all) - 1)
                    par_h[len(nodes_all) - 1] = h[:, r:r + 1, :]
            new.sort(key=lambda j: -nodes_all[j][0])
            frontier = new[:k1]
        # ---- final tree: the best `budget` nodes by cumulative log-prob, closed under ancestors, parents first
        order = sorted(range(len(nodes_all)), key=lambda j: -nodes_all[j][0])
        keep, seen = [], set()
        for j in order:
            if len(keep) >= self.budget:
                break
            chain = [x for x in reversed(self._ancestors(nodes_all, j)) if x not in seen] + [j]
            if len(keep) + len(chain) > self.budget:
                continue
            for x in chain:
                seen.add(x)
                keep.append(x)
        keep.sort(key=lambda j: (nodes_all[j][2], -nodes_all[j][0]))  # by depth: parents first
        idx = {j: n for n, j in enumerate(keep)}
        return [TreeNode(nodes_all[j][1], nodes_all[j][2], idx[nodes_all[j][3]] if nodes_all[j][3] >= 0 else -1,
                         nodes_all[j][4]) for j in keep]

    @staticmethod
    def _ancestors(nodes_all, i):
        out = []
        p = nodes_all[i][3]
        while p >= 0:
            out.append(p)
            p = nodes_all[p][3]
        return out
