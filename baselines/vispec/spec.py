""" ViSpec drafting for the strict tree verifier (sightline/tree_spec.py), the ViSpec rows of table 1's image-to-SVG
blocks. Same interface and tree growth as baselines/eagle3_spec.Eagle3Drafter (EAGLE-2 dynamic tree: depth 1 = the top-k of
the last context pair; at every further depth the top-k frontier nodes by cumulative log-prob are expanded with their
top-k children; the best `budget` nodes by cumulative log-prob, closed under ancestors, form the tree; = ViSpec's
topK_genrate), with ViSpec's draft (baselines/vispec_model.py):
  context pair t = (h_t = target hidden_states[-1] at t, x_{t+1}); its input embedding is the target's merged input
  embedding (the vision feature for image tokens), captured at the prefill with a pre-hook on the target's language
  model. The pairs whose x_{t+1} is an image token are compressed by the vision adaptor (ViSpec); the global image vector
  g then enters every later pair and every tree query (img_fc(cat(h, g))).
  tree query of a node v at depth d: hidden = output of the query that proposed v, token = v's token -> logits of v's
  children through the TARGET's lm_head (full vocabulary); RoPE position (start - 1) + d; attends to the absorbed
  context and to the queries of v's ancestors and itself.
Verification is tree_spec_generate's (strict sample-and-compare, any tree is lossless).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.tree_utils import TreeNode                                      # noqa: E402
from baselines.vispec.model import is_vispec_head, load_head                           # noqa: E402,F401


def language_model(target):
    inner = getattr(target, "model", None)
    return getattr(inner, "language_model", None)


class ViSpecDrafter:
    needs_hidden = True

    def __init__(self, head, target, topk: int = 8, depth: int = 4, budget: int = 29, image_token: int | None = None):
        self.m, self.target = head, target
        self.topk, self.depth, self.budget = topk, depth, budget
        self.image_token = image_token
        self.lm_head = target.get_output_embeddings()
        self.embed = target.get_input_embeddings()
        self.dev = next(head.parameters()).device
        self.dtype = next(head.parameters()).dtype
        self.K = self.V = None                       # (1, nkv, m_ctx, hd), RoPE applied
        self.absorbed = 0                            # pairs [0, absorbed) are in the draft KV
        self.pending = None                          # (1, P, H) target hidden states not yet absorbed
        self.g = None                                # global image vector after the image span
        self.prompt_emb = None
        self._cap = None
        lm = language_model(target)
        self._hook = lm.register_forward_pre_hook(self._grab, with_kwargs=True) if lm is not None else None

    def _grab(self, module, args, kwargs):
        if self._cap is None:                        # the first call = the prefill of tree_spec_generate
            self._cap = kwargs.get("inputs_embeds")

    # ---------------------------------------------------------------- target features
    def prefill_done(self, hidden_states):
        self.pending = hidden_states[-1]
        self.K = self.V = None
        self.absorbed = 0
        self.g = None
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self.prompt_emb = None if self._cap is None else self._cap[0]
        self._cap = None

    def update_hidden_kept(self, hidden_states, keep: list[int]):
        self.pending = hidden_states[-1][:, keep, :]

    def _emb_of(self, positions: list[int], tokens: list[int]):
        """input embeddings of tokens at stream positions (prompt positions: the captured merged embedding)"""
        n_p = 0 if self.prompt_emb is None else self.prompt_emb.shape[0]
        out = self.embed(torch.tensor(tokens, device=self.dev)).to(self.dtype)
        sel = [j for j, p in enumerate(positions) if p < n_p]
        if sel:
            idx = torch.tensor([positions[j] for j in sel], device=self.dev)
            out[sel] = self.prompt_emb[idx].to(self.dtype)
        return out

    def _logp(self, h):
        return torch.log_softmax(self.lm_head(h).float(), dim=-1)

    # ---------------------------------------------------------------- one round
    @torch.inference_mode()
    def propose_tree(self, committed: list[int], start: int, max_nodes_depth: int) -> list[TreeNode]:
        dev, dt = self.dev, self.dtype
        a, P = self.absorbed, self.pending.shape[1]
        assert a + P == start, (a, P, start)
        toks = committed[a + 1:start + 1]
        emb = self._emb_of(list(range(a + 1, start + 1)), toks)
        hid = self.pending[0].to(dt)
        span = None
        if self.image_token is not None and self.m.use_img:
            w = [j for j, t in enumerate(toks) if t == self.image_token]
            if w:
                assert w[-1] - w[0] + 1 == len(w), "image tokens not contiguous"
                span = (w[0], w[-1] + 1)           # chunk-relative pairs whose x_{t+1} is an image token
        X, pos, _, g = self.m.build_inputs(emb, hid, span, g0=self.g)
        self.g = g
        pos = pos + a
        q = X.shape[0]
        past_len = 0 if self.K is None else self.K.shape[2]
        neg = torch.finfo(dt).min
        mask = torch.cat((torch.zeros((q, past_len), dtype=dt), torch.triu(torch.full((q, q), neg, dtype=dt), 1)),
                         dim=1).to(dev)[None, None]
        h, k, v = self.m.layer(X[None].to(dt), pos, mask, past=None if self.K is None else (self.K, self.V))
        self.K = k if self.K is None else torch.cat((self.K, k), dim=2)
        self.V = v if self.V is None else torch.cat((self.V, v), dim=2)
        self.absorbed, self.pending = start, None
        h_last = h[:, -1:, :]
        D = max(1, min(self.depth, max_nodes_depth))
        k1 = self.topk
        # ---- EAGLE-2 growth (= Eagle3Drafter.propose_tree). node record: (cum logp, token, depth, parent, rank)
        lp = self._logp(h_last)[0, 0]
        top = torch.topk(lp, k1)
        vals, tids = top.values.tolist(), top.indices.tolist()
        nodes_all = [(vals[r], tids[r], 1, -1, r) for r in range(k1)]
        frontier = list(range(k1))
        par_h = {i: h_last for i in frontier}
        q_kv = {}
        for d in range(2, D + 1):
            if not frontier:
                break
            Fn = len(frontier)
            embq = self.embed(torch.tensor([nodes_all[i][1] for i in frontier], device=dev)).to(dt)
            hq = torch.cat([par_h[i] for i in frontier], dim=1)[0]
            x = self.m.input_proj(embq, hq, self.g)[None]
            posq = torch.full((Fn,), start - 1 + (d - 1), device=dev)
            anc_ids = sorted({j for i in frontier for j in self._ancestors(nodes_all, i) if j in q_kv})
            E = len(anc_ids)
            tail = torch.full((Fn, E + Fn), neg, dtype=dt)
            col = {j: c for c, j in enumerate(anc_ids)}
            for r, i in enumerate(frontier):
                for j in self._ancestors(nodes_all, i):
                    if j in col:
                        tail[r, col[j]] = 0
                tail[r, E + r] = 0
            maskq = torch.cat((torch.zeros((Fn, self.K.shape[2]), dtype=dt), tail), dim=1).to(dev)[None, None]
            Kp, Vp = self.K, self.V
            if anc_ids:
                Kp = torch.cat([Kp] + [q_kv[j][0] for j in anc_ids], dim=2)
                Vp = torch.cat([Vp] + [q_kv[j][1] for j in anc_ids], dim=2)
            hh, kq, vq = self.m.layer(x, posq, maskq, past=(Kp, Vp))
            for r, i in enumerate(frontier):
                q_kv[i] = (kq[:, :, r:r + 1], vq[:, :, r:r + 1])
            lpq = self._logp(hh)[0]
            top = torch.topk(lpq, k1, dim=-1)
            vals, tids = top.values.tolist(), top.indices.tolist()
            new = []
            for r, i in enumerate(frontier):
                base = nodes_all[i][0]
                for rk in range(k1):
                    nodes_all.append((base + vals[r][rk], tids[r][rk], d, i, rk))
                    new.append(len(nodes_all) - 1)
                    par_h[len(nodes_all) - 1] = hh[:, r:r + 1, :]
            new.sort(key=lambda j: -nodes_all[j][0])
            frontier = new[:k1]
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
        keep.sort(key=lambda j: (nodes_all[j][2], -nodes_all[j][0]))
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
