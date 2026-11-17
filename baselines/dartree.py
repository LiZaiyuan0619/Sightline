""" DARTree (2608.13524; official code refs/DARTree) as a table-A baseline inside OUR strict tree verifier
(sightline/tree_spec.py), so verification, the processing chain (OmniSVG's repetition penalty / top-k / top-p), timing and
records are the same as for every other row. DARTree builds training-free draft trees over a Domino head
(2605.29707; ours trained by train_domino.py with SpecForge's DominoDraftModel).

Port of refs/DARTree/eval_dartree.build_dartree_supertree, pruned variant, in plain batched PyTorch (the official
CUDA graphs / Triton kernels are speed-ups, not semantics; every row of table A runs in the same HF-eager harness):
  - backbone: the Domino head's DFlash backbone on [root, MASK x (D - 1)]; shift_label: block position j predicts
    depth j + 1, base logits = lm_head(h) (refs/DARTree/utils/draft_model.py spec_generate);
  - slot 0 (pure draft prefix): full-vocabulary log-softmax of the base logits;
  - slot s >= 1: corrected logits on the top-E base candidates of that slot, base[cand] + fc2(SiLU(fc1_z h_s +
    fc1_s g)), g = the GRU state after the path tokens (root, t1.. t_s); log-probs normalized within the candidates
    (DominoCorrectionScorer.candidate_topk_from_precomputed);
  - every depth keeps the W best children of the frontier by path score (sum of log-probs, prefix monotone);
  - the verification tree is the global top-B of all kept nodes by score + depth_bonus * depth, parents first.
Defaults = DARTree's: W 12, E 64, B 64, depth_bonus -0.2 (pruned). Design names "DART-W<w>-E<e>-N<budget>[-b<bonus>]"
(baselines/eval_dart.py). W = 1 with a large E is the Domino chain (checked against the reference loop by check_chain).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
import torch.nn.functional as F
from torch import nn
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "h2", REFS / "GLANCE"):
    sys.path.insert(0, str(p))
from sightline.tree_utils import TreeNode                         # noqa: E402
from sightline.tree_spec import TreeDrafter                # noqa: E402


@dataclass
class DominoExtras:
    gru: nn.GRU
    proj: nn.Sequential
    prefix_len: int
    shift_label: bool


def is_domino_head(path) -> bool:
    cfg = Path(path) / "config.json"
    return cfg.exists() and json.loads(cfg.read_text()).get("dflash_config", {}).get("projector_type") == "domino"


def load_domino(path, dtype=torch.bfloat16):
    """Domino head: the DFlash backbone through dflash.model (as every DFlash head of the harness) and the GRU +
    correction MLP of SpecForge's DominoDraftModel, read from the same safetensors (strict)."""
    from dflash.model import DFlashDraftModel
    from safetensors.torch import load_file
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    dc = cfg["dflash_config"]
    backbone = DFlashDraftModel.from_pretrained(path, torch_dtype=dtype).cuda().eval()
    sd = load_file(str(path / "model.safetensors"))
    H, G, Em = cfg["hidden_size"], dc["gru_hidden_dim"], dc["emb_dim"]
    gru = nn.GRU(input_size=H, hidden_size=G, num_layers=1, batch_first=True, bias=False)
    proj = nn.Sequential(nn.Linear(H + G, Em, bias=False), nn.SiLU(), nn.Linear(Em, cfg["vocab_size"], bias=False))
    gru.load_state_dict({k[len("prefix_gru."):]: v for k, v in sd.items() if k.startswith("prefix_gru.")}, strict=True)
    proj.load_state_dict({k[len("embed_proj."):]: v for k, v in sd.items() if k.startswith("embed_proj.")}, strict=True)
    backbone.domino = DominoExtras(gru.to(dtype).cuda().eval(), proj.to(dtype).cuda().eval(),
                                   int(dc.get("pure_draft_prefix_len", 0)), bool(dc.get("shift_label", False)))
    return backbone


class DominoTreeDrafter(TreeDrafter):
    """TreeDrafter (context features, draft KV, rollback) whose trees are DARTree's."""

    def __init__(self, model, target, block_size: int, width: int = 12, cand: int = 64, budget: int = 64,
                 depth_bonus: float = -0.2):
        super().__init__(model, target, block_size)
        ex = model.domino
        assert ex.shift_label and ex.prefix_len == 1, "port written for shift_label heads with one pure prefix slot"
        H = model.config.hidden_size
        fc1, fc2 = ex.proj[0], ex.proj[2]
        self.gru = ex.gru
        self.w_z, self.w_s, self.w2 = fc1.weight[:, :H], fc1.weight[:, H:], fc2.weight
        self.emb = target.get_input_embeddings().weight
        self.W, self.E, self.B, self.bonus = width, cand, budget, depth_bonus
        self.w_ih, self.w_hh = self.gru.weight_ih_l0, self.gru.weight_hh_l0          # (3G, H), (3G, G); no biases
        self.G = self.gru.hidden_size

    def _gru(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One GRU step, PyTorch's equations (gates r, z, n; bias=False), without the cuDNN RNN call overhead."""
        gi = F.linear(F.embedding(tokens, self.emb).to(self.w_ih.dtype), self.w_ih)
        gh = F.linear(h, self.w_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        zg = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - zg) * n + zg * h

    @torch.inference_mode()
    def block_states(self, committed: list[int], start: int, depth: int):
        """Backbone on [root, MASK x (depth - 1)]: hidden (depth, H) and base logits (depth, V) fp32."""
        from dflash.model import _crop_to, _output_head, _raw_input_embeddings
        dev = self.w_z.device
        block_ids = torch.tensor([[committed[-1]] + [self.mask_token_id] * (depth - 1)], device=dev)
        pos = torch.arange(start - self.target_hidden.shape[1], start + depth, device=dev).unsqueeze(0)
        noise = _raw_input_embeddings(self.target, block_ids, 1.0)
        h = self.model(target_hidden=self.target_hidden, noise_embedding=noise, position_ids=pos,
                       past_key_values=self.cache, use_cache=True)[:, -depth:, :]
        _crop_to(self.cache, start)
        base = self.model.compute_logits(h, _output_head(self.target))[0].float()
        return h[0], base

    @torch.inference_mode()
    def propose_tree(self, committed: list[int], start: int, max_depth: int) -> list[TreeNode]:
        D = max(1, min(self.block_size - 1, max_depth))
        h, base = self.block_states(committed, start, D)
        return self.tree_from_states(committed, h, base)

    @torch.inference_mode()
    def tree_from_states(self, committed: list[int], h: torch.Tensor, base: torch.Tensor) -> list[TreeNode]:
        D = h.shape[0]
        dev = h.device
        z = F.linear(h, self.w_z)                                                 # (D, Em)
        root = torch.tensor([committed[-1]], device=dev)
        g0 = self._gru(root, torch.zeros((1, self.gru.hidden_size), device=dev, dtype=self.w_z.dtype))
        # depth 1: the pure draft prefix slot, full-vocabulary log-probs
        lp0 = torch.log_softmax(base[0], dim=-1)
        sc, tok = torch.topk(lp0, min(self.W, self.E, lp0.numel()))
        rec_tok, rec_sc, rec_par, rec_dep = [tok], [sc], [torch.full_like(tok, -1)], [torch.ones_like(tok)]
        offset, n_prev = 0, tok.numel()
        g_par = g0.expand(tok.numel(), -1)
        if D > 1:                                                                 # candidate tables of slots 1.. D-1
            CV, CI = torch.topk(base[1:], self.E, dim=-1)                         # (D-1, E)
            W2 = self.w2[CI]                                                      # (D-1, E, Em)
        for d in range(2, D + 1):
            s = d - 1                                                             # slot predicting depth d
            g = self._gru(tok, g_par)                                             # states after each frontier token
            cv, ci = CV[s - 1], CI[s - 1]
            mid = F.silu(z[s].unsqueeze(0) + F.linear(g, self.w_s))               # (N, Em)
            logits = cv.unsqueeze(0) + (mid @ W2[s - 1].t()).float()              # (N, E)
            lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            child = (sc.unsqueeze(1) + lp).reshape(-1)
            k = min(self.W, child.numel())
            sc, flat = torch.topk(child, k)
            parent_local = flat // self.E
            tok = ci[flat % self.E]
            rec_tok.append(tok); rec_sc.append(sc); rec_dep.append(torch.full_like(tok, d))
            rec_par.append(parent_local + offset)
            offset += n_prev
            n_prev = k
            g_par = g[parent_local]
        toks = torch.cat(rec_tok); scores = torch.cat(rec_sc); pars = torch.cat(rec_par); deps = torch.cat(rec_dep)
        n = toks.numel()
        sel = torch.topk(scores + self.bonus * deps.float(), min(self.B, n)).indices
        toks, scores, pars, deps, sel = (x.tolist() for x in (toks, scores, pars, deps, sel))   # one host sync
        keep = set(sel)
        for j in list(sel):                                   # prefix closure (scores are prefix monotone; ties)
            p = pars[j]
            while p >= 0 and p not in keep:
                keep.add(p)
                p = pars[p]
        order = sorted(keep, key=lambda j: (deps[j], -scores[j]))
        idx = {j: i for i, j in enumerate(order)}
        return [TreeNode(int(toks[j]), int(deps[j]), idx[pars[j]] if pars[j] >= 0 else -1, 0) for j in order]


@torch.inference_mode()
def reference_chain(drafter: DominoTreeDrafter, committed: list[int], h: torch.Tensor, base: torch.Tensor) -> list[int]:
    """The Domino chain of refs/DARTree/utils/draft_model.spec_generate (temperature 0, full-vocabulary correction)
    on given backbone states: t1 = argmax base[0]; t_{i+1} = argmax(base[i] + embed_proj(cat(h_i, g_i))), g_i = the GRU
    state after (root, t1.. t_i)."""
    depth = h.shape[0]
    proj = drafter.model.domino.proj
    out = [int(base[0].argmax())]
    g = drafter._gru(torch.tensor([committed[-1]], device=h.device),
                     torch.zeros((1, drafter.gru.hidden_size), device=h.device, dtype=drafter.w_z.dtype))
    g = drafter._gru(torch.tensor([out[0]], device=h.device), g)
    for i in range(1, depth):
        bias = proj(torch.cat([h[i:i + 1], g], dim=-1)).float()[0]
        out.append(int((base[i] + bias).argmax()))
        if i + 1 < depth:
            g = drafter._gru(torch.tensor([out[-1]], device=h.device), g)
    return out
