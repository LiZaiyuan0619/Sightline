"""Tree-verified speculative decoding for the strict harness (H2 step 3, the design notes/§7.7).

Same invariants as sightline/spec.py (target KV holds committed[:-1]; every round feeds
[committed[-1]] + candidates), generalised from a chain to a prefix-closed candidate tree that
ONE ancestor-masked target pass verifies (the GLANCE construction: glance.tree.tree_mask, packed
positions start + depth, KV gathered down to the accepted path). The walk is the strict
sample-and-compare rule of sightline/verify.py, applied node by node:

    at node v (the root is committed[-1]):  y = draw(chain(logits_v, history_v))
        history_v = committed + the tokens on the path to v (the AR history of that position)
    a child of v carries y  -> move there (draft accepted)
    otherwise / v is a leaf -> emit y as the bonus and stop

One target draw per emitted token, from the target's own processed distribution at the right
history: the output law is the AR law for ANY tree, and under the same generator the realized
sequence equals the chain/AR one (up to bf16 near-ties between forward shapes). Any tree with
the chain's nodes therefore accepts at least as much as the chain.

Geometric tolerance (coord_tol = r > 0;  probe, NOT lossless; sightline/verify.py): when no
child carries y but y is a coordinate, the walk moves to the nearest drafted coordinate child
within Chebyshev distance r (ties: the earlier, higher-scored child) and commits that child's
token instead of y. coord_tol = 0 is the strict walk above.

Long period-copy branch (copy_long = C > 0;, lossless): when the committed stream ends
with two copies of a P-token cycle, a chain continuing the cycle for up to C tokens is merged
into the tree (add_copy_branch), so degenerate loops are verified far past the draft block.

Trees come from the H2 designs (simulate.py): the live draft log-probs of a block are
summarised exactly like the offline cache (cache_draft.fill_block), then the SAME
candidate code (simulate.block_candidates) and best-first growth (simulate.grow_tree) build the
tree, so the verifier's acceptance can be checked against the simulator on the same rows.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.tree_utils import gather_cache                      # noqa: E402
from sightline.tree_utils import TreeNode, children_of, tree_mask    # noqa: E402
from sightline.spec import DFlashDrafter, SpecResult, _chain_draw   # noqa: E402
from sightline.process import draw, eos_only, loop_rollback          # noqa: E402
from sightline.verify import coord_cheb                              # noqa: E402
from sightline.verify import ChainParams                       # noqa: E402
from sightline import cache_draft
from sightline import design as simulate
from sightline.svgtok.types import COORD_END, COORD_START              # noqa: E402


class StreamPositions:
    """1-D positions (text targets). A VLM adapter maps them to M-RoPE (H3a)."""

    def prefill_kwargs(self, n_prompt: int, dev) -> dict:
        return {"position_ids": torch.arange(n_prompt, device=dev).unsqueeze(0)}

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        return pos


class TreeDrafter(DFlashDrafter):
    """DFlashDrafter that returns the block's draft log-probs (the chain drafter's argmax is
    their top-1) and updates its context from arbitrary kept packed indices."""

    @torch.inference_mode()
    def draft_logprobs(self, start: int, block_first: int, verify_size: int) -> torch.Tensor:
        from dflash.model import _crop_to, _output_head, _raw_input_embeddings
        dev = next(self.model.parameters()).device
        K = verify_size - 1
        block_ids = torch.tensor([[block_first] + [self.mask_token_id] * K], device=dev)
        position_ids = torch.arange(start - self.target_hidden.shape[1], start + verify_size,
                                    device=dev).unsqueeze(0)
        noise = _raw_input_embeddings(self.target, block_ids, 1.0)
        h = self.model(target_hidden=self.target_hidden, noise_embedding=noise,
                       position_ids=position_ids, past_key_values=self.cache,
                       use_cache=True)[:, 1 - verify_size:, :]
        _crop_to(self.cache, start)
        logits = self.model.compute_logits(h, _output_head(self.target))[0]
        return torch.log_softmax(logits.float(), dim=-1)

    def update_hidden_kept(self, hidden_states, keep: list[int]):
        from dflash.model import extract_context_feature
        feats = extract_context_feature(hidden_states, self.model.target_layer_ids)
        self.target_hidden = feats[:, keep, :]

    def rollback(self, new_start: int, old_start: int):
        """Rollback escape: the next root is at new_start. The drafter's cache holds the
        context [0, old_start) and target_hidden the not-yet-consumed features [old_start,...)."""
        from dflash.model import _crop_to
        if new_start >= old_start:
            self.target_hidden = self.target_hidden[:, :new_start - old_start, :]
        else:
            _crop_to(self.cache, new_start)
            self.target_hidden = self.target_hidden[:, :0, :]


def recent_coords(committed: list[int], n_prompt: int, n: int = cache_draft.N_COPY) -> list[int]:
    """Last n distinct coordinate tokens of the generated part, most recent first."""
    out: list[int] = []
    seen = set()
    for tok in reversed(committed[n_prompt:]):
        if COORD_START <= tok <= COORD_END and tok not in seen:
            seen.add(tok)
            out.append(tok)
            if len(out) == n:
                break
    return out


def summarize_block(lp: torch.Tensor, copy_ids: list[int], K: int) -> dict:
    """The cache_draft fields for ONE block start, as a simulate-style row with t = 0."""
    dev = lp.device
    st = cache_draft.Stats(1, K, dev)
    buf = torch.full((cache_draft.N_COPY,), -1, dtype=torch.long, device=dev)
    if copy_ids:
        buf[:len(copy_ids)] = torch.tensor(copy_ids, device=dev)
    cache_draft.fill_block(st, 0, lp, torch.empty(0, dtype=torch.long, device=dev), buf,
                           cache_draft._nb_offsets(dev))
    return st.numpy()


def build_tree(design: "simulate.Design", lp: torch.Tensor, committed: list[int], n_prompt: int,
               K: int) -> list[TreeNode]:
    """Design -> prefix-closed nodes in packed order (parents first)."""
    k = lp.shape[0]
    if design.is_chain:
        top1 = lp.argmax(-1).tolist()
        return [TreeNode(int(top1[j]), j + 1, j - 1, 0) for j in range(k)]
    row = summarize_block(lp, recent_coords(committed, n_prompt), K)
    cands = simulate.block_candidates(row, 0, design)
    return [TreeNode(tok, depth, parent, rank)
            for tok, depth, parent, rank in simulate.grow_tree(cands, design.budget)]


def detect_period(committed: list[int], n_prompt: int, max_period: int = simulate.MAX_PERIOD) -> int:
    """Smallest P <= max_period such that the generated part ends with two copies of one P-token
    cycle (the online rule of simulate.periods); 0 if none."""
    gen = committed[n_prompt:]
    n = len(gen)
    for P in range(1, max_period + 1):
        if 2 * P > n:
            break
        if gen[n - P:] == gen[n - 2 * P:n - P]:
            return P
    return 0


COPY_RANK = 10 ** 6                      # TreeNode.rank of period-copy nodes


def add_copy_branch(nodes: list[TreeNode], committed: list[int], n_prompt: int,
                    length: int) -> tuple[list[TreeNode], int, int]:
    """Long period-copy branch (simulate.Design.copy_long): when the committed stream ends
    with two copies of a P-token cycle, the chain that continues the cycle for `length` tokens is
    merged into the tree as a trie path from the root (follow existing children that carry the
    chain's tokens, then append the rest), so it can run far past the draft block. Verification
    is unchanged, so the decoder stays exact. Returns (nodes, P, number of nodes added)."""
    P = detect_period(committed, n_prompt)
    if not P or length <= 0:
        return nodes, 0, 0
    cyc = committed[len(committed) - P:]
    chain = [cyc[j % P] for j in range(length)]
    children = children_of(nodes)
    cur, depth, j = -1, 0, 0
    while j < len(chain):
        hit = next((c for c in children.get(cur, []) if nodes[c].token == chain[j]), None)
        if hit is None:
            break
        cur, depth, j = hit, nodes[hit].depth, j + 1
    nodes = list(nodes)
    for tok in chain[j:]:
        depth += 1
        nodes.append(TreeNode(int(tok), depth, cur, COPY_RANK))
        cur = len(nodes) - 1
    return nodes, P, len(chain) - j


@dataclass
class TreeRound:
    start: int
    n_nodes: int
    n_accepted: int
    emitted: list


@torch.inference_mode()
def tree_spec_generate(model, drafter: TreeDrafter, prompt_ids: list[int], max_new_tokens: int,
                       eos_ids: list[int], params: ChainParams, design,
                       generator: torch.Generator | None = None, block_size: int = 16,
                       positions=None, prefill_kwargs: dict | None = None,
                       coord_tol: int = 0, copy_long: int = 0) -> SpecResult:
    """Tree-verified generation; returns a SpecResult whose per_block entries also carry the
    tree size. Same EOS / max_new semantics as spec_generate."""
    from transformers import DynamicCache
    positions = positions or StreamPositions()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_prompt = len(prompt_ids)
    max_len = n_prompt + max_new_tokens
    K = block_size - 1
    cache = DynamicCache()
    nfe = 0
    prompt = torch.tensor([prompt_ids], device=dev)
    kw = prefill_kwargs if prefill_kwargs is not None else positions.prefill_kwargs(n_prompt, dev)
    out = model(prompt, past_key_values=cache, use_cache=True, output_hidden_states=True, **kw)
    nfe += 1
    first = int(_chain_draw(out.logits[0, -1], list(prompt_ids), params, generator))
    drafter.prefill_done(out.hidden_states)
    committed = list(prompt_ids) + [first]
    acceptance: list[int] = []
    per_block: list[dict] = []
    stop_reason = "max_new_tokens"
    if first in eos_ids:
        return SpecResult(tokens=[first], nfe=nfe, acceptance=[1], per_block=[], stop_reason="eos")
    n_rb, ban = 0, None            # rollback escape state (process.loop_rollback), as in sightline.ar

    while len(committed) < max_len:
        verify_size = min(block_size, max_len - len(committed) + 1)
        start = len(committed) - 1
        if hasattr(drafter, "propose_tree"):          # EAGLE-3 baseline drafter (baselines/eagle3_spec.py)
            nodes = drafter.propose_tree(committed, start, verify_size - 1)
        else:
            lp = drafter.draft_logprobs(start, committed[-1], verify_size)
            nodes = build_tree(design, lp, committed, n_prompt, K)
        copy_P = n_copy = 0
        if copy_long:
            nodes, copy_P, n_copy = add_copy_branch(nodes, committed, n_prompt,
                                                    min(copy_long, max_len - len(committed)))
        packed = torch.tensor([[committed[-1]] + [n.token for n in nodes]], device=dev)
        packed_pos = torch.tensor([[start] + [start + n.depth for n in nodes]], device=dev)
        n_prefix = cache.get_seq_length()
        out = model(packed, position_ids=positions(packed_pos), past_key_values=cache,
                    use_cache=True, attention_mask=tree_mask(n_prefix, nodes, dev, dtype),
                    output_hidden_states=True)
        nfe += 1
        logits = out.logits[0]
        children = children_of(nodes)
        history = list(committed)
        path: list[int] = []
        emitted: list[int] = []
        cursor = -1
        n_snap = 0
        rollback = None
        while True:
            pos_logits = logits[0 if cursor == -1 else cursor + 1]
            act, new_len, btok = loop_rollback(history, params, n_rb, n_prompt)
            if act == "rollback":                    # no draw here; the round ends at this position
                rollback = (len(history), new_len, btok)
                break
            if act == "eos":
                y = int(draw(eos_only(pos_logits.float(), eos_ids[0]), params.greedy, generator))
            elif act == "ban":                       # escape-then-stop: every walked position is committed
                y = int(_chain_draw(pos_logits, history, params, generator, extra_ban=btok))
                n_rb += 1
            else:
                eb = ban[1] if ban is not None and ban[0] == len(history) else None
                y = int(_chain_draw(pos_logits, history, params, generator, extra_ban=eb))
            emitted.append(y)
            kids = children.get(cursor, [])
            hit = next((c for c in kids if nodes[c].token == y), None)
            if hit is None and coord_tol > 0:
                near = [(coord_cheb(nodes[c].token, y), k, c) for k, c in enumerate(kids)]
                near = [t for t in near if t[0] is not None and t[0] <= coord_tol]
                if near:
                    hit = min(near)[2]
                    emitted[-1] = nodes[hit].token          # snap to the drafted coordinate
                    n_snap += 1
            if hit is None:
                break
            path.append(hit)
            history.append(nodes[hit].token)
            cursor = hit
        n_acc = len(path)
        truncated = False
        for i, t in enumerate(emitted):
            if t in eos_ids:
                emitted = emitted[:i + 1]
                truncated = True
                break
        if len(committed) + len(emitted) > max_len:
            emitted = emitted[:max_len - len(committed)]
        committed += emitted
        acceptance.append(len(emitted))
        per_block.append({"start": start, "n_nodes": len(nodes), "n_accepted": n_acc,
                          "emitted": emitted, "n_snap": n_snap, "copy_P": copy_P,
                          "n_copy": n_copy,
                          "rollback": list(rollback[:2]) if rollback and not truncated else None})
        if truncated or len(committed) >= max_len:
            stop_reason = "eos" if truncated else "max_new_tokens"
            break
        # keep the root and the accepted path: cache = committed[:-1] again
        keep = [0] + [i + 1 for i in path]
        gather_cache(cache, n_prefix, keep)
        drafter.update_hidden_kept(out.hidden_states, keep)
        if rollback is not None:
            # the walk stopped before drawing at position rollback[0] == len(committed): no bonus
            # token, so the cache holds all of committed; cut both back to the kept cycle
            _, new_len, btok = rollback
            committed = committed[:new_len]
            cache.crop(new_len - 1)
            drafter.rollback(new_len - 1, start)
            n_rb, ban = n_rb + 1, (new_len, btok)
    return SpecResult(tokens=committed[n_prompt:], nfe=nfe, acceptance=acceptance,
                      per_block=per_block, stop_reason=stop_reason)
