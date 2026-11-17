""" classic speculative sampling (SpS, Leviathan / Chen 2023) baseline for OmniSVG-8B text2svg, with the official
smaller sibling OmniSVG-4B (exported text model, model/OmniSVG1.1_4B-qwen2-text) as the draft model. Training-free.

The draft proposes a chain of gamma tokens greedily (argmax of the 4B logits, the best guess under the strict
sample-and-compare verification of sightline/tree_spec.py, which keeps the output law the target's for ANY draft).

Token ids. Both sizes share the Qwen2.5 text vocabulary below 151936 and the fixed SVG BOS / EOS ids (196998 / 196999,
refs/OmniSVG/config.yaml "model"); every other SVG token is base_offset + local index, with base_offset 152064 (8B) and
151936 (4B) -> 8B id = 4B id + 128 for 4B ids in [151936, 196998). (The config's color_start_offset differs by 129,
192075 vs 191946, but both equal base_offset + 40010 + {1, 0}; the encoder adds the local index to base_offset, which
check_map verifies on real 4B outputs by rendering them with both decoders.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from sightline.tree_utils import TreeNode                                     # noqa: E402
from sightline.paths import MODELS

BASE_4B, BASE_8B, FIXED = 151936, 152064, (196998, 196999)
SHIFT = BASE_8B - BASE_4B
DRAFT_4B = (MODELS / "OmniSVG1.1_4B-qwen2-text")


def to_8b(t: int) -> int:
    return t if (t < BASE_4B or t in FIXED) else t + SHIFT


def to_4b(t: int) -> int:
    if t < BASE_4B or t in FIXED:
        return t
    if t < BASE_8B:
        raise ValueError(f"8B id {t} lies in the Qwen2.5-VL-7B padding range, no 4B counterpart")
    return t - SHIFT


def load_draft(path: Path = DRAFT_4B, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa").cuda().eval()


class SpSDrafter:
    """One sequence. The draft KV holds the 4B ids of committed[:n]; each round absorbs the new committed tokens,
    proposes gamma tokens greedily, then crops the KV back to the committed tokens (rejected drafts leave no trace)."""
    needs_hidden = False

    def __init__(self, model, gamma: int):
        from transformers import DynamicCache
        self.m, self.gamma = model, gamma
        self.dev = next(model.parameters()).device
        self.cache = DynamicCache()
        self.n = 0

    def prefill_done(self, hidden_states):
        pass

    def update_hidden_kept(self, hidden_states, keep):
        pass

    @torch.inference_mode()
    def propose_tree(self, committed: list[int], start: int, max_depth: int) -> list[TreeNode]:
        g = min(self.gamma, max_depth)
        if g <= 0:
            return []
        new = torch.tensor([[to_4b(t) for t in committed[self.n:]]], device=self.dev)
        out = self.m(new, past_key_values=self.cache, use_cache=True)
        nxt = int(out.logits[0, -1].argmax())
        draft = [nxt]
        for _ in range(g - 1):
            if nxt == FIXED[1]:                                  # EOS: nothing to draft after it
                break
            out = self.m(torch.tensor([[nxt]], device=self.dev), past_key_values=self.cache, use_cache=True)
            nxt = int(out.logits[0, -1].argmax())
            draft.append(nxt)
        self.cache.crop(len(committed))
        self.n = len(committed)
        return [TreeNode(to_8b(t), j + 1, j - 1, 0) for j, t in enumerate(draft)]

    def rollback(self, new_start: int, old_start: int):
        self.cache.crop(new_start)
        self.n = min(self.n, new_start)
