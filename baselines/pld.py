""" prompt-lookup decoding (PLD, Saxena 2023; HF `prompt_lookup_num_tokens`) as a training-free baseline row, run
through the strict tree verifier (sightline/tree_spec.py) as a chain, so timing, AR cache and records are eval_tree's.

Draft of a round: for n = max_ngram.. 1, find the most recent earlier occurrence of the last n committed tokens and copy
up to K tokens that followed it (HF searches for the first occurrence; the most recent one is the stronger choice on SVG,
whose repeats are local). No match: an empty draft (that round is one AR step). Strict verification keeps it lossless.
Design names in eval_tree.py: "PLD-n<max_ngram>-K<K>" (e.g. PLD-n3-K10, HF's defaults K = 10, max n = 3... 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from sightline.tree_utils import TreeNode                                     # noqa: E402


def lookup(seq: list[int], max_ngram: int, K: int) -> list[int]:
    L = len(seq)
    for n in range(min(max_ngram, L - 1), 0, -1):
        key = seq[L - n:]
        for s in range(L - n - 1, -1, -1):          # most recent earlier occurrence (start index s)
            if seq[s:s + n] == key:
                return seq[s + n:s + n + K]
    return []


class PLDDrafter:
    """One sequence (a new drafter per generation). n-gram -> start of the continuation after its most recent earlier
    occurrence, indexed incrementally as tokens are committed (O(1) per round instead of a scan of the whole stream,
    which would bias the baseline's wall clock)."""
    needs_hidden = False

    def __init__(self, max_ngram: int = 3, K: int = 10):
        self.max_ngram, self.K = max_ngram, K
        self.idx = {n: {} for n in range(1, max_ngram + 1)}
        self.done = 0                                # n-grams ending at positions < done are indexed

    def prefill_done(self, hidden_states):
        pass

    def update_hidden_kept(self, hidden_states, keep):
        pass

    def propose_tree(self, committed: list[int], start: int, max_depth: int) -> list[TreeNode]:
        L = len(committed)
        for i in range(self.done, L - 1):            # the n-grams ending at L - 1 are the query, not indexed yet
            for n in range(1, self.max_ngram + 1):
                if i - n + 1 >= 0:
                    self.idx[n][tuple(committed[i - n + 1:i + 1])] = i + 1
        self.done = max(self.done, L - 1)
        K = min(self.K, max_depth)
        draft: list[int] = []
        for n in range(min(self.max_ngram, L), 0, -1):
            s = self.idx[n].get(tuple(committed[L - n:]))
            if s is not None:
                draft = committed[s:s + K]
                break
        return [TreeNode(int(t), j + 1, j - 1, 0) for j, t in enumerate(draft)]
