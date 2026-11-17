""" REST-style retrieval drafting (He et al., "REST: Retrieval-Based Speculative Decoding",
NAACL 2024) as the training-free baseline family that table 1 has none of — every other baseline either trains a draft
head (DFlash, DDTree, DARTree, EAGLE-3, ViSpec), needs a smaller model of the same family (SpS) or copies from the
prompt (PLD). REST asks the question a reviewer will ask about SVG: the corpus is extremely repetitive, so how far does
plain retrieval from the training corpus get you without training anything?

Datastore (build_store): the same token streams the draft heads were trained on (one train.jsonl, or the i2svg unit
directories), response tokens only. For each match length n in LENS, a rolling hash of every n-gram is sorted once;
a query is a binary search over that array (numpy, no Python dict over millions of keys). Longest n first, as REST does.

Draft of a round (RESTDrafter.propose_tree): take the retrieved continuations of the longest matching suffix, cut them
to `depth`, count how often each prefix occurs among them, and keep the `budget` most frequent nodes, closed under
ancestors - REST's Trie weighted by continuation frequency. No match at any n: empty draft (that round is one AR step).
Verification stays the strict tree walk (sightline/tree_spec.py), so the row is lossless like every other row.

Design name in the evaluators: "REST-n<max match>-d<depth>-N<budget>" (e.g. REST-n8-d16-N127).
  python baselines/rest.py build --data <train.jsonl|dir> --out <store.npz> [--max-tokens 20000000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from sightline.tree_utils import TreeNode                                     # noqa: E402

LENS = (8, 6, 4, 3)                 # match lengths, longest first (REST: longest suffix that has continuations)
MOD = np.uint64(1000003)               # rolling-hash base; collisions are resolved by comparing the tokens


def _hashes(toks: np.ndarray, n: int) -> np.ndarray:
    """rolling hash of every n-gram starting at i, for i in [0, len - n]"""
    h = np.zeros(len(toks) - n + 1, dtype=np.uint64)
    for j in range(n):
        h = h * MOD + toks[j: len(toks) - n + 1 + j].astype(np.int64).astype(np.uint64)
    return h


def _key(seq, n: int) -> np.uint64:
    """= _hashes on the same n tokens (uint64 arithmetic wraps; Python ints masked to 64 bits)"""
    h = 0
    for t in seq[-n:]:
        h = (h * int(MOD) + int(t)) & 0xFFFFFFFFFFFFFFFF
    return np.uint64(h)


def build_store(data: Path, max_tokens: int, lens=LENS) -> dict:
    """Concatenate the response tokens of the training set (separated by -1) and index every n in lens."""
    parts, total = [], 0
    files = sorted(data.rglob("results.jsonl")) if data.is_dir() else [data]
    for f in files:
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "input_ids" in r:                      # train.jsonl of the text / image trainsets
                    ids, lm = r["input_ids"], r["loss_mask"]
                    gen = [t for t, m in zip(ids, lm) if m]
                else:                                     # raw generation units (gen_i2svg.py)
                    gen = r["candidates"][0]["token_ids"]
                if len(gen) < max(lens) + 1:
                    continue
                parts.append(np.asarray(gen, dtype=np.int64))
                parts.append(np.asarray([-1], dtype=np.int64))
                total += len(gen) + 1
                if total >= max_tokens:
                    break
        if total >= max_tokens:
            break
    toks = np.concatenate(parts).astype(np.int32)
    store = {"toks": toks, "lens": np.asarray(lens, dtype=np.int64)}
    for n in lens:
        h = _hashes(toks, n)
        order = np.argsort(h, kind="stable")
        store[f"h{n}"] = h[order]
        store[f"p{n}"] = order.astype(np.int32)           # start index of the n-gram (< 2^31 tokens)
    return store


class RESTDrafter:
    """One sequence. The datastore is shared (read-only numpy arrays)."""
    needs_hidden = False

    def __init__(self, store: dict, max_match: int = 8, depth: int = 16, budget: int = 127,
                 max_hits: int = 64, min_match: int = 2):
        self.s = store
        self.lens = [n for n in store["lens"].tolist() if min_match <= n <= max_match]
        self.depth, self.budget, self.max_hits = depth, budget, max_hits
        self.toks = store["toks"]

    def prefill_done(self, hidden_states):
        pass

    def update_hidden_kept(self, hidden_states, keep):
        pass

    def _continuations(self, committed: list[int]) -> list[list[int]]:
        for n in self.lens:
            if len(committed) < n:
                continue
            key = _key(committed, n)
            H, P = self.s[f"h{n}"], self.s[f"p{n}"]
            lo = int(np.searchsorted(H, key, side="left"))
            hi = int(np.searchsorted(H, key, side="right"))
            if hi <= lo:
                continue
            tail = committed[-n:]
            out = []
            for pos in P[lo:hi][: self.max_hits * 4]:
                pos = int(pos)
                if self.toks[pos: pos + n].tolist() != tail:        # hash collision
                    continue
                cont = self.toks[pos + n: pos + n + self.depth]
                cut = np.nonzero(cont < 0)[0]                        # do not cross a sequence boundary
                cont = cont[: int(cut[0])] if len(cut) else cont
                if len(cont):
                    out.append(cont.tolist())
                if len(out) >= self.max_hits:
                    break
            if out:
                return out
        return []

    def propose_tree(self, committed: list[int], start: int, max_depth: int) -> list[TreeNode]:
        conts = self._continuations(committed)
        if not conts:
            return []
        D = max(1, min(self.depth, max_depth))
        freq: dict[tuple, int] = {}
        for c in conts:
            for d in range(1, min(len(c), D) + 1):
                k = tuple(c[:d])
                freq[k] = freq.get(k, 0) + 1
        keep, seen = [], set()                                       # most frequent first, closed under prefixes
        for k in sorted(freq, key=lambda k: (-freq[k], len(k))):
            if len(seen) >= self.budget:
                break
            chain = [k[:d] for d in range(1, len(k) + 1) if k[:d] not in seen]
            if len(seen) + len(chain) > self.budget:
                continue
            for x in chain:
                seen.add(x)
                keep.append(x)
        keep.sort(key=lambda k: (len(k), -freq[k]))                  # packed order: parents first
        idx = {k: i for i, k in enumerate(keep)}
        return [TreeNode(int(k[-1]), len(k), idx[k[:-1]] if len(k) > 1 else -1, 0) for k in keep]


def load_store(path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=20_000_000)
    a = ap.parse_args()
    s = build_store(a.data, a.max_tokens)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **s)
    print("REST_STORE", a.out, "tokens", len(s["toks"]), "lens", s["lens"].tolist(),
          "MB", round(a.out.stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main()
