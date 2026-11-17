"""One table of every decoder the paper evaluates, keyed by the design names used on the
command line.

A design name says which drafter runs and with what window and node budget, so one evaluator
can serve every row of the main table. The names are the ones in configs/, for example

    D0-chain            the block head as a chain over its trained window
    D1-top16-N127       the block head as a tree: 16 candidates per position, 127 nodes
    E3-k8-d12-N127      EAGLE-3, top 8 per step, depth 12, 127 nodes
    DART-W12-E64-N127   DARTree, width 12, 64 candidates per branch, 127 nodes
    VS-k8-d4-N29        ViSpec, top 8 per step, depth 4, 29 nodes
    REST-n8-d16-N127    REST, matches of up to 8 tokens, depth 16, 127 nodes
    PLD-n3-K10          prompt lookup, up to 3 context tokens, 10 copied
    SPS-g7              speculative sampling with a smaller model of the same family, 7 drafted
    AR                  the original model (no drafting), the stopwatch every speed-up uses

Every entry returns a drafter for sightline.tree_spec.tree_spec_generate together with the
block size that drafter verifies, so the verification path, the timing and the records are the
same for all of them -- the paper's "every decoder runs on the same target with the same
verifier" (Appendix E.2). Modules are imported inside the builders, so a run that uses one
baseline does not need the others installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sightline import design as sim


@dataclass(frozen=True)
class Baseline:
    """How one family of design names turns into a drafter."""

    tag: str                                   # sightline.design.Design source tag
    pattern: str                               # full-match on the design name
    budget: Callable[[tuple], int]             # node budget from the match groups
    block: Callable[[tuple], int]              # window the target verifies per round
    build: Callable[..., Any]                  # (head, target, groups, block) -> drafter
    detect: Callable[[Path], bool] | None = None   # recognises a checkpoint of this family
    load: Callable[..., Any] | None = None         # loads such a checkpoint
    needs_head: bool = True                    # False: training-free (no draft head at all)


# --- builders -----------------------------------------------------------------------------------
# Each is tiny on purpose: it names the upstream recipe it follows and passes that recipe's
# own parameters through.

def _pld(head, target, g, block):
    from baselines.pld import PLDDrafter
    return PLDDrafter(max_ngram=int(g[0]), K=int(g[1]))


def _rest(head, target, g, block, store=None):
    from baselines.rest import RESTDrafter
    return RESTDrafter(store, max_match=int(g[0]), depth=int(g[1]), budget=int(g[2]))


def _sps(head, target, g, block):
    from baselines.sps import SpSDrafter
    return SpSDrafter(head, int(g[0]))


def _eagle3(head, target, g, block):
    from baselines.eagle3 import Eagle3Drafter
    return Eagle3Drafter(head, target, topk=int(g[0]), depth=int(g[1]), budget=int(g[2]))


def _dartree(head, target, g, block):
    from baselines.dartree import DominoTreeDrafter
    bonus = -0.2 if g[3] is None else float(g[3])      # DARTree's pruned default
    return DominoTreeDrafter(head, target, block, width=int(g[0]), cand=int(g[1]),
                             budget=int(g[2]), depth_bonus=bonus)


def _vispec(head, target, g, block):
    from baselines.vispec.spec import ViSpecDrafter
    return ViSpecDrafter(head, target, topk=int(g[0]), depth=int(g[1]), budget=int(g[2]))


def _block_head(head, target, g, block):
    from sightline.tree_spec import TreeDrafter
    return TreeDrafter(head, target, block)


def _detect_eagle3(path):
    from baselines.eagle3 import is_eagle3_head
    return is_eagle3_head(path)


def _load_eagle3(path, dtype):
    from baselines.eagle3 import load_eagle3
    return load_eagle3(path, dtype=dtype)


def _detect_domino(path):
    from baselines.dartree import is_domino_head
    return is_domino_head(path)


def _load_domino(path, dtype):
    from baselines.dartree import load_domino
    return load_domino(path, dtype).cuda().eval()


def _detect_vispec(path):
    from baselines.vispec.spec import is_vispec_head
    return is_vispec_head(path)


def _load_vispec(path, dtype):
    from baselines.vispec.model import load_head
    return load_head(path, dtype=dtype)


def _load_block_head(path, dtype):
    from dflash.model import DFlashDraftModel
    return DFlashDraftModel.from_pretrained(path, torch_dtype=dtype).cuda().eval()


def _load_sps_draft(path, dtype):
    from baselines.sps import load_draft
    return load_draft(Path(path), dtype)


# --- the table ----------------------------------------------------------------------------------
BASELINES: list[Baseline] = [
    Baseline("pld", r"PLD-n(\d+)-K(\d+)",
             budget=lambda g: int(g[1]), block=lambda g: int(g[1]) + 1,
             build=_pld, needs_head=False),
    Baseline("rest", r"REST-n(\d+)-d(\d+)-N(\d+)",
             budget=lambda g: int(g[2]), block=lambda g: int(g[1]) + 1,
             build=_rest, needs_head=False),
    Baseline("sps", r"SPS-g(\d+)",
             budget=lambda g: int(g[0]), block=lambda g: int(g[0]) + 1,
             build=_sps, detect=lambda p: "4B" in str(p) and (Path(p) / "config.json").exists(),
             load=_load_sps_draft),
    Baseline("eagle3", r"E3-k(\d+)-d(\d+)-N(\d+)",
             budget=lambda g: int(g[2]), block=lambda g: int(g[1]) + 1,
             build=_eagle3, detect=_detect_eagle3, load=_load_eagle3),
    # DARTree reads its Domino head's own block size: block_size + 1 draft depths plus the root.
    Baseline("dartree", r"DART-W(\d+)-E(\d+)-N(\d+)(?:-b(-?[\d.]+))?",
             budget=lambda g: int(g[2]), block=lambda g: 0,
             build=_dartree, detect=_detect_domino, load=_load_domino),
    Baseline("vispec", r"VS-k(\d+)-d(\d+)-N(\d+)",
             budget=lambda g: int(g[2]), block=lambda g: int(g[1]) + 1,
             build=_vispec, detect=_detect_vispec, load=_load_vispec),
]

TAGS = {b.tag: b for b in BASELINES}


# --- the three hooks an evaluator needs ---------------------------------------------------------
class Designs:
    """The designs of one run, and what each of them needs at decode time."""

    def __init__(self, spec: str):
        self.by_name: dict[str, tuple[Baseline, tuple]] = {}
        picked, rest = [], []
        for name in spec.split(","):
            name = name.strip()
            if not name:
                continue
            for b in BASELINES:
                m = re.fullmatch(b.pattern, name)
                if m:
                    g = m.groups()
                    self.by_name[name] = (b, g)
                    picked.append(sim.Design(name, b.budget(g), (b.tag,)))
                    break
            else:
                rest.append(name)                    # a block-head design, named in configs/
        self.designs = picked + (sim.pick_designs(",".join(rest)) if rest else [])

    def __iter__(self):
        return iter(self.designs)

    def __len__(self):
        return len(self.designs)

    @property
    def needs_store(self) -> bool:
        return any(b.tag == "rest" for b, _ in self.by_name.values())

    @property
    def needs_head(self) -> bool:
        """False only when every design in the run is training-free (PLD, REST)."""
        if len(self.by_name) < len(self.designs):
            return True                              # at least one block-head design
        return any(b.needs_head for b, _ in self.by_name.values())

    def load_head(self, path, dtype, default=_load_block_head):
        """Load the checkpoint, dispatching on which family recognises it."""
        if path is None or str(path) in ("none", "None"):
            return None
        for b in BASELINES:
            if b.detect is not None and b.load is not None and b.detect(path):
                return b.load(path, dtype)
        return default(path, dtype)

    def drafter(self, name: str, head, target, block_size: int, **kw):
        """Return (drafter, block size verified per round) for one design."""
        if name not in self.by_name:
            return _block_head(head, target, (), block_size), block_size
        b, g = self.by_name[name]
        block = b.block(g) or block_size
        if b.tag == "dartree":
            block = block_size + 1                   # the head's own block size plus the root
        drafter = b.build(head, target, g, block, **kw) if b.tag == "rest" \
            else b.build(head, target, g, block)
        return drafter, block
