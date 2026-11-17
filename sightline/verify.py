"""Block verification core for the strict verifier.

Pure logic, no model dependency: given a drafted block and the target's raw logits at
each block position, decide acceptance per position with the FULL HF-order processing
chain (sightline/process.py) and the correct penalty history, and produce the token(s)
that the strictly-lossless decoder must emit.

Verification rule (sample-and-compare; trivially lossless for ANY drafter):
    y_j = draw(chain(L_{j-1}, history_j))          # target's own draw at position j
    accept d_j iff d_j == y_j
    emitted token at j: y_j (which equals d_j when accepted)
On first rejection at position j: emit y_j as the bonus token and stop the block.
If all K accepted: bonus y_{K+1} = draw(chain(L_K, history incl. all drafts)).

Greedy mode: draw = argmax of the processed logits (HF greedy path: penalty applies;
warpers are no-ops for argmax).

Geometric tolerance (coord_tol = r > 0;  probe, NOT lossless): a drafted coordinate is
also accepted when the target's own draw y_j is a coordinate within Chebyshev distance r (grid
units) of it, and the DRAFTED coordinate is committed ("snap"). Guarantee: every emitted token
equals the target's draw at its prefix, except coordinates, which are within r of that draw.
Greedy and seeded sampling use the same rule. coord_tol = 0 is the strict rule above.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.process import apply_chain, draw
from sightline.svgtok.types import COORD_END, COORD_START, GRID


def coord_cheb(a: int, b: int) -> int | None:
    """Chebyshev distance (grid units) between two coordinate tokens; None unless both are
    coordinates (token - COORD_START = y * GRID + x)."""
    if not (COORD_START <= a <= COORD_END and COORD_START <= b <= COORD_END):
        return None
    ya, xa = divmod(a - COORD_START, GRID)
    yb, xb = divmod(b - COORD_START, GRID)
    return max(abs(xa - xb), abs(ya - yb))


@dataclass
class ChainParams:
    greedy: bool = True
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    loop_ban: int = 0               #  loop escape (process.loop_ban): max period, 0 = off
    loop_repeats: int = 2
    loop_min_period: int = 4
    loop_action: str = "ban"        # "ban" the continuation | "eos" stop at the detected loop
                                    # | "rollback" / "esc_stop" (process.loop_rollback; decoder-level)
    loop_max_rb: int = 1            # rollbacks / escapes before the next firing stops with EOS


@dataclass
class BlockResult:
    n_accepted: int                 # accepted draft tokens (0..K)
    emitted: list[int]              # tokens the decoder must commit (n_accepted + 1)
    per_position: list[dict] = field(default_factory=list)
    # per_position[j]: {"draft": d_j, "target": y_j, "accepted": bool, "history_len": n}
    stopped_on_eos: bool = False


def verify_block(draft: list[int], raw_logits: list[torch.Tensor], committed: list[int],
                 params: ChainParams, generator: torch.Generator | None = None,
                 coord_tol: int = 0) -> BlockResult:
    """Verify one drafted block.

    draft:        K proposed tokens d_1..d_K
    raw_logits:   K+1 raw logit tensors from the target's single block forward:
                  raw_logits[j-1] predicts the token at draft position j (j=1..K);
                  raw_logits[K] predicts the token after the block (bonus when all accepted).
    committed:    full history before the block (prompt + all previously committed tokens).
    coord_tol:    geometric tolerance in grid units (module docstring); 0 = strict.
    """
    K = len(draft)
    assert len(raw_logits) == K + 1, (len(draft), len(raw_logits))
    history = list(committed)
    emitted: list[int] = []
    per_position: list[dict] = []

    for j in range(1, K + 1):
        logits = raw_logits[j - 1]
        y = int(_chain_draw(logits, history, params, generator))
        accepted = (draft[j - 1] == y)
        snapped = False
        if not accepted and coord_tol > 0:
            dist = coord_cheb(draft[j - 1], y)
            snapped = accepted = dist is not None and dist <= coord_tol
        # margin on the PROCESSED logits (what the decision used): d - y, >=0 iff accepted
        processed = apply_chain(logits, history, penalty=params.repetition_penalty,
                                temperature_=params.temperature, top_k_=params.top_k,
                                top_p_=params.top_p, greedy=params.greedy,
                                loop_ban_=params.loop_ban, loop_repeats_=params.loop_repeats,
                            loop_min_period_=params.loop_min_period, loop_action_=params.loop_action)
        margin = float(processed[draft[j - 1]] - processed[y]) if draft[j - 1] != y else 0.0
        per_position.append({"draft": draft[j - 1], "target": y,
                             "accepted": accepted, "history_len": len(history),
                             "margin": margin, "snapped": snapped})
        if not accepted:
            emitted.append(y)                       # bonus token = the target's own draw
            return BlockResult(j - 1, emitted, per_position)
        emitted.append(draft[j - 1])
        history.append(draft[j - 1])

    # all K accepted -> bonus token from the position after the block
    bonus = int(_chain_draw(raw_logits[K], history, params, generator))
    emitted.append(bonus)
    per_position.append({"draft": None, "target": bonus, "accepted": None,
                         "history_len": len(history)})
    return BlockResult(K, emitted, per_position)


def _chain_draw(logits: torch.Tensor, history: list[int], params: ChainParams,
                generator: torch.Generator | None) -> torch.Tensor:
    processed = apply_chain(logits, history, penalty=params.repetition_penalty,
                            temperature_=params.temperature, top_k_=params.top_k,
                            top_p_=params.top_p, greedy=params.greedy,
                            loop_ban_=params.loop_ban, loop_repeats_=params.loop_repeats,
                            loop_min_period_=params.loop_min_period, loop_action_=params.loop_action)
    return draw(processed, params.greedy, generator)
