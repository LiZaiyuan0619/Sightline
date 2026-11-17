"""Shared τ calibers.

One function computes the four τ calibers × two aggregation styles so the
thermometer (offline exact simulation) and the harness evaluation (online
speculative runs) always report the same thing:

  Calibers (which tokens of a stream count):
    all: the full generated stream.
    eos: only streams that ended with EOS (loop-free streams, one definition).
    trunc: the stream truncated at the loop onset (loop-free prefix). Streams
            whose onset is <= MIN_HEALTHY_ONSET (looping essentially from the
            start — no loop-free prefix) are EXCLUDED and counted separately
            (C1 review §3.1: they used to fall back to the full-stream τ and
            inflated τ_trunc). Streams without a detected loop (EOS-ended,
            no-period, short) enter at full length.
    loop: only the loop segment [onset, end) — quantifies how much
            acceptance comes from looped tails.

  Aggregations (how per-stream numbers combine):
    per_stream: mean of per-stream τ = emitted_i / steps_i  (DFlash official).
    merged: Σ emitted / Σ steps over all streams in the caliber (GLANCE).

Framework-agnostic: callers turn their own runs into StreamRecord bundles;
this module only does selection + aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# A stream whose loop onset is at/below this position has no loop-free prefix
# worth reporting and is excluded from the trunc caliber (counted separately).
MIN_HEALTHY_ONSET = 2


@dataclass
class StreamRecord:
    """Per-stream raw material for τ: rounds used vs tokens committed."""
    steps: int        # verification rounds (= target forwards saved vs pure AR)
    emitted: int      # tokens committed by those rounds (incl. the bonus token)


@dataclass
class CaliberStream:
    """Per-stream bundle; `trunc`/`loop` may be None (not applicable/excluded)."""
    full: StreamRecord
    eos: bool
    onset: int | None          # loop onset position (same coords as the stream), None if no loop
    trunc: StreamRecord | None # None => excluded from trunc caliber (onset <= MIN_HEALTHY_ONSET)
    loop: StreamRecord | None  # None => stream has no loop segment


def tau_per_stream(records: Iterable[StreamRecord]) -> float:
    """DFlash official aggregation: mean of per-stream τ."""
    taus = [r.emitted / r.steps for r in records if r.steps > 0]
    return sum(taus) / len(taus) if taus else 0.0


def tau_merged(records: Iterable[StreamRecord]) -> float:
    """GLANCE aggregation: total emitted / total rounds."""
    records = list(records)
    steps = sum(r.steps for r in records)
    emitted = sum(r.emitted for r in records)
    return emitted / steps if steps else 0.0


def trunc_excluded(stream: CaliberStream) -> bool:
    """True when the stream has no loop-free prefix and is excluded from `trunc`."""
    return stream.onset is not None and stream.onset <= MIN_HEALTHY_ONSET


def caliber_table(streams: Iterable[CaliberStream]) -> dict:
    """The 4 × 2 τ table plus stream counts, from per-stream bundles."""
    streams = list(streams)
    eos_recs = [s.full for s in streams if s.eos]
    trunc_recs = [s.trunc for s in streams
                  if s.trunc is not None and not trunc_excluded(s)]
    loop_recs = [s.loop for s in streams if s.loop is not None and s.loop.steps > 0]
    return {
        "n_streams": len(streams),
        "n_eos": len(eos_recs),
        "n_loop": len(loop_recs),
        "n_trunc_excluded": sum(trunc_excluded(s) for s in streams),
        "calibers": {
            "all": {"per_stream": tau_per_stream(s.full for s in streams),
                    "merged": tau_merged(s.full for s in streams),
                    "n": len(streams)},
            "eos": {"per_stream": tau_per_stream(eos_recs),
                    "merged": tau_merged(eos_recs),
                    "n": len(eos_recs)},
            "trunc": {"per_stream": tau_per_stream(trunc_recs),
                      "merged": tau_merged(trunc_recs),
                      "n": len(trunc_recs)},
            "loop": {"per_stream": tau_per_stream(loop_recs),
                     "merged": tau_merged(loop_recs),
                     "n": len(loop_recs)},
        },
    }


def format_table(table: dict) -> str:
    """Compact one-drafter-per-line rendering of caliber_table."""
    c = table["calibers"]
    lines = [f"n={table['n_streams']} eos={table['n_eos']} "
             f"loop={table['n_loop']} trunc_excl={table['n_trunc_excluded']}"]
    for name in ("all", "eos", "trunc", "loop"):
        row = c[name]
        lines.append(f"  τ_{name:<5} per_stream={row['per_stream']:6.3f} "
                     f"merged={row['merged']:6.3f}  (n={row['n']})")
    return "\n".join(lines)
