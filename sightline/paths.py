"""The four roots every script in this package reads from, set by environment variable.

    SIGHTLINE_RUNS    evaluation outputs, draft-head checkpoints, cached references
    SIGHTLINE_MODELS  target model weights (OmniSVG, InternSVG, StarVector, ...)
    SIGHTLINE_OUT     where a fresh run writes
    SIGHTLINE_REFS    clones of the upstream repositories listed in docs/third_party.md

Nothing in the package writes outside SIGHTLINE_OUT. Defaults are directories next to the
package, so a reader can try a script without setting anything and get a clear error naming
the variable if the data it needs is absent.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent

RUNS = Path(os.environ.get("SIGHTLINE_RUNS", PKG / "data/runs"))
MODELS = Path(os.environ.get("SIGHTLINE_MODELS", PKG / "models"))
OUT = Path(os.environ.get("SIGHTLINE_OUT", PKG / "out"))
REFS = Path(os.environ.get("SIGHTLINE_REFS", PKG / "refs"))

_VAR = {"RUNS": "SIGHTLINE_RUNS", "MODELS": "SIGHTLINE_MODELS",
        "OUT": "SIGHTLINE_OUT", "REFS": "SIGHTLINE_REFS"}


def require(path: Path, what: str = "") -> Path:
    """Return `path`, or raise with the variable to set and what was expected there."""
    if path.exists():
        return path
    for name, var in _VAR.items():
        root = globals()[name]
        if path == root or root in path.parents:
            rel = path.relative_to(root)
            raise FileNotFoundError(
                f"{path} is missing. Set ${var} to the directory that holds '{rel}'"
                + (f" ({what})" if what else "")
                + f". Current ${var} = {root}")
    raise FileNotFoundError(f"{path} is missing" + (f" ({what})" if what else ""))
