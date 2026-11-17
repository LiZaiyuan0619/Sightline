"""Pick the checkpoint of a training run, by held-out tokens per round.

Training saves a checkpoint per epoch. The paper keeps, for every decoder including ours, the
checkpoint that writes the most tokens per round on held-out prompts -- not the one with the
lowest training loss, which is a different checkpoint often enough to matter (Appendix E.2).

Two steps, because the first needs a GPU and the second does not:

  1. decode the held-out rows with each checkpoint, one directory per checkpoint:

       python3 train/select_head.py decode --run <training dir> --design D0-chain \\
           --rows <held-out rows jsonl> --out <selection dir> [--profile internsvg]

  2. read those results and report the winner:

       python3 train/select_head.py pick --out <selection dir> --design D0-chain

Step 2 writes `summary.txt` (every checkpoint, overall and per subtype) and `best.txt` (the
winning checkpoint's path), which is what the evaluation jobs read.

Held-out rows: prompts of the training corpus's test split that no head was trained on. The
paper uses every tenth row of that split, balanced across subtypes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root

ROOT = Path(__file__).resolve().parents[1]


def tokens_per_round(records: list[dict]) -> float:
    """Tokens written per verification round, pooled over rows (not a mean of per-row ratios)."""
    written = sum(r["rec_full"][1] for r in records)
    rounds = sum(r["rec_full"][0] for r in records)
    return written / max(1, rounds)


def read(design_dir: Path) -> list[dict]:
    out = []
    for f in sorted(design_dir.glob("shard-*/results.jsonl")):
        out += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return out


def decode(a: argparse.Namespace) -> int:
    steps = sorted((a.run / "heads").glob("step*"), key=lambda p: int(p.name[4:]))
    if not steps:
        print(f"no checkpoints under {a.run / 'heads'}")
        return 2
    entry = ROOT / ("eval/eval_text.py" if a.task == "text" else f"eval/eval_image_{a.target}.py")
    for step in steps:
        dst = a.out / step.name
        if (dst / a.design).exists():
            print(f"  {step.name}: already decoded")
            continue
        cmd = [sys.executable, str(entry), "--head", str(step), "--designs", a.design,
               "--xrows", str(a.rows), "--sets", a.sets, "--modes", "greedy",
               "--no-ar", "--out", str(dst)]
        if a.profile:
            cmd += ["--profile", a.profile]
        print("  " + " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc:
            return rc
    return 0


def pick(a: argparse.Namespace) -> int:
    lines, best = [], None
    for d in sorted(a.out.glob("step*"), key=lambda p: int(p.name[4:])):
        records = read(d / a.design)
        if not records:
            continue
        overall = tokens_per_round(records)
        per_type = {t: tokens_per_round([r for r in records if r.get("subtype") == t])
                    for t in sorted({r.get("subtype") for r in records} - {None})}
        lines.append(f"{d.name} n={len(records)} tokens_per_round={overall:.3f} "
                     + " ".join(f"{t}={v:.3f}" for t, v in per_type.items()))
        if best is None or overall > best[1]:
            best = (d.name, overall)
    if not lines:
        print(f"no results under {a.out} for design {a.design}")
        return 2
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if best:
        (a.out / "best.txt").write_text(f"{best[0]}\n")
        print(f"BEST {best[0]} with {best[1]:.3f} tokens per round")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decode", help="decode the held-out rows with every checkpoint (GPU)")
    d.add_argument("--run", type=Path, required=True, help="training directory, holding heads/step*")
    d.add_argument("--design", required=True, help="the design to select on, e.g. D0-chain")
    d.add_argument("--rows", type=Path, required=True, help="held-out rows (eval/rows/)")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--sets", default="extra", help="which set names of the row file to use")
    d.add_argument("--task", default="text", choices=["text", "image"])
    d.add_argument("--target", default="omnisvg", choices=["omnisvg", "internsvg", "starvector"])
    d.add_argument("--profile", default=None, choices=[None, "internsvg"])

    p = sub.add_parser("pick", help="read the decoded results and report the winner (CPU)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--design", required=True)

    a = ap.parse_args()
    return decode(a) if a.cmd == "decode" else pick(a)


if __name__ == "__main__":
    sys.exit(main())
