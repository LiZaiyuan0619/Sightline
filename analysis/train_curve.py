"""Training curves of the B3 DFlash heads from the a batch GPU job logs of src/a batch GPU job/t7_train.sh.

Parses the timestamped `step N: {...}` lines (log_interval 50) and reports, per epoch,
the mean over the epoch's last 10 logged steps of: acc, ce_loss and SpecForge's
teacher-forced proxies `dflash/hard_label/unary_greedy_accepted_length` (chain) and
`unary_top16_oracle_accepted_length` (if the top-16 drafts of every offset were all
verified) — accepted lengths on the TRAINING streams under the data's own tokens, not
the harness tokens per round — plus the step rate.

Usage: python3 train_curve.py <run_dir> [<run_dir>...]   (TRAIN_ROOT/<run_id>)
Writes <run_dir>/train_curve.json and prints a table.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) step (\d+): (\{.*\})\s*$")
KEYS = {"acc": "acc", "ce_loss": "ce_loss",
        "proxy_len": "dflash/hard_label/unary_greedy_accepted_length",
        "top16_oracle_len": "dflash/hard_label/unary_top16_oracle_accepted_length"}
VAL = {k: re.compile(r"'" + re.escape(v) + r"': ([-+0-9.eE]+|nan|inf)") for k, v in KEYS.items()}


def curve(run_dir: Path) -> dict:
    plan = json.loads((run_dir / "plan.json").read_text())
    pts = []
    for line in Path(plan["log"]).read_text(errors="replace").split("\n"):
        m = LINE.match(line)
        if m:
            vals = {k: float(v.group(1)) for k, rx in VAL.items() if (v := rx.search(m.group(3)))}
            pts.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2)), vals))
    spe = plan["steps_per_epoch"]
    epochs = []
    for e in range(1, plan["epochs"] + 1):
        lo, hi = (e - 1) * spe, min(e * spe, plan["total_steps"])
        inside = [p for p in pts if lo < p[1] <= hi]
        if not inside:
            continue
        tail = inside[-10:]
        row = {"epoch": e, "last_step": inside[-1][1]}
        for k in KEYS:
            vals = [p[2][k] for p in tail if k in p[2]]
            row[k] = round(st.fmean(vals), 4) if vals else None
        if len(inside) > 1:
            dt = (inside[-1][0] - inside[0][0]).total_seconds()
            row["s_per_step"] = round(dt / max(1, inside[-1][1] - inside[0][1]), 3)
        epochs.append(row)
    out = {"run_id": plan["run_id"], "variant": plan["variant"], "global_batch": plan["global_batch"],
           "trainers": plan["trainers"], "n_logged": len(pts), "epochs": epochs}
    try:
        (run_dir / "train_curve.json").write_text(json.dumps(out, indent=2))
    except PermissionError:                      # run dir still root-owned while training
        pass
    return out


def main():
    for d in sys.argv[1:]:
        c = curve(Path(d))
        print(f"== {c['run_id']} (global batch {c['global_batch']}, {c['trainers']} trainers, "
              f"{c['n_logged']} logged steps)")
        print("| epoch | step | acc | ce_loss | proxy len (chain) | proxy len (top-16 oracle) | s/step |")
        print("|---|---|---|---|---|---|---|")
        for r in c["epochs"]:
            print(f"| {r['epoch']} | {r['last_step']} | {r['acc']} | {r['ce_loss']} | "
                  f"{r['proxy_len']} | {r['top16_oracle_len']} | {r.get('s_per_step')} |")


if __name__ == "__main__":
    main()
