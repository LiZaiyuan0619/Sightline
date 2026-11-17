""" the acceptance profile and the token-class acceptance of EVERY method in table 1, not only the four the
 /  scripts were wired to (DFlash, DDTree, EAGLE-3, ours). The figures compare the full baseline set, so
the analysis has to as well: PLD, REST, DFlash, DDTree, EAGLE-3, DARTree, ours (ViSpec is added when it trains).

Per method and block, from the result files only (0 GPU):
  rounds, tokens per round (all tokens), median emitted, share of rounds that hit the speculation limit, tree nodes per accepted
  token, the histogram of tokens emitted per round and its survival curve;
  and, for the SVG-token targets, the per-class draft accuracy and the class's share of all rejections (the
  caliber: loop-free positions, before any loop onset).

  python3 analysis/profiles_all.py --blocks t2 isvg i2 i2s 4b --modes greedy sample --out runs/2026-10-30/analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from sightline.svgtok.types import CMD_ARGS, CMD_BY_ID, token_type  # noqa: E402
from sightline.paths import MODELS, RUNS

U = RUNS
CLASSES = ["cmd", "coord:M", "coord:L", "coord:C-ctrl", "coord:C-end", "coord:Z", "arc/A", "color", "eos", "other"]

# block -> method -> (roots relative to U, design, speculation limit (block size, or depth + 1))
# roots are the table-1 row sources (runs/2026-10-30/tableA_v9.py), so these are the very rows the table reports.
B = {
    "t2": {
        "PLD": (["tableA/t2-own/pld", "tableA/t2-x/pld"], "PLD-n3-K10", 11),
        "REST": (["r1029/rest-eval/t2/own", "r1029/rest-eval/t2/x"], "REST-n8-d16-N127", 17),
        "DFlash": (["tableA/t2-own/b16", "tableA/t2-x/b16"], "D0-chain", 16),
        "DDTree": (["tableA/t2-own/b16", "tableA/t2-x/b16"], "D1-top16-N127", 16),
        "EAGLE-3": (["tableA/t2-own/e3", "tableA/t2-x/e3"], "E3-k8-d12-N127", 13),
        "DARTree": (["tableA/dart8-fix/t2-own", "tableA/dart8-fix/t2-x"], "DART-W12-E64-N127", 17),
        "Ours": (["tableA/t2-own/ours", "tableA/t2-x/ours"], "D1-top16-N127", 32),
    },
    "4b": {
        "PLD": (["tableA/t2-4b/own/pld", "tableA/t2-4b/x/pld"], "PLD-n3-K10", 11),
        "REST": (["r1029/rest-eval/4b/own", "r1029/rest-eval/4b/x"], "REST-n8-d16-N127", 17),
        "DFlash": (["tableA/t2-4b/own/b16", "tableA/t2-4b/x/b16"], "D0-chain", 16),
        "DDTree": (["tableA/t2-4b/own/b16", "tableA/t2-4b/x/b16"], "D1-top16-N127", 16),
        "EAGLE-3": (["tableA/t2-4b/own/e3", "tableA/t2-4b/x/e3"], "E3-k8-d12-N127", 13),
        "DARTree": (["tableA/dart-4b/own", "tableA/dart-4b/x"], "DART-W12-E64-N127", 17),
        "Ours": (["tableA/t2-4b/own/ours", "tableA/t2-4b/x/ours"], "D1-top16-N127", 32),
    },
    "isvg": {
        "PLD": (["tableA/isvg-own/pld", "tableA/isvg-x/pld"], "PLD-n3-K10", 11),
        "REST": (["r1029/rest-eval/isvg/own", "r1029/rest-eval/isvg/x"], "REST-n8-d16-N127", 17),
        "DFlash": (["tableA/isvg-own/b16", "tableA/isvg-x/b16"], "D0-chain", 16),
        "DDTree": (["tableA/isvg-own/b16", "tableA/isvg-x/b16"], "D1-top16-N127", 16),
        "EAGLE-3": (["tableA/isvg-own/e3", "tableA/isvg-x/e3"], "E3-k8-d12-N127", 13),
        "DARTree": (["tableA/dart-isvg/own", "tableA/dart-isvg/x"], "DART-W12-E64-N127", 17),
        "Ours": (["tableA/isvg-own/ours", "tableA/isvg-x/ours"], "D1-top16-N127", 32),
    },
    "i2": {
        "PLD": (["tableA/i2-own/pld", "tableA/i2-x-b127/pld"], "PLD-n3-K10", 11),
        "REST": (["r1029/rest-eval/i2/own", "r1029/rest-eval/i2/x"], "REST-n8-d16-N127", 17),
        "DFlash": (["tableA/i2-own-b127/b16", "tableA/i2-x-b127/b16"], "D0-chain", 16),
        "DDTree": (["tableA/i2-own-b127/b16", "tableA/i2-x-b127/b16"], "D1-top16-N127", 16),
        "EAGLE-3": (["tableA/e3-i2/own", "tableA/e3-i2/x"], "E3-k8-d12-N127", 13),
        "DARTree": (["tableA/dart-i2-v3/own", "tableA/dart-i2-v3/x"], "DART-W12-E64-N127", 17),
        "Ours": (["tableA/i2-own-b127/ours", "tableA/i2-x-b127/ours"], "D1-top16-N127", 32),
    },
    "i2s": {
        "PLD": (["tableA/isvg-i2/own/b16", "tableA/isvg-i2/x/b16"], "PLD-n3-K10", 11),
        "REST": (["r1029/rest-eval/i2s/own", "r1029/rest-eval/i2s/x"], "REST-n8-d16-N127", 17),
        "DFlash": (["tableA/isvg-i2/own/b16", "tableA/isvg-i2/x/b16"], "D0-chain", 16),
        "DDTree": (["tableA/isvg-i2/own/b16", "tableA/isvg-i2/x/b16"], "D1-top16-N127", 16),
        "EAGLE-3": (["tableA/isvg-i2/own/e3", "tableA/isvg-i2/x/e3"], "E3-k8-d12-N127", 13),
        "DARTree": (["tableA/isvg-i2/own/dart", "tableA/isvg-i2/x/dart"], "DART-W12-E64-N127", 17),
        "Ours": (["tableA/isvg-i2/own/ours", "tableA/isvg-i2/x/ours"], "D1-top16-N127", 32),
    },
}
SVG_TOKENS = {"t2", "4b", "i2"}          # OmniSVG vocabulary: the token classes of sightline/sightline/svgtok/types.py apply
ISVG_TOKENS = {"isvg", "i2s"}            # InternSVG vocabulary: classes from the decoded string (ISVG_CLASSES below)
# the InternSVG classifier of runs/2026-10-22/analysis/token_type_accept_isvg.py, which ran on 4 methods and
# one block. Having it here puts the class panel of claim M1 on all FIVE blocks x 7 methods instead of three blocks.
ISVG_CLASSES = ["number", "path command", "color", "markup", "space-punct", "other text", "eos"]
ISVG_EOS = 151645
_NUM = re.compile(r"^[\s,]*-?[0-9.]+[\s,]*$")
_CMD = re.compile(r"^\s*[MmLlCcQqAaZzHhVvSsTt]\s*$")
_ISVG_TOK = f"{MODELS}/InternSVG-8B-qwen2-text"
_W: dict = {}


def isvg_class(tid: int) -> str:
    """Class of one InternSVG token id (cached; the tokenizer is loaded on first use)."""
    if tid in _W.setdefault("cache", {}):
        return _W["cache"][tid]
    if "tok" not in _W:
        from transformers import AutoTokenizer
        _W["tok"] = AutoTokenizer.from_pretrained(_ISVG_TOK)
    s = _W["tok"].decode([tid])
    if tid == ISVG_EOS:
        c = "eos"
    elif _NUM.match(s):
        c = "number"
    elif _CMD.match(s):
        c = "path command"
    elif "#" in s:
        c = "color"
    elif any(ch in s for ch in '<>="/:'):
        c = "markup"
    elif not s.strip() or all(not ch.isalnum() for ch in s):
        c = "space-punct"
    else:
        c = "other text"
    _W["cache"][tid] = c
    return c


def rows(roots, design, mode):
    out = []
    for rt in roots:
        for f in sorted(glob.glob(str(U / rt / design / "shard-*/results.jsonl"))):
            out += [r for r in map(json.loads, open(f)) if r and r["mode"] == mode]
    return out


def profile(rs, cap):
    hist: Counter = Counter()
    nodes = emitted = full = 0
    per_row = []                                            # per row: (ceiling share, tokens per round) for distribution panels
    for r in rs:
        acc = r["acceptance"]
        for n in acc:
            hist[n] += 1
            emitted += n
            full += int(n >= cap)
        if acc:
            per_row.append((sum(n >= cap for n in acc) / len(acc), sum(acc) / len(acc)))
        if r.get("mean_tree_nodes"):
            nodes += r["mean_tree_nodes"] * len(r["acceptance"])
    tot = sum(hist.values())
    if not tot:
        return None
    surv, run = {}, tot
    for j in range(1, cap + 2):
        surv[j] = run / tot
        run -= hist.get(j, 0)
    return {"rows": len(rs), "cap": cap, "rounds": tot, "tau_all": emitted / tot,
            "median": st.median([n for n, c in sorted(hist.items()) for _ in range(c)]) if tot else 0,
            "full_block_share": full / tot, "nodes_per_token": (nodes / emitted) if emitted and nodes else None,
            "hist": {str(k): v / tot for k, v in sorted(hist.items())},
            "survival": {str(k): v for k, v in surv.items()},
            "per_row_ceiling": [round(a, 4) for a, _ in per_row],
            "per_row_tau": [round(b, 3) for _, b in per_row],
            "first_token_rejected": hist.get(1, 0) / tot}


def token_classes(tokens):
    """Class label per position, following the official grammar (sightline/sightline/svgtok/types.py)."""
    out, cmd, pos = [], None, 0
    for t in tokens:
        ty = token_type(t)
        if ty == "color":
            out.append("color")
            cmd, pos = None, 0
        elif ty == "cmd":
            out.append("cmd")
            cmd, pos = CMD_BY_ID[t], 0
        elif ty in ("coord", "arc"):
            if cmd is None:
                out.append("other")
                continue
            if cmd == "M":
                lab = "coord:M"
            elif cmd == "L":
                lab = "coord:L"
            elif cmd == "C":
                lab = "coord:C-end" if pos == 2 else "coord:C-ctrl"
            elif cmd == "Z":
                lab = "coord:Z"
            else:
                lab = "arc/A"
            out.append(lab)
            pos += 1
            if pos >= len(CMD_ARGS[cmd]):
                cmd, pos = None, 0
        elif ty == "eos":
            out.append("eos")
            cmd, pos = None, 0
        else:
            out.append("other")
            cmd, pos = None, 0
    return out


def class_accuracy(rs, cap, labels=None, classes=None):
    """Per class: share of positions, P(draft right | round reached it), share of all rejections (loop-free only).
    labels(tokens) -> one label per position (default: the OmniSVG grammar classes); classes = the labels to report."""
    labels = labels or token_classes
    classes = classes or CLASSES
    seen, acc, rej = Counter(), Counter(), Counter()
    for r in rs:
        toks = r.get("spec_tokens")
        if not toks:
            continue
        stop = r["onset"] if r.get("onset") else len(toks)
        lab = labels(toks)
        s = 1                                              # spec_tokens[0] comes from the prefill
        for e in r["acceptance"]:
            for i in range(s, min(s + e - 1, stop)):       # accepted draft tokens
                seen[lab[i]] += 1
                acc[lab[i]] += 1
            j = s + e - 1                                  # the target's token: a rejection unless the round was full
            if j < stop and e < cap:
                seen[lab[j]] += 1
                rej[lab[j]] += 1
            s += e
            if s >= stop:
                break
    tot_seen, tot_rej = sum(seen.values()), sum(rej.values())
    return {c: {"share": seen[c] / tot_seen if tot_seen else 0.0,
                "acc": acc[c] / seen[c] if seen[c] else None,
                "rej_share": rej[c] / tot_rej if tot_rej else 0.0} for c in classes if seen[c]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", default=["t2", "isvg", "i2", "i2s", "4b"])
    ap.add_argument("--modes", nargs="+", default=["greedy", "sample"])
    ap.add_argument("--out", type=Path, default=RUNS / "2026-10-30/analysis")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for blk in a.blocks:
        res = defaultdict(dict)
        for mode in a.modes:
            for meth, (roots, design, cap) in B[blk].items():
                rs = rows(roots, design, mode)
                if not rs:
                    print(f"[{blk}/{mode}] {meth}: no rows", flush=True)
                    continue
                p = profile(rs, cap)
                if blk in SVG_TOKENS:
                    p["classes"] = class_accuracy(rs, cap)
                elif blk in ISVG_TOKENS:
                    p["classes"] = class_accuracy(rs, cap, lambda ts: [isvg_class(t) for t in ts], ISVG_CLASSES)
                res[mode][meth] = p
                print(f"[{blk}/{mode}] {meth}: {p['rows']} rows, {p['rounds']} rounds, tau {p['tau_all']:.2f}, "
                      f"ceiling {p['full_block_share']:.3f}", flush=True)
        (a.out / f"profile_{blk}.json").write_text(json.dumps(res, indent=1))
        print(f"wrote {a.out / f'profile_{blk}.json'}", flush=True)


if __name__ == "__main__":
    main()
