""" the entropy-acceptance law on SVG (GLANCE Fig 4 / T12-T15 style), the
mechanism behind "the SVG token stream is predictable, so long speculative blocks pay off".

For every generated position of an already-decoded run we need two things:
  - whether the verifier took that token from the draft or had to spend the round's bonus draw on it;
  - how uncertain the TARGET itself was there (entropy of its next-token distribution).
Both come without re-decoding. The results files store `spec_tokens` (the whole output) and `acceptance` (how many
tokens each round emitted), and the strict walk emits, per round, `n_accepted` drafted tokens followed by one bonus
token drawn from the target - so the last token of a round is a rejection point unless the whole block was accepted
(acceptance[r] == block_size, the "full block" case, where the walk ran out of tree instead of being corrected).
Entropy comes from ONE teacher-forced forward per row over prompt + output.

Outputs (JSON + markdown): acceptance rate per entropy bucket, mean entropy at accepted vs rejected positions, the same
split by SVG token type (src/svgtok), and the entropy spectrum (quantiles) per token type.

  python analysis/entropy_accept.py --rows <results dir> [<dir>...] --target <text model> \
      --out <prefix> [--limit 200]
  (image-to-SVG runs: --task i2, the VLM target of vlm.py; InternSVG: --profile internsvg)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "src/h2", "src/t7", "src/h3a"):
    sys.path.insert(0, str(ROOT / p))

BUCKETS = [0.0, 0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 1e9]


def token_class(t: int, profile: str) -> str:
    """OmniSVG: the parametric token ranges (sightline/sightline/svgtok/types.py). InternSVG / others: raw code, by decoded text."""
    if profile != "omnisvg":
        return "all"
    from sightline.svgtok.types import COORD_END, COORD_START
    if COORD_START <= t <= COORD_END:
        return "coord"
    if 196_900 <= t <= 196_999:
        return "special"
    if t >= 192_072:
        return "color"
    return "cmd"


def labels_from_acceptance(acc: list[int], block: int) -> list[int]:
    """1 = taken from the draft, 0 = the round's bonus draw (the position the draft got wrong)."""
    out = []
    for n in acc:
        if n <= 0:
            continue
        if n >= block:                      # whole block accepted: no bonus token in this round
            out += [1] * n
        else:
            out += [1] * (n - 1) + [0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True, help="design directories with shard-*/results.jsonl")
    ap.add_argument("--labels", nargs="*", default=None, help="one name per --rows entry")
    ap.add_argument("--target", type=Path, default=None)
    ap.add_argument("--profile", default="omnisvg", choices=["omnisvg", "internsvg"])
    ap.add_argument("--task", default="t2", choices=["t2", "i2"])
    ap.add_argument("--block", type=int, default=32, help="block size of the run (full-block rounds have no bonus)")
    ap.add_argument("--xrows", type=Path, default=None,
                    help="text tasks: the row file with prompt_input_ids (results rows do not store them)")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = a.labels or [Path(r).parent.name + "/" + Path(r).name for r in a.rows]
    PROMPTS, IMAGES = {}, {}
    if a.task == "i2":
        src = a.xrows or (RUNS / "h3a-data/eval_rows.jsonl")
        for x in src.read_text().splitlines():
            if x.strip():
                q = json.loads(x)
                IMAGES[q["gidx"]] = q["image_path"]
    if a.xrows is not None and a.task != "i2":
        for x in a.xrows.read_text().splitlines():
            if x.strip():
                q = json.loads(x)
                PROMPTS[q["gidx"]] = q["prompt_input_ids"]

    if a.task == "i2":
        import vlm
        official = vlm.load_official()
        target, note = vlm.load_target(torch.bfloat16)
        tmp = a.out.parent / ".entropy-input.png"
    else:
        from transformers import AutoModelForCausalLM
        mp = a.target or (MODELS /
                          ("InternSVG-8B-qwen2-text" if a.profile == "internsvg" else "OmniSVG1.1_8B-qwen2-text"))
        target = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16,
                                                      attn_implementation="sdpa").cuda().eval()
        note = str(mp)
    print("[target]", note, flush=True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    stats = {n: {"buck": defaultdict(lambda: [0, 0]),            # bucket -> [accepted, total]
                 "cls": defaultdict(lambda: [0.0, 0, 0.0, 0]),   # class -> [H_acc, n_acc, H_rej, n_rej]
                 "spec": defaultdict(list)} for n in names}      # class -> entropies (for the spectrum)
    for name, root in zip(names, a.rows):
        rows = [json.loads(x) for f in sorted(glob.glob(f"{root}/shard-*/results.jsonl"))
                for x in open(f) if x.strip()]
        rows = [r for r in rows if r["mode"] == a.mode][:a.limit]
        print(f"[rows] {name}: {len(rows)}", flush=True)
        for i, r in enumerate(rows):
            lab = labels_from_acceptance(r["acceptance"], a.block)
            toks = r["spec_tokens"][:len(lab)]
            lab = lab[:len(toks)]
            if len(toks) < 8:
                continue
            with torch.inference_mode():
                if a.task == "i2":
                    img = r.get("image_path") or IMAGES[r["gidx"]]      # image rows keep the path in the row file
                    inp = vlm.build_inputs(official, target, img, tmp, torch.bfloat16)
                    ids = torch.tensor([inp.prompt_ids + toks], device="cuda")
                    n_p = len(inp.prompt_ids)
                    pos = torch.cat([inp.prefill_positions,
                                     (torch.arange(n_p, ids.shape[1], device="cuda") + inp.delta)[None, None]
                                     .expand(3, 1, -1)], dim=2)
                    out = target(ids, position_ids=pos, **{k: v for k, v in inp.prefill_kwargs().items()
                                                           if k != "position_ids"})
                else:
                    full = r.get("prompt_input_ids") or PROMPTS.get(r["gidx"])
                    assert full is not None, f"no prompt ids for gidx {r['gidx']}; pass --xrows"
                    ids = torch.tensor([full + toks], device="cuda")
                    n_p = len(full)
                    out = target(ids)
                lg = out.logits[0, n_p - 1: ids.shape[1] - 1].float()             # distribution BEFORE each output token
                lp = torch.log_softmax(lg, dim=-1)
                H = (-(lp.exp() * lp).sum(-1)).tolist()
            S = stats[name]
            for j, (t, l) in enumerate(zip(toks, lab)):
                h = H[j]
                b = next(k for k in range(len(BUCKETS) - 1) if BUCKETS[k] <= h < BUCKETS[k + 1])
                S["buck"][b][0] += l
                S["buck"][b][1] += 1
                c = token_class(t, a.profile)
                S["cls"][c][0 if l else 2] += h
                S["cls"][c][1 if l else 3] += 1
                if len(S["spec"][c]) < 200_000:
                    S["spec"][c].append(round(h, 4))
            if (i + 1) % 20 == 0:
                print(f"  {name} {i + 1}/{len(rows)}", flush=True)

    res = {}
    for n in names:
        S = stats[n]
        res[n] = {
            "buckets": [{"lo": BUCKETS[b], "hi": BUCKETS[b + 1], "accept": v[0] / max(v[1], 1), "n": v[1]}
                        for b, v in sorted(S["buck"].items())],
            "by_class": {c: {"H_accepted": v[0] / max(v[1], 1), "n_accepted": v[1],
                             "H_rejected": v[2] / max(v[3], 1), "n_rejected": v[3],
                             "accept_rate": v[1] / max(v[1] + v[3], 1)} for c, v in S["cls"].items()},
            "spectrum": {c: {q: float(torch.tensor(v).quantile(q / 100)) for q in (10, 25, 50, 75, 90)}
                         for c, v in S["spec"].items() if v},
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    L = ["# Entropy and acceptance (the target's own uncertainty at every generated position)", ""]
    for n in names:
        L += [f"## {n}", "", "| target entropy (nats) | acceptance rate | positions |", "|---|---|---|"]
        for b in res[n]["buckets"]:
            hi = "inf" if b["hi"] > 1e8 else f"{b['hi']:g}"
            L.append(f"| {b['lo']:g} - {hi} | {b['accept']:.3f} | {b['n']} |")
        L += ["", "| token class | mean H accepted | mean H rejected | acceptance | n |", "|---|---|---|---|---|"]
        for c, v in sorted(res[n]["by_class"].items()):
            L.append(f"| {c} | {v['H_accepted']:.3f} | {v['H_rejected']:.3f} | {v['accept_rate']:.3f} | "
                     f"{v['n_accepted'] + v['n_rejected']} |")
        L.append("")
    a.out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("ENTROPY_ACCEPT_DONE", a.out.with_suffix(".md"))


if __name__ == "__main__":
    main()
