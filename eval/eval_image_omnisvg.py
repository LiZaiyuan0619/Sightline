""" copy of eval_i2svg_plus.py (not edited: the width / seed jobs of run it)
with the sampling overrides the text evaluator already has (eval_tree.py --temperature / --top-p / --top-k), so
the image-to-SVG block can be swept over temperature like the text blocks. Defaults unchanged = the official
image-to-svg setting (T 0.3, top-p 0.90, top-k 50, repetition penalty 1.05).
The eval_i2svg_plus.py docstring follows.

copy of eval_i2svg_dart.py (itself a copy of eval_i2svg.py; neither is edited: the DARTree
jobs of runs/2026-10-24/jobs_1024b.sh use the former) + the EAGLE-3 baseline for table 1's image-to-SVG block:
  - --head may also be an EAGLE-3 head (baselines/train_eagle3_i2.py; loaded with baselines/eagle3_spec.load_eagle3);
  - designs "E3-k<topk>-d<depth>-N<budget>" build Eagle3Drafter (block size = depth + 1), as eval_tree.py.
The eval_i2svg_dart.py docstring follows.

copy of eval_i2svg.py (not edited: queued / running jobs import it) with the DARTree baseline, for
table 1's image-to-SVG block. Differences from eval_i2svg.py, nothing else:
  - --head may be a Domino head (train_domino.py; loaded with baselines/dartree_spec.load_domino);
  - designs "DART-W<width>-E<cand>-N<budget>[-b<depth bonus>]" (as baselines/eval_dart.py) build DominoTreeDrafter with
    block size = head.block_size + 1 (shift_label: block_size draft depths + the root), as eval_dart.py does;
  - --xrows <jsonl> replaces the row file (schema of $SIGHTLINE_RUNS/h3a-data/eval_rows.jsonl), as eval_i2svg_x.py.
The original docstring follows.

H3a S1/S2: image-to-SVG evaluation in the strict harness (the glance environment, transformers 5.15).

Rows: $SIGHTLINE_RUNS/h3a-data/eval_rows.jsonl (900; bench 0-299, held-out 300-899; build_sets.py).
Modes: greedy (official chain incl. repetition penalty 1.05, then argmax) and the official
image-to-svg sampling (T 0.3, top-p 0.90, top-k 50, rp 1.05); seed 42 + gidx, as B3.

  --ar-only          head-independent AR arm -> <out>/ar.jsonl (resumable), the I2SVG AR cache
  --head... --designs D0-chain,...   speculative arm per design through
                     sightline.tree_spec.tree_spec_generate (VLM prefill + M-RoPE positions);
                     records in the B3 schema under <out>/<design>/shard-i-of-n/, so
                     eval_b3_merge.py summarises them; AR from --ar-cache, else inline
  --hf-check N       self-check 1: our VLM AR loop vs HF generate (greedy) on N bench rows
  --audit            fp32: every design vs fp32 AR greedy on bench rows (+ AR self-control)

Usage (GPU, the glance environment; refs/SpecForge not needed here):
  python eval_i2svg.py --ar-only --shard i/n --out <ar dir>
  python eval_i2svg.py --head <dir> --designs D0-chain --out <dir> \
      [--ar-cache <ar dir>] [--sets bench,extra] [--modes greedy,sample] [--shard i/n]

Image-to-SVG decoding of every method in the main table's OmniSVG-8B block. One file per target
family, because the targets differ in their processor, end-of-sequence id, token cap and loop
window; which decoder a design name selects is shared, in eval/drafters.py.

Merged, with no change to what runs, from src/h3a/eval_i2svg{,_dart,_plus,_t,_x}.py, src/rest/eval_i2svg_rest.py, src/vispec/eval_i2svg_vs.py; see _build/MERGES.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from sightline import design as sim                                               # noqa: E402
from targets import omnisvg as vlm
from eval_b3 import rec, spec_records                               # noqa: E402
from eval.drafters import Designs
from sightline.tree_spec import tree_spec_generate       # noqa: E402
from sightline.verify import ChainParams                              # noqa: E402
from sightline.paths import RUNS

EOS, MAX_NEW = 196999, 1536
REST_STORE: dict = {}      # the REST datastore, loaded once per process
ROWS = (RUNS / "h3a-data/eval_rows.jsonl")
I2SVG = {"temperature": 0.3, "top_p": 0.90, "top_k": 50, "repetition_penalty": 1.05}


def chain_params(mode: str) -> ChainParams:
    if mode == "greedy":
        return ChainParams(greedy=True, repetition_penalty=I2SVG["repetition_penalty"])
    return ChainParams(greedy=False, temperature=I2SVG["temperature"], top_p=I2SVG["top_p"],
                       top_k=I2SVG["top_k"], repetition_penalty=I2SVG["repetition_penalty"])


def load_rows(sets, shard):
    i, n = (int(x) for x in shard.split("/"))
    rows = [json.loads(l) for l in ROWS.read_text().split("\n") if l.strip()]
    return [r for r in rows if r["set"] in sets and r["gidx"] % n == i]


def load_ar_cache(d: Path | None) -> dict:
    out = {}
    if d is None:
        return out
    for f in sorted(d.rglob("ar.jsonl")):
        for line in f.read_text().split("\n"):
            if line.strip():
                a = json.loads(line)
                out[(a["mode"], a["gidx"])] = a
    return out


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--head", type=Path, default=None)
    ap.add_argument("--designs", default="D0-chain")
    ap.add_argument("--ar-only", action="store_true")
    ap.add_argument("--ar-cache", type=Path, default=None)
    ap.add_argument("--ar-save", type=Path, default=None,
                    help="append AR runs computed inline here (an ar.jsonl dir) for later heads")
    ap.add_argument("--hf-check", type=int, default=0)
    ap.add_argument("--rest-store", type=Path, default=None,
                    help="REST datastore (baselines/rest.py build); the REST-* designs need it")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--n-audit", type=int, default=5)
    ap.add_argument("--sets", default="bench,extra")
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gidx", default=None, help="comma-separated row subset (smoke tests)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--coord-tol", type=int, default=0,
                    help="geometric tolerance in grid units (sightline/verify.py); 0 = strict")
    ap.add_argument("--copy-long", type=int, default=0,
                    help="10-13: long period-copy branch of up to this many tokens (lossless); 0 = off")
    ap.add_argument("--loop-ban", type=int, default=0,
                    help="10-13 loop escape (changes the target): max period, 0 = off")
    ap.add_argument("--loop-repeats", type=int, default=2)
    ap.add_argument("--loop-min-period", type=int, default=4)
    ap.add_argument("--loop-action", default="ban", choices=["ban", "eos", "rollback", "esc_stop"])
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="10-14 multi-sample runs: seed = 42 + gidx + offset (sample mode)")
    ap.add_argument("--loop-max-rb", type=int, default=1,
                    help="rollback / esc_stop: rollbacks or escapes before the next firing stops (10-14)")
    ap.add_argument("--no-cudnn-sdp", action="store_true",
                    help="10-15: disable torch's cuDNN SDPA backend (it rebuilds a graph per new shape, ~30 ms); use for timing")
    ap.add_argument("--fp32", action="store_true",
                    help="10-15: run the whole evaluation in fp32 (fp32 reference samples), not only the audit")
    ap.add_argument("--xrows", type=Path, default=None, help="10-24: row file instead of ROWS")
    ap.add_argument("--temperature", type=float, default=None, help="sample-mode temperature override")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    for k in ("temperature", "top_p", "top_k"):          # before any chain_params call
        v = getattr(args, k)
        if v is not None:
            I2SVG[k] = v
    if args.xrows is not None:
        global ROWS
        ROWS = args.xrows

    def params_for(mode):
        """chain_params plus the loop escape (process.loop_ban) when --loop-ban > 0."""
        p = chain_params(mode)
        p.loop_ban, p.loop_repeats = args.loop_ban, args.loop_repeats
        p.loop_min_period, p.loop_action = args.loop_min_period, args.loop_action
        p.loop_max_rb = args.loop_max_rb
        return p
    args.out.mkdir(parents=True, exist_ok=True)
    if args.no_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    fp32 = args.audit or args.fp32
    if fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float32 if fp32 else torch.bfloat16
    official = vlm.load_official()
    model, note = vlm.load_target(dtype)
    print(f"[load] {note}", flush=True)
    tmp = args.out / f".proc-{args.shard.replace('/', 'of')}.png"
    rows = load_rows(args.sets.split(","), args.shard)
    if args.gidx:
        keep = {int(x) for x in args.gidx.split(",")}
        rows = [r for r in rows if r["gidx"] in keep]
    if args.limit:
        rows = rows[:args.limit]

    if args.hf_check:
        rows = [r for r in load_rows(["bench"], "0/1")][:args.hf_check]
        res = []
        for r in rows:
            inp = vlm.build_inputs(official, model, r["image_path"], tmp, dtype)
            ours = vlm.vlm_ar_generate(model, inp, args.max_new, [EOS], chain_params("greedy"))
            pre = official.prepare_inputs("image-to-svg", str(tmp))
            dev = next(model.parameters()).device
            pre = {k: (v.to(dev, dtype=dtype) if k == "pixel_values" else v.to(dev)) for k, v in pre.items()}
            if "mm_token_type_ids" not in pre:
                pre["mm_token_type_ids"] = (pre["input_ids"] == vlm.IMAGE_TOKEN_ID).long()
            model.model.rope_deltas = None
            with torch.inference_mode():
                hf = model.generate(**pre, max_new_tokens=args.max_new, do_sample=False,
                                    repetition_penalty=1.05, eos_token_id=EOS, pad_token_id=151643)
            hf_tok = hf[0, pre["input_ids"].shape[1]:].tolist()
            hf_tok = hf_tok[:hf_tok.index(EOS) + 1] if EOS in hf_tok else hf_tok
            div = next((j for j in range(min(len(ours.tokens), len(hf_tok)))
                        if ours.tokens[j] != hf_tok[j]), None)
            res.append({"gidx": r["gidx"], "n_ours": len(ours.tokens), "n_hf": len(hf_tok),
                        "identical": ours.tokens == hf_tok, "first_div": div, "delta": inp.delta})
            print(json.dumps(res[-1]), flush=True)
        (args.out / "hf_check.json").write_text(json.dumps(res, indent=1))
        print("HF_CHECK", sum(x["identical"] for x in res), "/", len(res))
        return

    if args.ar_only:
        ar_file = args.out / "ar.jsonl"
        done = set(load_ar_cache(args.out))
        with ar_file.open("a") as af:
            for mode in args.modes.split(","):
                for r in rows:
                    if (mode, r["gidx"]) in done:
                        continue
                    inp = vlm.build_inputs(official, model, r["image_path"], tmp, dtype)
                    gen = torch.Generator(device="cuda").manual_seed(42 + r["gidx"])
                    ar, t = timed(lambda: vlm.vlm_ar_generate(model, inp, args.max_new, [EOS],
                                                              chain_params(mode), gen))
                    af.write(json.dumps({"mode": mode, "gidx": r["gidx"], "id": r["id"],
                                         "set": r["set"], "subtype": r["subtype"],
                                         "seed": 42 + r["gidx"], "max_new": args.max_new,
                                         "prompt_len": len(inp.prompt_ids), "rope_delta": inp.delta,
                                         "ar_tokens": ar.tokens, "stop_reason": ar.stop_reason,
                                         "nfe_ar": ar.nfe, "t_ar": round(t, 3)}) + "\n")
                    af.flush()
                    print(f"[ar] {mode} gidx {r['gidx']} {len(ar.tokens)} tok {t:.1f}s", flush=True)
        print("AR_ONLY_DONE", ar_file)
        return

    designs_spec = Designs(args.designs)
    designs = designs_spec.designs
    if designs_spec.needs_store:
        assert args.rest_store is not None, "the REST-* designs need --rest-store"
        from baselines.rest import load_store
        REST_STORE["s"] = load_store(args.rest_store)
    head = designs_spec.load_head(args.head, dtype)
    block = int(getattr(head, "block_size", 0) or 0)    # EAGLE-3 / ViSpec: depth + 1 per design

    def drafter_for(d):
        """(drafter, block size verified per round); eval/drafters.py holds the table."""
        kw = {"store": REST_STORE.get("s")} if REST_STORE else {}
        return designs_spec.drafter(d.name, head, model, block, **kw)

    if args.audit:
        bench = load_rows(["bench"], "0/1")
        sel = ([r for r in bench if r["subtype"] == "icon"][:args.n_audit]
               + [r for r in bench if r["subtype"] == "illustration"][:args.n_audit])
        if args.gidx:                               # explicit rows (e.g. looping ones) instead
            sel = list(rows)
        out = []
        for r in sel:
            inp = vlm.build_inputs(official, model, r["image_path"], tmp, dtype)
            a1 = vlm.vlm_ar_generate(model, inp, args.max_new, [EOS], params_for("greedy"))
            a2 = vlm.vlm_ar_generate(model, inp, args.max_new, [EOS], params_for("greedy"))
            x = {"gidx": r["gidx"], "n_ar": len(a1.tokens), "ar_self": a1.tokens == a2.tokens,
                 "designs": {}}
            for d in designs:
                dr, bsz = drafter_for(d)
                res = tree_spec_generate(model, dr, inp.prompt_ids,
                                         args.max_new, [EOS], params_for("greedy"), d,
                                         block_size=bsz, positions=inp,
                                         prefill_kwargs=inp.prefill_kwargs(),
                                         copy_long=args.copy_long)
                x["designs"][d.name] = {"identical": res.tokens == a1.tokens,
                                        "tau": sum(res.acceptance) / max(1, len(res.acceptance))}
            out.append(x)
            print(json.dumps(x), flush=True)
        (args.out / "audit_fp32_i2svg.json").write_text(json.dumps(out, indent=1))
        print("AUDIT", {d.name: sum(x["designs"][d.name]["identical"] for x in out) for d in designs},
              "ar_self", sum(x["ar_self"] for x in out), "/", len(out))
        return

    ar_cache = {k: a for k, a in load_ar_cache(args.ar_cache).items() if a.get("max_new") == args.max_new}
    sub = f"shard-{args.shard.replace('/', '-of-')}"
    files = {}
    for d in designs:
        (args.out / d.name / sub).mkdir(parents=True, exist_ok=True)
        (args.out / d.name / sub / "protocol.json").write_text(json.dumps({
            "drafter": str(args.head), "design": d.name, "task": "image2svg", "sets": args.sets,
            "modes": args.modes, "shard": args.shard, "seed_rule": f"42 + gidx + {args.seed_offset}", "block_size": block,
            "max_new_tokens": args.max_new, "params": I2SVG, "env": "the glance environment 5.15",
            "coord_tol": args.coord_tol, "copy_long": args.copy_long, "loop_ban": args.loop_ban,
            "loop_repeats": args.loop_repeats, "loop_min_period": args.loop_min_period,
            "loop_action": args.loop_action, "loop_max_rb": args.loop_max_rb,
            "ar_note": ("AR = the ORIGINAL target (no loop ban): time / NFE baseline only"
                        if args.loop_ban else "")}, indent=1))
        files[d.name] = (args.out / d.name / sub / "results.jsonl").open("w")
    for mode in args.modes.split(","):
        params = params_for(mode)
        for r in rows:
            inp = vlm.build_inputs(official, model, r["image_path"], tmp, dtype)
            seed = 42 + r["gidx"] + args.seed_offset
            cached = ar_cache.get((mode, r["gidx"]))
            if cached is not None:
                ar = SimpleNamespace(tokens=cached["ar_tokens"], nfe=cached["nfe_ar"],
                                     stop_reason=cached["stop_reason"])
                t_ar, src = cached["t_ar"], "cache"
            else:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                # the AR baseline is the ORIGINAL target even under --loop-ban
                ar, t_ar = timed(lambda: vlm.vlm_ar_generate(model, inp, args.max_new, [EOS],
                                                             chain_params(mode), gen))
                src = "inline"
                if args.ar_save is not None:
                    args.ar_save.mkdir(parents=True, exist_ok=True)
                    with (args.ar_save / "ar.jsonl").open("a") as af:
                        af.write(json.dumps({"mode": mode, "gidx": r["gidx"], "id": r["id"],
                                             "set": r["set"], "subtype": r["subtype"], "seed": seed,
                                             "max_new": args.max_new, "prompt_len": len(inp.prompt_ids),
                                             "rope_delta": inp.delta, "ar_tokens": ar.tokens,
                                             "stop_reason": ar.stop_reason, "nfe_ar": ar.nfe,
                                             "t_ar": round(t_ar, 3)}) + "\n")
            for d in designs:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                dr, bsz = drafter_for(d)
                res, t_spec = timed(lambda: tree_spec_generate(
                    model, dr, inp.prompt_ids, args.max_new, [EOS],
                    params, d, generator=gen, block_size=bsz, positions=inp,
                    prefill_kwargs=inp.prefill_kwargs(), coord_tol=args.coord_tol,
                    copy_long=args.copy_long))
                eos_flag = res.stop_reason == "eos"
                full, trunc, loop, onset = spec_records(res, eos_flag, len(inp.prompt_ids),
                                                        cap=args.max_new)
                files[d.name].write(json.dumps({
                    "key": f"{mode}/{r['set']}/{r['subtype']}", "id": r["id"], "gidx": r["gidx"],
                    "set": r["set"], "subtype": r["subtype"], "mode": mode, "seed": seed,
                    "n_spec": len(res.tokens), "n_ar": len(ar.tokens), "eos": eos_flag,
                    "eos_ar": ar.stop_reason == "eos", "onset": onset, "rec_full": rec(full),
                    "rec_trunc": rec(trunc), "rec_loop": rec(loop), "acceptance": res.acceptance,
                    "t_ar": round(t_ar, 3), "t_spec": round(t_spec, 3), "t_hook": 0.0,
                    "t_spec_net": round(t_spec, 3), "nfe_spec": res.nfe, "nfe_ar": ar.nfe,
                    "ar_tokens": ar.tokens, "spec_tokens": res.tokens,
                    "first_div_vs_ar": (next((j for j in range(min(len(res.tokens), len(ar.tokens)))
                                              if res.tokens[j] != ar.tokens[j]), None)
                                        if mode == "greedy" and not args.loop_ban else None),
                    "ar_source": src, "prompt_len": len(inp.prompt_ids),
                    "n_copy": int(sum(b.get("n_copy", 0) for b in res.per_block)),
                    "max_accept": max(res.acceptance) if res.acceptance else 0,
                    "n_snap": int(sum(b.get("n_snap", 0) for b in res.per_block))}) + "\n")
                files[d.name].flush()
                print(f"[spec] {mode} gidx {r['gidx']} {d.name}: tau "
                      f"{sum(res.acceptance) / len(res.acceptance):.2f} {len(res.tokens)} tok "
                      f"{t_spec:.1f}s (AR {t_ar:.1f}s {src})", flush=True)
    for d in designs:
        files[d.name].close()
        (args.out / d.name / sub / "summary_shard.json").write_text("{}")
        (args.out / d.name / sub / "coord_positions.jsonl").write_text("")
    print("EVAL_I2SVG_DONE", sub, flush=True)


if __name__ == "__main__":
    main()
