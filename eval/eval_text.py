"""Text-to-SVG decoding of every method in the main table, on one GPU.

One row, one mode, one node. The original model and every speculative decoder run here, in this
order, so a speed-up is a ratio of two times measured in the same job on the same prompts:

  - the original model (sightline/ar.py), decoded alongside with --time-ar, or taken from the
    cache of an earlier identical run when only tokens per round and forward passes are needed;
  - every requested design through sightline/tree_spec.py: the chain designs go down the same
    code path as the trees, so the window is the only thing that differs between them;
  - optionally the offline prediction for the same draft head (--cache), which checks the real
    decoder's per-round acceptance against sightline/design.py row by row.

Which decoder a design name selects is defined once, in eval/drafters.py; this file knows only
about rows, protocol and timing. Records follow eval/records.py's schema, one directory per
design, so eval/merge.py summarises every method the same way.

Losslessness is checked by --audit: target and head in fp32, greedy; every design must reproduce
the fp32 output of the original model token for token wherever the original model reproduces
itself. Appendix F.5 (Table 24) of the paper collects the results of those audits.

  python eval/eval_text.py --head <dir> --designs D0-chain,D1-top16-N127 --out <dir> \
      [--profile internsvg] [--sets bench] [--modes greedy,sample] [--shard i/n] [--time-ar]
  python eval/eval_text.py --head <dir> --designs ... --out <dir> --audit
  python eval/eval_text.py --head none --designs REST-n8-d16-N127 --rest-store <dir> --out <dir>
  python eval/eval_text.py ... --xrows <jsonl>     # rows of another benchmark, eval/rows/
  python eval/eval_text.py ... --fast              # faster logit processing, same outputs

--profile internsvg switches the target, its end-of-sequence id, the 4,000-token cap, the
released decoding settings and the benchmark rows (targets/profiles.py).

Merged, with no change to what runs, from eval_tree.py and the six wrappers that monkey-patched
it one baseline at a time (eval_tree_x, eval_tree_fast, eval_dart, eval_sps, eval_tree_rest,
eval_tree_rest_x); see _build/MERGES.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from eval.drafters import Designs                                          # noqa: E402
from eval.records import (EOS, MAX_NEW, MODELS, chain_params, load_ar_cache,  # noqa: E402
                          load_rows, rec, spec_records)
from sightline.ar import ar_generate                                       # noqa: E402
from sightline.tree_spec import tree_spec_generate                         # noqa: E402
from sightline.paths import RUNS

AR_CACHE = (RUNS / "b3-eval/ar-cache")
LOOP_KW = None
DESIGNS: Designs | None = None       # the run's designs; eval/drafters.py builds their drafters
REST_STORE: dict = {}                # REST's datastore, loaded once per process


def load_head(path, dtype):
    """Load the checkpoint, whichever family it belongs to (eval/drafters.py decides)."""
    return DESIGNS.load_head(path, dtype)


def use_profile(name: str):
    """Rebind the target-specific module globals (default: OmniSVG, as before)."""
    global EOS, MAX_NEW, AR_CACHE, LOOP_KW, chain_params, load_rows
    if name == "internsvg":
        from sightline.process import set_eos
        from targets import profiles as P
        EOS, MAX_NEW, AR_CACHE, LOOP_KW = P.EOS, P.MAX_NEW, P.AR_CACHE, P.LOOP_KW
        set_eos(P.EOS)                                    # the loop rule's stop action (default: OmniSVG's EOS)
        chain_params, load_rows = P.chain_params, P.load_rows
        return P.TARGET
    return None


def use_xrows(path: Path, profile: str) -> None:
    """Decode the rows of `path` instead of the profile's own benchmark.

    The prompts in such a file are already tokenised for their target (eval/rows/ builds them).
    No row of another benchmark has a cached run of the original model, so the cache is pointed
    at an empty directory: the original model is decoded in the same job (--time-ar) or skipped
    (--no-ar), never read from a cache that was measured elsewhere.
    """
    global load_rows, AR_CACHE
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

    def load_xrows(sets):
        return [dict(r) for r in rows if r["set"] in sets]

    empty = RUNS / "empty-ar-cache"
    empty.mkdir(parents=True, exist_ok=True)
    load_rows, AR_CACHE = load_xrows, empty
    if profile == "internsvg":                     # use_profile() rebinds from the profile module
        from targets import profiles as P
        P.load_rows, P.AR_CACHE = load_xrows, empty


def run_design(target, head, d, prompt, params, seed, max_new, block_size, coord_tol=0, copy_long=0):
    kw = {"store": REST_STORE.get("s")} if REST_STORE else {}
    drafter, block_size = DESIGNS.drafter(d.name, head, target, block_size, **kw)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = tree_spec_generate(target, drafter, prompt, max_new, [EOS], params, d,
                             generator=gen, block_size=block_size, coord_tol=coord_tol,
                             copy_long=copy_long)
    torch.cuda.synchronize()
    return res, time.perf_counter() - t0


def base_params(args, mode, subtype):
    """chain_params with the sample-mode overrides --temperature / --top-p (InternSVG: no official sampling)."""
    p = chain_params(mode, subtype)
    if mode == "sample" and args.temperature is not None:
        p.temperature = args.temperature
        if args.top_p is not None:
            p.top_p = args.top_p
    return p


def params_for(args, mode, subtype):
    """chain_params plus the loop escape (process.loop_ban) when --loop-ban > 0."""
    p = base_params(args, mode, subtype)
    p.loop_ban, p.loop_repeats = args.loop_ban, args.loop_repeats
    p.loop_min_period, p.loop_action = args.loop_min_period, args.loop_action
    p.loop_max_rb = args.loop_max_rb
    return p


def audit(args, designs):
    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel
    torch.backends.cuda.matmul.allow_tf32 = False
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    head = load_head(args.head, torch.float32)
    bench = load_rows(["bench"])
    rows = ([r for r in bench if r["subtype"] == "icon"][:args.n_audit]
            + [r for r in bench if r["subtype"] == "illustration"][:args.n_audit])
    if args.gidx:                                   # explicit rows (e.g. looping ones) instead
        keep = {int(x) for x in args.gidx.split(",")}
        rows = [r for r in load_rows(["bench", "extra"]) if r["gidx"] in keep]
    out = []
    for r in rows:
        params = params_for(args, "greedy", r["subtype"])
        ar1 = ar_generate(target, r["prompt_input_ids"], args.max_new, [EOS], params)
        ar2 = ar_generate(target, r["prompt_input_ids"], args.max_new, [EOS], params)
        rec_ = {"gidx": r["gidx"], "subtype": r["subtype"], "n_ar": len(ar1.tokens),
                "ar_self_identical": ar1.tokens == ar2.tokens, "designs": {}}
        for d in designs:
            res, dt = run_design(target, head, d, r["prompt_input_ids"], params, 0, args.max_new,
                                 int(getattr(head, "block_size", 0) or 0), copy_long=args.copy_long)
            rec_["designs"][d.name] = {"identical_to_ar": res.tokens == ar1.tokens,
                                       "tau": sum(res.acceptance) / max(1, len(res.acceptance)),
                                       "n": len(res.tokens)}
        out.append(rec_)
        print(json.dumps(rec_), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {d.name: sum(x["designs"][d.name]["identical_to_ar"] for x in out) for d in designs}
    (args.out / "audit_fp32_tree.json").write_text(json.dumps(
        {"rows": out, "identical": summary, "ar_self": sum(x["ar_self_identical"] for x in out),
         "n": len(out)}, indent=1))
    print("AUDIT", summary, "ar_self", sum(x["ar_self_identical"] for x in out), "/", len(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=None,
                    help='the draft head; omit (or "none") for the training-free rows, PLD and REST')
    ap.add_argument("--designs", required=True,
                    help="comma-separated design names; eval/drafters.py lists them")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="bench")
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--gidx", default=None, help="comma-separated row subset (smoke tests)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time-ar", action="store_true", help="run AR on this node (same-node wall clock)")
    ap.add_argument("--cache", type=Path, default=None, help="H2 cache of this head (sim check)")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--n-audit", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=None, help="default: the profile's cap (OmniSVG 1536, InternSVG 4000)")
    ap.add_argument("--target", type=Path, default=None, help="default: the profile's target")
    ap.add_argument("--profile", default="omnisvg", choices=["omnisvg", "internsvg"])
    ap.add_argument("--no-cudnn-sdp", action="store_true",
                    help="10-15: disable torch's cuDNN SDPA backend (it rebuilds a graph for every new shape, ~30 ms; "
                         "on an interactive node AR fell to 48 tok/s and SD rounds to 65 ms with it on)")
    ap.add_argument("--no-ar", action="store_true",
                    help="no AR baseline when the cache lacks the row (strict designs; nfe_ar = spec token count)")
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
    ap.add_argument("--temperature", type=float, default=None,
                    help="10-16: sample-mode temperature override (the InternSVG profile samples purely at 1.0); "
                         "the AR cache is not used with it")
    ap.add_argument("--top-p", type=float, default=None, help="sample-mode top-p override (with --temperature)")
    ap.add_argument("--rest-store", type=Path, default=None,
                    help="REST datastore (baselines/rest.py build); required by the REST-* designs")
    ap.add_argument("--xrows", type=Path, default=None,
                    help="decode the rows of this file instead of the profile's own benchmark "
                         "(eval/rows/ builds them); --sets selects their set names")
    ap.add_argument("--fast", action="store_true",
                    help="faster logit processing for every arm (sightline/fastchain.py); "
                         "outputs are unchanged")
    args = ap.parse_args()
    if args.no_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    if args.fast:
        from sightline import fastchain
        fastchain.enable()
    prof_target = use_profile(args.profile)
    if args.xrows is not None:
        use_xrows(args.xrows, args.profile)
    args.target = args.target or prof_target or MODELS / "OmniSVG1.1_8B-qwen2-text"
    args.max_new = args.max_new or MAX_NEW
    # --no-ar: no AR baseline (the run's own output stands in for it). also allowed with loop variants, which
    # are compared with the strict run of the same head (total forwards, paired quality), not with AR; never with --time-ar.
    assert not (args.no_ar and args.time_ar), "--no-ar cannot be combined with --time-ar"
    global DESIGNS
    DESIGNS = Designs(args.designs)
    designs = DESIGNS.designs
    assert designs, args.designs
    if DESIGNS.needs_store:
        assert args.rest_store is not None, "the REST-* designs need --rest-store"
        from baselines.rest import load_store
        REST_STORE["s"] = load_store(args.rest_store)
    assert args.head is not None or not DESIGNS.needs_head, \
        "every design but PLD and REST needs --head"
    if args.audit:
        return audit(args, designs)

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    rows = [r for r in load_rows(args.sets.split(",")) if r["gidx"] % shard_n == shard_i]
    if args.gidx:
        keep = {int(x) for x in args.gidx.split(",")}
        rows = [r for r in rows if r["gidx"] in keep]
    if args.limit:
        rows = rows[:args.limit]
    from transformers import AutoModelForCausalLM
    from dflash.model import DFlashDraftModel
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    head = load_head(args.head, torch.bfloat16)
    block = int(getattr(head, "block_size", 0) or 0)                # EAGLE-3: per design (depth + 1)
    ar_cache = {k: a for k, a in load_ar_cache(AR_CACHE).items() if a.get("max_new") == args.max_new}
    if args.temperature is not None:                # the cache holds the profile's own chain only
        ar_cache = {}
    print(f"head {args.head} block={block}; {len(rows)} rows; designs {[d.name for d in designs]}; "
          f"AR {'same node' if args.time_ar else 'from cache'}", flush=True)
    sub = f"shard-{shard_i}-of-{shard_n}"
    files = {}
    for d in designs:
        (args.out / d.name / sub).mkdir(parents=True, exist_ok=True)
        (args.out / d.name / sub / "protocol.json").write_text(json.dumps({
            "drafter": str(args.head), "design": d.name, "sets": args.sets, "modes": args.modes,
            "shard": args.shard, "seed_rule": f"42 + gidx + {args.seed_offset}", "block_size": block,
            "temperature": args.temperature, "top_p": args.top_p,
            "max_new_tokens": args.max_new, "ar": "same node" if args.time_ar else "cache",
            "verifier": "sightline/tree_spec.py", "coord_tol": args.coord_tol,
            "copy_long": args.copy_long, "loop_ban": args.loop_ban,
            "loop_repeats": args.loop_repeats, "loop_min_period": args.loop_min_period,
            "loop_action": args.loop_action, "loop_max_rb": args.loop_max_rb,
            "ar_note": ("AR = the ORIGINAL target (no loop ban): time / NFE baseline only"
                        if args.loop_ban else "")}, indent=1))
        files[d.name] = (args.out / d.name / sub / "results.jsonl").open("w")
    sim_check = []
    for mode in args.modes.split(","):
        for r in rows:
            prompt = r["prompt_input_ids"]
            params = params_for(args, mode, r["subtype"])
            seed = 42 + r["gidx"] + args.seed_offset
            cached = ar_cache.get((mode, r["gidx"]))
            if args.no_ar and cached is None:
                ar = None
                t_ar, ar_src = 0.0, "none"
            elif args.time_ar or cached is None:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                # the AR baseline is the ORIGINAL target even under --loop-ban
                ar = ar_generate(target, prompt, args.max_new, [EOS], base_params(args, mode, r["subtype"]),
                                 generator=gen)
                torch.cuda.synchronize()
                t_ar, ar_src = time.perf_counter() - t0, "inline"
            else:
                ar = SimpleNamespace(tokens=cached["ar_tokens"], nfe=cached["nfe_ar"],
                                     stop_reason=cached["stop_reason"])
                t_ar, ar_src = cached["t_ar"], "cache"
            cache_row = None
            if args.cache is not None:
                f = args.cache / mode / f"{r['gidx']:04d}.npz"
                if f.exists():
                    cache_row = sim.load_row(f)
            for d in designs:
                res, t_spec = run_design(target, head, d, prompt, params, seed, args.max_new, block,
                                         coord_tol=args.coord_tol, copy_long=args.copy_long)
                eos_flag = res.stop_reason == "eos"
                if ar is None or ar_src == "spec":                   # --no-ar: the strict output stands in
                    ar = SimpleNamespace(tokens=res.tokens, nfe=len(res.tokens), stop_reason=res.stop_reason)
                    ar_src = "spec"
                full, trunc, loop, onset = spec_records(res, eos_flag, len(prompt), cap=args.max_new,
                                                        loop_kw=LOOP_KW)
                if cache_row is not None and [int(x) for x in cache_row["tokens"]] == res.tokens:
                    simulated = sim.walk(cache_row, sim.replace(d, copy_long=args.copy_long))
                    sim_check.append({"design": d.name, "mode": mode, "gidx": r["gidx"],
                                      "same_rounds": simulated == res.acceptance,
                                      "tau_real": sum(res.acceptance) / len(res.acceptance),
                                      "tau_sim": sum(simulated) / len(simulated)})
                files[d.name].write(json.dumps({
                    "key": f"{mode}/{r['set']}/{r['subtype']}", "id": r["id"], "gidx": r["gidx"],
                    "set": r["set"], "subtype": r["subtype"], "mode": mode, "seed": seed,
                    "n_spec": len(res.tokens), "n_ar": len(ar.tokens),
                    "eos": eos_flag, "eos_ar": ar.stop_reason == "eos", "onset": onset,
                    "rec_full": rec(full), "rec_trunc": rec(trunc), "rec_loop": rec(loop),
                    "acceptance": res.acceptance, "t_ar": round(t_ar, 3), "t_spec": round(t_spec, 3),
                    "t_hook": 0.0, "t_spec_net": round(t_spec, 3), "nfe_spec": res.nfe,
                    "nfe_ar": ar.nfe, "ar_tokens": ar.tokens, "spec_tokens": res.tokens,
                    "first_div_vs_ar": (next((j for j in range(min(len(res.tokens), len(ar.tokens)))
                                              if res.tokens[j] != ar.tokens[j]), None)
                                        if mode == "greedy" and not args.loop_ban else None),
                    "ar_source": ar_src,
                    "n_copy": int(sum(b.get("n_copy", 0) for b in res.per_block)),
                    "max_accept": max(res.acceptance) if res.acceptance else 0,
                    "n_snap": int(sum(b.get("n_snap", 0) for b in res.per_block)),
                    "mean_tree_nodes": float(np.mean([b["n_nodes"] for b in res.per_block])
                                             ) if res.per_block else 0.0}) + "\n")
                files[d.name].flush()
                print(f"    {d.name}: tau {sum(res.acceptance) / len(res.acceptance):.2f} "
                      f"nfe {res.nfe} {t_spec:.1f}s", flush=True)
            print(f"[done] {mode} gidx {r['gidx']} ({r['subtype']}): {len(ar.tokens)} tok, "
                  f"AR {t_ar:.1f}s ({ar_src})", flush=True)
    for f in files.values():
        f.close()
    # per-shard summary files (eval_b3_merge reads summary_shard.json presence)
    for d in designs:
        (args.out / d.name / sub / "summary_shard.json").write_text("{}")
        (args.out / d.name / sub / "coord_positions.jsonl").write_text("")
    if sim_check:
        (args.out / f"sim_check-{sub}.json").write_text(json.dumps(sim_check))
        for d in designs:
            xs = [c for c in sim_check if c["design"] == d.name]
            if xs:
                print(f"SIMCHECK {d.name}: {sum(c['same_rounds'] for c in xs)}/{len(xs)} rows with "
                      f"identical rounds; tau real {np.mean([c['tau_real'] for c in xs]):.3f} "
                      f"vs sim {np.mean([c['tau_sim'] for c in xs]):.3f}", flush=True)
    print("EVAL_TREE_DONE", sub, flush=True)


if __name__ == "__main__":
    main()
