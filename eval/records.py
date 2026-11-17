"""A2 step 8: the B3 evaluation — first plain-DFlash tokens per round on
OmniSVG text2svg, per-caliber, with H2's coordinate-near-miss evidence.

Eval sets: MMSVGBench text2svg icon+illustration (150 each) + the held-out extra
pool (300 each, runs/2026-10-09-prompt-pool/eval-ids). Settings: greedy (official
chain incl. repetition penalty, then argmax) and official sampling (per subtype).

Arms per (row, mode): same-stack AR (sightline/ar.py — also SAVES the 5.15 AR
reference outputs) and spec (sightline/spec.py + the trained OmniSVG DFlash
head). Metrics: tokens per round 4 calibers x 2 aggregations (analysis/calibers.py), NFE,
wall-clock speedup (batch 1), per-position logs; coordinate positions
additionally log pixel distance of rejected drafts and chain-processed
probability mass within +-1/+-2/+-4 px of (a) the draft coord and (b) the
processed argmax (H2 premises). Greedy lossless spot-check: spec vs AR token
identity with first divergence (bf16 near-ties expected to be rare here since
both arms run the same block/step forward structure... they do NOT: AR is
per-token, spec is block forward — near-tie divergences are possible and are
reported, with fp32 audit as the definitive check, done separately).

AR cache: the AR arm does not depend on the head (same target, seed 42 + gidx,
same env), so it is run once with --ar-only (writes ar.jsonl, resumable) and every head
evaluation reads it back with --ar-cache; rows missing from the cache run AR inline.
The wall-clock speedup then compares AR and spec timed on different (identical H200)
nodes — tokens per round and NFE are unaffected.

Usage (GPU, the dflash environment):
  python eval/records.py --drafter <head-ckpt-dir> \
      [--sets bench,extra] [--modes greedy,sample] [--limit 20] [--out runs/<date>-b3-eval] \
      [--ar-cache <dir with ar.jsonl files>]
  python eval/records.py --ar-only [--shard i/n] --out <ar-cache>/shard-i-of-n
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import torch
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]

POOL = RUNS / "2026-10-09-prompt-pool"
OUT_DEFAULT = RUNS / "2026-10-09-b3-eval"

BOS, EOS, PAD, MAX_NEW = 196998, 196999, 151643, 1536
GRID = 200
from sightline.svgtok.types import COORD_START, COORD_END, coord_xy      # noqa: E402
from sightline.svgtok.loops import analyze as loop_analyze                # noqa: E402
from analysis.calibers import (CaliberStream, StreamRecord, caliber_table,  # noqa: E402
                               format_table, MIN_HEALTHY_ONSET)
from sightline.ar import ar_generate                              # noqa: E402
from sightline.spec import spec_generate, DFlashDrafter           # noqa: E402
from sightline.verify import ChainParams                          # noqa: E402
from sightline.process import apply_chain                         # noqa: E402

# official per-subclass params (refs/OmniSVG config.yaml)
PARAMS = {
    "icon":         {"temperature": 0.5, "top_p": 0.88, "top_k": 50,  "repetition_penalty": 1.05},
    "illustration": {"temperature": 0.6, "top_p": 0.90, "top_k": 60,  "repetition_penalty": 1.03},
}


def chain_params(mode: str, subtype: str) -> ChainParams:
    p = PARAMS[subtype]
    if mode == "greedy":
        return ChainParams(greedy=True, repetition_penalty=p["repetition_penalty"])
    return ChainParams(greedy=False, temperature=p["temperature"], top_p=p["top_p"],
                       top_k=p["top_k"], repetition_penalty=p["repetition_penalty"])


def is_coord(t: int) -> bool:
    return COORD_START <= t <= COORD_END


def neighborhood_ids(x: int, y: int, r: int) -> list[int]:
    ids = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < GRID and 0 <= yy < GRID:
                ids.append(COORD_START + yy * GRID + xx)
    return ids


class CoordCollector:
    """on_block hook: coordinate near-miss + local-mass evidence (H2 premises).

    Its own GPU time is accumulated in t_hook (synchronized at entry and exit) so
    the wall-clock speedup can exclude the analysis overhead."""

    def __init__(self, prompt_ids: list[int], params: ChainParams):
        self.n_prompt = len(prompt_ids)
        self.mirror = list(prompt_ids)        # committed mirror for penalty history
        self.params = params
        self.rows = []                        # per-analyzed-position records
        self.t_hook = 0.0

    def _mass(self, logits, history, center_xy):
        processed = apply_chain(logits, history, penalty=self.params.repetition_penalty,
                                temperature_=self.params.temperature,
                                top_k_=self.params.top_k, top_p_=self.params.top_p,
                                greedy=False,         # mass needs the distribution
                                loop_ban_=getattr(self.params, "loop_ban", 0),
                                loop_repeats_=getattr(self.params, "loop_repeats", 2),
                                loop_min_period_=getattr(self.params, "loop_min_period", 4),
                                loop_action_=getattr(self.params, "loop_action", "ban"))
        probs = torch.softmax(processed.float(), dim=-1)
        out = {}
        for r in (1, 2, 4):
            out[f"mass_pm{r}"] = float(probs[neighborhood_ids(*center_xy, r)].sum())
        return int(torch.argmax(processed)), out

    def on_block(self, info, raw_logits, res):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._on_block(info, raw_logits, res)
        torch.cuda.synchronize()
        self.t_hook += time.perf_counter() - t0

    def _on_block(self, info, raw_logits, res):
        drafts = info["drafts"]
        for i, pp in enumerate(res.per_position):
            d, y = pp["draft"], pp["target"]
            if d is None:                     # bonus entry after a fully accepted block
                continue
            if not (is_coord(d) or is_coord(y)):
                continue
            # penalty history at this position: mirror + accepted drafts before i
            hist = list(self.mirror)
            for i2 in range(i):
                if res.per_position[i2]["accepted"]:
                    hist.append(drafts[i2])
            # position relative to the generated stream (same coordinates as the loop onset)
            row = {"pos": info["start"] + 1 + i - self.n_prompt, "draft": d, "target": y,
                   "accepted": pp["accepted"]}
            if not pp["accepted"] and is_coord(d) and is_coord(y):
                dx, dy = coord_xy(d)
                tx, ty = coord_xy(y)
                row["cheb"] = max(abs(dx - tx), abs(dy - ty))
                row["euc"] = round(((dx - tx) ** 2 + (dy - ty) ** 2) ** 0.5, 2)
            if is_coord(d):
                argmax, mass = self._mass(raw_logits[i], hist, coord_xy(d))
                row["mass_draft"] = mass
                row["argmax"] = int(argmax)
                if is_coord(int(argmax)):
                    _, mass2 = self._mass(raw_logits[i], hist, coord_xy(int(argmax)))
                    row["mass_argmax"] = mass2
            self.rows.append(row)
        self.mirror += res.emitted


def spec_records(res, eos_flag: bool, n_prompt: int, cap: int = MAX_NEW, loop_kw: dict | None = None):
    """(full, trunc, loop) StreamRecords + onset for one spec run.

    Positions are relative to the generated stream (res.tokens; tokens[0] came from
    the prefill), the coordinates of the loop onset. per_block[b]["start"] is a
    GLOBAL position (prompt included), hence the n_prompt shift. A block counts
    toward `trunc` if the first token it commits lies before the onset (emitted
    clipped at the onset), toward `loop` otherwise — the thermometer's rule
    (step position >= onset), including streams that loop from the start.
    Streams with onset <= MIN_HEALTHY_ONSET get no trunc record (excluded)."""
    onset = None
    info = loop_analyze(res.tokens, eos_flag, cap=cap, **(loop_kw or {}))   # InternSVG profile settings
    if info.label in ("strict-loop", "approx-loop"):
        onset = info.onset
    full = StreamRecord(steps=len(res.acceptance), emitted=sum(res.acceptance))
    if onset is None:
        return full, full, None, None
    steps_t = emitted_t = steps_l = emitted_l = 0
    for acc, blk in zip(res.acceptance, res.per_block):
        s = blk["start"] + 1 - n_prompt          # first token this block commits
        if s < onset:
            steps_t += 1
            emitted_t += max(0, min(s + acc, onset) - s)
        else:
            steps_l += 1
            emitted_l += acc
    trunc = (StreamRecord(steps=steps_t, emitted=emitted_t)
             if onset > MIN_HEALTHY_ONSET else None)
    loop = StreamRecord(steps=steps_l, emitted=emitted_l) if steps_l else None
    return full, trunc, loop, onset


def rec(r: StreamRecord | None):
    return None if r is None else [r.steps, r.emitted]


def load_rows(sets: list[str]) -> list[dict]:
    rows = []
    if "bench" in sets:
        rows += [json.loads(l) for l in
                 (POOL / "bench-ids-text2svg.jsonl").read_text().split("\n") if l.strip()]
        for r in rows:
            r["set"] = "bench"
    if "extra" in sets:
        for cls in ("icon", "illustration"):
            extra = []
            for l in (POOL / f"eval-ids/{cls}.jsonl").read_text().split("\n"):
                if l.strip():
                    r = json.loads(l)
                    extra.append({"id": r["id"], "text": r["text"], "subtype": cls,
                                  "prompt_input_ids": r["prompt_input_ids"], "set": "extra"})
            rows += extra
    # stable global index: seeds and shard membership do not depend on how the run is split
    for g, r in enumerate(rows):
        r["gidx"] = g
    return rows


def load_ar_cache(cache_dir: Path | None) -> dict:
    """(mode, gidx) -> AR record from every ar.jsonl under cache_dir."""
    cache = {}
    if cache_dir is None:
        return cache
    for f in sorted(cache_dir.rglob("ar.jsonl")):
        for line in f.read_text().split("\n"):
            if line.strip():
                a = json.loads(line)
                cache[(a["mode"], a["gidx"])] = a
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", type=Path, default=None,
                    help="trained OmniSVG DFlash head dir (not used with --ar-only)")
    ap.add_argument("--ar-only", action="store_true",
                    help="run only the head-independent AR arm, append to <out>/ar.jsonl (resumable)")
    ap.add_argument("--ar-cache", type=Path, default=None,
                    help="dir holding ar.jsonl files from --ar-only runs")
    ap.add_argument("--sets", default="bench,extra")
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: run the rows with gidx %% n == i")
    ap.add_argument("--max-new", type=int, default=MAX_NEW, help="protocol: 1536 (lower only for tests)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--target", type=Path, default=MODELS / "OmniSVG1.1_8B-qwen2-text")
    ap.add_argument("--block-size", type=int, default=None,
                    help="default: the head's own block_size (B32 heads, 10-11)")
    args = ap.parse_args()
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    if not args.ar_only and args.drafter is None:
        ap.error("--drafter is required unless --ar-only")

    from transformers import AutoModelForCausalLM
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    rows = [r for r in load_rows(args.sets.split(",")) if r["gidx"] % shard_n == shard_i]
    if args.limit:
        rows = rows[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    if args.ar_only:
        ar_file = args.out / "ar.jsonl"
        done = {(a["mode"], a["gidx"]) for a in load_ar_cache(args.out).values()}
        print(f"ar-only: {len(rows)} rows x {args.modes}; {len(done)} already in {ar_file}")
        with ar_file.open("a") as af:
            for mode in args.modes.split(","):
                for r in rows:
                    if (mode, r["gidx"]) in done:
                        continue
                    seed = 42 + r["gidx"]
                    gen = torch.Generator(device="cuda").manual_seed(seed)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    ar = ar_generate(target, r["prompt_input_ids"], args.max_new, [EOS],
                                     chain_params(mode, r["subtype"]), generator=gen)
                    torch.cuda.synchronize()
                    af.write(json.dumps({
                        "mode": mode, "gidx": r["gidx"], "id": r["id"], "set": r["set"],
                        "subtype": r["subtype"], "seed": seed, "max_new": args.max_new,
                        "ar_tokens": ar.tokens, "stop_reason": ar.stop_reason,
                        "nfe_ar": ar.nfe, "t_ar": round(time.perf_counter() - t0, 3)}) + "\n")
                    af.flush()
        print("AR_ONLY_DONE", ar_file)
        return

    from dflash.model import DFlashDraftModel
    head = DFlashDraftModel.from_pretrained(args.drafter, torch_dtype=torch.bfloat16).cuda().eval()
    if args.block_size is None:
        args.block_size = int(head.block_size)
    print(f"head loaded: block={head.block_size} mask={head.mask_token_id} "
          f"layers={list(head.target_layer_ids)}")
    ar_cache = load_ar_cache(args.ar_cache)
    ar_cache = {k: a for k, a in ar_cache.items() if a.get("max_new") == args.max_new}
    print(f"AR cache: {len(ar_cache)} (mode, row) entries from {args.ar_cache}")
    (args.out / "protocol.json").write_text(json.dumps({
        "drafter": str(args.drafter), "target": str(args.target),
        "sets": args.sets, "modes": args.modes, "n_rows": len(rows),
        "shard": args.shard, "seed_rule": "42 + gidx (AR and spec arms alike)",
        "ar_cache": str(args.ar_cache) if args.ar_cache else None,
        "block_size": args.block_size, "max_new_tokens": args.max_new,
        "params": PARAMS, "env": "the dflash environment torch2.13/transformers5.15",
        "speedup": "per-token: (t_ar/n_ar) / (t_spec_net/n_spec); t_spec_net excludes the H2 collector"},
        indent=2))

    summary = {}
    results_file = args.out / "results.jsonl"
    coord_file = args.out / "coord_positions.jsonl"
    with results_file.open("w") as rf, coord_file.open("w") as cf:
        for mode in args.modes.split(","):
            for set_name in [s for s in ("bench", "extra") if s in args.sets]:
                for subtype in ("icon", "illustration"):
                    key = f"{mode}/{set_name}/{subtype}"
                    group = [r for r in rows if r["set"] == set_name and r["subtype"] == subtype]
                    bundles, speeds = [], []
                    for r in group:
                        prompt = r["prompt_input_ids"]
                        params = chain_params(mode, subtype)
                        seed = 42 + r["gidx"]

                        cached = ar_cache.get((mode, r["gidx"]))
                        if cached is not None:
                            ar = SimpleNamespace(tokens=cached["ar_tokens"], nfe=cached["nfe_ar"],
                                                 stop_reason=cached["stop_reason"])
                            t_ar = cached["t_ar"]
                        else:
                            gen = torch.Generator(device="cuda").manual_seed(seed)   # as harness check4
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            ar = ar_generate(target, prompt, args.max_new, [EOS], params, generator=gen)
                            torch.cuda.synchronize()
                            t_ar = time.perf_counter() - t0

                        drafter = DFlashDrafter(head, target, args.block_size)
                        collector = CoordCollector(prompt, params)
                        gen2 = torch.Generator(device="cuda").manual_seed(seed)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        spec = spec_generate(target, drafter, prompt, args.max_new, [EOS],
                                             params, generator=gen2,
                                             block_size=args.block_size,
                                             on_block=collector.on_block)
                        torch.cuda.synchronize()
                        t_spec = time.perf_counter() - t0
                        t_spec_net = t_spec - collector.t_hook

                        eos_flag = spec.stop_reason == "eos"
                        full, trunc, loop, onset = spec_records(spec, eos_flag, len(prompt),
                                                                cap=args.max_new)
                        bundles.append(CaliberStream(full=full, eos=eos_flag, onset=onset,
                                                     trunc=trunc, loop=loop))
                        speeds.append((t_ar / max(len(ar.tokens), 1)) /
                                      max(t_spec_net / max(len(spec.tokens), 1), 1e-9))
                        div = next((j for j in range(min(len(spec.tokens), len(ar.tokens)))
                                    if spec.tokens[j] != ar.tokens[j]), None) \
                            if mode == "greedy" else None
                        for c in collector.rows:
                            cf.write(json.dumps({"key": key, "id": r["id"], "gidx": r["gidx"],
                                                 "onset": onset, **c}) + "\n")

                        rf.write(json.dumps({
                            "key": key, "id": r["id"], "gidx": r["gidx"], "set": set_name,
                            "subtype": subtype, "mode": mode, "seed": seed,
                            "n_spec": len(spec.tokens), "n_ar": len(ar.tokens),
                            "eos": eos_flag, "eos_ar": ar.stop_reason == "eos",
                            "onset": onset, "rec_full": rec(full), "rec_trunc": rec(trunc),
                            "rec_loop": rec(loop), "acceptance": spec.acceptance,
                            "t_ar": round(t_ar, 3), "t_spec": round(t_spec, 3),
                            "t_hook": round(collector.t_hook, 3),
                            "t_spec_net": round(t_spec_net, 3),
                            "nfe_spec": spec.nfe, "nfe_ar": ar.nfe,
                            "ar_tokens": ar.tokens, "spec_tokens": spec.tokens,
                            "first_div_vs_ar": div,
                            "ar_source": "cache" if cached is not None else "inline",
                        }, ensure_ascii=False) + "\n")
                        rf.flush()
                        cf.flush()

                    if not group:
                        continue
                    table = caliber_table(bundles)
                    summary[key] = {"n": len(group), "calibers": table["calibers"],
                                    "n_trunc_excluded": table["n_trunc_excluded"],
                                    "speedup_median": st.median(speeds)}
                    print(f"== {key} (n={len(group)}, shard {args.shard})")
                    print(format_table(table))
                    print(f"  per-token speedup median={st.median(speeds):.2f}x (collector excluded)")

    (args.out / "summary_shard.json").write_text(json.dumps(summary, indent=2))
    print("wrote", results_file, "and", coord_file,
          "(per-shard summary only; merge all shards with eval/merge.py)")


if __name__ == "__main__":
    main()
