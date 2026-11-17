""" copy of sightline/bon_decode.py (not edited: its jobs may still run) with one more
task, --task i2s = InternSVG-8B image-to-SVG (table 1's 5th block), so the quality tier (component 3) covers four
settings instead of three. i2s is the intersection of the two tasks that already exist:
  - input / target like --task i2 but through vlm_i2s.py (InternVL, one 448 tile, 1-D positions);
  - output decoding like --task isvg (token ids -> SVG source, closed with close_svg) because InternSVG writes raw SVG;
  - selection by DINO against the input image, as --task i2 (sightline/score_server.py already takes {"svg", "gt"} items).
Rows come from --xrows ($SIGHTLINE_RUNS/tableA/xbench/internsvg-i2-own.jsonl, build_i2s_rows.py); sampling is the
block's sampling column (T 0.6, top-p 0.9); the head and tree are the block's own (--head, D1-top16-N127).
Lives in sightline/ (not ): a script in src/internsvg shadows the stdlib `profile` module that transformers
imports lazily (the same trap eval_i2s.py guards against).
The bon_decode.py docstring follows.

best-of-N text2svg decoding on one GPU, end to end and timed (the real counterpart of the exact accounting in
runs/2026-10-15-harvest/analysis/bon_sd.py).

Per prompt (OmniSVG text2svg rows of eval_b3.load_rows), on the same node, in this order:
  ar1    plain AR, one sample (seed offset 0; sightline.ar), the official model (no loop stop)
  sd1    strict SD, one sample (top-16@127 tree; sightline.tree_spec), no loop stop: the strongest single-sample decoder
  ar8    batched AR x 8 (sightline.batch_spec ar_mode), no loop stop; the official first-valid choice (and best by CLIP)
  sd8    batched chain SD x 8 + loop stop (3 repeats, period 8..128), best by CLIP among the valid
  sd8p   the same, pruned after --prune-rounds rounds to the top-4 by partial-render CLIP (finished = final output)
  sd4    batched chain SD x 4 + loop stop, best by CLIP among the valid
  sd<N>p<R>k<K>  N candidates + loop stop, pruned after R rounds to the top K (e.g. sd8p24k4, sd8p12k2)
  the fair baselines and the latency-quality frontier (every AR / SD method with the same loop stop):
  ar1s   ar1 + loop stop (same seed: = ar1 cut at the first firing + EOS)
  sd1s   sd1 + loop stop
  ar<N>s batched AR x N + loop stop, best by score (e.g. ar2s, ar4s, ar8s)
  sd<N>  batched chain SD x N + loop stop, best by score, any N >= 2 (e.g. sd2, sd16; sd4 / sd8 as before)
  sd<N>f<k> / ar<N>f<k>   night, "first k finish": N candidates (SD / batched AR) + loop stop decode together; as
         soon as >= k have finished, the rest are dropped and the best of the finished ones is chosen (e.g. sd8f4)
Candidate k uses seed 42 + gidx + 1000 k (the multi-sample runs). The batched SD decoders use rng_mode "block" (each
candidate is still an exact sample of its target; seeds do not reproduce the sequential tokens). Rendering + CLIP run in
sightline/score_server.py (the svgmetrics environment) and their time is part of every multi-candidate method. cuDNN SDPA is off.
Writes one JSON line per (prompt, method) to <out>/shard-i-of-n/results.jsonl.
--task isvg (the dflash environment): InternSVG-8B text2svg (profile.py rows, the exported text model, --head required,
cap 4000, EOS 151645), sampling at --temperature / --top-p (no official setting: InternSVG's protocol is greedy); outputs
are decoded to SVG source, closed (close_svg) and rendered with the official OmniSVG render + validity rule; CLIP uses the
caption after "Instruction:".
--task i2 (the glance environment): image2svg on the VLM target (vlm.py: official preprocessing, M-RoPE prefill and positions),
the I2SVG block-32 head (3 epochs), official I2SVG sampling (T 0.3, top-p 0.9, top-k 50, penalty 1.05), sd1 = top-16@63,
selection by DINO(input image, render).

  python sightline/bon_decode.py --out <dir> [--sets bench] [--shard i/n] [--methods ar1,sd1,ar8,sd8,sd8p,sd4]

Merged, with no change to what runs, from src/bon/bon_decode.py and its three task wrappers
(bon_decode_x, bon_decode_i2x, bon_decode_i2s): --xrows now serves every task, which is all the
wrappers did. See _build/MERGES.md.
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("src", "src/h2", "src/t7", "src/tol", "src/h3a"):
    sys.path.insert(0, str(ROOT / p))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
from sightline import design as sim                                              # noqa: E402
from eval.records import EOS, MODELS, chain_params, load_rows             # noqa: E402
from sightline.ar import ar_generate                                  # noqa: E402
from sightline.batch_spec import BatchChainSD                         # noqa: E402
from sightline.tree_spec import TreeDrafter, tree_spec_generate       # noqa: E402
from loop_quality import snap                                       # noqa: E402
from sightline.paths import RUNS

HEAD = (RUNS / "b3-train/omnisvg-dflash-main300k-b32-6ep-1012/heads/step150000")
HEAD_I2 = (RUNS / "h3a-train/i2svg-b32-m3warm-100k-3ep/heads/step18750")  # the final 100k head (was the 30k head)
TASK = {"name": "t2", "eos": 196999, "max_new": 1536}
_TAG = None


def close_svg(text: str) -> str:
    """Make a partial / capped SVG source renderable: cut after the last complete tag, close the open elements."""
    import re
    global _TAG
    _TAG = _TAG or re.compile(r"<(/?)([A-Za-z][\w:.-]*)([^<>]*?)(/?)>")
    i = text.rfind(">")
    t = text[:i + 1] if i >= 0 else ""
    stack = []
    for m in _TAG.finditer(t):
        closing, name, _, selfclose = m.groups()
        if closing:
            if stack and stack[-1] == name:
                stack.pop()
        elif not selfclose:
            stack.append(name)
    return t + "".join(f"</{n}>" for n in reversed(stack))
MAX_NEW = 1536          # OmniSVG; TASK["max_new"] is the active cap


class Scorer:
    def __init__(self, workers: int, log: Path):
        self.p = subprocess.Popen([str(ROOT / "python"), str(ROOT / "bon/score_server.py"),
                                   "--workers", str(workers)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=log.open("a"), text=True, bufsize=1)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"score server died (see {log})")
            if line.startswith("{") and "ready" in line:
                break

    def __call__(self, items: list[dict]) -> dict:
        self.p.stdin.write(json.dumps({"items": items,
                                       "metric": "dino" if TASK["name"] in ("i2", "i2s") else "clip"}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("score server died")
        return json.loads(line)

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"op": "quit"}) + "\n")
            self.p.stdin.flush()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()


def strip(tokens: list[int]) -> list[int]:
    e = TASK["eos"]
    return tokens[:-1] if tokens and tokens[-1] == e else tokens


def partial(tokens: list[int]) -> list[int]:
    """A running candidate's prefix that ends before its last (possibly incomplete) command / color token."""
    return tokens[:snap(tokens, max(1, len(tokens) - 1))] if len(tokens) > 1 else tokens


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def params(mode: str, subtype: str, stop: bool):
    if TASK["name"] in ("isvg", "i2s"):
        from sightline.verify import ChainParams
        p = ChainParams(greedy=False, temperature=TASK["temperature"], top_p=TASK["top_p"])
        if stop:
            p.loop_ban, p.loop_repeats, p.loop_min_period, p.loop_action = 128, 3, 8, "eos"
        return p
    if TASK["name"] == "i2":
        from eval_i2svg import chain_params as i2_params
        p = i2_params(mode)
    else:
        p = chain_params(mode, subtype)
    if stop:
        p.loop_ban, p.loop_repeats, p.loop_min_period, p.loop_action = 128, 3, 8, "eos"
    return p


def gens(gidx: int, n: int):
    return [torch.Generator(device="cuda").manual_seed(42 + gidx + 1000 * k) for k in range(n)]


def pick(scores, valid, cands=None) -> int:
    cands = list(range(len(scores))) if cands is None else list(cands)
    pool = [i for i in cands if valid[i]] or cands
    return max(pool, key=lambda i: scores[i])


def item(r, tokens):
    if TASK["name"] == "i2s":                  # InternSVG writes SVG source; selection is against the input image
        text = TASK["tok"].decode(tokens, skip_special_tokens=True)
        return {"svg": close_svg(text), "subtype": r["subtype"], "gt": r["image_path"]}
    if TASK["name"] == "isvg":
        text = TASK["tok"].decode(tokens, skip_special_tokens=True)
        return {"svg": close_svg(text), "subtype": r["subtype"], "text": r["text"].split("Instruction:")[-1].strip()}
    if TASK["name"] == "i2":
        return {"tokens": tokens, "subtype": r["subtype"], "gt": r["image_path"]}
    return {"tokens": tokens, "subtype": r["subtype"], "text": r["text"]}


def run_batched(target, head, r, n, stop, scorer, ar_mode=False, prune_rounds=None, keep=4, mode="sample", first_k=None):
    """Returns the record fields of one multi-candidate method."""
    p = params(mode, r["subtype"], stop)
    dec = BatchChainSD(target, head, [TASK["eos"]], TASK["max_new"], rng_mode="sequential" if ar_mode else "block",
                       ar_mode=ar_mode)
    t_prune_score = t_final_score = 0.0
    t0 = sync()
    inp = r.get("_vlm")
    dec.start(r["prompt_input_ids"], [p] * n, gens(r["gidx"], n),
              prefill_kwargs=inp.prefill_kwargs() if inp else None, positions=inp)
    pruned_info = None
    if prune_rounds:
        dec.run(max_rounds=prune_rounds)
        ts = sync()
        toks = [strip(s.generated(dec.n_prompt)) if s.stop_reason else partial(s.generated(dec.n_prompt)) for s in dec.seqs]
        rep = scorer([item(r, t) for t in toks])
        t_prune_score = sync() - ts
        order = sorted(range(n), key=lambda i: -rep["scores"][i])
        kept = set(order[:keep])
        pruned_info = {"partial_scores": rep["scores"], "kept": sorted(kept),
                       "partial_len": [len(t) for t in toks], "round": dec.rounds}
        dec.prune(kept)
    if first_k:                                    # race -- once >= k candidates finished, drop the rest
        while dec.rows and sum(s.stop_reason in ("eos", "max_new_tokens") for s in dec.seqs) < first_k:
            dec.run(max_rounds=1)
        dec.prune(set())
    dec.run()
    t_dec_end = sync()
    alive = [i for i, s in enumerate(dec.seqs) if s.stop_reason != "pruned"]
    ts = sync()
    finals = [strip(dec.seqs[i].generated(dec.n_prompt)) for i in alive]
    rep = scorer([item(r, t) for t in finals])
    t_final_score = sync() - ts
    scores, valid = rep["scores"], rep["valid"]
    choice_local = pick(scores, valid)
    rec = {"n": n, "t_total": round(sync() - t0, 4), "t_score": round(t_prune_score + t_final_score, 4),
           "t_decode": round(t_dec_end - t0 - t_prune_score, 4), "t_prune_score": round(t_prune_score, 4),
           "nfe": dec.nfe, "rounds": dec.rounds, "phase": {k: round(v, 4) for k, v in dec.t.items()},
           "alive": alive, "final_scores": scores, "final_valid": valid,
           "choice": alive[choice_local], "clip": scores[choice_local], "valid": valid[choice_local],
           "cand_tokens": [s.generated(dec.n_prompt) for s in dec.seqs],
           "cand_stop": [s.stop_reason for s in dec.seqs], "prune": pruned_info, "first_k": first_k}
    if ar_mode:                                    # the official rule: the first valid candidate (else candidate 0)
        fv = next((k for k, v in enumerate(valid) if v), 0)
        rec.update({"first_valid": alive[fv], "clip_first_valid": scores[fv], "valid_first_valid": valid[fv]})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="bench")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gidx", default=None, help="comma-separated rows (smoke)")
    ap.add_argument("--methods", default="ar1,sd1,ar8,sd8,sd8p,sd4")
    ap.add_argument("--prune-rounds", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--task", default="t2", choices=["t2", "i2", "isvg", "i2s"])
    ap.add_argument("--xrows", type=Path, default=None,
                    help="decode the rows of this file instead of the task's own benchmark "
                         "(eval/rows/ builds them); required for --task i2s")
    ap.add_argument("--head", type=Path, default=None,
                    help="isvg: the InternSVG head (required); t2 / i2: overrides HEAD / HEAD_I2 (10-15 night)")
    ap.add_argument("--temperature", type=float, default=0.6, help="isvg sampling")
    ap.add_argument("--top-p", type=float, default=0.9, help="isvg sampling")
    a = ap.parse_args()
    TASK["name"] = a.task
    if a.task == "i2s":
        from sightline.process import set_eos
        from targets import internsvg as V
        from transformers import AutoTokenizer
        TASK.update({"eos": V.EOS, "max_new": V.MAX_NEW, "temperature": a.temperature, "top_p": a.top_p,
                     "tok": AutoTokenizer.from_pretrained(V.MODEL)})
        set_eos(V.EOS)
        assert a.head is not None and a.xrows is not None, "--task i2s needs --head and --xrows"
    if a.task == "isvg":
        from sightline.process import set_eos
        from targets import profiles as P
        from transformers import AutoTokenizer
        TASK.update({"eos": P.EOS, "max_new": P.MAX_NEW, "temperature": a.temperature, "top_p": a.top_p,
                     "tok": AutoTokenizer.from_pretrained(P.TARGET)})
        set_eos(P.EOS)
        assert a.head is not None, "--task isvg needs --head"
    from dflash.model import DFlashDraftModel
    from transformers import AutoModelForCausalLM
    torch.backends.cuda.enable_cudnn_sdp(False)
    si, sn = (int(x) for x in a.shard.split("/"))
    if a.xrows is not None:            # any task: decode the rows of this file (eval/rows/)
        allr = [json.loads(x) for x in a.xrows.read_text().splitlines() if x.strip()]
        rows = [r for r in allr if r["set"] in a.sets.split(",") and r["gidx"] % sn == si]
    elif a.task == "isvg":
        from targets.profiles import load_rows as isvg_rows
        rows = [r for r in isvg_rows(a.sets.split(",")) if r["gidx"] % sn == si]
    elif a.task == "i2":
        from eval.eval_image_omnisvg import load_rows as i2_rows
        rows = i2_rows(a.sets.split(","), a.shard)
    else:
        rows = [r for r in load_rows(a.sets.split(",")) if r["gidx"] % sn == si]
    if a.gidx:
        keep = {int(x) for x in a.gidx.split(",")}
        rows = [r for r in rows if r["gidx"] in keep]
    rows = rows[:a.limit]
    out = a.out / f"shard-{si}-of-{sn}"
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    res_path = out / "results.jsonl"
    if res_path.exists():
        done = {(x["gidx"], x["method"]) for x in map(json.loads, res_path.read_text().splitlines()) if x}
    if a.task == "i2s":
        from targets import internsvg as V
        proc = V.load_processor()
        target, _ = V.load_target(torch.bfloat16)
        head = DFlashDraftModel.from_pretrained(a.head, dtype=torch.bfloat16).cuda().eval()
        tree = sim.pick_designs("D1-top16-N127")[0]

        def prep(r):
            inp = V.build_inputs(proc, target, r, torch.bfloat16)
            return {**r, "prompt_input_ids": inp.prompt_ids, "_vlm": inp}
    elif a.task == "isvg":
        from targets import profiles as P
        target = AutoModelForCausalLM.from_pretrained(P.TARGET, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        head = DFlashDraftModel.from_pretrained(a.head, dtype=torch.bfloat16).cuda().eval()
        tree = sim.pick_designs("D1-top16-N127")[0]

        def prep(r):
            return r
    elif a.task == "i2":
        import vlm
        official = vlm.load_official()
        target, _ = vlm.load_target(torch.bfloat16)
        target = target.cuda().eval() if next(target.parameters()).device.type != "cuda" else target.eval()
        head = DFlashDraftModel.from_pretrained(a.head or HEAD_I2, dtype=torch.bfloat16).cuda().eval()
        tree = sim.pick_designs("D1-top16-N63")[0]
        tmp = out / "tmp-input.png"

        def prep(r):
            inp = vlm.build_inputs(official, target, r["image_path"], tmp, torch.bfloat16)
            return {**r, "prompt_input_ids": inp.prompt_ids, "_vlm": inp}
    else:
        target = AutoModelForCausalLM.from_pretrained(MODELS / "OmniSVG1.1_8B-qwen2-text", dtype=torch.bfloat16,
                                                      attn_implementation="sdpa").cuda().eval()
        head = DFlashDraftModel.from_pretrained(a.head or HEAD, dtype=torch.bfloat16).cuda().eval()
        tree = sim.pick_designs("D1-top16-N127")[0]

        def prep(r):
            return r
    block = int(head.block_size)
    scorer = Scorer(a.workers, out / "score_server.log")
    methods = a.methods.split(",")
    (out / "protocol.json").write_text(json.dumps({
        "task": a.task, "head": str(a.head or (HEAD_I2 if a.task == "i2" else HEAD)), "methods": methods,
        "temperature": a.temperature if a.task == "isvg" else None, "top_p": a.top_p if a.task == "isvg" else None, "prune_rounds": a.prune_rounds, "max_new": TASK["max_new"],
        "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(), "cudnn_sdp": False,
        "seeds": "42 + gidx + 1000 k", "rng_mode": "batched SD: block; ar8: sequential"}, indent=1))
    # warm-up (kernels, allocator, score server) outside the timed rows
    w = prep(rows[0])
    if a.task in ("t2", "isvg"):
        ar_generate(target, w["prompt_input_ids"], 32, [TASK["eos"]], params("sample", w["subtype"], False),
                    generator=gens(0, 1)[0])
    run_batched(target, head, w, 2, True, scorer)
    with res_path.open("a") as f:
        for r in rows:
            if all((r["gidx"], m) in done for m in methods):
                continue
            r = prep(r)
            for m in methods:
                if (r["gidx"], m) in done:
                    continue
                base = {"gidx": r["gidx"], "id": r["id"], "set": r["set"], "subtype": r["subtype"], "method": m,
                        "task": a.task}
                stop1 = m in ("ar1s", "sd1s")      # one sample + the loop stop (ar1s = the fair AR baseline)
                if m in ("ar1", "ar1s") and a.task in ("i2", "i2s"):
                    import vlm
                    t0 = sync()
                    res = vlm.vlm_ar_generate(target, r["_vlm"], TASK["max_new"], [TASK["eos"]],
                                              params("sample", r["subtype"], stop1), gens(r["gidx"], 1)[0])
                    rec = {"t_total": round(sync() - t0, 4), "nfe": res.nfe, "tokens": res.tokens}
                elif m in ("ar1", "ar1s"):
                    t0 = sync()
                    res = ar_generate(target, r["prompt_input_ids"], TASK["max_new"], [TASK["eos"]],
                                      params("sample", r["subtype"], stop1), generator=gens(r["gidx"], 1)[0])
                    rec = {"t_total": round(sync() - t0, 4), "nfe": res.nfe, "tokens": res.tokens}
                elif m in ("sd1", "sd1s"):
                    t0 = sync()
                    dr = TreeDrafter(head, target, block)
                    inp = r.get("_vlm")
                    res = tree_spec_generate(target, dr, r["prompt_input_ids"], TASK["max_new"], [TASK["eos"]],
                                             params("sample", r["subtype"], stop1), tree,
                                             generator=gens(r["gidx"], 1)[0], block_size=block, positions=inp,
                                             prefill_kwargs=inp.prefill_kwargs() if inp else None)
                    rec = {"t_total": round(sync() - t0, 4), "nfe": res.nfe, "tokens": res.tokens}
                elif m == "ar8":
                    rec = run_batched(target, head, r, 8, False, scorer, ar_mode=True)
                elif re.fullmatch(r"(sd|ar)(\d+)f(\d+)", m):           # N candidates + loop stop, keep
                    kind, n_c, k_f = re.fullmatch(r"(sd|ar)(\d+)f(\d+)", m).groups()   # the first k that finish
                    rec = run_batched(target, head, r, int(n_c), True, scorer, ar_mode=kind == "ar", first_k=int(k_f))
                elif re.fullmatch(r"ar(\d+)s", m):                      # batched AR x N + loop stop, best
                    rec = run_batched(target, head, r, int(m[2:-1]), True, scorer, ar_mode=True)   # by score
                elif m == "sd8p":
                    rec = run_batched(target, head, r, 8, True, scorer, prune_rounds=a.prune_rounds)
                elif re.fullmatch(r"sd(\d+)p(\d+)k(\d+)", m):          # e.g. sd8p24k4: 8 candidates, prune after
                    n_c, R, K = (int(x) for x in re.fullmatch(r"sd(\d+)p(\d+)k(\d+)", m).groups())  # 24 rounds to 4
                    rec = run_batched(target, head, r, n_c, True, scorer, prune_rounds=R, keep=K)
                elif re.fullmatch(r"sd(\d+)", m) and int(m[2:]) >= 2:   # batched chain SD x N + loop stop (sd4, sd8;
                    rec = run_batched(target, head, r, int(m[2:]), True, scorer)   # any N, e.g. sd2, sd16)
                else:
                    raise ValueError(m)
                f.write(json.dumps({**base, **rec}) + "\n")
                f.flush()
                print(f"gidx {r['gidx']} {m}: {rec['t_total']:.2f}s nfe {rec['nfe']}"
                      + (f" clip {rec['clip']:.4f}" if "clip" in rec else ""), flush=True)
    scorer.close()
    (out / "DONE").write_text(json.dumps({"shard": a.shard, "methods": methods}) + "\n")   # watchers
    print("BON_DECODE_DONE", a.shard)


if __name__ == "__main__":
    main()
