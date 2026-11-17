""" T3 (theory experiments brief): a draft cache on NATURAL LANGUAGE, the control for "statistics, not domain".

Target Qwen3-8B, drafter DFlash's released Qwen3-8B-DFlash-b16 (block 16), the anchor's setting (dflash benchmark
transformers --reasoning off --temperature 0, max_new_tokens 2048). Per prompt:
  1. decode with the official dflash.model.dflash_generate (greedy, block 16) -> the target's own greedy stream and the
     official per-round acceptance lengths (stored as ref_acceptance; their histogram is checked against the anchor logs);
  2. cache_draft.cache_row on that stream (unchanged; a non-OmniSVG vocabulary keeps the generic fields only):
     n_off, top_ids / top_lp (top-32), real_rank (full-vocabulary rank of the realized token), real_lp, per block start;
  3. one teacher-forced target forward over prompt + stream: per-position entropy H, top-1 p1, top-k mass (k = 1, 2, 4,
     8, 16), target argmax == realized (greedy self-check), the same fields as runs/2026-11-05/theory/entropy_dump.py.
Prompts, sample order and counts are the benchmark's (dflash.benchmark.DATASETS formats, chat template with
enable_thinking False, random.Random(42) shuffle, first n). MT-Bench is multi-turn: every turn is one row (its prompt
holds the earlier turns and the official answers, as the benchmark builds them).
  python cache_draft_nl.py --target <Qwen3-8B dir> --draft <Qwen3-8B-DFlash-b16 dir> \
      --dataset gsm8k --n 128 --out <dir>      [--limit 20: smoke]   -> <out>/greedy/<idx>.npz + <out>/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
KS = (1, 2, 4, 8, 16)


def chain_replay(rr: np.ndarray, n_off: np.ndarray, n_gen: int, W: int) -> list[int]:
    """Rounds of a chain with window W on the cached labels (top-1 hit = real_rank 0), as runs/2026-11-05/theory/
    window_budget_replay.py does for SVG caches."""
    out, t = [], 0
    while t < n_gen - 1:
        k = min(int(n_off[t]), W - 1)
        n = 0
        while n < k and int(rr[t, n]) == 0:
            n += 1
        e = min(n + 1, n_gen - 1 - t)
        out.append(e)
        t += e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke)")
    a = ap.parse_args()
    from dflash.benchmark import (_select_dataset, apply_chat_template, load_and_process_dataset,
                                  load_transformers_models, stop_token_ids)
    from dflash.model import dflash_generate
    import cache_draft as CD

    torch.manual_seed(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    dev = torch.device("cuda:0")
    target, draft, tok = load_transformers_models(a.target, a.draft, dev)
    block = int(draft.block_size)
    stops = stop_token_ids(target, tok)
    data = _select_dataset(load_and_process_dataset(a.dataset), a.n)
    (a.out / "greedy").mkdir(parents=True, exist_ok=True)
    summ = {"dataset": a.dataset, "n": a.n, "block": block, "rows": 0, "official_rounds": 0, "official_tokens": 0,
            "replay_rounds": 0, "replay_tokens": 0, "same_rounds_rows": 0, "positions": 0, "argmax_ok": 0,
            "hist": [0] * (block + 1)}
    idx = 0
    t_all = time.perf_counter()
    for inst in data:
        messages = []
        for turn, content in enumerate(inst["turns"]):
            messages.append({"role": "user", "content": content})
            text = apply_chat_template(tok, messages, "off")
            ids = tok.encode(text, return_tensors="pt", add_special_tokens=False).to(dev)
            res = dflash_generate(draft, target=target, input_ids=ids, max_new_tokens=a.max_new, stop_token_ids=stops,
                                  temperature=0.0, top_p=1.0, top_k=0, block_size=block, return_stats=True)
            gen = res.output_ids[0, res.num_input_tokens:]
            gen = gen[gen >= 0].tolist() if (gen < 0).any() else gen.tolist()
            messages.append({"role": "assistant", "content": tok.decode(gen, skip_special_tokens=True)})
            prompt = ids[0].tolist()
            if a.limit is not None and idx >= a.limit:
                break
            f = a.out / "greedy" / f"{idx:04d}.npz"
            arrays = CD.cache_row(target, draft, prompt, gen, max_new=a.max_new, block_size=block)
            with torch.inference_mode():
                full = torch.tensor([prompt + gen], device=dev)
                lg = target(full).logits[0, len(prompt) - 1: full.shape[1] - 1].float()
                lp = torch.log_softmax(lg, dim=-1)
                pr = lp.exp()
                H = (-(pr * lp).sum(-1)).cpu().numpy().astype(np.float32)
                top = pr.topk(max(KS), dim=-1).values.cumsum(-1)
                topk = torch.stack([top[:, k - 1] for k in KS], -1).cpu().numpy().astype(np.float32)
                am = lg.argmax(-1).cpu().numpy() == np.asarray(gen)
            acc = [int(x) for x in res.acceptance_lengths]
            eos = bool(gen and gen[-1] in stops)
            np.savez(f, **arrays, gidx=idx, set=a.dataset, subtype=f"turn{turn}", n_prompt=len(prompt),
                     tokens=np.asarray(gen, dtype=np.int32), ref_acceptance=np.asarray(acc, dtype=np.int16),
                     ref_onset=-1, ref_eos=eos, H=H, p1=topk[:, 0], topk=topk, ks=np.asarray(KS), argmax_ok=am)
            rp = chain_replay(arrays["real_rank"], arrays["n_off"], len(gen), block)
            summ["rows"] += 1
            summ["official_rounds"] += len(acc)
            summ["official_tokens"] += sum(acc)
            summ["replay_rounds"] += len(rp)
            summ["replay_tokens"] += sum(rp)
            summ["same_rounds_rows"] += rp == acc
            summ["positions"] += len(gen)
            summ["argmax_ok"] += int(am.sum())
            for x in acc:
                summ["hist"][min(x, block)] += 1
            print(f"[done] {a.dataset} row {idx} (turn {turn}): {len(gen)} tok, official tau "
                  f"{sum(acc) / max(len(acc), 1):.2f}, replay tau {sum(rp) / max(len(rp), 1):.2f}, "
                  f"argmax_ok {am.mean():.4f}", flush=True)
            idx += 1
        if a.limit is not None and idx >= a.limit:
            break
    summ["official_tau"] = summ["official_tokens"] / max(summ["official_rounds"], 1)
    summ["replay_tau"] = summ["replay_tokens"] / max(summ["replay_rounds"], 1)
    summ["argmax_ok_rate"] = summ["argmax_ok"] / max(summ["positions"], 1)
    n = sum(summ["hist"])
    summ["hist_share"] = [x / n for x in summ["hist"]] if n else []
    summ["wall_s"] = time.perf_counter() - t_all
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print("CACHE_NL_DONE", json.dumps({k: summ[k] for k in ("rows", "official_tau", "replay_tau", "same_rounds_rows",
                                                             "argmax_ok_rate")}), flush=True)


if __name__ == "__main__":
    main()
