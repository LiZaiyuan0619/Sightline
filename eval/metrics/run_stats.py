"""Model-free statistics of a run.py output dir: validity, EOS, length-cap hits, lengths, generate time."""
import json
import statistics
import sys


def main(run, cap=1536):
    rows = [json.loads(l) for l in open(f"{run}/results.jsonl")]
    cands = [c for r in rows for c in r["candidates"]]
    lens = sorted(c["length"] for c in cands)
    q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    out = {"run": run, "prompts": len(rows), "streams": len(cands),
           "valid": sum(c["valid"] for c in cands) / len(cands),
           "eos": sum(c["eos_found"] for c in cands) / len(cands),
           "hit_cap": sum((not c["eos_found"]) and c["length"] >= cap for c in cands) / len(cands),
           "len_mean": statistics.mean(lens), "len_p50": statistics.median(lens), "len_p90": q(0.9),
           "gen_s_mean": statistics.mean(r["generate_seconds"] for r in rows),
           "prompt_with_valid": sum(r["selected_index"] is not None for r in rows) / len(rows)}
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}))


if __name__ == "__main__":
    for run in sys.argv[1:]:
        main(run)
