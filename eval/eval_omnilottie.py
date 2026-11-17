""" (C1): eval_tree.py for OmniLottie Text-to-Lottie (= eval_fam.py with OmniLottie's settings):
binds the target-specific globals of eval_tree's use_profile: target = model/OmniLottie-qwen2-text (fp32-identical to
the official LottieDecoder, check_lottie.py), EOS 192399, cap 4096 new tokens (the official inference_hf.py
default), greedy = the official default (no penalty), sample = the official sampling (T 0.9, top-p 0.25, top-k 5), loop
detector at the cap (tail window 1024, max period 512), rows from --xrows (build_lottie_rows.py:
$SIGHTLINE_RUNS/tableA/xbench/omnilottie-bench.jsonl, sets mmlottie-bench, subtypes real / synthetic). Every other flag and the decode /
timing loop are eval_tree.main (not edited). fp32 audit: --audit --gidx <rows> (eval_tree's default audit picks
"icon" / "illustration" rows, which this benchmark does not have).
  PYTHONPATH=refs/SpecForge python eval_lottie.py --xrows <jsonl> --head <dir> \
      --designs D0-chain,D1-top16-N127 --sets mmlottie-bench --modes greedy,sample --time-ar --out <dir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import json
import sys
from pathlib import Path
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "h2", ROOT / "src", ROOT / "t7"):
    sys.path.insert(0, str(p))
EMPTY = (RUNS / "tableA/xbench/empty-ar-cache")
TARGET = (MODELS / "OmniLottie-qwen2-text")
EOS, CAP, TEMP, TOP_P, TOP_K = 192399, 4096, 0.9, 0.25, 5


def pop(argv, flag):
    i = argv.index(flag)
    v = argv[i + 1]
    del argv[i:i + 2]
    return v


def main():
    argv = sys.argv[1:]
    xrows = Path(pop(argv, "--xrows"))
    if "--max-new" not in argv:
        argv += ["--max-new", str(CAP)]
    sys.argv = [sys.argv[0]] + argv
    allr = [json.loads(x) for x in xrows.read_text().splitlines() if x.strip()]
    import eval_tree as E
    import cache_draft
    # OmniLottie's 192,400 ids pass cache_draft.fill_block's OmniSVG-vocabulary test (V >= COORD_START + N_COORD =
    # 192,072), which then reads OmniSVG's coordinate / token-type ranges and EOS 196999 (IndexError). Force its generic
    # branch, the one InternSVG / SVGen / HiVG take (top-k, realized, copy fields; the top-k trees read nothing else).
    cache_draft.N_COORD = 1 << 30
    from sightline.process import set_eos
    from sightline.verify import ChainParams

    def load_rows(sets):
        sets = set(sets) | ({"mmlottie-bench"} if {"bench", "extra"} & set(sets) else set())   # eval_tree's audit asks
        return [dict(r) for r in allr if r["set"] in sets]                                       # for bench / extra rows

    def chain_params(mode, subtype):
        if mode == "greedy":
            return ChainParams(greedy=True)
        return ChainParams(greedy=False, temperature=TEMP, top_p=TOP_P, top_k=TOP_K)

    def use_profile(name):
        E.EOS, E.MAX_NEW, E.AR_CACHE = EOS, CAP, EMPTY
        E.LOOP_KW = {"tail_window": 1024, "max_period": 512}
        E.chain_params, E.load_rows = chain_params, load_rows
        set_eos(EOS)
        return TARGET
    E.use_profile = use_profile
    E.main()


if __name__ == "__main__":
    main()
