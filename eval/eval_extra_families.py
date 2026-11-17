""" eval_tree.py for the third-family targets (transfer matrix): --family svgen7b|svgen3b|hivg3b replaces
the target-specific globals that eval_tree's use_profile binds (EOS, cap, sampling chain, rows, AR cache, loop
detector settings) and the target path; every other flag and the whole decode / timing loop are eval_tree.main
(eval_tree.py itself is not edited: running jobs import it). Rows: --xrows <$SIGHTLINE_RUNS/tableA/xbench/<fam>-eval.jsonl> from
build_fam_rows.py. Cap 2048 new tokens (= the drafters' training length; --max-new overrides). Sampling = the
family's official setting (SVGen T 0.2 / top-p 0.7; HiVG T 0.7 / top-p 0.9 / top-k 50).
  PYTHONPATH=refs/SpecForge python eval_fam.py --family svgen7b --xrows <jsonl> --head <dir> \
      --designs D1-top16-N127 --sets mmsvg-bench,sarena-bench --modes greedy,sample --time-ar --out <dir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import json
import sys
from pathlib import Path
from sightline.paths import MODELS, RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "h2", ROOT / "src", ROOT / "t7"):
    sys.path.insert(0, str(p))
EMPTY = (RUNS / "tableA/xbench/empty-ar-cache")
FAM = {"svgen7b": (f"{MODELS}/SVGen-Qwen2.5-Coder-7B-Instruct", 151645, 0.2, 0.7, 0),
       "svgen3b": (f"{MODELS}/SVGen-Llama-3.2-3B-Instruct", 128009, 0.2, 0.7, 0),
       "hivg3b": (f"{MODELS}/HiVG-3B-Base-qwen2-text", 151645, 0.7, 0.9, 50)}
CAP = 2048


def pop(argv, flag):
    i = argv.index(flag)
    v = argv[i + 1]
    del argv[i:i + 2]
    return v


def main():
    argv = sys.argv[1:]
    fam = pop(argv, "--family")
    xrows = Path(pop(argv, "--xrows"))
    if "--max-new" not in argv:
        argv += ["--max-new", str(CAP)]
    sys.argv = [sys.argv[0]] + argv
    target, eos, temp, top_p, top_k = FAM[fam]
    allr = [json.loads(x) for x in xrows.read_text().splitlines() if x.strip()]
    import eval_tree as E
    from sightline.process import set_eos
    from sightline.verify import ChainParams

    def load_rows(sets):
        return [dict(r) for r in allr if r["set"] in sets]

    def chain_params(mode, subtype):
        if mode == "greedy":
            return ChainParams(greedy=True)
        return ChainParams(greedy=False, temperature=temp, top_p=top_p, top_k=top_k)

    def use_profile(name):
        E.EOS, E.MAX_NEW, E.AR_CACHE = eos, CAP, EMPTY
        E.LOOP_KW = {"tail_window": 1024, "max_period": 512}
        E.chain_params, E.load_rows = chain_params, load_rows
        set_eos(eos)
        return Path(target)
    E.use_profile = use_profile
    E.main()


if __name__ == "__main__":
    main()
