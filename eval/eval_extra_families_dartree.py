""" the DARTree baseline on the extension families (generality table).

baselines/eval_dart.py adds DARTree to eval_tree.py by replacing that module's pick / load_head / run_design at
import time, and eval_fam.py and eval_lottie.py both end in the same eval_tree.main after binding
their target's globals. So the whole bridge is: import eval_dart first, then hand over to the family's own evaluator --
no evaluator is edited (running jobs import them), and the DART design string, the Domino head loader and the tree
builder are the same code table 1's DARTree rows went through.

StarVector's evaluator (eval_sv.py) already parses DART designs itself and is not routed through here.

  python eval_dart_fam.py --evaluator fam --family svgen7b --xrows <jsonl> \
      --head <domino head> --designs DART-W12-E64-N127 --sets mmsvg-bench,sarena-bench --modes greedy,sample --out <dir>
  python eval_dart_fam.py --evaluator lottie --xrows <jsonl> --head <domino head>...
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "h2", ROOT / "e3", ROOT / "src", ROOT / "t7"):
    sys.path.insert(0, str(p))


def main():
    argv = sys.argv[1:]
    i = argv.index("--evaluator")
    which = argv[i + 1]
    del argv[i:i + 2]
    sys.argv = [sys.argv[0]] + argv

    import eval_dart  # noqa: F401  (patches eval_tree.pick / load_head / run_design)
    import eval_tree
    assert eval_tree.pick is eval_dart.pick, "eval_dart did not patch eval_tree"

    if which == "fam":
        from fam import eval_fam
        eval_fam.main()
    elif which == "lottie":
        sys.path.insert(0, str(ROOT / "lottie"))
        import eval_lottie
        eval_lottie.main()
    else:
        raise SystemExit(f"unknown evaluator {which!r} (fam | lottie)")


if __name__ == "__main__":
    main()
