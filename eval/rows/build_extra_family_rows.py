""" evaluation rows of the third-family targets for the transfer matrix, as xrows files for eval_fam.py:
  mmsvg-bench: the 300 MMSVG-Bench captions of table A (150 icon + 150 illustration; OmniSVG's official test set)
  sarena-bench: the 300 SArena text2svg descriptions of table A's cross-benchmark rows (150 + 150)
each wrapped in the family's official prompt (families.py; hivg3b: refs/HiVG text2svg prompt + chat template).
gidx: svgen7b 70000+, svgen3b 80000+, hivg3b 90000+.
  python build_fam_rows.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from fam.families import FAMILIES, encode  # noqa: E402
from sightline.paths import MODELS, RUNS

X = (RUNS / "tableA/xbench")
BASE = {"svgen7b": 70000, "svgen3b": 80000, "hivg3b": 90000}
HIVG = f"{MODELS}/HiVG-3B-Base-qwen2-text"


def main():
    from transformers import AutoTokenizer
    own = [json.loads(x) for x in (X / "omnisvg-own.jsonl").read_text().splitlines() if x.strip()]
    sa = [json.loads(x) for x in (X / "omnisvg-sarena.jsonl").read_text().splitlines() if x.strip()]
    src = ([("mmsvg-bench", r["subtype"], r["id"], r["text"]) for r in own if r["set"] == "bench"]
           + [("sarena-bench", r["subtype"], r["id"], r["text"]) for r in sa if r["set"] == "sarena-bench"])
    for fam in ("svgen7b", "svgen3b", "hivg3b"):
        path = HIVG if fam == "hivg3b" else FAMILIES[fam]["model"]
        tok = AutoTokenizer.from_pretrained(path)
        with (X / f"{fam}-eval.jsonl").open("w") as f:
            for i, (s, st, rid, text) in enumerate(src):
                if fam == "hivg3b":
                    msgs = [{"role": "user", "content": f"[T2ST] Generate SVG tokens from text.\n{text}"}]
                    ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                              add_special_tokens=False).input_ids
                else:
                    ids = encode(tok, fam, text)
                f.write(json.dumps({"gidx": BASE[fam] + i, "set": s, "subtype": st, "id": rid, "text": text,
                                    "prompt_input_ids": ids}) + "\n")
        print(fam, len(src), "rows ->", X / f"{fam}-eval.jsonl")


if __name__ == "__main__":
    main()
