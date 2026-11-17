""" one-sentence descriptions of the 300 MMSVG-Bench image-to-SVG inputs (150 icon + 150 illustration; they ship
without text), so table 1's 5th block (InternSVG-8B image-to-SVG, whose prompt is image + instruction) can keep the
MMSVG-Bench columns of the OmniSVG image block. Style = the MMSVG `description` column the other own rows use: the
prompt shows 4 MMSVG descriptions of the same subtype (from the MMSVG-2M held-out rows of the eval file) as style
examples; temperature 0. Model: DeepSeek deepseek-flash (OpenAI-compatible API; key from $DEEPSEEK_API_KEY, never
written to disk), through the proxy. Resumable: one JSON line per image in --out; reruns skip done ids.
  source ~/bin/the site proxy helper && pm && DEEPSEEK_API_KEY=... python3 caption_bench.py --out <jsonl> [--workers 8]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import argparse
import base64
import io
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sightline.paths import RUNS

U = RUNS
ROWS = U / "h3a-data/eval_rows.jsonl"
OWN = U / "tableA/xbench/internsvg-i2-own.jsonl"
BASE = os.environ["DEEPSEEK_BASE_URL"]      # an OpenAI-compatible endpoint
MODEL = "deepseek-flash"


def style_examples(sub: str, n: int = 4) -> list[str]:
    rows = [json.loads(x) for x in open(OWN) if x.strip()]
    ex = [r["prompt"].split("Instruction: ", 1)[1] for r in rows if r["subtype"] == sub and r["set"] == "extra"]
    return ex[:n]


def png_b64(path: str) -> str:
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    im.thumbnail((512, 512))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def prompt_for(sub: str) -> str:
    kind = "icon" if sub == "icon" else "illustration"
    ex = "\n".join(f"- {e}" for e in style_examples(sub))
    return (f"Describe this {kind} image in ONE sentence, in the same style as these example descriptions of other "
            f"{kind}s:\n{ex}\nState what is depicted, its main colors and layout. Output only the sentence.")


def call(key: str, text: str, b64: str, timeout: int = 180) -> str:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 3000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                                                        {"type": "image_url",
                                                         "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    key = os.environ["DEEPSEEK_API_KEY"]
    rows = [json.loads(x) for x in open(ROWS) if x.strip()]
    bench = [r for r in rows if r["set"] == "bench"][:a.limit]
    done = set()
    if a.out.exists():
        done = {json.loads(x)["id"] for x in open(a.out) if x.strip()}
    todo = [r for r in bench if r["id"] not in done]
    prompts = {s: prompt_for(s) for s in ("icon", "illustration")}
    lock = threading.Lock()
    print(f"{len(bench)} bench images, {len(done)} done, {len(todo)} to caption", flush=True)

    def work(r):
        b64 = png_b64(r["image_path"])
        for attempt in range(5):
            try:
                cap = call(key, prompts[r["subtype"]], b64)
                if cap:
                    break
            except Exception as e:                       # noqa: BLE001  (network / rate limit: retry)
                cap = ""
                print("retry", r["gidx"], repr(e)[:120], flush=True)
            time.sleep(5 * (attempt + 1))
        cap = cap.splitlines()[0].strip().strip('"') if cap else ""
        with lock, a.out.open("a") as f:
            f.write(json.dumps({"id": r["id"], "gidx": r["gidx"], "subtype": r["subtype"], "caption": cap,
                                "model": MODEL}) + "\n")
        return bool(cap)

    with ThreadPoolExecutor(a.workers) as pool:
        ok = sum(pool.map(work, todo))
    print(f"CAPTION_DONE {ok} / {len(todo)} non-empty", flush=True)


if __name__ == "__main__":
    main()
