""" render + score server for the best-of-N decoder (sightline/bon_decode.py), run in the svgmetrics environment.

The decoder (the dflash environment) cannot import the official OmniSVG decoder (deepsvg needs IPython, qwen_vl_utils,...), so it
talks to this process over stdin / stdout, one JSON object per line:
  request  {"id": <any>, "metric": "clip"|"dino", "items": [{"tokens": [...] | "svg": "<svg...>", "subtype": "icon"|"illustration",
            "text": "<prompt>" (clip) | "gt": "<input image path>" (dino)},...]}
  reply    {"id": <same>, "scores": [CLIP or null], "valid": [bool], "t_render": s, "t_score": s}
Rendering = the official decode (run.decode: tokens -> SVG -> 1024 px image, official validity), in a pool of
--workers processes; a failed render is scored as a white image (the metrics.py convention). Score = CLIP(prompt, render)
with the steps of metrics.clip_scores (77-token truncation, normalized features, dot product), the model
loaded once; DINO = refs/OmniSVG/metrics/compute_dion (DINOv2-base mean-pooled last hidden state, (cos + 1) / 2, the input
image composited on white), the model loaded once and the input's features cached. Partial outputs are rendered the same
way (the caller snaps cuts to a command / color boundary).

  python sightline/score_server.py [--workers 8] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from sightline.paths import MODELS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script

_OFF = None
# cairosvg 2.7.1's path parser loops forever on numbers after a closepath with no current subpath
# ("... z 5.33 5.33 z m...", an InternSVG candidate), and pool.map below has no timeout, so n31-i2sbon-18 hung for 47 h.
# A render that runs past this many seconds is a failed render (white image, invalid), the convention above; the slowest
# render + score of the 15,871 table-2 x8 rows before it took 0.65 s, so this is ~8x the worst loop-free case and the
# timed row is inflated by at most 5 s (harvest: flag rows with t_score > 2 s). The parser loop is pure Python, so
# SIGALRM stops it.
RENDER_TIMEOUT = 5


def _alarm(signum, frame):
    raise TimeoutError(f"render exceeded {RENDER_TIMEOUT} s")


def _init():
    global _OFF
    signal.signal(signal.SIGALRM, _alarm)
    from omnisvg.run import load_official
    _OFF = load_official()
    _OFF.svg_tokenizer = _OFF.SVGTokenizer(_OFF.CONFIG_PATH, model_size="8B")


def _render(item):
    from PIL import Image
    from omnisvg.run import decode
    signal.alarm(RENDER_TIMEOUT)
    try:
        if "svg" in item:                 # SVG source text (InternSVG-8B): the official render + validity rule
            svg = item["svg"]
            img = _OFF.render_svg_to_image(svg, size=_OFF.RENDER_SIZE)
            valid, _ = _OFF.is_valid_candidate(svg, img, item["subtype"])
        else:
            svg, img, valid, _ = decode(_OFF, item["tokens"], item["subtype"])
    except Exception as e:
        if isinstance(e, TimeoutError):
            print(f"render timeout ({RENDER_TIMEOUT} s): scored as a failed render", file=sys.stderr, flush=True)
        img, valid = None, False
    finally:
        signal.alarm(0)
    if img is None:
        img = Image.new("RGB", (1024, 1024), (255, 255, 255))
    return img.convert("RGB"), bool(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    # the official render prints "Render error:..." to stdout, and the forked render workers share this stdout
    # (the protocol pipe), so a malformed SVG broke the protocol (all  InternSVG BoN shards died). The protocol gets a
    # private dup of fd 1; fd 1 itself (this process, the workers forked below, C code) now goes to stderr (the log).
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    import torch
    from multiprocessing import Pool
    from transformers import CLIPModel, CLIPProcessor
    from omnisvg.metrics import MODELS
    # the render workers must be forked before CUDA is initialized here (the official module touches CUDA on import)
    pool = Pool(a.workers, initializer=_init)
    path = MODELS / "clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(path).to(a.device).eval()
    proc = CLIPProcessor.from_pretrained(path)
    dino = {}

    def dino_feats(images):
        if "model" not in dino:
            from transformers import AutoImageProcessor, AutoModel
            dino["model"] = AutoModel.from_pretrained(MODELS / "dinov2-base").to(a.device).eval()
            dino["proc"] = AutoImageProcessor.from_pretrained(MODELS / "dinov2-base")
        inp = dino["proc"](images=images, return_tensors="pt").to(a.device)
        return dino["model"](**inp).last_hidden_state.mean(dim=1)

    def gt_image(path):
        from PIL import Image
        im = Image.open(path)
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        return im.convert("RGB")

    gt_cache = {}
    print(json.dumps({"ready": True}), file=proto, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        items = req["items"]
        t0 = time.perf_counter()
        rendered = pool.map(_render, items, chunksize=1)
        t1 = time.perf_counter()
        if req.get("metric", "clip") == "dino":
            with torch.no_grad():
                g = items[0]["gt"]
                assert all(it["gt"] == g for it in items)
                if g not in gt_cache:
                    gt_cache.clear()
                    gt_cache[g] = dino_feats(gt_image(g))
                f_g = gt_cache[g]
                f = dino_feats([img for img, _ in rendered])
                scores = ((torch.nn.functional.cosine_similarity(f, f_g.expand_as(f), dim=1) + 1) / 2).tolist()
            t2 = time.perf_counter()
            print(json.dumps({"id": req.get("id"), "scores": scores, "valid": [v for _, v in rendered],
                              "t_render": round(t1 - t0, 4), "t_score": round(t2 - t1, 4)}), file=proto, flush=True)
            continue
        with torch.no_grad():                       # one batched forward (scores may differ from the per-image
            inp = proc(text=[it["text"] for it in items], images=[img for img, _ in rendered],  # path in the last bits)
                       return_tensors="pt", padding=True, truncation=True, max_length=77).to(a.device)
            f_i = model.get_image_features(pixel_values=inp["pixel_values"])
            f_t = model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            f_i = f_i / f_i.norm(dim=-1, keepdim=True)
            f_t = f_t / f_t.norm(dim=-1, keepdim=True)
            scores = (f_i * f_t).sum(-1).tolist()
        t2 = time.perf_counter()
        print(json.dumps({"id": req.get("id"), "scores": scores, "valid": [v for _, v in rendered],
                          "t_render": round(t1 - t0, 4), "t_score": round(t2 - t1, 4)}), file=proto, flush=True)
    pool.close()


if __name__ == "__main__":
    main()
