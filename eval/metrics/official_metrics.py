"""A1 evaluation: the official OmniSVG metrics (refs/OmniSVG/metrics) on one run.py output directory.

Models, preprocessing and formulas follow upstream; upstream functions are imported wherever they work.
Changes: local weight paths, each model loaded once instead of per image, and the broken entry points
(compute_clip.py contains Markdown, base_metric.py imports a missing `util`). HPS builds ViT-H-14 without
the laion weights because HPS_v2_compressed.pt overwrites every parameter (strict load).

Selection: `first` = candidate 0 (paper protocol, issue #49); `first-valid` = official CLI choice for 8 streams.
Missing outputs (no rendered PNG) are reported and either excluded or scored as a white image.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import argparse
import json
import os
import sys
import types
from pathlib import Path
from sightline.paths import REFS, RUNS

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REFS / "OmniSVG/metrics"
MODELS = MODELS
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch-hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

TEXT_METRICS = ["clip", "aesthetic", "hps", "fid"]
IMAGE_METRICS = ["dino", "ssim_lpips_mse"]


def upstream(name):
    sys.path.insert(0, str(UPSTREAM))
    if name == "base_metric" or name == "compute_dion":
        # base_metric.py imports AverageMeter from a `util` module that upstream never shipped.
        class AverageMeter:
            def __init__(self):
                self.reset()

            def reset(self):
                self.sum, self.count, self.avg = 0.0, 0, 0.0

            def update(self, val, n=1):
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
        sys.modules.setdefault("util", types.SimpleNamespace(AverageMeter=AverageMeter))
    return __import__(name)


def select(records, run, policy):
    rows = []
    for r in records:
        cands = r["candidates"]
        if policy == "first":
            pick = cands[0]
        else:
            pick = next((c for c in cands if c.get("valid")), None)
        png = run / f'{r["id"]}-candidate-{pick["index"]}.png' if pick else None
        rows.append({"id": r["id"], "text": r.get("text", ""), "gt": r.get("image_path"),
                     "candidate": pick["index"] if pick else None,
                     "png": str(png) if png and png.exists() else None})
    return rows


def clip_scores(rows, device):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    path = MODELS / "clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(path).to(device).eval()
    processor = CLIPProcessor.from_pretrained(path)
    out = []
    for r in rows:  # Same steps as calculate_clip_score in compute_clip.py.
        image = Image.open(r["png"]).convert("RGB")
        # truncate to CLIP's 77-token context (standard CLIPScore). Upstream has no truncation;
        # the MMSVGBench prompts are <= 13 words, so this changes nothing for them, but held-out
        # MMSVG prompts reach ~100 words and crash the text encoder otherwise.
        inputs = processor(text=[r["text"]], images=image, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device)
        with torch.no_grad():
            img = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            out.append((img @ txt.T).item())
    return out


def aesthetic_scores(rows, device):
    import clip
    import torch
    from PIL import Image
    mod = upstream("compute_aesthetic")
    model = mod.MLP(768)
    model.load_state_dict(torch.load(MODELS / "aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth",
                                     map_location=device, weights_only=True))
    model.to(device).eval()
    clip_model, preprocess = clip.load("ViT-L/14", device=device, download_root=str(MODELS / "openai-clip"))
    clip_model.eval()
    out = []
    for r in rows:  # Same steps as compute_aesthetic_score; its second output (L/14 CLIP) is not reported.
        image = preprocess(Image.open(r["png"]).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = clip_model.encode_image(image)
            emb = mod.normalized(feats.cpu().detach().numpy())
            tensor_type = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
            out.append(model(torch.from_numpy(emb).to(device).type(tensor_type)).cpu().item())
    return out


def hps_scores(rows, device):
    import torch
    from PIL import Image
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    mod = upstream("compute_hps")
    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14", None, precision="amp", device=device, jit=False, force_quick_gelu=False,
        force_custom_text=False, force_patch_dropout=False, force_image_size=None, pretrained_image=False,
        image_mean=None, image_std=None, light_augmentation=True, aug_cfg={}, output_dict=True,
        with_score_predictor=False, with_region_predictor=False)
    checkpoint = torch.load(MODELS / "HPSv2/HPS_v2_compressed.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    model = model.to(device).eval()
    out = []
    for r in rows:  # Same steps as score in compute_hps.py (HPS v2.0).
        with torch.no_grad():
            image = mod.convert_rgba_to_rgb_with_white_bg(Image.open(r["png"]))
            image = preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
            text = tokenizer([r["text"]]).to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast():
                o = model(image, text)
                logits = o["image_features"] @ o["text_features"].T
                out.append(float(torch.diagonal(logits).cpu().numpy()[0]))
    return out


def fid_score(rows, ref, device):
    import numpy as np
    mod = upstream("compute_fid")
    stats = np.load(ref)
    extractor = mod.InceptionV3Feature(device)
    feats = extractor.extract_features_batch([r["png"] for r in rows], batch_size=50)
    mu, sigma = mod.calculate_activation_statistics(feats)
    return float(mod.calculate_frechet_distance(stats["mu"], stats["sigma"], mu, sigma))


def dino_scores(rows, device):
    from transformers import AutoImageProcessor, AutoModel
    mod = upstream("compute_dion")

    class Local(mod.DINOScoreCalculator):
        def get_DINOv2_model(self, model_size):
            assert model_size == "base"
            path = MODELS / "dinov2-base"
            return AutoModel.from_pretrained(path), AutoImageProcessor.from_pretrained(path)

    calc = Local(device=device)
    return [calc.calculate_DINOv2_similarity_score(gt_im=r["gt"], gen_im=r["png"]) for r in rows]


def ssim_lpips_mse_scores(rows, device):
    mod = upstream("compute_ssim_lpips_mse")
    cache = {}
    make = mod.lpips.LPIPS

    def cached(**kw):  # evaluate_images builds a new LPIPS(net='alex') per pair; build it once.
        key = tuple(sorted(kw.items()))
        if key not in cache:
            cache[key] = make(**kw)
        return cache[key]
    mod.lpips.LPIPS = cached
    out = []
    for r in rows:
        mse, ssim, lp = mod.evaluate_images(r["gt"], r["png"])
        out.append({"mse": mse, "ssim": ssim, "lpips": lp})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--selection", choices=["first", "first-valid"], required=True)
    parser.add_argument("--missing", choices=["exclude", "white"], default="exclude")
    parser.add_argument("--fid-ref", type=Path, help="npz with mu/sigma from fid_reference.py (text tasks)")
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    records = [json.loads(l) for l in (args.run / "results.jsonl").read_text().splitlines() if l.strip()]
    task = records[0]["task"]
    metrics = args.metrics or (TEXT_METRICS if task == "text2svg" else IMAGE_METRICS)
    if "fid" in metrics and not args.fid_ref:
        metrics = [m for m in metrics if m != "fid"]
        print("no --fid-ref given: skipping FID", flush=True)
    # Generation dirs may still be root-owned while their a batch GPU job runs, so results go to a parallel tree.
    out = args.out or RUNS / "2026-10-07-omnisvg-metrics" / args.run.parent.name / \
        f"{args.run.name}-{args.selection}-{args.missing}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = select(records, args.run, args.selection)
    missing = [r["id"] for r in rows if r["png"] is None]
    if args.missing == "white":
        blank = out.parent / "white-1024.png"
        Image.new("RGB", (1024, 1024), (255, 255, 255)).save(blank)
        for r in rows:
            r["png"] = r["png"] or str(blank)
    scored = [r for r in rows if r["png"]]
    summary = {"run": str(args.run), "task": task, "selection": args.selection, "missing_policy": args.missing,
               "n_prompts": len(rows), "n_missing": len(missing), "n_scored": len(scored), "missing_ids": missing}
    for m in metrics:
        print("computing", m, flush=True)
        if m == "fid":
            summary["fid"] = fid_score(scored, args.fid_ref, "cuda")
            continue
        values = globals()[f"{m}_scores"](scored, "cuda")
        if m == "ssim_lpips_mse":
            for k in ("ssim", "lpips", "mse"):
                for r, v in zip(scored, values):
                    r[k] = v[k]
                summary[k] = float(np.mean([v[k] for v in values]))
        else:
            for r, v in zip(scored, values):
                r[m] = v
            summary[m] = float(np.mean(values))
    print(json.dumps({k: v for k, v in summary.items() if k != "missing_ids"}, ensure_ascii=False), flush=True)
    out.write_text(json.dumps({"summary": summary, "per_sample": rows}, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
