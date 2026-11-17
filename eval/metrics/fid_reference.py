"""FID reference statistics for the text-to-SVG benchmark (issue #39: 3% of MMSVG-{Icon,Illustration}, seed 42).

Reproduces download_gt_dataset + extract_features_from_pil_images from refs/OmniSVG/metrics/compute_fid.py,
but loads a pinned local snapshot instead of the moving hub head, and saves the sampled indices and ids.
The MMSVG PNGs are RGBA with black (0,0,0) under transparent pixels, so upstream's convert('RGB') yields
black-background references against white-background generations. Default = upstream behaviour (for comparison
with the paper); --white-bg composites onto white first (sanity variant, report separately).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import argparse
import os
import sys
from pathlib import Path
from sightline.paths import REFS

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT.parents[1] / "model/torch-hub"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/hf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="local snapshot of OmniSVG/MMSVG-*")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-percentage", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--white-bg", action="store_true", help="composite RGBA references onto white")
    args = parser.parse_args()
    import numpy as np
    from datasets import load_dataset
    sys.path.insert(0, str(REFS / "OmniSVG/metrics"))
    import compute_fid as mod

    ds = load_dataset(str(args.data_dir))
    data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    total = len(data)
    size = int(total * args.sample_percentage)
    np.random.seed(args.seed)
    indices = sorted(np.random.choice(total, size=size, replace=False))
    image_col = next(c for c in ["image", "png", "img", "picture", "rendered_image"] if c in data.column_names)
    subset = data.select([int(i) for i in indices])
    images, ids = list(subset[image_col]), list(subset["id"])
    if args.white_bg:
        from PIL import Image
        def white(im):
            if im.mode != "RGBA":
                return im
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            return bg
        images = [white(im) for im in images]
    print(f"total={total} sampled={size} column={image_col} modes={sorted({im.mode for im in images[:200]})}", flush=True)
    feats = mod.InceptionV3Feature("cuda").extract_features_from_pil_images(images, batch_size=50)
    mu, sigma = mod.calculate_activation_statistics(feats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, mu=mu, sigma=sigma, indices=np.array(indices), ids=np.array(ids), total=total,
             n_features=len(feats), white_bg=args.white_bg)
    print(f"saved {args.out}: {len(feats)} features", flush=True)


if __name__ == "__main__":
    main()
