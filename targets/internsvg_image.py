""" the one image preprocessing of InternSVG-8B image-to-SVG in this project (data generation, draft training,
evaluation all call it, so the target sees identical pixels everywhere). SArena's img2svg test inputs are RGB 448 x 448
renders on white; the MMSVG training previews ($SIGHTLINE_RUNS/h3a-data/train-images) are RGBA with a transparent background at 448
or 896 px, and the official loader's plain convert('RGB') would turn that background black. So: alpha-composite onto
white, RGB, resize to 448 x 448 (bicubic, the InternVL image processor's resample) when needed. The model's processor
(GotOcr2ImageProcessorFast, crop_to_patches false) then gives one 448 tile = 256 image tokens."""
from PIL import Image

SIZE = 448


def load_i2s_image(path) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
    return im
