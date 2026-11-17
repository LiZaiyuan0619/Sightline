# Environments

Seven environments, because the upstream repositories this work builds on pin incompatible
versions of `transformers`: StarVector needs 4.49, the OmniSVG metrics need 4.51, our decoder and
training need 5.15, and the serving engine needs 5.19. Each `requirements-*.txt` is the exact
package list of the environment a given stage ran in, exported from it.

| environment | used for | key versions |
|---|---|---|
| `dflash` | text-to-SVG decoding, draft-head training, the offline simulator | torch 2.13, transformers 5.15 |
| `glance` | image-to-SVG decoding (the vision-language targets), rendering | torch 2.13, transformers 5.15, cairosvg 2.9 |
| `specforge` | the third-party trainer our training loop is built around | torch 2.13, transformers 5.12.1, sglang 0.5.18 |
| `sglang` | the serving experiment (Figure 6l) | torch 2.14.1, transformers 5.19, sglang 0.5.22 |
| `svgmetrics` | the official SVG metrics, CLIP and DINOv2 scoring | torch 2.3, transformers 4.51.3, lpips 0.1.4 |
| `starvector` | the StarVector reference dump (generality experiment only) | torch 2.3, transformers 4.49 |
| `omnisvg` | the official OmniSVG stack, for batch generation of training data | torch 2.3, transformers 4.51.3 |

```bash
conda create -p ./envs/dflash python=3.12 -y
conda activate ./envs/dflash
pip install -r env/requirements-dflash.txt
```

Three packages in these lists are the upstream projects themselves, installed from source rather
than from an index: `dflash`, `specforge` and `sglang`. `docs/third_party.md` says where each
comes from and which of our patches to apply. Install them from their own repositories, then
install the rest of the list.

Nothing in this package needs all seven. The checks in the README's first section
(`paper/check_paper_numbers.py`, `paper/` table builders, `tests/`) run on a plain Python 3.10+
with `numpy`; only decoding, training and metrics need a GPU environment.
