# Third-party code

No upstream repository is copied into this package. Each one below is named with the commit we
built against, its licence, and what we did with it. Clone them yourself and point
`$SIGHTLINE_REFS` at the parent directory; the scripts that need one say so in their docstring.

Every repository is the official release of the paper or project named, and the commit identifies
the revision exactly.

## Used at run time

| project | commit | licence | how we use it |
|---|---|---|---|
| DFlash ([paper](https://openreview.net/forum?id=Oz335dV48X), `dflash` package) | released wheel 0.1.0 | MIT | the block draft-head architecture and its training objective. Our heads are this architecture, trained by `train/train_block_head.py` on each target's own outputs |
| [SpecForge](https://github.com/sgl-project/SpecForge) | `ed64d27` | MIT | the trainer our training loop is built around: the draft model, the online objective, the bf16 optimiser. Three patches in `train/patches/` |
| [GLANCE](https://github.com/js-lee-AI/GLANCE) | `2818244` | MIT | **the only code we include verbatim**: the four tree utilities in `sightline/tree_utils.py` (node record, budget-tree construction, ancestor mask, KV gather), with the MIT notice reproduced in that file. Including them lets the decoder run without a clone |
| [SGLang](https://github.com/sgl-project/sglang) | `0.5.22` (released) | Apache-2.0 | the serving experiment of Figure 6l. One patch in `train/patches/` adds the block-head capture hook |
| [OmniSVG](https://github.com/OmniSVG/OmniSVG) | `308b49b` | Apache-2.0 | target model: its official preprocessing, decoding configuration and evaluation metrics. `eval/metrics/official_metrics.py` calls the upstream metric implementations |
| [InternSVG](https://github.com/hmwang2002/InternSVG) | `1f20a72` | Apache-2.0 | target model and the SArena benchmark protocol |
| [StarVector](https://github.com/joanrod/star-vector) | `0e083c1` | Apache-2.0 | target model of the generality experiment; `targets/extra/starvector_official.py` loads it through the official code |
| [OmniLottie](https://huggingface.co/OmniLottie/OmniLottie) | (release) | Apache-2.0 | target model of the generality experiment (Lottie JSON output) |
| [SVGen](https://github.com/gitcat-404/SVGen) | `f0454ef` | not stated | target model of the generality experiment |
| [HiVG](https://github.com/ximinng/HiVG) | `2ada95d` | MIT | target model of the generality experiment |

## Baselines we reimplemented

Each of these is our own implementation of a published method, written against its paper and its
released code, so that every baseline runs inside the same verifier as our method and is timed in
the same job (Appendix E.2 of the paper). They are in `baselines/`.

| baseline | our file | lines | note |
|---|---|---|---|
| PLD | `baselines/pld.py` | 60 | the semantics of the `prompt_lookup_num_tokens` option in the transformers library, with the most recent match rather than the first |
| REST | `baselines/rest.py` | 175 | datastore construction and retrieval drafting |
| SpS | `baselines/sps.py` | 85 | classic speculative sampling with a smaller model of the same family |
| EAGLE-3 | `baselines/eagle3.py` | 215 | drafting against our verifier; the head is trained by `train/train_eagle3.py` using SpecForge's objective |
| [ViSpec](https://github.com/KangJialiang/ViSpec) | `baselines/vispec/` | 876 | the two-stage recipe and the image-compression head |
| [DARTree](https://github.com/VILA-Lab/DARTree) | `baselines/dartree.py` | 182 | the autoregressive correction head over a block draft |

DARTree's released repository (`5bb3240`) states no licence. Our port is an independent
implementation: of its 123 substantive lines, 6 also occur in the upstream source, and the
longest identical run is 3 lines. Those six are `import` boilerplate and the standard GRU
update equations, which are the same in any implementation. We measured this before release.

## Datasets and benchmarks

MMSVG, MMSVG-2M and SArena are used under their own terms, from the releases of the OmniSVG and
InternSVG projects. This package ships **prompt identifiers only** (`data/prompts/`), never
benchmark content, so that a reader can reproduce the exact prompt selection from their own copy
of each benchmark.

The one piece of data we generated ourselves and do include is `data/descriptions/`: a
one-sentence description of each of the 300 MMSVG image-to-SVG inputs, written by a DeepSeek
model at temperature 0 (`eval/rows/write_image_descriptions.py`), because MMSVG ships those
images without text and the InternSVG image-to-SVG prompt needs one. The paper discloses this in
Appendix E.1 and in its LLM usage statement.

## This repository

Released under the MIT License (`LICENSE`). `sightline/tree_utils.py` keeps GLANCE's MIT notice.
