# How the experiments were run

The job scripts that submitted these runs to a particular cluster are not included: they are
scheduler-specific and say nothing about the method. This file records what the runs cost and how
they were split, so the work can be planned on another machine.

## Hardware

Every timing in the paper is one H200, bf16, eager attention, batch size 1 (except best-of-N,
which batches its N samples, and the serving experiment). A speedup always divides the original
model's time per token, measured in the same job on the same prompts, by the decoder's own.
Seconds from different jobs are put on one node by the node factors of Appendix E.4, and the
only times from other hardware are those InternSVG reports for other generators (Appendix F.3).

## Shape of the work

| stage | unit | roughly |
|---|---|---|
| training data | one target, 300k prompts, one sample each | 8-16 GPU-hours, shardable |
| one draft head | block length 32, 6 epochs on 300k samples | 6-12 GPU-hours on 4 GPUs |
| one main-table cell | one decoder, one benchmark set, one mode (150-450 prompts) | 0.5-3 GPU-hours |
| the full main table | 9 decoders x 6 sets x 2 modes x 5 blocks | the bulk of the compute |
| the quality table | 4 blocks x the full SArena test sets (8,014 prompts) | split into 80 one-GPU jobs |
| fp32 audits | 10 prompts per design | minutes |
| the theory measurements | offline, from cached draft distributions | CPU, after one caching pass per head |

## Sharding

Every evaluator takes `--shard i/n` and selects rows by `row index % n == i`, so a block can be
split across as many jobs as convenient and merged afterwards with `eval/merge.py`. Two rules
matter for the numbers to stay comparable:

1. **The original model must be timed in the same job as the decoder it is compared with**
   (`--time-ar`), or its time must come from a cache measured on the same node for the same rows.
   `eval/eval_text.py` refuses to mix the two.
2. **One seed per prompt**, `42 + row index`, the same for every decoder. Re-running a shard
   reproduces its outputs.

## Offline work needs no GPU

The theory measurements and the mechanism statistics read a cached draft distribution instead of
re-decoding: `sightline/cache_draft.py` writes the cache once per head on a GPU, and everything in
`analysis/` reads it on CPU. `sightline/design.py` then simulates any tree design exactly from the
cache, which is how the paper compares designs without running each of them.
