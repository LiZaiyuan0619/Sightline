# numbers behind round 1 (one candidate per position, block-16 footing)

| corpus | prompts plotted (of) | pooled miss rate | pooled gap CV | median per-prompt CV (n) | runs of 1-2 tokens (share of runs) | tokens in runs >= 16 / 32 / 64 | gain 8 -> 16 (pooled) | decision density | tokens in stretches >= 16 | g(15)/g(1) |
|---|---|---|---|---|---|---|---|---|---|---|
| SVG, block-16 head | 89 (90) | 6.6% | 3.62 | 1.55 (63) | 40% | 77 / 62 / 48% | ×1.494 | 14.6% | 67% | 0.967 |
| HumanEval | 164 (164) | 6.6% | 1.79 | 1.55 (164) | 26% | 74 / 54 / 33% | ×1.531 | 24.6% | 56% | 0.840 |
| SVG, block-32 head | 270 (277) | 7.2% | 2.96 | 1.66 (206) | 40% | 76 / 59 / 43% | ×1.471 | 14.8% | 66% | 0.975 |
| GSM8K | 128 (128) | 7.0% | 1.45 | 1.29 (127) | 25% | 72 / 47 / 17% | ×1.515 | 21.2% | 43% | 0.771 |

All pooled values equal nl_vs_svg.json (asserted in make.py).
