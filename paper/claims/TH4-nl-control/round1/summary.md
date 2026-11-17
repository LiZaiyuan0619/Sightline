# TH4 per-prompt data, round 1 (11-07)

Pooled over rows (k = 1, offset-1 labels, block-16 footing). Must equal nl_vs_svg.json (asserted in build).

| corpus | rows | positions | miss rate | mean run (tokens) | gap CV | tokens in runs >= 8 / 16 / 32 | tau_4 / tau_8 / tau_16 |
|---|---|---|---|---|---|---|---|
| OmniSVG-8B text-to-SVG, block-16 head | 100 | 51678 | 0.0665 | 14.7 | 3.62 | 0.87 / 0.77 / 0.62 | 3.57 / 6.09 / 9.10 |
| OmniSVG-8B text-to-SVG, block-32 head | 300 | 161696 | 0.0716 | 13.7 | 2.96 | 0.86 / 0.76 / 0.59 | 3.54 / 5.97 / 8.78 |
| Qwen3-8B HumanEval | 164 | 75857 | 0.0658 | 14.7 | 1.79 | 0.88 / 0.74 / 0.54 | 3.58 / 6.18 / 9.46 |
| Qwen3-8B GSM8K | 128 | 37201 | 0.0703 | 13.6 | 1.45 | 0.88 / 0.72 / 0.47 | 3.55 / 6.09 / 9.22 |

Matched pairs (from nl_vs_svg.json): OmniSVG-8B t2svg, b16 head vs Qwen3-8B HumanEval: lambda 0.0665 vs 0.0658, CV 3.62 vs 1.79; OmniSVG-8B t2svg, b32 head vs Qwen3-8B GSM8K: lambda 0.0716 vs 0.0703, CV 2.96 vs 1.45
