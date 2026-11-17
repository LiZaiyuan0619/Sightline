# Table 3 (generality): six SVG / Lottie models, six decoding methods, official decoding settings

Same training budget for every trained draft (100k rows x 2 epochs, global batch 16) and, per method, the design it has in table 1. Speed-up = per-token wall clock against the AR of the row's own job (same node); tau = accepted tokens per verification round over all tokens. Sampling = each model's official setting.

| Model | Backbone, tokens | Decoding | Subset | PLD | DFlash | EAGLE-3 | DDTree | DARTree | Ours | tau PLD / DFlash / EAGLE-3 / DDTree / DARTree / Ours | Ours / DDTree | rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVGen-3B | Llama-3.2-3B, raw SVG code (text-to-SVG) | greedy | icon | 1.39× | 4.96× | 2.93× | 6.61× | 5.47× | **7.37×** | 1.61 / 7.34 / 8.87 / 10.27 / 10.72 / 12.00 | 1.11 | 300/300/300/300/300/300 |
| SVGen-3B | Llama-3.2-3B, raw SVG code (text-to-SVG) | greedy | illustration | 1.48× | 5.34× | 3.16× | 6.95× | 5.95× | **8.10×** | 1.69 / 7.95 / 9.40 / 10.90 / 11.42 / 13.16 | 1.16 | 300/300/300/300/300/300 |
| SVGen-3B | Llama-3.2-3B, raw SVG code (text-to-SVG) | sample (T 0.2, top-p 0.7) | icon | 1.42× | 4.32× | 2.88× | 5.65× | 5.14× | **6.34×** | 1.63 / 7.16 / 8.90 / 10.21 / 10.86 / 11.67 | 1.12 | 300/300/300/300/300/300 |
| SVGen-3B | Llama-3.2-3B, raw SVG code (text-to-SVG) | sample (T 0.2, top-p 0.7) | illustration | 1.46× | 4.52× | 3.00× | 5.88× | 5.43× | **6.86×** | 1.68 / 7.62 / 9.33 / 10.86 / 11.54 / 12.87 | 1.17 | 300/300/300/300/300/300 |
| SVGen-7B | Qwen2.5-Coder-7B, raw SVG code (text-to-SVG) | greedy | icon | 1.47× | 4.94× | 3.41× | 6.98× | 6.44× | **8.25×** | 1.69 / 7.51 / 10.12 / 11.34 / 12.41 / 13.63 | 1.18 | 300/300/300/300/300/300 |
| SVGen-7B | Qwen2.5-Coder-7B, raw SVG code (text-to-SVG) | greedy | illustration | 1.49× | 4.74× | 3.39× | 6.72× | 6.26× | **7.81×** | 1.72 / 7.20 / 9.96 / 11.04 / 12.16 / 12.95 | 1.16 | 300/300/300/300/300/300 |
| SVGen-7B | Qwen2.5-Coder-7B, raw SVG code (text-to-SVG) | sample (T 0.2, top-p 0.7) | icon | 1.46× | 4.42× | 3.28× | 5.98× | 5.68× | **7.07×** | 1.69 / 7.42 / 10.01 / 11.20 / 12.29 / 13.66 | 1.18 | 300/300/300/300/300/300 |
| SVGen-7B | Qwen2.5-Coder-7B, raw SVG code (text-to-SVG) | sample (T 0.2, top-p 0.7) | illustration | 1.49× | 4.27× | 3.28× | 5.86× | 5.61× | **6.74×** | 1.72 / 7.16 / 10.05 / 11.07 / 12.09 / 13.01 | 1.15 | 300/300/300/300/300/300 |
| HiVG-3B | Qwen2.5-VL-3B, hierarchical compressed tokens (text-to-SVG) | greedy | icon | 3.93× | 8.48× | 4.46× | 9.26× | 8.29× | **13.78×** | 4.61 / 12.09 / 11.91 / 13.85 / 15.17 / 21.66 | 1.49 | 300/300/300/300/300/300 |
| HiVG-3B | Qwen2.5-VL-3B, hierarchical compressed tokens (text-to-SVG) | greedy | illustration | 5.14× | 9.65× | 4.76× | 9.90× | 8.70× | **15.97×** | 6.05 / 13.77 / 12.47 / 14.87 / 15.85 / 25.11 | 1.61 | 300/300/300/300/300/300 |
| HiVG-3B | Qwen2.5-VL-3B, hierarchical compressed tokens (text-to-SVG) | sample (T 0.7, top-p 0.9, top-k 50) | icon | 2.74× | 5.57× | 3.77× | 6.72× | 6.04× | **7.99×** | 3.32 / 8.90 / 10.50 / 11.72 / 12.51 / 14.87 | 1.19 | 300/300/300/300/300/300 |
| HiVG-3B | Qwen2.5-VL-3B, hierarchical compressed tokens (text-to-SVG) | sample (T 0.7, top-p 0.9, top-k 50) | illustration | 3.00× | 5.78× | 3.89× | 7.04× | 6.33× | **8.83×** | 3.68 / 9.22 / 10.90 / 12.28 / 13.26 / 16.87 | 1.26 | 300/300/300/300/300/300 |
| OmniLottie-3B | Qwen2.5-VL-3B, Lottie JSON tokens (text-to-Lottie) | greedy | real | 2.07× | 5.62× | 3.33× | 7.00× | 6.53× | **8.61×** | 2.41 / 8.14 / 8.70 / 10.64 / 11.43 / 13.46 | 1.23 | 150/150/150/150/150/150 |
| OmniLottie-3B | Qwen2.5-VL-3B, Lottie JSON tokens (text-to-Lottie) | greedy | synthetic | 1.92× | 4.64× | 3.44× | 6.71× | 6.19× | **7.73×** | 2.24 / 6.64 / 8.62 / 10.16 / 10.70 / 11.94 | 1.15 | 150/150/150/150/150/150 |
| OmniLottie-3B | Qwen2.5-VL-3B, Lottie JSON tokens (text-to-Lottie) | sample (T 0.9, top-p 0.25, top-k 5) | real | 2.01× | 4.95× | 3.16× | 6.15× | 5.73× | **7.18×** | 2.40 / 7.97 / 8.51 / 10.90 / 11.44 / 13.39 | 1.17 | 150/150/150/150/150/150 |
| OmniLottie-3B | Qwen2.5-VL-3B, Lottie JSON tokens (text-to-Lottie) | sample (T 0.9, top-p 0.25, top-k 5) | synthetic | 1.94× | 4.73× | 3.07× | 6.19× | 5.41× | **6.83×** | 2.32 / 7.59 / 8.45 / 10.99 / 10.94 / 12.57 | 1.10 | 150/150/150/150/150/150 |
| StarVector-1B | StarCoderBase-1B + CLIP, raw SVG code (image-to-SVG) | greedy | icon | 1.99× | 4.14× | 1.90× | 5.18× | 4.23× | **6.35×** | 2.48 / 8.41 / 10.27 / 11.53 / 12.98 / 14.60 | 1.23 | 300/300/300/300/300/300 |
| StarVector-1B | StarCoderBase-1B + CLIP, raw SVG code (image-to-SVG) | greedy | illustration | 2.78× | 5.20× | 2.11× | 5.87× | 4.72× | **7.98×** | 3.48 / 10.59 / 11.31 / 13.19 / 14.49 / 18.46 | 1.36 | 300/300/300/300/300/300 |
| StarVector-1B | StarCoderBase-1B + CLIP, raw SVG code (image-to-SVG) | sample (T 0.2, top-p 0.95, top-k 50) | icon | 1.75× | 3.37× | 1.82× | 4.23× | 3.59× | **4.84×** | 2.28 / 7.61 / 9.85 / 10.89 / 12.15 / 13.52 | 1.14 | 300/300/300/300/300/300 |
| StarVector-1B | StarCoderBase-1B + CLIP, raw SVG code (image-to-SVG) | sample (T 0.2, top-p 0.95, top-k 50) | illustration | 2.35× | 4.05× | 1.99× | 4.62× | 3.92× | **5.63×** | 3.14 / 9.44 / 10.58 / 12.28 / 13.27 / 16.32 | 1.22 | 300/300/300/300/300/300 |
| StarVector-8B | StarCoder2-7B + SigLIP, raw SVG code (image-to-SVG) | greedy | icon | 2.14× | 4.69× | 2.82× | 5.73× | 4.75× | **6.71×** | 2.86 / 8.10 / 9.70 / 10.83 / 11.11 / 12.80 | 1.17 | 300/300/300/300/300/300 |
| StarVector-8B | StarCoder2-7B + SigLIP, raw SVG code (image-to-SVG) | greedy | illustration | 2.86× | 4.93× | 2.86× | 5.16× | 4.94× | **6.54×** | 3.96 / 8.79 / 9.83 / 10.04 / 11.70 / 12.91 | 1.27 | 300/300/300/300/300/300 |
| StarVector-8B | StarCoder2-7B + SigLIP, raw SVG code (image-to-SVG) | sample (T 0.7, top-p 0.95, top-k 50) | icon | 1.72× | 2.43× | 1.95× | 3.41× | 2.81× | **3.35×** | 2.30 / 4.37 / 6.73 / 6.80 / 6.70 / 6.75 | 0.98 | 300/300/300/300/300/300 |
| StarVector-8B | StarCoder2-7B + SigLIP, raw SVG code (image-to-SVG) | sample (T 0.7, top-p 0.95, top-k 50) | illustration | 1.96× | 2.46× | 1.91× | 3.20× | 3.05× | **3.71×** | 2.71 / 4.49 / 6.54 / 6.50 / 7.35 / 7.66 | 1.16 | 300/300/300/300/300/300 |

Ours is the fastest method in 23 of 24 complete cells: the exceptions are StarVector-8B sample icon: DDTree 3.41× vs Ours 3.35×.

## Appendix: StarVector sampling at T 1.0 / top-p 0.9 (generate_im2svg's code defaults without its beams)

These were the table's StarVector sampling rows until 11-04; at this temperature every method's tau falls to about 2.9 on StarVector-8B illustrations.

| Model | Backbone, tokens | Decoding | Subset | PLD | DFlash | EAGLE-3 | DDTree | DARTree | Ours | tau PLD / DFlash / EAGLE-3 / DDTree / DARTree / Ours | Ours / DDTree | rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| StarVector-1B | StarCoderBase-1B + CLIP, raw SVG code (image-to-SVG) | sample (T 1.0, top-p 0.9) | icon | 1.51× | 2.30× | 1.43× | 3.44× | 2.72× | **3.65×** | 1.88 / 4.72 / 7.38 / 8.07 / 8.41 / 8.65 | 1.06 | 300/300/300/300/300/300 |
| StarVector-1B | StarCoderBase-1B + CLIP, raw SVG code (image-to-SVG) | sample (T 1.0, top-p 0.9) | illustration | 1.52× | 1.99× | 1.32× | 3.15× | 2.39× | **3.25×** | 1.89 / 4.00 / 6.83 / 7.26 / 7.33 / 7.62 | 1.03 | 300/300/300/300/300/300 |
| StarVector-8B | StarCoder2-7B + SigLIP, raw SVG code (image-to-SVG) | sample (T 1.0, top-p 0.9) | icon | 1.47× | 1.72× | 1.23× | 2.33× | 1.84× | **2.34×** | 2.05 / 3.05 / 4.17 / 4.58 / 4.39 / 4.60 | 1.00 | 300/300/300/300/300/300 |
| StarVector-8B | StarCoder2-7B + SigLIP, raw SVG code (image-to-SVG) | sample (T 1.0, top-p 0.9) | illustration | 1.56× | 1.19× | 0.85× | 1.48× | 1.25× | **1.46×** | 2.20 / 2.12 / 2.86 / 2.91 / 2.90 / 2.88 | 0.99 | 300/300/300/300/300/300 |

Ours is the fastest method in 3 of 4 complete cells: the exceptions are StarVector-8B sample-T1.0 illustration: PLD 1.56× vs Ours 1.46×.
