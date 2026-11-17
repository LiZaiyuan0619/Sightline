# Selection signals on InternSVG-8B text2svg (isvg-bench300-T1.0-ep1; 300 prompts, the 8 batched-AR samples)

Quality of the chosen candidate minus the random pick (mean over the 8), paired per prompt, 95% bootstrap CI. CLIP is the render-score selection signal (not independent for that row); Aesthetic / HPS are independent.

| rule | clip Δ [CI] | aesthetic Δ [CI] | hps Δ [CI] |
|---|---|---|---|
| random (mean of 8) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| max likelihood | -0.0058 [-0.0095, -0.0019] | +0.0352 [+0.0015, +0.0708] | -0.0014 [-0.0027, +0.0000] |
| min likelihood | +0.0004 [-0.0029, +0.0036] | -0.0389 [-0.0730, -0.0046] | -0.0021 [-0.0034, -0.0009] |
| text consensus (MBR) | +0.0020 [-0.0010, +0.0046] | +0.0112 [-0.0189, +0.0416] | +0.0024 [+0.0014, +0.0034] |
| render score (max CLIP) | +0.0365 [+0.0346, +0.0384] | +0.0257 [-0.0061, +0.0597] | +0.0067 [+0.0057, +0.0078] |

random-pick absolute: clip 0.2927, aesthetic 4.5625, hps 0.2555
