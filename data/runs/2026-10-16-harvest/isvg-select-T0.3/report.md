# Selection signals on InternSVG-8B text2svg (isvg-bench300-T0.3-ep1; 300 prompts, the 8 batched-AR samples)

Quality of the chosen candidate minus the random pick (mean over the 8), paired per prompt, 95% bootstrap CI. CLIP is the render-score selection signal (not independent for that row); Aesthetic / HPS are independent.

| rule | clip Δ [CI] | aesthetic Δ [CI] | hps Δ [CI] |
|---|---|---|---|
| random (mean of 8) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| max likelihood | -0.0139 [-0.0180, -0.0100] | -0.0685 [-0.0975, -0.0405] | -0.0056 [-0.0071, -0.0042] |
| min likelihood | +0.0085 [+0.0046, +0.0122] | +0.0301 [-0.0037, +0.0620] | +0.0025 [+0.0011, +0.0039] |
| text consensus (MBR) | +0.0045 [+0.0011, +0.0079] | +0.0207 [-0.0050, +0.0481] | +0.0028 [+0.0017, +0.0040] |
| render score (max CLIP) | +0.0397 [+0.0367, +0.0427] | +0.0698 [+0.0361, +0.1036] | +0.0088 [+0.0076, +0.0101] |

random-pick absolute: clip 0.2734, aesthetic 4.5419, hps 0.2500
