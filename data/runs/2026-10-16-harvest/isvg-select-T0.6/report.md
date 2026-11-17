# Selection signals on InternSVG-8B text2svg (isvg-bench300-T0.6-ep1; 300 prompts, the 8 batched-AR samples)

Quality of the chosen candidate minus the random pick (mean over the 8), paired per prompt, 95% bootstrap CI. CLIP is the render-score selection signal (not independent for that row); Aesthetic / HPS are independent.

| rule | clip Δ [CI] | aesthetic Δ [CI] | hps Δ [CI] |
|---|---|---|---|
| random (mean of 8) | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| max likelihood | -0.0174 [-0.0217, -0.0133] | -0.0581 [-0.0919, -0.0242] | -0.0059 [-0.0076, -0.0045] |
| min likelihood | +0.0060 [+0.0022, +0.0100] | +0.0485 [+0.0142, +0.0797] | +0.0029 [+0.0015, +0.0042] |
| text consensus (MBR) | +0.0053 [+0.0015, +0.0088] | +0.0408 [+0.0098, +0.0690] | +0.0028 [+0.0016, +0.0039] |
| render score (max CLIP) | +0.0411 [+0.0384, +0.0437] | +0.0687 [+0.0355, +0.1019] | +0.0089 [+0.0077, +0.0103] |

random-pick absolute: clip 0.2848, aesthetic 4.5756, hps 0.2533
