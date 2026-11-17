# bf16 token identity against the bf16 original model, greedy, runs of Figure 4

| block | method | rows with AR ref | token-identical | median first divergence (token) | median first divergence (share of output) |
|---|---|---|---|---|---|
| OmniSVG-8B, text-to-SVG | PLD | 900 | 516 (57%) | 140.0 | 19% |
| OmniSVG-8B, text-to-SVG | REST | 1800 | 1052 (58%) | 131.0 | 16% |
| OmniSVG-8B, text-to-SVG | SpS | 1800 | 1064 (59%) | 127.5 | 17% |
| OmniSVG-8B, text-to-SVG | EAGLE-3 | 900 | 514 (57%) | 127.5 | 19% |
| OmniSVG-8B, text-to-SVG | DFlash | 900 | 537 (60%) | 118 | 15% |
| OmniSVG-8B, text-to-SVG | DDTree | 900 | 532 (59%) | 129.0 | 16% |
| OmniSVG-8B, text-to-SVG | DARTree | 1800 | 1057 (59%) | 132 | 17% |
| OmniSVG-8B, text-to-SVG | Ours (lossless) | 1800 | 1070 (59%) | 131.0 | 17% |
| OmniSVG-4B, text-to-SVG | REST | 1800 | 763 (42%) | 68 | 8% |
| OmniSVG-4B, text-to-SVG | DARTree | 1800 | 732 (41%) | 74.5 | 8% |
| OmniSVG-4B, text-to-SVG | Ours (lossless) | 1800 | 746 (41%) | 69.0 | 8% |
| InternSVG-8B, text-to-SVG | REST | 1800 | 542 (30%) | 64.5 | 4% |
| InternSVG-8B, text-to-SVG | DARTree | 1800 | 523 (29%) | 67 | 5% |
| InternSVG-8B, text-to-SVG | Ours (lossless) | 1800 | 583 (32%) | 67 | 4% |
| OmniSVG-8B, image-to-SVG | PLD | 1800 | 850 (47%) | 135.5 | 14% |
| OmniSVG-8B, image-to-SVG | REST | 1800 | 852 (47%) | 133.5 | 13% |
| OmniSVG-8B, image-to-SVG | EAGLE-3 | 1800 | 857 (48%) | 136 | 14% |
| OmniSVG-8B, image-to-SVG | DFlash | 1800 | 846 (47%) | 138.0 | 14% |
| OmniSVG-8B, image-to-SVG | DDTree | 1800 | 838 (47%) | 132.0 | 14% |
| OmniSVG-8B, image-to-SVG | DARTree | 1800 | 834 (46%) | 136.0 | 15% |
| OmniSVG-8B, image-to-SVG | ViSpec | 1800 | 848 (47%) | 139.5 | 14% |
| OmniSVG-8B, image-to-SVG | Ours (lossless) | 1800 | 854 (47%) | 139.0 | 15% |
| InternSVG-8B, image-to-SVG | PLD | 1800 | 257 (14%) | 81 | 5% |
| InternSVG-8B, image-to-SVG | REST | 1800 | 261 (14%) | 79 | 6% |
| InternSVG-8B, image-to-SVG | EAGLE-3 | 1800 | 276 (15%) | 77.5 | 5% |
| InternSVG-8B, image-to-SVG | DFlash | 1800 | 281 (16%) | 83 | 5% |
| InternSVG-8B, image-to-SVG | DDTree | 1800 | 267 (15%) | 82 | 6% |
| InternSVG-8B, image-to-SVG | DARTree | 1800 | 255 (14%) | 80 | 6% |
| InternSVG-8B, image-to-SVG | ViSpec | 1800 | 300 (17%) | 80.0 | 5% |
| InternSVG-8B, image-to-SVG | Ours (lossless) | 1800 | 269 (15%) | 79 | 5% |
