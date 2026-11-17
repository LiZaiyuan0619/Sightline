# Progressive preview: when does the drawing become watchable, and when is it nearly done (InternSVG-8B, text-to-SVG, first 40 MMSVG-Icon prompts, greedy)

Similarity = pixel IoU of the ink mask against that method's own final image. Times are seconds.

| run | n | first valid render | 50% of final | 75% | 90% | 95% | full decode |
|---|---|---|---|---|---|---|---|
| ar | 40 | 6.92 | 2.57 | 2.96 | 3.25 | 3.48 | 17.03 |
| ddtree | 40 | 0.66 | 0.39 | 0.45 | 0.48 | 0.49 | 1.86 |
| ours | 40 | 0.64 | 0.31 | 0.37 | 0.45 | 0.46 | 1.46 |

Mean similarity to the final image at a fixed wall clock:

| t (s) | ar | ddtree | ours |
|---|---|---|---|
| 0.1 | 0.000 | 0.069 | 0.204 |
| 0.2 | 0.000 | 0.269 | 0.451 |
| 0.3 | 0.000 | 0.580 | 0.623 |
| 0.5 | 0.000 | 0.734 | 0.785 |
| 0.75 | 0.119 | 0.899 | 0.893 |
| 1.0 | 0.183 | 0.927 | 0.962 |
| 1.5 | 0.289 | 0.974 | 0.989 |
| 2 | 0.517 | 0.974 | 0.995 |
| 3 | 0.766 | 1.000 | 0.998 |
| 4 | 0.813 | 1.000 | 1.000 |
| 6 | 0.913 | 1.000 | 1.000 |
| 8 | 0.917 | 1.000 | 1.000 |
| 12 | 0.996 | 1.000 | 1.000 |
| 16 | 0.998 | 1.000 | 1.000 |
| 24 | 1.000 | 1.000 | 1.000 |
| 32 | 1.000 | 1.000 | 1.000 |
