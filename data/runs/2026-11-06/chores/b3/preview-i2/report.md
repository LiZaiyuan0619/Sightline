# Progressive preview: when does the drawing become watchable, and when is it nearly done (OmniSVG-8B, image-to-SVG, first 40 MMSVG-Icon prompts, greedy)

Similarity = pixel IoU of the ink mask against that method's own final image. Times are seconds.

| run | n | first valid render | 50% of final | 75% | 90% | 95% | full decode |
|---|---|---|---|---|---|---|---|
| ar | 40 | 0.36 | 1.13 | 2.01 | 3.31 | 4.00 | 8.19 |
| ddtree | 40 | 0.09 | 0.28 | 0.60 | 0.82 | 0.99 | 1.47 |
| ours | 40 | 0.09 | 0.36 | 0.56 | 0.81 | 0.94 | 1.39 |

Mean similarity to the final image at a fixed wall clock:

| t (s) | ar | ddtree | ours |
|---|---|---|---|
| 0.1 | 0.000 | 0.354 | 0.358 |
| 0.2 | 0.063 | 0.548 | 0.511 |
| 0.3 | 0.194 | 0.667 | 0.637 |
| 0.5 | 0.416 | 0.765 | 0.748 |
| 0.75 | 0.525 | 0.869 | 0.866 |
| 1.0 | 0.620 | 0.919 | 0.907 |
| 1.5 | 0.741 | 0.938 | 0.944 |
| 2 | 0.794 | 0.966 | 0.964 |
| 3 | 0.868 | 0.992 | 0.999 |
| 4 | 0.914 | 0.998 | 1.000 |
| 6 | 0.955 | 1.000 | 1.000 |
| 8 | 0.979 | 1.000 | 1.000 |
| 12 | 0.991 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 | 1.000 |
| 24 | 1.000 | 1.000 | 1.000 |
| 32 | 1.000 | 1.000 | 1.000 |
