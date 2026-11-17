# Progressive preview: when does the drawing become watchable, and when is it nearly done

Similarity = pixel IoU of the ink mask against that method's own final image. Times are seconds.

| run | n | first valid render | 50% of final | 75% | 90% | 95% | full decode |
|---|---|---|---|---|---|---|---|
| ar | 40 | 0.79 | 1.19 | 1.74 | 2.34 | 2.52 | 8.50 |
| ours-ets | 40 | 0.09 | 0.24 | 0.35 | 0.40 | 0.43 | 0.73 |
| ours | 40 | 0.15 | 0.22 | 0.32 | 0.36 | 0.38 | 0.96 |

Mean similarity to the final image at a fixed wall clock:

| t (s) | ar | ours-ets | ours |
|---|---|---|---|
| 0.1 | 0.025 | 0.466 | 0.474 |
| 0.2 | 0.122 | 0.686 | 0.697 |
| 0.3 | 0.370 | 0.732 | 0.743 |
| 0.5 | 0.459 | 0.789 | 0.819 |
| 0.75 | 0.559 | 0.907 | 0.951 |
| 1.0 | 0.620 | 0.964 | 0.977 |
| 1.5 | 0.759 | 0.995 | 0.994 |
| 2 | 0.801 | 1.000 | 1.000 |
| 3 | 0.883 | 1.000 | 1.000 |
| 4 | 0.938 | 1.000 | 1.000 |
| 6 | 0.964 | 1.000 | 1.000 |
| 8 | 0.981 | 1.000 | 1.000 |
| 12 | 0.987 | 1.000 | 1.000 |
| 16 | 0.996 | 1.000 | 1.000 |
| 24 | 0.999 | 1.000 | 1.000 |
| 32 | 0.999 | 1.000 | 1.000 |
