# Same wall clock: what each method has delivered at instant t (OmniSVG-8B, image-to-SVG, the first 40 MMSVG-Icon prompts of table 1, greedy, one H200; lanes ar / ddtree / ours)

Progress = pixel IoU of the rendered prefix against that method's own final image; a prompt past its own decode time counts as finished (1.0). Rows are the preview records of src/analysis/preview_curve.py.

| t (s) | ar done | ar mean progress | ours done | ours mean progress | ddtree done | ddtree mean progress |
|---|---|---|---|---|---|---|
| 0.25 | 0/40 | 0.156 | 1/40 | 0.552 | 1/40 | 0.581 |
| 0.5 | 0/40 | 0.408 | 7/40 | 0.706 | 8/40 | 0.705 |
| 0.75 | 0/40 | 0.518 | 12/40 | 0.813 | 11/40 | 0.838 |
| 1.0 | 0/40 | 0.587 | 17/40 | 0.892 | 17/40 | 0.902 |
| 1.5 | 1/40 | 0.707 | 25/40 | 0.927 | 26/40 | 0.915 |
| 2.0 | 8/40 | 0.741 | 32/40 | 0.964 | 29/40 | 0.964 |
| 3.0 | 11/40 | 0.821 | 37/40 | 0.999 | 37/40 | 0.989 |
| 4.0 | 12/40 | 0.907 | 39/40 | 1.000 | 39/40 | 0.998 |
| 6.0 | 20/40 | 0.935 | 40/40 | 1.000 | 40/40 | 1.000 |
| 8.0 | 26/40 | 0.977 | 40/40 | 1.000 | 40/40 | 1.000 |
| 12.0 | 31/40 | 0.972 | 40/40 | 1.000 | 40/40 | 1.000 |
| 16.0 | 32/40 | 0.981 | 40/40 | 1.000 | 40/40 | 1.000 |

| run | mean decode (s) | finished drawings per prompt-minute |
|---|---|---|
| ar | 8.19 | 7.3 |
| ours | 1.39 | 43.2 |
| ddtree | 1.47 | 40.7 |
