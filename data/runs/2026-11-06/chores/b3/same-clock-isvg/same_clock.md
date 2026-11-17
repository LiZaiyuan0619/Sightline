# Same wall clock: what each method has delivered at instant t (InternSVG-8B, text-to-SVG, the first 40 MMSVG-Icon prompts of table 1, greedy, one H200; lanes ar / ddtree / ours)

Progress = pixel IoU of the rendered prefix against that method's own final image; a prompt past its own decode time counts as finished (1.0). Rows are the preview records of src/analysis/preview_curve.py.

| t (s) | ar done | ar mean progress | ours done | ours mean progress | ddtree done | ddtree mean progress |
|---|---|---|---|---|---|---|
| 0.25 | 0/40 | 0.000 | 9/40 | 0.541 | 8/40 | 0.461 |
| 0.5 | 0/40 | 0.000 | 17/40 | 0.780 | 19/40 | 0.709 |
| 0.75 | 0/40 | 0.119 | 24/40 | 0.893 | 25/40 | 0.871 |
| 1.0 | 1/40 | 0.183 | 25/40 | 0.962 | 26/40 | 0.927 |
| 1.5 | 4/40 | 0.289 | 25/40 | 0.989 | 28/40 | 0.974 |
| 2.0 | 12/40 | 0.517 | 27/40 | 0.970 | 28/40 | 0.949 |
| 3.0 | 18/40 | 0.766 | 29/40 | 0.994 | 29/40 | 0.975 |
| 4.0 | 21/40 | 0.788 | 39/40 | 1.000 | 29/40 | 0.975 |
| 6.0 | 23/40 | 0.913 | 40/40 | 1.000 | 39/40 | 1.000 |
| 8.0 | 23/40 | 0.915 | 40/40 | 1.000 | 40/40 | 1.000 |
| 12.0 | 26/40 | 0.965 | 40/40 | 1.000 | 40/40 | 1.000 |
| 16.0 | 26/40 | 0.994 | 40/40 | 1.000 | 40/40 | 1.000 |

| run | mean decode (s) | finished drawings per prompt-minute |
|---|---|---|
| ar | 17.03 | 3.5 |
| ours | 1.46 | 41.1 |
| ddtree | 1.86 | 32.3 |
