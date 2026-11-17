# Same wall clock: what each method has delivered at instant t (OmniSVG-8B, text-to-SVG, the first 40 MMSVG-Icon prompts of table 1, greedy, one H200; lanes ar / ddtree / ours)

Progress = pixel IoU of the rendered prefix against that method's own final image; a prompt past its own decode time counts as finished (1.0). Rows are the preview records of src/analysis/preview_curve.py.

| t (s) | ar done | ar mean progress | ours done | ours mean progress | ddtree done | ddtree mean progress |
|---|---|---|---|---|---|---|
| 0.25 | 0/40 | 0.172 | 13/40 | 0.705 | 10/40 | 0.635 |
| 0.5 | 0/40 | 0.427 | 15/40 | 0.816 | 14/40 | 0.809 |
| 0.75 | 0/40 | 0.531 | 20/40 | 0.949 | 18/40 | 0.922 |
| 1.0 | 2/40 | 0.566 | 21/40 | 0.973 | 21/40 | 0.950 |
| 1.5 | 7/40 | 0.693 | 30/40 | 0.990 | 22/40 | 0.964 |
| 2.0 | 13/40 | 0.750 | 37/40 | 1.000 | 23/40 | 0.992 |
| 3.0 | 17/40 | 0.882 | 40/40 | 1.000 | 39/40 | 1.000 |
| 4.0 | 18/40 | 0.927 | 40/40 | 1.000 | 40/40 | 1.000 |
| 6.0 | 23/40 | 0.961 | 40/40 | 1.000 | 40/40 | 1.000 |
| 8.0 | 24/40 | 0.981 | 40/40 | 1.000 | 40/40 | 1.000 |
| 12.0 | 24/40 | 0.988 | 40/40 | 1.000 | 40/40 | 1.000 |
| 16.0 | 25/40 | 0.996 | 40/40 | 1.000 | 40/40 | 1.000 |

| run | mean decode (s) | finished drawings per prompt-minute |
|---|---|---|
| ar | 8.56 | 7.0 |
| ours | 0.96 | 62.2 |
| ddtree | 1.28 | 46.8 |
