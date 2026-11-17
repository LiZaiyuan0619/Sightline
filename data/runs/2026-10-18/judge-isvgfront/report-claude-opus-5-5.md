# VLM-judge preference (claude-opus-5-5; runs/2026-10-18/judge-isvgfront)

pref(A) = mean over prompts of (1 A wins, 0.5 tie, 0 B wins), averaged over both orders; 95% bootstrap CI. consistent = both orders give the same verdict.

| pair A:B | prompts | pref(A) | 95% CI | A wins / ties / B wins (per order) | consistent | unparsed |
|---|---|---|---|---|---|---|
| ar2s:ar1s | 300 | 0.602 | [0.552, 0.652] | 322 / 78 / 200 | 279/300 | 0 |
| ar4s:ar1s | 300 | 0.650 | [0.598, 0.698] | 366 / 48 / 186 | 283/300 | 0 |
| ar8s:ar1s | 300 | 0.689 | [0.634, 0.739] | 400 / 27 / 173 | 289/300 | 0 |
| sd16:ar1s | 300 | 0.699 | [0.650, 0.748] | 411 / 17 / 172 | 283/300 | 0 |
| sd1s:ar1s | 300 | 0.517 | [0.468, 0.562] | 242 / 136 / 222 | 286/300 | 0 |
| sd2:ar1s | 300 | 0.614 | [0.564, 0.665] | 354 / 29 / 217 | 280/300 | 0 |
| sd4:ar1s | 300 | 0.672 | [0.620, 0.722] | 392 / 22 / 186 | 284/300 | 0 |
| sd8:ar1s | 300 | 0.678 | [0.628, 0.728] | 399 / 16 / 185 | 289/300 | 0 |
| sd8:ar8s | 300 | 0.502 | [0.448, 0.556] | 292 / 19 / 289 | 273/300 | 0 |
