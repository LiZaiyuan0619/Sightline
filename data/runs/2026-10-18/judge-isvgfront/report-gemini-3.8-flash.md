# VLM-judge preference (gemini-3.8-flash; runs/2026-10-18/judge-isvgfront)

pref(A) = mean over prompts of (1 A wins, 0.5 tie, 0 B wins), averaged over both orders; 95% bootstrap CI. consistent = both orders give the same verdict.

| pair A:B | prompts | pref(A) | 95% CI | A wins / ties / B wins (per order) | consistent | unparsed |
|---|---|---|---|---|---|---|
| ar2s:ar1s | 300 | 0.610 | [0.561, 0.658] | 324 / 84 / 192 | 277/300 | 0 |
| ar4s:ar1s | 300 | 0.611 | [0.557, 0.661] | 341 / 51 / 208 | 268/300 | 0 |
| ar8s:ar1s | 300 | 0.688 | [0.635, 0.733] | 396 / 33 / 171 | 277/300 | 0 |
| sd16:ar1s | 300 | 0.656 | [0.604, 0.708] | 388 / 11 / 201 | 276/300 | 0 |
| sd1s:ar1s | 300 | 0.513 | [0.464, 0.559] | 233 / 150 / 217 | 271/300 | 0 |
| sd2:ar1s | 300 | 0.590 | [0.537, 0.642] | 336 / 36 / 228 | 277/300 | 0 |
| sd4:ar1s | 300 | 0.647 | [0.597, 0.699] | 377 / 23 / 200 | 279/300 | 0 |
| sd8:ar1s | 300 | 0.650 | [0.599, 0.700] | 380 / 20 / 200 | 280/300 | 0 |
| sd8:ar8s | 300 | 0.527 | [0.477, 0.582] | 303 / 27 / 270 | 264/300 | 0 |
