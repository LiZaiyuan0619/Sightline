# VLM-judge preference (gpt-6.1-sol; runs/2026-10-18/judge-t2front)

pref(A) = mean over prompts of (1 A wins, 0.5 tie, 0 B wins), averaged over both orders; 95% bootstrap CI. consistent = both orders give the same verdict.

| pair A:B | prompts | pref(A) | 95% CI | A wins / ties / B wins (per order) | consistent | unparsed |
|---|---|---|---|---|---|---|
| ar2s:ar1s | 300 | 0.563 | [0.516, 0.609] | 265 / 146 / 189 | 277/300 | 0 |
| ar4s:ar1s | 300 | 0.661 | [0.613, 0.708] | 345 / 103 / 152 | 283/300 | 0 |
| ar8s:ar1s | 300 | 0.677 | [0.627, 0.724] | 376 / 60 / 164 | 278/300 | 0 |
| sd16:ar1s | 300 | 0.700 | [0.648, 0.748] | 402 / 36 / 162 | 273/300 | 0 |
| sd16:ar8s | 300 | 0.538 | [0.489, 0.590] | 291 / 64 / 245 | 269/300 | 0 |
| sd1s:ar1s | 300 | 0.499 | [0.463, 0.535] | 144 / 311 / 145 | 273/300 | 0 |
| sd2:ar1s | 300 | 0.561 | [0.512, 0.614] | 310 / 53 / 237 | 276/300 | 0 |
| sd4:ar1s | 300 | 0.623 | [0.574, 0.674] | 353 / 42 / 205 | 279/300 | 0 |
| sd8:ar1s | 300 | 0.686 | [0.636, 0.737] | 387 / 49 / 164 | 279/300 | 0 |
| sd8:ar8s | 300 | 0.498 | [0.449, 0.552] | 264 / 70 / 266 | 268/300 | 0 |
