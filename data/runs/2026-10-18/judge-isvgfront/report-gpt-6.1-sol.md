# VLM-judge preference (gpt-6.1-sol; runs/2026-10-18/judge-isvgfront)

pref(A) = mean over prompts of (1 A wins, 0.5 tie, 0 B wins), averaged over both orders; 95% bootstrap CI. consistent = both orders give the same verdict.

| pair A:B | prompts | pref(A) | 95% CI | A wins / ties / B wins (per order) | consistent | unparsed |
|---|---|---|---|---|---|---|
| ar2s:ar1s | 300 | 0.598 | [0.547, 0.647] | 318 / 81 / 201 | 287/300 | 0 |
| ar4s:ar1s | 300 | 0.639 | [0.586, 0.686] | 361 / 45 / 194 | 280/300 | 0 |
| ar8s:ar1s | 300 | 0.693 | [0.638, 0.743] | 403 / 26 / 171 | 283/300 | 0 |
| sd16:ar1s | 300 | 0.694 | [0.643, 0.743] | 409 / 15 / 176 | 280/300 | 0 |
| sd1s:ar1s | 300 | 0.521 | [0.472, 0.566] | 244 / 137 / 219 | 276/300 | 0 |
| sd2:ar1s | 300 | 0.568 | [0.516, 0.618] | 327 / 28 / 245 | 274/300 | 0 |
| sd4:ar1s | 300 | 0.636 | [0.583, 0.687] | 370 / 23 / 207 | 282/300 | 0 |
| sd8:ar1s | 300 | 0.685 | [0.633, 0.734] | 401 / 20 / 179 | 282/300 | 0 |
| sd8:ar8s | 300 | 0.489 | [0.437, 0.542] | 284 / 19 / 297 | 263/300 | 0 |
