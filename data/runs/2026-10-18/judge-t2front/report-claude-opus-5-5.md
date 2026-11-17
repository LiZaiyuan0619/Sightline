# VLM-judge preference (claude-opus-5-5; runs/2026-10-18/judge-t2front)

pref(A) = mean over prompts of (1 A wins, 0.5 tie, 0 B wins), averaged over both orders; 95% bootstrap CI. consistent = both orders give the same verdict.

| pair A:B | prompts | pref(A) | 95% CI | A wins / ties / B wins (per order) | consistent | unparsed |
|---|---|---|---|---|---|---|
| ar2s:ar1s | 300 | 0.581 | [0.536, 0.625] | 254 / 189 / 157 | 271/300 | 0 |
| ar4s:ar1s | 300 | 0.647 | [0.598, 0.693] | 321 / 134 / 145 | 284/300 | 0 |
| ar8s:ar1s | 300 | 0.689 | [0.641, 0.733] | 370 / 87 / 143 | 274/300 | 0 |
| sd16:ar1s | 300 | 0.693 | [0.644, 0.739] | 386 / 60 / 154 | 272/300 | 0 |
| sd16:ar8s | 300 | 0.555 | [0.505, 0.605] | 290 / 86 / 224 | 272/300 | 0 |
| sd1s:ar1s | 300 | 0.491 | [0.456, 0.524] | 116 / 357 / 127 | 277/300 | 0 |
| sd2:ar1s | 300 | 0.593 | [0.545, 0.646] | 313 / 86 / 201 | 271/300 | 0 |
| sd4:ar1s | 300 | 0.667 | [0.618, 0.715] | 361 / 78 / 161 | 278/300 | 0 |
| sd8:ar1s | 300 | 0.710 | [0.662, 0.757] | 393 / 66 / 141 | 270/300 | 0 |
| sd8:ar8s | 300 | 0.487 | [0.438, 0.540] | 251 / 82 / 267 | 272/300 | 0 |
