# VLM-judge preference (gemini-3.8-flash; runs/2026-10-18/judge-t2front)

pref(A) = mean over prompts of (1 A wins, 0.5 tie, 0 B wins), averaged over both orders; 95% bootstrap CI. consistent = both orders give the same verdict.

| pair A:B | prompts | pref(A) | 95% CI | A wins / ties / B wins (per order) | consistent | unparsed |
|---|---|---|---|---|---|---|
| ar2s:ar1s | 300 | 0.558 | [0.517, 0.599] | 216 / 238 / 146 | 268/300 | 0 |
| ar4s:ar1s | 300 | 0.630 | [0.586, 0.675] | 292 / 172 / 136 | 267/300 | 0 |
| ar8s:ar1s | 300 | 0.658 | [0.612, 0.703] | 337 / 116 / 147 | 255/300 | 0 |
| sd16:ar1s | 300 | 0.683 | [0.637, 0.726] | 362 / 96 / 142 | 263/300 | 0 |
| sd16:ar8s | 300 | 0.550 | [0.501, 0.595] | 271 / 118 / 211 | 260/300 | 0 |
| sd1s:ar1s | 300 | 0.458 | [0.426, 0.486] | 79 / 391 / 130 | 271/300 | 0 |
| sd2:ar1s | 300 | 0.592 | [0.545, 0.643] | 292 / 126 / 182 | 270/300 | 0 |
| sd4:ar1s | 300 | 0.629 | [0.583, 0.679] | 323 / 109 / 168 | 268/300 | 0 |
| sd8:ar1s | 300 | 0.676 | [0.628, 0.723] | 353 / 105 / 142 | 266/300 | 0 |
| sd8:ar8s | 300 | 0.487 | [0.443, 0.537] | 236 / 113 / 251 | 261/300 | 0 |
