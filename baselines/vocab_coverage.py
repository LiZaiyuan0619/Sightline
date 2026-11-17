""" loss-bearing token coverage of the top-N draft vocabulary (EAGLE-3 d2t / t2d), per training set."""
import json, sys
from collections import Counter
path, max_len = sys.argv[1], int(sys.argv[2])
cnt = Counter()
n = 0
with open(path) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        ids, lm = r["input_ids"][:max_len], r["loss_mask"][:max_len]
        cnt.update(t for t, m in zip(ids, lm) if m)
        n += 1
tot = sum(cnt.values())
freq = sorted(cnt.values(), reverse=True)
print(f"{path}: {n} rows, {tot} loss tokens, {len(cnt)} distinct")
for N in (16000, 32000, 48000, 64000):
    print(f"  top {N}: coverage {sum(freq[:N]) / tot:.5f}")
json.dump(cnt.most_common(), open(sys.argv[3], "w"))
