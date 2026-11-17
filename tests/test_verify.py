"""Synthetic unit tests for verify_block: acceptance logic + penalty-history growth.

Run: <env>/bin/python sightline/test_verify.py
No model needed; logits are hand-crafted.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sightline.verify import verify_block, ChainParams

VOCAB = 50


def onehot_logits(token: int, boost=10.0):
    logits = torch.zeros(VOCAB)
    logits[token] = boost
    return logits


def run(name, check):
    ok = check()
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
    return ok


def main():
    params = ChainParams(greedy=True)
    all_ok = True

    # 1. all accepted -> K + 1 emitted, bonus from last logits
    drafts = [3, 7, 9]
    logits = [onehot_logits(3), onehot_logits(7), onehot_logits(9), onehot_logits(11)]
    r = verify_block(drafts, logits, [1, 2], params)
    all_ok &= run("all accepted: n=3, emitted=[3,7,9,11]",
                  lambda: r.n_accepted == 3 and r.emitted == [3, 7, 9, 11])

    # 2. reject at position 1: emit target's token only
    logits = [onehot_logits(5), onehot_logits(7), onehot_logits(9), onehot_logits(11)]
    r = verify_block(drafts, logits, [1, 2], params)
    all_ok &= run("reject@1: n=0, emitted=[5]",
                  lambda: r.n_accepted == 0 and r.emitted == [5])

    # 3. reject at position 2: one accepted + bonus
    logits = [onehot_logits(3), onehot_logits(8), onehot_logits(9), onehot_logits(11)]
    r = verify_block(drafts, logits, [1, 2], params)
    all_ok &= run("reject@2: n=1, emitted=[3,8]",
                  lambda: r.n_accepted == 1 and r.emitted == [3, 8])

    # 4. penalty history growth: token 4 is in history after position 1 is accepted.
    #    Craft: raw argmax at pos 2 would be 4 (boost 10) vs 7 (boost 9.5);
    #    penalty 2.0 at pos2 must use history containing accepted draft 4 to flip argmax to 7.
    drafts2 = [4, 7]
    l_0 = torch.full((VOCAB,), -10.0); l_0[4] = 10.0
    l_1 = torch.full((VOCAB,), -10.0); l_1[4] = 10.0; l_1[7] = 9.5
    l_2 = torch.full((VOCAB,), -10.0); l_2[0] = 5.0
    params_pen = ChainParams(greedy=True, repetition_penalty=2.0)
    r = verify_block(drafts2, [l_0, l_1, l_2], [1, 2], params_pen)
    # After accepting 4 (pos1), history has 4; at pos2: penalty 2.0 halves 4's logit to 5.0
    # (positive -> /2), 7 stays 9.5 -> argmax = 7 -> accept. With wrong history (no 4),
    # argmax at pos2 = 4 -> reject. So n_accepted=2 proves history growth.
    def check_hist():
        if r.n_accepted != 2:
            print(f"      n_accepted={r.n_accepted}, per_pos={r.per_position}")
            return False
        return r.per_position[1]["history_len"] == len([1, 2]) + 1
    all_ok &= run("penalty history includes accepted drafts", check_hist)

    # 5. per-position log shape
    drafts = [3, 7]
    logits = [onehot_logits(5), onehot_logits(7), onehot_logits(11)]
    r = verify_block(drafts, logits, [1], params)
    # subset check: the log also carries "margin" (since ) and "snapped"
    all_ok &= run("per-position log has draft/target/accepted",
                  lambda: {k: r.per_position[0][k] for k in ("draft", "target", "accepted",
                                                             "history_len")}
                  == {"draft": 3, "target": 5, "accepted": False, "history_len": len([1])})

    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
