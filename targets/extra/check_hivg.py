""" the exported HiVG-3B text model (export_hivg_text.py) vs the original Qwen2.5-VL checkpoint on the
official text-to-SVG prompt (refs/HiVG/hivg_infer/pipeline.py: '[T2ST] Generate SVG tokens from text.\\n<desc>', the
model's chat template): next-token logits on the prompt + greedy 96 tokens, bf16, same GPU.
  python check_hivg_export.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from sightline.paths import MODELS

SRC = f"{MODELS}/HiVG-3B-Base"
DST = f"{MODELS}/HiVG-3B-Base-qwen2-text"
DESCS = ["A black circular outline encloses two thick black lines forming an 'X' shape.",
         "A cute cartoon fox sitting under a tree with orange leaves, flat illustration."]


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(SRC)
    tok = AutoTokenizer.from_pretrained(DST)
    a = Qwen2_5_VLForConditionalGeneration.from_pretrained(SRC, dtype=torch.bfloat16).cuda().eval()
    b = AutoModelForCausalLM.from_pretrained(DST, dtype=torch.bfloat16).cuda().eval()
    for d in DESCS:
        msgs = [{"role": "user", "content": [{"type": "text", "text": f"[T2ST] Generate SVG tokens from text.\n{d}"}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids_a = proc(text=[text], return_tensors="pt")["input_ids"].cuda()
        ids_b = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        assert ids_a.tolist() == ids_b.tolist(), "prompt ids differ"
        with torch.no_grad():
            la = a(input_ids=ids_a).logits[0, -1].float()
            lb = b(input_ids=ids_b).logits[0, -1].float()
            ga = a.generate(input_ids=ids_a, max_new_tokens=96, do_sample=False)[0, ids_a.shape[1]:].tolist()
            gb = b.generate(input_ids=ids_b, max_new_tokens=96, do_sample=False)[0, ids_b.shape[1]:].tolist()
        same = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), None)
        print(f"max |dlogit| {float((la - lb).abs().max()):.4f}  argmax {int(la.argmax())}/{int(lb.argmax())}  "
              f"greedy 96: first diff at {same}  prompt len {ids_a.shape[1]}")
        print("   text:", repr(proc.decode(ga[:40], skip_special_tokens=False))[:300])


if __name__ == "__main__":
    main()
