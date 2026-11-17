""" is the exported OmniSVG1.1_4B text model (export_qwen2_text_any.py) the same function as the official
4B SketchDecoder transformer on text-to-SVG prompts? Same env (the omnisvg environment, transformers 4.51), fp32:
prompt logits max |diff| and plain greedy continuations (no repetition penalty: model equivalence only), token by token,
on --n MMSVG-Bench text2svg prompts (half icon, half illustration; prompt ids from the T6 pool, Qwen2.5 tokenizer).
The official model is built the way run.py does for 4B (config-only base, every weight from OmniSVG1.1_4B).
  python check_omnisvg4b.py [--n 4] [--new 256]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # run as a script from the package root
import torch
from sightline.paths import MODELS, RUNS

ROOT = Path(__file__).resolve().parents[1]
M = MODELS
EXPORT = M / "OmniSVG1.1_4B-qwen2-text"
POOL = RUNS / "2026-10-09-prompt-pool"


def official_4b():
    sys.path.insert(0, str(ROOT / "omnisvg"))
    import run as R
    official = R.load_official()
    import decoder as official_decoder
    cls = official_decoder.Qwen2_5_VLForConditionalGeneration

    def from_config_only(path, torch_dtype=None, attn_implementation=None, device_map=None, **kw):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path)
        return cls._from_config(cfg, torch_dtype=torch_dtype, attn_implementation=attn_implementation).to("cuda")
    official_decoder.Qwen2_5_VLForConditionalGeneration = type(
        "Qwen2_5_VLFromConfig", (cls,), {"from_pretrained": staticmethod(from_config_only)})
    official.load_models("4B", str(M / "OmniSVG1.1_4B"), str(M / "Qwen2.5-VL-3B-Instruct-configs"))
    return official.sketch_decoder.transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--new", type=int, default=256)
    a = ap.parse_args()
    rows = [json.loads(x) for x in (POOL / "bench-ids-text2svg.jsonl").read_text().splitlines() if x.strip()]
    pick = [r for r in rows if r["subtype"] == "icon"][:a.n // 2] + \
           [r for r in rows if r["subtype"] == "illustration"][:a.n - a.n // 2]
    outs = {}
    for name in ("orig", "export"):
        if name == "orig":
            m = official_4b().float().eval()
        else:
            # build from the config and load the state dict once (from_pretrained of transformers 4.51 kept re-reading
            # the.bin from the shared disk: ~1 TB of reads in 25 min, )
            from transformers import AutoConfig, Qwen2ForCausalLM
            cfg = AutoConfig.from_pretrained(EXPORT)
            cfg._attn_implementation = "sdpa"
            m = Qwen2ForCausalLM(cfg)
            from safetensors.torch import load_file
            sd = load_file(str(EXPORT / "model.safetensors"))
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
            m = m.float().cuda().eval()
        print(f"[{name}] loaded", flush=True)
        res = []
        for r in pick:
            ids = torch.tensor([r["prompt_input_ids"]], device="cuda")
            with torch.no_grad():
                logits = m(input_ids=ids).logits[0].float().cpu()
                g = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                               max_new_tokens=a.new, eos_token_id=196999, pad_token_id=151643)[0, ids.shape[1]:].tolist()
            res.append((logits, g, ids.shape[1]))
        outs[name] = res
        del m
        torch.cuda.empty_cache()
    report = []
    for i, ((lo, go, n), (le, ge, _)) in enumerate(zip(outs["orig"], outs["export"])):
        first = next((j for j in range(min(len(go), len(ge))) if go[j] != ge[j]), None)
        report.append({"i": i, "subtype": pick[i]["subtype"], "prompt_len": n,
                       "logit_maxdiff": (lo - le).abs().max().item(), "len_orig": len(go), "len_export": len(ge),
                       "identical": go == ge, "first_diff": first})
        print(report[-1], flush=True)
    print("ALL_IDENTICAL" if all(r["identical"] for r in report) else "DIFFERENCES")
    out = RUNS / "2026-10-20/omnisvg4b-export-check.json"
    out.write_text(json.dumps({"new": a.new, "rows": report}, indent=1))


if __name__ == "__main__":
    main()
