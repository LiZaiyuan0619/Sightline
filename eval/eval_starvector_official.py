""" (project decision): the generality table's StarVector sampling column follows StarVector's OFFICIAL evaluation
setting, as the other four families' columns already do. eval_sv.py hard-codes T 1.0 / top-p 0.9 (the code
defaults of generate_im2svg minus its 2 beams); the official evaluation configs are
refs/StarVector/configs/generation/hf/starvector-{1b,8b}/im2svg.yaml: num_beams 1, do_sample, T 0.2 (1B) / 0.7 (8B),
top-p 0.95, no repetition penalty -- and top-k 50, because those configs go through HF generate and the model dirs ship
no generation_config.json, so transformers' default top-k applies. Same order as HF (temperature -> top-k -> top-p,
sightline/process.py). The T 1.0 column stays, for the appendix.

eval_sv.py is not edited (the fill shards import it): this wrapper loads it as a module, replaces its I2SVG dict
(written into every protocol.json) and chain_params -- which every arm reads at call time: the inline AR, the AR
cache records and every SD design -- and runs its main. AR records carry no sampling parameters, so every --ar-save
dir gets a sampling.json stamp and an --ar-cache dir with a different (or no) stamp is refused: a T 1.0 cache would
otherwise be reused silently. Only --modes sample is accepted (greedy needs no wrapper).
  STARVEC=8b python eval_sv_off.py --modes sample <other eval_sv.py arguments>
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFICIAL = {"1b": {"temperature": 0.2, "top_p": 0.95, "top_k": 50},
            "8b": {"temperature": 0.7, "top_p": 0.95, "top_k": 50}}


def argval(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def main():
    which = os.environ.get("STARVEC", "1b")
    s = OFFICIAL[which]
    stamp = {"target": f"StarVector-{which}", "mode": "sample", **s,
             "source": f"refs/StarVector/configs/generation/hf/starvector-{which}/im2svg.yaml + HF default top-k 50"}
    if argval("--modes") != "sample":
        sys.exit(f"[eval_sv_off] needs --modes sample, got {argval('--modes')}")
    for flag in ("--ar-cache", "--ar-save"):
        d = argval(flag)
        if d is None:
            continue
        f = Path(d) / "sampling.json"
        got = json.loads(f.read_text()) if f.exists() else None
        if flag == "--ar-save" and got is None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(stamp, indent=1))
        elif got != stamp:
            sys.exit(f"[eval_sv_off] {flag} {d} was not made with this sampling setting: {got} != {stamp}")

    spec = importlib.util.spec_from_file_location("eval_sv", HERE / "eval_sv.py")
    E = importlib.util.module_from_spec(spec)
    sys.modules["eval_sv"] = E
    spec.loader.exec_module(E)
    E.I2SVG.update(s, setting="StarVector official HF evaluation config (eval_sv_off.py, 11-04)")

    def chain_params(mode):
        if mode == "greedy":
            return E.ChainParams(greedy=True)
        return E.ChainParams(greedy=False, temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"])
    E.chain_params = chain_params
    print(f"[eval_sv_off] StarVector-{which} sampling: {s}", flush=True)
    E.main()


if __name__ == "__main__":
    main()
