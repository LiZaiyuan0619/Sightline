"""B4 / B5 (B line): appendix A22 (training cost and break-even) and A24 (training budget and
hyper-parameters), for the draft heads the PAPER actually reports. 0 GPU, trainer logs + result files only.

Which heads: runs/2026-11-06/chores/b4/heads_in_paper.py walks the result directories table 1 and table 3 read and
takes each run's protocol.json `drafter`, so heads trained and dropped are excluded by construction. The head path
maps back to its training run (its parent of `heads/stepN`), which carries plan.json and train_curve.jsonl.

Cost: GPU hours = world_size x wall; wall = sum over logged intervals of s_per_step x log_every (the method,
runs/2026-10-22/analysis/train_cost.py). The b3-train runs (SpecForge era) have no train_curve.jsonl; there wall is
the plan.json -> TRAIN_RC mtime span, as in. Where reported the same run, its wall hours are asserted.

Hyper-parameters come from the plan.json the trainer WROTE (the values that took effect), never from the draft config
file: the 4B heads' config says target_layer_ids [1,9,17,25,33] while train_dflash.py hard-codes [1,7,13,19,25]
(project memory smoke-must-cover-formal-paths). Layers are therefore read from the trained head's own config.json,
and for the 4B DFlash heads from the evaluated copy (tableA/t2-4b/heads-fixed/..., which jobs_1022b.sh rewrote to the
layers the trainer used); a '!' marks heads whose own config.json disagrees with the draft config it was trained from.

Break-even: training GPU hours / seconds saved per prompt, with the saving measured on this block's own table-1 rows
(mean original-model seconds minus mean decoder seconds per prompt, the shared clock of
runs/2026-11-06/chores/b3/hero5.py). Reported for our heads and, for comparison, DDTree's.
  python3 runs/2026-11-06/chores/b4/train_cost_v2.py
    -> train_cost.{json,md} (this dir) + tex/tables/app_a22_train_cost.tex + app_a24_hparams.tex
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import importlib.util
import json
import statistics as st
import sys
from pathlib import Path
from sightline.paths import MODELS, RUNS

ROOT = RUNS.parent            # data/, which holds runs/
HERE = Path(__file__).resolve().parent
U = RUNS


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B1 = _load("b1", "runs/2026-11-05/appendix/make_tables_b1.py")
B3 = _load("b3", "runs/2026-11-05/appendix/make_tables_b3.py")
sys.path.insert(0, str(RUNS / "2026-11-06/chores/b3"))
import hero5 as H  # noqa: E402

# the paper's blocks, in table order; table 3 families after table 1's five
BLOCK_ORDER = ["OmniSVG-8B text", "OmniSVG-4B text", "InternSVG-8B text", "OmniSVG-8B image", "InternSVG-8B image",
               "SVGen-3B", "SVGen-7B", "HiVG-3B", "OmniLottie-3B", "StarVector-1B", "StarVector-8B"]
METHOD_ORDER = ["EAGLE-3", "ViSpec", "DFlash", "DARTree", "DDTree", "Ours"]
T1_BLOCK = {"OmniSVG-8B text": "OmniSVG-8B, text-to-SVG", "OmniSVG-4B text": "OmniSVG-4B, text-to-SVG",
            "InternSVG-8B text": "InternSVG-8B, text-to-SVG", "OmniSVG-8B image": "OmniSVG-8B, image-to-SVG",
            "InternSVG-8B image": "InternSVG-8B, image-to-SVG"}
# runs/2026-10-22/analysis/train_cost.md, as a regression check: run -> (steps done then, wall h then).
# A run that had already finished must reproduce its wall hours exactly; one that was still training on must
# now show MORE steps and more wall (that is why the snapshot cannot be used as the appendix table).
T_1022 = {"b3-train/omnisvg-dflash-main300k-b32-6ep-1012": (150000, 33.4),
          "b3-train/omnisvg-dflash-main300k-6ep-1011": (150000, 17.3),
          "e3-train/omnisvg-e3-300k-6ep": (112500, 15.8), "h3a-train/i2svg-b32-m3warm-100k-3ep": (18750, 9.7),
          "h3a-train/i2svg-b16-m3warm-100k-3ep": (18750, 5.7), "e3-train/isvg-e3-300k-2ep": (37500, 5.7),
          "isvg-train/isvg-dflash-b32-300k-2ep": (37500, 21.8), "isvg-train/isvg-dflash-b16-300k-2ep": (37500, 12.4),
          "isvg-train/isvg-dflash-b32-300k-2ep-mix1ep": (11150, 6.9),
          "isvg-train/isvg-dflash-b16-300k-2ep-mix1ep": (19300, 6.9),
          "e3-train/isvg-e3-300k-2ep-mix1ep": (37500, 6.7), "omnisvg4b/train/dflash-b32-300k-6ep": (43350, 9.8),
          "omnisvg4b/train/dflash-b16-300k-6ep": (56000, 7.1), "omnisvg4b/train/e3-300k-6ep": (59350, 7.0),
          "domino-train/omnisvg-domino-b16-300k-2ep": (33300, 6.9)}


# the 4B DFlash heads are evaluated from the config-fixed copies jobs_1022b.sh wrote (one per shard, byte-identical
# weights); they carry no plan.json, so they map back to the run that produced them
FIXED_4B = {"b32": U / "omnisvg4b/train/dflash-b32-300k-6ep", "b16": U / "omnisvg4b/train/dflash-b16-300k-6ep"}
NO_TRAINING = {f"{MODELS}/OmniSVG1.1_4B-qwen2-text":
               "SpS: no draft head, the 4B model itself drafts for the 8B one"}


def run_dir(head: str) -> Path | None:
    p = Path(head)
    if "heads-fixed" in str(p):
        return FIXED_4B.get(p.name)
    if p.parent.name.startswith("step") or p.name.startswith("step"):
        q = p if p.name.startswith("step") else p.parent
        return q.parent.parent                      # <run>/heads/stepN
    return None


def chain(run: Path, seen=None) -> list[Path]:
    """[the run, its warm start,...]: a continued or warm-started head costs what it cost PLUS its ancestor."""
    seen = seen or []
    if run in seen or not (run / "plan.json").exists():
        return seen
    seen = seen + [run]
    init = json.loads((run / "plan.json").read_text()).get("init")
    if init and init != "None":
        prev = run_dir(init)
        if prev is not None:
            return chain(prev, seen)
    return seen


def cost(run: Path) -> dict:
    plan = json.loads((run / "plan.json").read_text())
    ws = int(plan.get("world_size") or plan.get("trainers") or 1)
    tot = plan.get("total_steps")
    curve = run / "train_curve.jsonl"
    if curve.exists():
        L = [json.loads(x) for x in curve.read_text().splitlines() if x.strip()]
        le = int(plan.get("log_every", 50))
        wall = sum(r["s_per_step"] * le for r in L) / 3600
        done = L[-1]["step"] if L else 0
        source = "train_curve"
    else:
        rc = run / "TRAIN_RC"
        wall = (rc.stat().st_mtime - (run / "plan.json").stat().st_mtime) / 3600
        done = tot
        source = "plan -> TRAIN_RC"
    cfg = {}
    y = run / "config.yaml"
    if y.exists():                       # SpecForge-era runs (b3-train): plan.json has no lr / max_length
        cur = None
        for line in y.read_text().splitlines():
            if line and not line[0].isspace():
                cur = line.split(":")[0]
            elif cur in ("training", "data") and ":" in line:
                k, _, v = line.strip().partition(":")
                cfg[k.strip()] = v.strip()
    def pick(*keys, src=plan):
        for k in keys:
            v = src.get(k) if isinstance(src, dict) else None
            if v not in (None, "None"):
                return v
        for k in keys:
            if cfg.get(k) not in (None, "None"):
                return cfg[k]
        return None
    return {"gpus": ws, "wall_h": wall, "gpu_h": ws * wall, "steps_done": done, "steps_planned": tot,
            "global_batch": pick("global_batch"), "lr": pick("lr_used", "lr", "learning_rate"),
            "epochs": pick("epochs", "num_epochs"),
            "n_samples": pick("n_samples") or (plan.get("n_train") if plan.get("n_train") else None),
            "init": plan.get("init"), "max_length": pick("max_length_used", "max_length"),
            "seed": pick("seed"), "draft_cfg": plan.get("draft_cfg") or cfg.get("draft_model_config"),
            "source": source, "plan": plan}


def head_layers(head: str) -> tuple[list | None, int | None, str]:
    """(target layers, draft layers, note) from the TRAINED head's own config.json."""
    c = Path(head) / "config.json"
    if not c.exists():
        return None, None, "no config.json"
    d = json.loads(c.read_text())
    sub = d.get("dflash_config") or d.get("domino_config") or {}
    return sub.get("target_layer_ids"), d.get("num_hidden_layers"), ""


# ViSpec is not in tableA_v9's method list (table 1 adds its row separately), so its rows are read here
VISPEC = {"OmniSVG-8B image": ["tableA/vispec/omnisvg/own/VS-k8-d4-N29", "tableA/vispec/omnisvg/x/VS-k8-d4-N29"],
          "InternSVG-8B image": ["tableA/vispec/internsvg/own/VS-k8-d12-N127",
                                 "tableA/vispec/internsvg/x/VS-k8-d12-N127"]}
_vis: dict = {}


def vispec_rows(block):
    if block not in _vis:
        rows = {}
        for rel in VISPEC[block]:
            for f in sorted((U / rel).glob("shard-*/results.jsonl")):
                for x in f.read_text().splitlines():
                    if x.strip():
                        r = json.loads(x)
                        if r["mode"] == "greedy":
                            rows[(r["set"], r["subtype"], r["gidx"])] = r
        _vis[block] = rows
    return _vis[block]


def saving_per_prompt(cap, block, method):
    """mean original-model seconds - mean decoder seconds, over this block's table-1 rows (greedy, all six columns)."""
    name = T1_BLOCK.get(block)
    if name is None:
        return None
    tot_ar = tot_m = n = 0.0
    for j in range(6):
        col = H.column(cap, name, "greedy", j)
        ms = col["methods"]
        key = "Ours" if method == "Ours" else method
        if key == "ViSpec" and block in VISPEC:
            lab, sets, sb = cap[(name, "greedy")]["cols"][j]
            sets = (sets,) if isinstance(sets, str) else tuple(sets)
            vr = vispec_rows(block)
            ar_t = ms["original model"]["t"]
            ids = [g for g in ar_t if any((st, sb, g) in vr for st in sets)]
            if not ids:
                return None
            tot_ar += sum(ar_t[g] for g in ids)
            tot_m += sum(next(vr[(st, sb, g)] for st in sets if (st, sb, g) in vr)["t_spec"] for g in ids)
            n += len(ids)
            continue
        if key not in ms:
            return None
        ids = sorted(set(ms["original model"]["t"]) & set(ms[key]["t"]))
        tot_ar += sum(ms["original model"]["t"][g] for g in ids)
        tot_m += sum(ms[key]["t"][g] for g in ids)
        n += len(ids)
    return (tot_ar - tot_m) / n


def main():
    heads = json.loads((RUNS / "2026-11-05/appendix/heads_in_paper.json").read_text())["heads"]
    cap = H.capture(modes=("greedy",))
    rows, by_run, skipped = [], {}, []
    for h in heads:
        run = run_dir(h["head"])
        if run is None or not (run / "plan.json").exists():
            skipped.append({"head": h["head"], "why": NO_TRAINING.get(h["head"], "no plan.json"),
                            "used_in": h["used_in"]})
            continue
        uses = sorted({(b, m) for _, b, m in h["used_in"]})
        rel = str(run).replace(str(U) + "/", "").replace(f"{RUNS}/", "g2/")
        if rel not in by_run:
            c = cost(run)
            tl, dl, note = head_layers(h["head"])
            dc = c.get("draft_cfg")
            if dc and Path(dc).exists() and tl is not None:
                want = (json.loads(Path(dc).read_text()).get("dflash_config")
                        or json.loads(Path(dc).read_text()).get("domino_config") or {}).get("target_layer_ids")
                if want and list(want) != list(tl):
                    note = (note + f" head config layers {tl} differ from the draft config {want} "
                            "(the trainer's hard-coded layers took effect)").strip()
            ch = chain(run)
            anc = [cost(x) for x in ch[1:]]
            by_run[rel] = dict(run=rel, head=h["head"], layers=tl, draft_layers=dl, note=note,
                               warm_start_from=[str(x) for x in ch[1:]],
                               gpu_h_total=c["gpu_h"] + sum(a["gpu_h"] for a in anc), **c)
        by_run[rel].setdefault("uses", set()).update(uses)
    same, grown = [], []
    for rel, v in by_run.items():
        if rel not in T_1022:
            continue
        st_old, wall_old = T_1022[rel]
        if v["steps_done"] == st_old:
            assert abs(v["wall_h"] - wall_old) < 0.1, (rel, v["wall_h"], wall_old)
            same.append(rel)
        else:
            assert v["steps_done"] > st_old and v["wall_h"] > wall_old, (rel, v["steps_done"], st_old)
            grown.append((rel, st_old, v["steps_done"], round(wall_old, 1), round(v["wall_h"], 1)))
    print(f"{len(by_run)} training runs behind the paper's heads; {len(same)} reproduce the 10-22 wall hours, "
          f"{len(grown)} kept training after that snapshot:")
    for g in grown:
        print("   ", g[0], f"{g[1]:,} -> {g[2]:,} steps, {g[3]} -> {g[4]} wall h")
    # break-even, per block x method (table 1 blocks only: table 3 has no per-prompt AR timing in this capture)
    for v in by_run.values():
        v["uses"] = sorted(v["uses"])
        v["break_even"] = {}
        for b, m in v["uses"]:
            s = saving_per_prompt(cap, b, m)
            if s:
                v["break_even"][f"{b}|{m}"] = {"seconds_saved_per_prompt": s,
                                               "prompts_to_repay": v["gpu_h_total"] * 3600 / s}
    out = {"runs": list(by_run.values()), "skipped": skipped}
    (RUNS / "2026-11-05/appendix/train_cost.json").write_text(json.dumps(out, indent=1, default=str))
    return out, by_run


# ---------------------------------------------------------------------------------------------- tables
def fmt_lr(v):
    if v is None:
        return "--"
    x = float(v)
    e = 0
    while x < 1:
        x *= 10
        e -= 1
    return f"${x:.0f}\\!\\times\\!10^{{{e}}}$"


def tables(out, by_run):
    rows = out["runs"]
    use = {}                                            # (block, method) -> run record
    for r in rows:
        for b, m in r["uses"]:
            use[(b, m)] = r
    # ---- A22: cost and break-even, table-1 blocks (the only ones with per-prompt original-model timing here)
    lines = []
    for bi, b in enumerate([x for x in BLOCK_ORDER if x in T1_BLOCK]):
        ms = [m for m in METHOD_ORDER if (b, m) in use]
        for i, m in enumerate(ms):
            r = use[(b, m)]
            be = r["break_even"].get(f"{b}|{m}")
            lead = f"\\multirow{{{len(ms)}}}{{*}}{{{b}}}" if i == 0 else ""
            name = B1.method_name("Ours") if m == "Ours" else m
            warm = f" + {r['gpu_h_total'] - r['gpu_h']:.0f}" if r["gpu_h_total"] > r["gpu_h"] + 0.5 else ""
            lines.append(f"{lead} & {name} & {r['gpus']} & {r['wall_h']:.1f} & {r['gpu_h']:.0f}{warm} & "
                         + (f"{be['seconds_saved_per_prompt']:.1f} & {be['prompts_to_repay'] / 1000:.0f}k \\\\"
                            if be else "-- & -- \\\\"))
        if bi < 4:
            lines.append("\\midrule")
    ours = [use[(b, "Ours")]["break_even"][f"{b}|Ours"]["prompts_to_repay"] for b in T1_BLOCK]
    cap22 = ("\\caption{\\textbf{Training cost and break-even.} Wall time and GPU hours come from each trainer's own "
             "log (H200); \\emph{+ n} is the cost of the checkpoint a head was warm-started or continued from. "
             "\\emph{Saved} is the mean seconds a prompt takes off the original model's decoding time, over that "
             "block's table~\\ref{tab:main} rows (greedy, all six benchmarks); \\emph{break-even} is training GPU "
             f"hours divided by it. Our heads repay their training in {min(ours) / 1000:.0f}k--{max(ours) / 1000:.0f}k "
             "prompts, a few days of a single server's traffic, and the head is then reused for every later prompt.}")
    head22 = ("Block & Method & GPUs & Wall (h) & GPU-h & Saved (s) & Break-even \\\\")
    B1.write("app_a22_train_cost", B3.table("table*", cap22, "app:traincost", "llccccc", head22, lines))
    # ---- A24: budget and hyper-parameters, every head in the paper
    lines, flagged = [], []
    for bi, b in enumerate(BLOCK_ORDER):
        ms = [m for m in METHOD_ORDER if (b, m) in use]
        for i, m in enumerate(ms):
            r = use[(b, m)]
            lead = f"\\multirow{{{len(ms)}}}{{*}}{{{b}}}" if i == 0 else ""
            name = B1.method_name("Ours") if m == "Ours" else m
            n = r["n_samples"]
            n = n[0] if isinstance(n, list) else n
            data = f"{int(n) / 1000:.0f}k" if n else "--"
            ep = r["epochs"]
            ep = f"{float(ep):g}" if ep not in (None, "None") else "--"
            ml = r["max_length"] or "--"
            L = r["layers"]
            layers = ",".join(str(x) for x in L) if L else "--"
            if r["note"]:
                layers += "$^{\\dagger}$"
                flagged.append((b, m))
            warm = "--"
            if r["warm_start_from"]:
                anc = cost(Path(r["warm_start_from"][0]) if Path(r["warm_start_from"][0]).is_absolute()
                           else U / r["warm_start_from"][0])
                an = anc["n_samples"]
                an = an[0] if isinstance(an, list) else an
                ae = anc["epochs"]
                warm = (f"{int(an) / 1000:.0f}k" if an else "?") + (f", {float(ae):g} ep" if ae not in (None, "None") else "")
            lines.append(f"{lead} & {name} & {data} & {ep} & {r['global_batch']} & {fmt_lr(r['lr'])} & {ml} & "
                         f"{layers} & {warm} \\\\")
        if bi < len(BLOCK_ORDER) - 1 and ms:
            lines.append("\\midrule")
    n_heads = len({r["run"] for r in rows})
    cap24 = ("\\caption{\\textbf{Training budget and hyper-parameters of every draft head in the paper.} Values are "
             "the ones the trainer actually used, read from the run's own plan and from the trained head's "
             "configuration, not from the draft config file it started from. \\emph{Data} is the number of generated "
             "training prompts and \\emph{epochs} the passes over them; \\emph{layers} are the target layers whose "
             "hidden states the head reads. \\emph{Warm start} marks a head initialised from an earlier head "
             "(its cost is carried in table~\\ref{app:traincost}). Our block drafters and the DFlash / DDTree "
             "baselines share every setting except the block length (32 against 16), so the comparison isolates it. "
             f"Every row is a head the paper reports; the {n_heads} training runs behind them are all that is "
             "counted, heads trained during development and not reported are left out (DFlash and DDTree share one "
             "head per model, so they share a row's settings). $\\dagger$: the trainer's built-in layer choice "
             "overrode the draft config file.}")
    head24 = "Block & Method & Data & Ep. & Batch & LR & Len. & Layers & Warm start from \\\\"
    B1.write("app_a24_hparams", B3.table("table*", cap24, "app:hparams", "ll*{7}{c}", head24, lines))
    print(f"A22: {len([x for x in BLOCK_ORDER if x in T1_BLOCK])} blocks; A24: {n_heads} heads, "
          f"layer-override flag on {flagged}")


if __name__ == "__main__":
    tables(*main())
