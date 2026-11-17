"""Tidy data pack for figure batch 3 -- L (loop control), P5 (serving throughput), Q1 (quality frontier).

Why: the figure session draws, it does not re-derive numbers (guide SS7.5 / SS8). Every value below is read from a
result file written by the experiment line, and each block asserts one number that is already printed in
logs/STATUS.md or in the source report, so a wrong read fails here rather than on the canvas.

Writes runs/2026-11-04/figdata_b3/{l_rules,p5_sglang,q1_frontier}.json.
Run: python3 runs/2026-11-04/figdata_b3.py
"""
import json
import re
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[2]
U = RUNS
OUT = Path(__file__).resolve().parent / "figdata_b3"
OUT.mkdir(exist_ok=True)

CI = re.compile(r"([+-]?\d+\.\d+)\s*\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]")


def md_tables(path):
    """-> list of (heading, header_cells, [row_cells]) for every pipe table under a '## ' heading."""
    out, head, hdr, rows = [], None, None, []
    for line in Path(path).read_text().splitlines():
        h = re.match(r"#+\s+(\S.*)", line)
        if h:
            if hdr:
                out.append((head, hdr, rows))
            head, hdr, rows = h.group(1).strip(), None, []
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        if hdr is None:
            hdr = cells
        else:
            rows.append(cells)
    if hdr:
        out.append((head, hdr, rows))
    return out


# ---------------------------------------------------------------------------------------------------- L
# One report per target: the loop rules on top of the same lossless head, paired quality against the lossless run.
L_SRC = [
    # target, mode->report, rule name in the report -> our name, primary quality metric
    ("OmniSVG-8B, text-to-SVG", U / "tol-quality/variant_report_rollback_t2.md", "clip"),
    ("InternSVG-8B, text-to-SVG", U / "tol-quality/variant_report_loops_isvg_final.md", "clip"),
    ("OmniSVG-8B, image-to-SVG", U / "tol-quality/variant_report_loops_i2_100k.md", "dino"),
]
L_I2S = [("InternSVG-8B, image-to-SVG", U / "tol-quality/variant_report_loops_i2s_greedy.md", "greedy"),
         ("InternSVG-8B, image-to-SVG", U / "tol-quality/variant_report_loops_i2s_sample.md", "sample")]
RULE = {"strict": "lossless", "rb1": "rollback", "ets3": "escape-then-stop", "ets": "escape-then-stop",
        "stop3": "stop", "stop": "stop"}


def l_rows(path, target, metric, only_mode=None):
    rows = []
    for head, hdr, body in md_tables(path):
        mode = head if only_mode is None else only_mode
        if mode not in ("greedy", "sample"):
            continue
        for c in body:
            r = dict(zip(hdr, c))
            if r["run"] not in RULE:
                continue
            q = {}
            for m in hdr[8:]:
                hit = CI.search(r[m])
                # the lossless row prints the absolute value, the variants print "delta [lo, hi]"
                q[m] = ([float(hit.group(1)), float(hit.group(2)), float(hit.group(3))] if hit
                        else [float(r[m]), None, None])
            rows.append(dict(target=target, mode=mode, rule=RULE[r["run"]], rows=int(r["rows"]),
                             eos=int(r["EOS"]), loops=int(r["loops"]), mean_len=int(r["mean len"]),
                             nfe_vs_lossless=float(r["NFE vs strict"].rstrip("x")),
                             nfe_vs_ar=float(r["NFE vs orig. AR"].rstrip("x")),
                             quality=q, primary=metric))
    return rows


def build_l():
    ladder = []
    for target, path, metric in L_SRC:
        ladder += l_rows(path, target, metric)
    for target, path, mode in L_I2S:
        ladder += l_rows(path, target, "dino", only_mode=mode)
    # anchors: logs/STATUS.md SS1.2 (text stop x1.32 / x1.25, image stop x1.24, InternSVG stop x1.34)
    got = {(r["target"], r["mode"], r["rule"]): r["nfe_vs_lossless"] for r in ladder}
    assert got[("OmniSVG-8B, text-to-SVG", "greedy", "stop")] == 1.32, got
    assert got[("OmniSVG-8B, text-to-SVG", "sample", "stop")] == 1.25
    assert got[("OmniSVG-8B, image-to-SVG", "greedy", "stop")] == 1.24
    assert got[("InternSVG-8B, text-to-SVG", "greedy", "stop")] == 1.34
    assert len(ladder) == 4 * 2 * 4, len(ladder)

    # speed of the rule on new targets, same job / node / rows (runs/2026-10-31/loops_new_targets.md)
    new_t = []
    for head, hdr, body in md_tables(RUNS / "2026-10-31/loops_new_targets.md"):
        tgt, mode = head.split("-")[0], head.split("-")[1].split(" ")[0]
        for c in body:
            r = dict(zip(hdr, c))
            name = {"lossless": "lossless", "rollback (rb1)": "rollback",
                    "escape-then-stop": "escape-then-stop", "stop": "stop"}[r["Rule"]]
            new_t.append(dict(target=tgt, mode=mode, rule=name, rows=int(r["Rows"]), eos=int(r["EOS"]),
                              mean_len=int(r["Mean tokens"]),
                              nfe_vs_lossless=float(r["Forward vs strict"].lstrip("x").lstrip("×")),
                              wall_vs_ar=float(r["Wall clock vs AR"].rstrip("x").rstrip("×"))))
    assert any(r["target"] == "sv8b" and r["rule"] == "stop" and r["wall_vs_ar"] == 10.17 for r in new_t)

    # measured wall clock on the full official benchmark (table 2): the same rule on the strongest baseline
    t2 = json.loads((RUNS / "2026-11-02/figdata/table2.json").read_text())["blocks"]
    sarena = []
    for blk, rows in t2.items():
        for row in ("AR", "DDTree", "DDTree (fast)", "Ours", "Ours (fast)"):
            for sub, v in rows[row].items():
                sarena.append(dict(block=blk, subset=sub, method=row, seconds=v["seconds"], n=v["n"],
                                   pred_tokens=v["pred_tokens"], metrics=v["metrics"]))
    s = {(r["block"], r["subset"], r["method"]): r["seconds"] for r in sarena}
    assert round(s[("OmniSVG-8B, text-to-SVG", "icon", "DDTree (fast)")], 2) == 0.94      # runs/2026-11-03/t2_ddtree_fast.md
    assert round(s[("OmniSVG-8B, text-to-SVG", "illustration", "DDTree (fast)")], 2) == 1.76

    # the same rule on the original model (bench 300, timed best-of-N job): ar1 -> ar1s
    ar = []
    for blk, f in (("OmniSVG-8B, text-to-SVG", "t2-bench300-front"),
                   ("OmniSVG-8B, image-to-SVG", "i2-bench300-front"),
                   ("InternSVG-8B, text-to-SVG", "isvg-bench300-T0.6-front")):
        tab = md_tables(ROOT / f"runs/2026-10-15-harvest/bon_decode_report_{f}.md")[0]
        r = {d[0]: dict(zip(tab[1], d)) for d in tab[2]}
        ar.append(dict(block=blk, ar1=float(r["ar1"]["t total"]), ar1_stop=float(r["ar1s"]["t total"]),
                       sd1=float(r["sd1"]["t total"]), sd1_stop=float(r["sd1s"]["t total"])))
    assert round(ar[0]["ar1"] / ar[0]["ar1_stop"], 2) == 1.68, ar[0]

    # does the rule cut the right outputs (exact replay against the offline labeller)
    audit = []
    tab = md_tables(RUNS / "2026-10-22/analysis/loop_rule_triggers.md")
    hdr, body = tab[0][1], tab[0][2]
    for c in body:
        r = dict(zip(hdr, c))
        num = lambda s: int(s.split(" ")[0])
        audit.append(dict(model=r["model"], subset=r["subset"], outputs=int(r["outputs"]),
                          labelled=num(r["labelled loop"]), fires=num(r["rule fires"]),
                          true_triggers=int(r["true triggers"]), premature=num(r["premature stops (EOS-ended)"]),
                          misses=int(r["misses"]),
                          cut_share=float(r["tokens cut on premature stops"].split("%")[0]) / 100,
                          tokens_saved=int(r["tokens saved on true triggers"].split(" ")[0])))
    assert audit[0]["true_triggers"] == 644 and audit[0]["fires"] == 816

    doc = dict(meta=dict(
        ladder="900 bench rows (300 for InternSVG image), paired against the same head's lossless run, "
               "95% bootstrap CI; NFE vs lossless = forward passes saved. Sources: "
               "$SIGHTLINE_RUNS/tol-quality/variant_report_{rollback_t2,loops_isvg_final,loops_i2_100k,loops_i2s_*}.md",
        new_targets="runs/2026-10-31/loops_new_targets.md -- same job, same node, same 300 rows",
        sarena="runs/2026-11-02/figdata/table2.json -- full official benchmark, measured seconds per prompt; "
               "the (fast) rows are escape-then-stop on the lossless row above them",
        ar_stop="runs/2026-10-15-harvest/bon_decode_report_*-front.md -- the same rule on the original model",
        audit="runs/2026-10-22/analysis/loop_rule_triggers.md -- exact replay of the rule on AR outputs"),
        ladder=ladder, new_targets=new_t, sarena=sarena, ar_stop=ar, audit=audit)
    (OUT / "l_rules.json").write_text(json.dumps(doc, indent=1))
    print(f"l_rules.json: {len(ladder)} ladder rows, {len(sarena)} SArena cells, {len(audit)} audit rows")


# ---------------------------------------------------------------------------------------------------- P5
def build_p5():
    raw = json.loads((RUNS / "2026-10-29/analysis/b2_sweep.json").read_text())
    arms = {"ar": "AR", "b16": "DFlash (block 16)", "b32": "Ours (window 32)", "b32w16": "Ours (window 16)"}
    rows = []
    for mode, byarm in raw.items():
        for key, label in arms.items():
            if key not in byarm:
                raise KeyError((mode, key, list(byarm)))
            for c, v in byarm[key].items():
                rows.append(dict(mode=mode, arm=label, conc=int(c), tok_per_s=v["tok_per_s"],
                                 latency_s=v["mean_latency_s"], speedup=v["speedup"], accept=v["accept"]))
    g = {(r["mode"], r["arm"], r["conc"]): r for r in rows}
    assert round(g[("greedy", "Ours (window 32)", 1)]["speedup"], 2) == 9.29       # report_b2_sweep.md
    assert round(g[("greedy", "Ours (window 16)", 64)]["speedup"], 2) == 1.78
    assert g[("greedy", "AR", 64)]["tok_per_s"] == 8778.1
    doc = dict(meta=dict(
        source="runs/2026-10-29/analysis/b2_sweep.json (report_b2_sweep.md)",
        setup="SGLang 0.5.22, InternSVG-8B text-to-SVG, one H200, one node; tok/s is the whole client loop, "
              "latency is the mean per-request wall clock, accept is the engine's mean accepted tokens per "
              "verification step; speed-up is against AR at the same concurrency",
        arms="'Ours (window 16)' is the same block-32 head, verifying only 16 tokens per step -- no retraining"),
        rows=rows)
    (OUT / "p5_sglang.json").write_text(json.dumps(doc, indent=1))
    print(f"p5_sglang.json: {len(rows)} cells")


# ---------------------------------------------------------------------------------------------------- Q1
Q1_BLOCKS = [("OmniSVG-8B, text-to-SVG", "t2-bench300-front", "clip", 1),
             ("OmniSVG-8B, image-to-SVG", "i2-bench300-front", "lpips", -1),
             ("InternSVG-8B, text-to-SVG", "isvg-bench300-T0.6-front", "clip", 1)]
Q1_ARMS = {"ar1": ("AR", 1, False), "ar1s": ("AR", 1, True), "ar2s": ("AR", 2, True), "ar4s": ("AR", 4, True),
           "ar8s": ("AR", 8, True), "ar8": ("AR", 8, False), "ar8-fv": ("AR official", 8, False),
           "sd1": ("Ours", 1, False), "sd1s": ("Ours", 1, True), "sd2": ("Ours", 2, True),
           "sd4": ("Ours", 4, True), "sd8": ("Ours", 8, True), "sd16": ("Ours", 16, True)}


def build_q1():
    rows = []
    for blk, f, primary, sign in Q1_BLOCKS:
        head, hdr, body = md_tables(ROOT / f"runs/2026-10-15-harvest/bon_decode_report_{f}.md")[0]
        for c in body:
            r = dict(zip(hdr, c))
            if r["method"] not in Q1_ARMS:
                continue
            fam, n, stop = Q1_ARMS[r["method"]]
            q = {}
            for m in hdr:
                if not m.endswith(" Δ"):
                    continue
                hit = CI.search(r[m])
                q[m[:-2]] = [float(hit.group(1)), float(hit.group(2)), float(hit.group(3))]
            rows.append(dict(block=blk, arm=r["method"], family=fam, n=n, loop_stop=stop,
                             seconds=float(r["t total"]), seconds_decode=float(r["t decode"]),
                             seconds_score=float(r["t render+score"]), target_forward=int(r["target fwd"]),
                             valid=float(r["valid"]), quality=q, primary=primary, better=sign))
    g = {(r["block"], r["arm"]): r for r in rows}
    assert g[("OmniSVG-8B, text-to-SVG", "sd8")]["seconds"] == 4.71                 # logs/STATUS.md SS1.3
    assert g[("OmniSVG-8B, text-to-SVG", "ar1")]["target_forward"] == 1016
    assert g[("OmniSVG-8B, image-to-SVG", "sd16")]["seconds"] == 7.75
    doc = dict(meta=dict(
        source="runs/2026-10-15-harvest/bon_decode_report_{t2,i2,isvg-...T0.6}-bench300-front.md",
        setup="300 benchmark prompts present in every arm, one H200 per shard, HF eager, bf16; seconds are wall "
              "clock per prompt including rendering and scoring for the multi-candidate arms; quality deltas are "
              "paired per prompt against one AR sample, 95% bootstrap CI",
        fairness="arms with loop_stop=true carry the same loop rule, AR and ours alike; ar8-fv is the model "
                 "authors' own pipeline (8 samples, take the first that renders)"),
        rows=rows)
    (OUT / "q1_frontier.json").write_text(json.dumps(doc, indent=1))
    print(f"q1_frontier.json: {len(rows)} arms over {len(Q1_BLOCKS)} blocks")


if __name__ == "__main__":
    build_l()
    build_p5()
    build_q1()
