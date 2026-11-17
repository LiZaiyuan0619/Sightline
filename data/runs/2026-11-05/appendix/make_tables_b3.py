""" appendix batch 3: four positive-but-minor results as tables (user appendix mostly tables).

  A9   selection signals on InternSVG-8B text-to-SVG at T 0.3 / 0.6 / 1.0   <- runs/2026-10-16-harvest/isvg-select-T*/report.md
  A13  acceptance against the target's own entropy, three models (Ours)      <- $SIGHTLINE_RUNS/r1029/entropy/{t2,i2,isvg}-ours.json
  A14  where one speculative round's time goes, seven methods                 <- runs/2026-10-30/analysis/timing_split_all.json
  A20  the slowest, middle and fastest rows                                   <- runs/2026-10-29/analysis/hard_{t2,i2}.json

A13 shows Ours only: its claim is about the target (near-deterministic positions are almost always accepted), and the
per-position rate is not comparable across methods with different block lengths (a block-16 draft scores higher per
position while keeping fewer tokens per round). Caption above the table; best bold / second underlined only in
columns where higher is better.
  python3 runs/2026-11-05/appendix/make_tables_b3.py  -> tex/tables/app_a{9,13,14,20}_*.tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
U = RUNS
OUT = ROOT / "tex/tables"
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location("b1", RUNS / "2026-11-05/appendix/make_tables_b1.py")
B1 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(B1)
marks, cell, write = B1.marks, B1.cell, B1.write


def table(env, cap, label, colspec, head, lines, width="\\textwidth", sep="2.0pt"):
    return "\n".join([f"\\begin{{{env}}}[t]", cap, f"\\label{{{label}}}", "\\centering", "\\footnotesize",
                      f"\\setlength{{\\tabcolsep}}{{{sep}}}",
                      f"\\begin{{tabular*}}{{{width}}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}", "\\toprule",
                      head, "\\midrule", *lines, "\\bottomrule", "\\end{tabular*}", f"\\end{{{env}}}", ""])


# ---------------------------------------------------------------------------------------------- A9
RULES = [("random (mean of 8)", "random pick"), ("max likelihood", "highest likelihood"),
         ("min likelihood", "lowest likelihood"), ("text consensus (MBR)", "text consensus"),
         ("render score (max CLIP)", "\\textbf{render score}")]
NUM = r"([+-]\d+\.\d+)"


def a9():
    data = {}
    for T in ("0.3", "0.6", "1.0"):
        md = (ROOT / f"runs/2026-10-16-harvest/isvg-select-T{T}/report.md").read_text()
        for key, _ in RULES:
            m = re.search(re.escape(f"| {key} |") + r"\s*" + r"\s*\|\s*".join(
                [NUM + r" \[" + NUM + ", " + NUM + r"\]"] * 3), md)
            assert m, (T, key)
            v = [float(x) for x in m.groups()]
            data[(T, key)] = [(v[0], v[1], v[2]), (v[3], v[4], v[5]), (v[6], v[7], v[8])]   # clip, aes, hps
    # the claim the caption makes, checked: render score's interval excludes 0 everywhere except aesthetic at T 1.0
    excl = {(T, j): data[(T, RULES[-1][0])][j][1] > 0 for T in ("0.3", "0.6", "1.0") for j in range(3)}
    assert [k for k, v in excl.items() if not v] == [("1.0", 1)], excl
    lines = []
    cols = [(T, j) for T in ("0.3", "0.6", "1.0") for j in range(3)]
    mk = {}
    for (T, j) in cols:
        for (key, _), k in zip(RULES, marks([data[(T, key)][j][0] for key, _ in RULES], 3)):
            mk[(T, j, key)] = k
    for key, lab in RULES:
        cells = []
        for (T, j) in cols:
            v = data[(T, key)][j][0]
            s = f"{v:+.3f}".replace("+0.000", "0.000").replace("-0.000", "0.000")
            s = {"b": f"\\textbf{{{s}}}", "u": f"\\underline{{{s}}}"}.get(mk[(T, j, key)], s)
            cells.append(s)
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    head = (" & \\multicolumn{3}{c}{$T$ 0.3} & \\multicolumn{3}{c}{$T$ 0.6 (official)} & \\multicolumn{3}{c}{$T$ 1.0} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
            "Pick by & " + " & ".join(["CLIP", "Aesthetic", "HPS"] * 3) + " \\\\")
    cap = ("\\caption{\\textbf{Choosing among eight samples: three temperatures.} InternSVG-8B text-to-SVG, 300 "
           "prompts, eight samples each. Quality of the chosen sample minus a random pick, paired per prompt. CLIP is "
           "the render score's own signal; Aesthetic and HPS are independent of it. The render score's 95\\% "
           "bootstrap interval excludes zero in every column except Aesthetic at $T$ 1.0. Best in bold, second "
           "underlined.}")
    write("app_a9_selection_temps", table("table*", cap, "app:selection-temps", "l*{9}{c}", head, lines))


# ---------------------------------------------------------------------------------------------- A13
ENT = [("t2", "OmniSVG-8B, text-to-SVG"), ("i2", "OmniSVG-8B, image-to-SVG"), ("isvg", "InternSVG-8B, text-to-SVG")]


def a13():
    B = {}
    for k, _ in ENT:
        d = json.loads((U / f"r1029/entropy/{k}-ours.json").read_text())["ours"]["buckets"]
        B[k] = d
    edges = [(b["lo"], b["hi"]) for b in B["t2"]]
    assert all([(b["lo"], b["hi"]) for b in B[k]] == edges for k, _ in ENT)
    assert abs(B["t2"][0]["accept"] - 0.968) < 5e-4 and B["t2"][0]["n"] == 128850      # entropy/t2-ours.md
    tot = {k: sum(b["n"] for b in B[k]) for k, _ in ENT}
    lines = []
    for i, (lo, hi) in enumerate(edges):
        hs = f"{lo:g}--{hi:g}" if hi not in (None, float("inf")) and hi < 1e9 else f"$\\geq${lo:g}"
        cells = []
        for k, _ in ENT:
            b = B[k][i]
            cells += [f"{100 * b['n'] / tot[k]:.1f}", f"{b['accept']:.3f}"]
        lines.append(f"{hs} & " + " & ".join(cells) + " \\\\")
    head = (" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in ENT) + " \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Target entropy (nats) & " + " & ".join(["positions (\\%)", "accepted"] * 3) + " \\\\")
    sh = [100 * B[k][0]["n"] / tot[k] for k, _ in ENT]
    ac = [100 * B[k][0]["accept"] for k, _ in ENT]
    cap = ("\\caption{\\textbf{Acceptance follows the target's own certainty.} For every generated position, the "
           "entropy of the target's next-token distribution and whether our draft's token was kept (greedy, the "
           f"outputs of Table~\\ref{{tab:main}}). {min(sh):.0f}--{max(sh):.0f}\\% of the positions are "
           f"near-deterministic (below 0.01 nats) and are kept {min(ac):.0f}--{max(ac):.0f}\\% of the time.}}")
    write("app_a13_entropy", table("table*", cap, "app:entropy", "l*{6}{c}", head, lines, sep="3.0pt"))


# ---------------------------------------------------------------------------------------------- A14
T14 = ["PLD", "REST", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]


def a14():
    d = json.loads((RUNS / "2026-10-30/analysis/timing_split_all.json").read_text())
    M = d["methods"]
    ar = d["ar_ms_per_token"]
    assert abs(M["Ours"]["tau"] - 19.02) < 0.01 and abs(ar - 10.92) < 0.01        # timing_split_all.md
    sp = {m: M[m]["tok_per_s"] / (1000.0 / ar) for m in T14}
    assert abs(sp["Ours"] - 9.66) < 0.01 and abs(sp["EAGLE-3"] - 3.44) < 0.01
    mt = dict(zip(T14, marks([M[m]["tau"] for m in T14], 2)))
    ms = dict(zip(T14, marks([sp[m] for m in T14], 2)))
    lines = []
    for m in T14:
        v = M[m]
        lines.append(f"{B1.method_name(m)} & {cell(v['tau'], mt[m])} & {v['ms_round']:.1f} & {v['draft']:.1f} & "
                     f"{v['tree']:.1f} & {v['verify']:.1f} & {v['other']:.1f} & {cell(sp[m], ms[m], x=True)} \\\\")
    head = (" & & & \\multicolumn{4}{c}{ms per round, split} & \\\\\n\\cmidrule(lr){4-7}\n"
            "Method & $\\tau$ & ms / round & draft & tree build & verify & other & speed-up \\\\")
    cap = (f"\\caption{{\\textbf{{Where one speculative round's time goes.}} OmniSVG-8B text-to-SVG, 20 icons and 20 "
           f"illustrations, greedy, one H200; the original model takes {ar:.2f} ms per token on the same rows. "
           "Instrumented timing on this subset, so the speed-ups differ slightly from Table~\\ref{tab:main}. Every "
           "method's verify forward costs about the same; EAGLE-3 loses on its layer-by-layer draft. Bold and "
           "underline mark the best and second $\\tau$ and speed-up.}")
    write("app_a14_timing_split", table("table", cap, "app:timing-split", "l*{7}{c}", head, lines,
                                        width="\\columnwidth", sep="1.6pt"))


# ---------------------------------------------------------------------------------------------- A20
def a20():
    lines = []
    for k, name in (("t2", "OmniSVG-8B, text-to-SVG"), ("i2", "OmniSVG-8B, image-to-SVG")):
        d = json.loads((ROOT / f"runs/2026-10-29/analysis/hard_{k}.json").read_text())
        assert d["n"] == 900
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{name}}}}} \\\\")
        for b in d["deciles"]:
            lab = {"bottom 10%": "slowest 10\\%", "middle": "middle 80\\%", "top 10%": "fastest 10\\%"}[b["bucket"]]
            lines.append(f"{lab} & {b['n']} & {b['speedup']:.2f}$\\times$ & {b['tau']:.2f} & {b['len']:.0f} & "
                         f"{100 * b['full_block']:.0f} & {100 * b['eos_rate']:.0f} & {100 * b['ar_capped']:.0f} \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    head = ("Rows & $n$ & speed-up & $\\tau$ & tokens & full-block rounds (\\%) & ended normally (\\%) & "
            "original model hit the cap (\\%) \\\\")
    cap = ("\\caption{\\textbf{The hardest and the easiest prompts.} Our decoder's rows (greedy, the outputs of "
           "Table~\\ref{tab:main}) ranked by their own speed-up over the original model and split into the slowest "
           "10\\%, the middle 80\\% and the fastest 10\\%. The slowest tenth is still 2.6--2.8$\\times$ faster; the "
           "fastest tenth is mostly outputs that loop to the length cap.}")
    write("app_a20_hard_rows", table("table*", cap, "app:hard-rows", "l*{7}{c}", head, lines, sep="3.0pt"))


if __name__ == "__main__":
    a9()
    a13()
    a14()
    a20()
