""" appendix batch 1: full versions of the main-text tables (LaTeX, generated, never typed by hand).

  A1  table 1 in full, one table per decoding mode: every benchmark gets (speed-up, tokens per round)  <- runs/2026-11-02/figdata/table1.json
  A2  table 1 on loop-free content: tokens per round of the rounds before each output's loop onset     <- healthy_tau.json (this dir)
  A3  table 3 in full: settings / speed-up / tokens per round, plus StarVector's code-default T 1.0     <- runs/2026-11-04/ext_v4.json
  A7  P5 in full: every load x every arm, speed-up and engine acceptance                  <- runs/2026-11-04/b2w/b2w.json

House rules (guide SS6): caption ABOVE the table, best bold / second underlined in every column (by printed
value, ties share the mark), one number per cell, no notes under the table, method order = house.METHODS (families
together, DDTree next to Ours). Every source number is re-checked against its source before anything is written.
  python3 runs/2026-11-05/appendix/make_tables_b1.py  -> tex/tables/app_*.tex
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
HERE = Path(__file__).resolve().parent
OUT = ROOT / "tex/tables"
ORDER = ["PLD", "REST", "SpS", "EAGLE-3", "ViSpec", "DFlash", "DARTree", "DDTree", "Ours"]
T1KEY = {"Ours": "Ours (lossless)"}
BENCH = [("MMSVG", ["MMSVG-Icon", "MMSVG-Illus."]), ("MMSVG-2M$^\\dagger$", ["MMSVG-2M-Icon†", "MMSVG-2M-Illus.†"]),
         ("SArena", ["SArena-Icon", "SArena-Illus."])]
COLS = [c for _, cs in BENCH for c in cs]


# ---------------------------------------------------------------------------------------------- formatting
def marks(vals, nd):
    """'b' for the best printed value, 'u' for the second (ties share), '' otherwise; None stays None."""
    pr = [None if v is None else round(v, nd) for v in vals]
    uniq = sorted({p for p in pr if p is not None}, reverse=True)
    best = uniq[0] if uniq else None
    second = uniq[1] if len(uniq) > 1 else None
    return [None if p is None else ("b" if p == best else "u" if p == second else "") for p in pr]


def cell(v, mk, nd=2, x=False):
    s = f"{v:.{nd}f}"
    s = {"b": f"\\textbf{{{s}}}", "u": f"\\underline{{{s}}}"}.get(mk, s)
    return s + ("$\\times$" if x else "")


def method_name(m):
    return "\\textbf{Ours}" if m == "Ours" else m


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", OUT / f"{name}.tex")


# ---------------------------------------------------------------------------------------------- A1 / A2
def t1_blocks():
    t1 = json.loads((RUNS / "2026-11-02/figdata/table1.json").read_text())
    blocks = {}
    for b in t1["blocks"]:
        assert [c[0] for c in b["columns"]] == COLS, b["block"]
        blocks.setdefault(b["block"], {})[b["mode"]] = b["rows"]
    return blocks


def bench_header(ncol_per_bench, metric_names, extra_label="Mean"):
    """Three header rows: benchmark / subset / metric."""
    k = ncol_per_bench
    h1 = " & " + " & ".join(f"\\multicolumn{{{2 * k}}}{{c}}{{{b}}}" for b, _ in BENCH) + \
         f" & \\multicolumn{{{k}}}{{c}}{{{extra_label}}} \\\\"
    rules1 = "".join(f"\\cmidrule(lr){{{2 + 2 * k * i}-{1 + 2 * k * (i + 1)}}}" for i in range(3)) + \
        f"\\cmidrule(lr){{{2 + 6 * k}-{1 + 7 * k}}}"
    h2 = " & " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{{s}}}" for _ in BENCH for s in ("Icon", "Illus.")) + \
         (f" & \\multicolumn{{{k}}}{{c}}{{}} \\\\" if k > 1 else " & \\\\")
    rules2 = "".join(f"\\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(6)) if k > 1 else ""
    h3 = "Method & " + " & ".join(metric_names * 7) + " \\\\"
    return "\n".join([h1, rules1, h2, rules2, h3]) if k > 1 else "\n".join([h1, rules1, h3.replace("Method", "Method")])


def a1(blocks):
    for mode, title, label in (("greedy", "greedy decoding", "app:t1-greedy"),
                               ("sample", "each model's official sampling", "app:t1-sample")):
        lines = []
        for bname, modes in blocks.items():
            R = modes[mode]
            ms = [m for m in ORDER if T1KEY.get(m, m) in R and any(R[T1KEY.get(m, m)])]
            rows = {m: R[T1KEY.get(m, m)] for m in ms}
            assert all(all(c is not None for c in rows[m]) for m in ms), (bname, mode)
            sp = {m: [c["speedup"] for c in rows[m]] + [st.mean(c["speedup"] for c in rows[m])] for m in ms}
            ta = {m: [c["tau"] for c in rows[m]] + [st.mean(c["tau"] for c in rows[m])] for m in ms}
            msp = {m: [] for m in ms}
            mta = {m: [] for m in ms}
            for j in range(7):
                for m, k in zip(ms, marks([sp[m][j] for m in ms], 2)):
                    msp[m].append(k)
                for m, k in zip(ms, marks([ta[m][j] for m in ms], 2)):
                    mta[m].append(k)
            lines.append(f"\\multicolumn{{15}}{{l}}{{\\textit{{{bname}}}}} \\\\")
            for m in ms:
                cells = []
                for j in range(7):
                    cells += [cell(sp[m][j], msp[m][j], x=True), cell(ta[m][j], mta[m][j])]
                lines.append(method_name(m) + " & " + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        lines = lines[:-1]
        cap = (f"\\caption{{\\textbf{{Table~\\ref{{tab:main}} in full, {title}.}} For every benchmark, the wall-clock "
               "speed-up over the model's own decoding on the same H200 (Sp.) and the mean acceptance length "
               "($\\tau$, tokens kept per verification round, all tokens). Mean is over the six benchmarks. "
               "$^\\dagger$Held-out captions of the drafts' training source. SpS needs a smaller model of the same "
               "family, which only OmniSVG-8B has; ViSpec is a vision-language method and runs on the image-to-SVG "
               "models. Best in bold, second underlined, within each model.}")
        body = "\n".join([
            "\\begin{table*}[t]", cap, f"\\label{{{label}}}", "\\centering", "\\footnotesize",
            "\\setlength{\\tabcolsep}{1.5pt}",
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{14}{c}@{}}", "\\toprule",
            bench_header(2, ["Sp.", "$\\tau$"]), "\\midrule", *lines, "\\bottomrule", "\\end{tabular*}",
            "\\end{table*}", ""])
        write(f"app_a1_table1_{mode}", body)


def a2(blocks):
    H = json.loads((HERE / "healthy_tau.json").read_text())["cells"]
    lines = []
    for bname, modes in blocks.items():
        ms = [m for m in ORDER if m in H[bname]["greedy"]]
        vals = {}
        for m in ms:
            v = []
            for mode in ("greedy", "sample"):
                cs = H[bname][mode][m]
                for c, want in zip(cs, modes[mode][T1KEY.get(m, m)]):      # same rows as table 1 (checked upstream)
                    assert c is not None and abs(c["tau_table1"] - want["tau"]) < 1e-9 and c["n"] == want["n"]
                h = [c["tau_healthy"] for c in cs]
                v += h + [st.mean(h)]
            vals[m] = v
        mk = {m: [] for m in ms}
        for j in range(14):
            for m, k in zip(ms, marks([vals[m][j] for m in ms], 2)):
                mk[m].append(k)
        lines.append(f"\\multicolumn{{15}}{{l}}{{\\textit{{{bname}}}}} \\\\")
        for m in ms:
            lines.append(method_name(m) + " & " + " & ".join(cell(vals[m][j], mk[m][j]) for j in range(14)) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    sub = " & ".join(["Icon", "Illus."] * 3)
    h1 = (" & \\multicolumn{7}{c}{Greedy} & \\multicolumn{7}{c}{Official sampling} \\\\\n"
          "\\cmidrule(lr){2-8}\\cmidrule(lr){9-15}\n"
          " & \\multicolumn{2}{c}{MMSVG} & \\multicolumn{2}{c}{MMSVG-2M$^\\dagger$} & \\multicolumn{2}{c}{SArena} & "
          "& \\multicolumn{2}{c}{MMSVG} & \\multicolumn{2}{c}{MMSVG-2M$^\\dagger$} & \\multicolumn{2}{c}{SArena} & \\\\\n"
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}"
          "\\cmidrule(lr){13-14}\n"
          f"Method & {sub} & Mean & {sub} & Mean \\\\")
    cap = ("\\caption{\\textbf{Acceptance length on loop-free content.} Mean acceptance length $\\tau$ over the "
           "verification rounds that start before the output's loop onset, on the same outputs as "
           "Table~\\ref{tab:main}; loop segments hold 13--75\\% of the generated tokens, depending on model and benchmark. Speed-up is not split this "
           "way because the wall clock is not recorded per segment. Best in bold, second underlined, within each "
           "model.}")
    body = "\n".join(["\\begin{table*}[t]", cap, "\\label{app:t1-healthy}", "\\centering", "\\footnotesize",
                      "\\setlength{\\tabcolsep}{2.2pt}",
                      "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{14}{c}@{}}", "\\toprule", h1,
                      "\\midrule", *lines, "\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""])
    write("app_a2_table1_healthy", body)


# ---------------------------------------------------------------------------------------------- A3
T3M = ["PLD", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]


def t3_settings():
    """Backbone / tokens / task and the official sampling string per family, parsed from ext_tables_v4.md."""
    md = (RUNS / "2026-11-04/ext_tables_v4.md").read_text().splitlines()
    out, appendix = {}, False
    for ln in md:
        if ln.startswith("## Appendix"):            # StarVector's T 1.0 rows (code defaults), not the official setting
            appendix = True
        p = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(p) < 13 or p[0] in ("Model", "---") or p[0].startswith("-"):
            continue
        fam, bb, dec = p[0], p[1], p[2]
        o = out.setdefault(fam, {"backbone": bb, "sampling": None, "sampling_t1": None})
        assert o["backbone"] == bb
        if dec.startswith("sample"):
            o["sampling_t1" if appendix else "sampling"] = dec[len("sample"):].strip().strip("()")
    assert out["StarVector-8B"]["sampling"].startswith("T 0.7") and out["StarVector-1B"]["sampling"].startswith("T 0.2")
    assert out["StarVector-8B"]["sampling_t1"] == "T 1.0, top-p 0.9"
    return out


def a3():
    d = json.loads((RUNS / "2026-11-04/ext_v4.json").read_text())
    S = t3_settings()
    fams = list(d)
    assert set(fams) == set(S), (fams, list(S))
    # A3a settings
    rows = []
    for f in fams:
        bb, rest = S[f]["backbone"].split(", ", 1)
        tok, task = re.match(r"(.*) \((.*)\)", rest).groups()
        subs = sorted({k.split("|")[2] for k in d[f]})
        n = {s: d[f][f"Ours|greedy|{s}"]["n"] for s in subs}
        rows.append(f"{f} & {bb} & {tok} & {task} & {S[f]['sampling']} & " + " / ".join(str(n[s]) for s in subs)
                    + " \\\\")
    cap = ("\\caption{\\textbf{The six models of Table~\\ref{tab:generality}.} Official sampling is each model's "
           "released setting (StarVector: its released evaluation configuration, with the library's default "
           "top-$k$). Prompts per subset: icon / illustration (OmniLottie-3B: real / synthetic). Every trained "
           "draft has the same budget (100k prompts, 2 epochs).}")
    write("app_a3_table3_settings", "\n".join([
        "\\begin{table*}[t]", cap, "\\label{app:t3-settings}", "\\centering", "\\footnotesize",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llllll@{}}", "\\toprule",
        "Model & Backbone & Output tokens & Task & Official sampling & Prompts \\\\", "\\midrule", *rows,
        "\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""]))
    # A3b / A3c speed-up and tokens per round
    for key, nm, x, label, what in (("speedup", "speedup", True, "app:t3-speed", "Wall-clock speed-up over the model's "
                                     "own decoding on the same node"),
                                    ("tau", "tau", False, "app:t3-tau", "Mean acceptance length $\\tau$ (all tokens)")):
        lines = []
        for f in fams:
            subs = sorted({k.split("|")[2] for k in d[f]})
            modes = [("greedy", "greedy"), ("sample", "official sampling")]
            if any("sample-T1.0" in k for k in d[f]):
                modes.append(("sample-T1.0", "sampling, $T$ 1.0"))
            for i, (mo, mlab) in enumerate(modes):
                cells = []
                for s in subs:
                    v = [d[f][f"{m}|{mo}|{s}"][key] for m in T3M]
                    cells += [cell(a, k, x=x) for a, k in zip(v, marks(v, 2))]
                lines.append((f + (" " if i else "") if i == 0 else "") + f" & {mlab} & " + " & ".join(cells) + " \\\\")
            lines.append("\\addlinespace[2pt]")
        lines = lines[:-1]
        head = (" & & \\multicolumn{6}{c}{Icon (OmniLottie-3B: real)} & "
                "\\multicolumn{6}{c}{Illustration (OmniLottie-3B: synthetic)} \\\\\n"
                "\\cmidrule(lr){3-8}\\cmidrule(lr){9-14}\n"
                "Model & Decoding & " + " & ".join([method_name(m) for m in T3M] * 2) + " \\\\")
        cap = (f"\\caption{{\\textbf{{Table~\\ref{{tab:generality}} in full: {('speed-up' if x else 'acceptance length')}.}} "
               f"{what}. StarVector's last row samples at $T$ 1.0, top-$p$ 0.9, the defaults of its released generation "
               "code (the official row is its released evaluation configuration). Best in bold, second underlined, "
               "per row and subset.}")
        write(f"app_a3_table3_{nm}", "\n".join([
            "\\begin{table*}[t]", cap, f"\\label{{{label}}}", "\\centering", "\\footnotesize",
            "\\setlength{\\tabcolsep}{2.0pt}",
            "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll*{12}{c}@{}}", "\\toprule", head, "\\midrule",
            *lines, "\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""]))


# ---------------------------------------------------------------------------------------------- A7
P5ARMS = [("b16", "DFlash", "16", "16"), ("b16w8", "DFlash", "16", "8"), ("b32", "Ours", "32", "32"),
          ("b32w24", "Ours", "32", "24"), ("b32w16", "Ours", "32", "16"), ("b32w8", "Ours", "32", "8")]
P5PUB = {("greedy", "b16", 1): 7.22, ("greedy", "b32", 1): 9.29, ("greedy", "b16", 8): 6.19,
         ("sample", "b32", 1): 5.53, ("sample", "b16w8", 64): 1.36}         # report_b2w.md, spot checks


def a7():
    d = json.loads((RUNS / "2026-11-04/b2w/b2w.json").read_text())["cells"]
    for (mo, arm, c), v in P5PUB.items():
        assert abs(d[mo][arm][str(c)]["speedup"] - v) < 0.006, (mo, arm, c)
    for key, nm, x, label, nd in (("speedup", "speedup", True, "app:p5-speed", 2),
                                  ("accept", "accept", False, "app:p5-accept", 1)):
        lines = []
        for mo, mlab in (("greedy", "Greedy"), ("sample", "Official sampling")):
            lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{mlab}}}}} \\\\")
            for c in sorted(d[mo]["b16"], key=int):
                ar = d[mo]["b16"][c]["tok_per_s"] / d[mo]["b16"][c]["speedup"]
                v = [d[mo][a][c][key] for a, *_ in P5ARMS]
                cells = [cell(a, k, nd=nd, x=x) for a, k in zip(v, marks(v, nd))]
                lines.append(f"{c} & {ar:.0f} & " + " & ".join(cells) + " \\\\")
            lines.append("\\midrule")
        lines = lines[:-1]
        head = (" & & \\multicolumn{2}{c}{DFlash (block 16)} & \\multicolumn{4}{c}{\\textbf{Ours} (block 32)} \\\\\n"
                "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n"
                " & & \\multicolumn{6}{c}{verification window} \\\\\n"
                "Concurrency & AR tok/s & " + " & ".join(w for *_, w in P5ARMS) + " \\\\")
        what = ("Throughput speed-up over autoregressive decoding at the same concurrency" if x else
                "Mean accepted tokens per verification step reported by the engine")
        cap = (f"\\caption{{\\textbf{{Serving in SGLang, every load and window (speed-up).}} "
               f"{what}, InternSVG-8B text-to-SVG on one H200. The verification window is set separately from the "
               "draft's block length. Arms measured in a later job are put on the main run's node by their own "
               "re-run of DFlash block 16, window 16 (factors 0.96--1.07). AR tok/s is the autoregressive "
               "throughput the speed-ups divide by. Best in bold, second underlined, per row.}") if x else (
               f"\\caption{{\\textbf{{Serving in SGLang, every load and window (acceptance).}} {what}, for the runs "
               "of Table~\\ref{app:p5-speed}. Best in bold, second underlined, per row.}")
        write(f"app_a7_p5_{nm}", "\n".join([
            "\\begin{table}[t]", cap, f"\\label{{{label}}}", "\\centering", "\\footnotesize",
            "\\setlength{\\tabcolsep}{1.2pt}",
            "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}rr*{6}{c}@{}}", "\\toprule", head, "\\midrule",
            *lines, "\\bottomrule", "\\end{tabular*}", "\\end{table}", ""]))


if __name__ == "__main__":
    B = t1_blocks()
    a1(B)
    a2(B)
    a3()
    a7()
