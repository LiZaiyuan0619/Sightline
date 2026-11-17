"""Table 3 draft v1: the generality table, compact. Rows = model x decoding (12), columns = two subsets
x six methods (12), one speed-up per cell; bold = best, underline = second, within each subset of each row.
tokens per round and the official sampling settings go to the appendix table (all 24 cells x 6 methods x (speed-up, tokens per round)).

  python3 paper/claims/T3-generality/round1/make_table.py
  cd tex && PATH=python latexmk -pdf table3_preview.tex
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import t3data as T  # noqa: E402
from sightline.paths import RUNS

ROOT = T.ROOT
SHOWM = {"PLD": "PLD", "DFlash": "DFlash", "EAGLE-3": "EAGLE-3", "DDTree": "DDTree", "DARTree": "DARTree",
         "Ours": r"\textbf{Ours}"}
DESC = {f: T.TASK[f] for f in T.FAMILIES}


def fmt_row(vals):
    order = sorted(range(len(vals)), key=lambda i: -round(vals[i], 2))
    best, second = round(vals[order[0]], 2), round(vals[order[1]], 2)
    out = []
    for v in vals:
        s = f"{v:.2f}"
        if round(v, 2) == best:
            s = rf"\textbf{{{s}}}"
        elif round(v, 2) == second:
            s = rf"\underline{{{s}}}"
        out.append(s)
    return out


def table():
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\caption{\textbf{Generality: six more SVG and Lottie models.} Speed-up of the wall clock per token over "
             r"the model's own decoding on the same node; sampling uses each model's released settings. Every trained "
             r"draft has the same budget (100k prompts, 2 epochs). Subsets are icon and illustration (OmniLottie-3B: real "
             r"and synthetic). Best in bold, second underlined. \textcolor{red}{[Draft 11-04 v1 from "
             r"runs/2026-11-04/ext\_v4.json; acceptance lengths and sampling settings in the appendix table.]}}")
    L.append(r"\label{tab:generality}")
    L.append(r"\centering")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{2.0pt}")
    L.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll*{12}{c}@{}}")
    L.append(r"\toprule")
    L.append(r" & & \multicolumn{6}{c}{Icon (OmniLottie-3B: real)} & \multicolumn{6}{c}{Illustration (OmniLottie-3B: synthetic)} \\")
    L.append(r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}")
    L.append(r"Model & Decoding & " + " & ".join(SHOWM[m] for m in T.METHODS) + " & "
             + " & ".join(SHOWM[m] for m in T.METHODS) + r" \\")
    L.append(r"\midrule")
    for k, f in enumerate(T.FAMILIES):
        for j, mode in enumerate(T.MODES):
            cells = []
            for sub in T.SUBSETS[f]:
                cells += fmt_row([T.spd(f, m, mode, sub) for m in T.METHODS])
            name = (r"\multirow{2}{*}{\shortstack[l]{" + f + r"\\[-1pt]\scriptsize\color{gray!70!black} "
                    + DESC[f] + "}}" if j == 0 else "")
            dec = "greedy" if mode == "greedy" else "sampling"
            L.append(f"{name} & {dec} & " + " & ".join(cells) + r" \\")
        if k < len(T.FAMILIES) - 1:
            L.append(r"\addlinespace[2pt]")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular*}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    out = RUNS.parent / "tables/table3_draft_v1.tex"
    out.write_text(table())
    pv = ROOT / "tex/table3_preview.tex"
    pv.write_text((ROOT / "tex/tables_preview.tex").read_text().split(r"\begin{document}")[0]
                  + "\\begin{document}\n\\input{tables/table3_draft_v1}\n\\end{document}\n")
    # check: every printed number comes from the data, 144 cells
    import re
    nums = re.findall(r"\d+\.\d\d", out.read_text().split(r"\midrule")[1])
    assert len(nums) == 144, len(nums)
    print("wrote", out, "and", pv, "-", len(nums), "cells")
