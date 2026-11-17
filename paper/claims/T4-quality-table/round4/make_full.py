"""Quality table, round 4 (main-line session 8): the appendix tables with every metric, for the merged table.

The merged table (round 3, main.pdf Table 1) prints two metrics per subset and says the appendix gives every metric.
The old main-text tables (table2a / table2b) had every metric but only the rows published before the fill runs, and
their labels are now aliases of the merged table. These two tables replace them in the appendix:
  table_full_t2s.tex   text-to-SVG,  FID, FID-C, CLIP-T2I, CLIP-I2I and seconds per subset, label app:quality-t2s
  table_full_i2s.tex   image-to-SVG, DINO, LPIPS, SSIM, PSNR and seconds per subset,        label app:quality-i2s
Rows are the merged table's three groups, plus two best-of-8 rows that stay out of the main text (user, ):
OmniSVG's released pipeline (eight samples, first valid one), and AR x8 as officially decoded, without the loop stop.
Marks rank the seconds within a group only, as in the merged table.

Data: runs/2026-11-09/table-fill/figdata/table2_v2.json. The printed numbers of the old tables are asserted below.
Run: python3 paper/claims/T4-quality-table/round4/make_full.py   (writes the two.tex and page_full.pdf/png)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))   # run as a script from the package root
import json
import re
import shutil
import subprocess
from pathlib import Path
from sightline.paths import RUNS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
V0 = ROOT / "draft/v0"
TEX = "pdflatex"
BUILD = Path("/tmp/sd8/t4full")
D = json.loads((RUNS / "2026-11-09/table-fill/figdata/table2_v2.json").read_text())["blocks"]

MODELS = [("OmniSVG-8B", "its released sampling settings"),
          ("InternSVG-8B", "greedy; best of 8 at $T{=}0.6$, top-$p$ 0.9")]
SUBS = [("icon", "SArena-Icon", "6,013"), ("illustration", "SArena-Illus.", "2,001")]
TASKS = {"text-to-SVG": ("t2s", [("FID", "FID", 2, "$\\downarrow$"), ("FID-C", "FID-C", 2, "$\\downarrow$"),
                                  ("CLIP-T2I", "CLIP-T2I", 2, "$\\uparrow$"), ("CLIP-I2I", "CLIP-I2I", 2, "$\\uparrow$")]),
         "image-to-SVG": ("i2s", [("DINO-Score", "DINO", 3, "$\\uparrow$"), ("LPIPS", "LPIPS", 3, "$\\downarrow$"),
                                   ("SSIM", "SSIM", 3, "$\\uparrow$"), ("PSNR", "PSNR", 2, "$\\uparrow$")])}
OURS = "\\textbf{Ours}"
GROUPS = [("same\\\\outputs", [("AR", "AR"), ("EAGLE-3", "EAGLE-3"), ("DFlash", "DFlash"), ("DDTree", "DDTree"),
                                ("DARTree", "DARTree"), (OURS, "Ours")]),
          ("loop\\\\control", [("AR", "AR (fast)"), ("EAGLE-3", "EAGLE-3 (fast)"), ("DFlash", "DFlash (fast)"),
                               ("DDTree", "DDTree (fast)"), ("DARTree", "DARTree (fast)"), (OURS, "Ours (fast)")]),
          ("best\\\\of 8", [("Authors' pipeline", "Official 8, first valid"), ("AR, no loop stop", "AR x8"),
                            ("AR", "AR x8s"), ("DFlash", "DFlash x8"), (OURS, "Ours x8")])]
CAPTION = {
    "text-to-SVG": ("\\textbf{Text-to-SVG on the full SArena test sets, every official metric.} Rows and groups as in "
                    "Table~\\ref{tab:quality}, with two more best-of-8 rows. The authors' pipeline of OmniSVG samples "
                    "eight and keeps the first valid one, and AR without the loop stop is the original model's best of 8 "
                    "as officially decoded. Seconds per prompt on one H200, with \\textbf{fastest} in bold and "
                    "\\underline{second}{} underlined within a group."),
    "image-to-SVG": ("\\textbf{Image-to-SVG on the full SArena test sets, every official metric.} Layout as in "
                     "Table~\\ref{app:quality-t2s}, with candidates scored by DINO similarity to the input image."),
}
LABEL = {"text-to-SVG": "app:quality-t2s", "image-to-SVG": "app:quality-i2s"}
# printed in main.pdf's old Tables 1 and 2 (draft/v0/tables/table2a.tex, table2b.tex), which these tables replace
CHECK = {("OmniSVG-8B", "text-to-SVG", "AR", "icon", "FID-C"): "2.91",
         ("OmniSVG-8B", "text-to-SVG", "Official 8, first valid", "illustration", "s"): "24.95",
         ("OmniSVG-8B", "text-to-SVG", "AR x8", "icon", "CLIP-I2I"): "80.27",
         ("OmniSVG-8B", "image-to-SVG", "DARTree", "icon", "PSNR"): "14.54",
         ("OmniSVG-8B", "image-to-SVG", "Ours", "illustration", "s"): "2.94",
         ("InternSVG-8B", "text-to-SVG", "AR", "illustration", "s"): "43.58"}


def row(model, task, key):
    return D[f"{model}, {task}"].get(key)


def cell(model, task, key, sub, metric):
    """(value, printed) or None when the row does not exist for this model (the authors' pipeline is OmniSVG's)."""
    r = row(model, task, key)
    if not r or sub not in r:
        return None
    c = r[sub]
    if metric == "s":
        return c["seconds"], f"{c['seconds']:.2f}"
    dec = next(d for k, _, d, _ in TASKS[task][1] if k == metric)
    v = c["metrics"][metric]
    return v, f"{v:.{dec}f}"


for (m, t, r, s, k), want in CHECK.items():
    got = cell(m, t, r, s, k)[1]
    assert got == want, (m, t, r, s, k, got, want)


def columns(task):
    return [(sub, k, lab, arr) for sub, _, _ in SUBS for k, lab, _, arr in TASKS[task][1] + [("s", "s", None, "$\\downarrow$")]]


def marks(model, task, keys, cols):
    out = {}
    for j, (sub, k, _, _) in enumerate(cols):
        if k != "s":
            continue
        vals = {key: float(cell(model, task, key, sub, k)[1]) for key in keys if cell(model, task, key, sub, k)}
        order = sorted(set(vals.values()))
        for key, v in vals.items():
            if order.index(v) < 2:
                out[(key, j)] = "bu"[order.index(v)]
    return out


def table(task):
    cols = columns(task)
    n = len(cols)
    lines = ["\\begin{table*}[t]", f"\\caption{{{CAPTION[task]}}}", f"\\label{{{LABEL[task]}}}", "\\centering", "\\small",
             "\\setlength{\\tabcolsep}{2.0pt}",
             f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}ll*{{{n}}}{{c}}@{{}}}}", "\\toprule",
             " &  & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{lab} ({task}, {num})}}" for _, lab, num in SUBS) + " \\\\",
             "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}",
             " & Method & " + " & ".join(f"{lab}{arr}" for _, _, lab, arr in cols) + " \\\\", "\\midrule"]
    for mi, (model, setting) in enumerate(MODELS):
        if mi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{n + 2}}}{{l}}{{\\textbf{{{model}}}, {setting}}} \\\\")
        for gi, (short, rows) in enumerate(GROUPS):
            rows = [(name, key) for name, key in rows if row(model, task, key)]
            if gi:
                lines.append(f"\\cmidrule(lr){{1-{n + 2}}}")
            mk = marks(model, task, [key for _, key in rows], cols)
            for ri, (name, key) in enumerate(rows):
                g = f"\\multirow{{{len(rows)}}}{{1.05cm}}{{\\centering\\itshape\\shortstack{{{short}}}}}" if ri == 0 else ""
                vals = []
                for j, (sub, k, _, _) in enumerate(cols):
                    s = cell(model, task, key, sub, k)[1]
                    vals.append({"b": f"\\textbf{{{s}}}", "u": f"\\underline{{{s}}}"}.get(mk.get((key, j)), s))
                lines.append(" & ".join([g, name] + vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


FILL = "\\ph{" + ("This gray text stands in for the two columns of prose under the table, so that the share of the page "
                   "the table takes is visible. It has no content. " * 110) + "}"


def page(texs):
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True)
    for n in ["preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"]:
        shutil.copy(V0 / n, BUILD / n)
    body = "\n".join(t.replace("Table~\\ref{tab:quality}", "Table~1").replace("Table~\\ref{app:quality-t2s}", "Table~A")
                     for t in texs)
    doc = (open(V0 / "main.tex").read().split("\\begin{document}")[0]
           + "\\begin{document}\n" + body + "\n" + FILL + "\n\\end{document}\n")
    (BUILD / "page_full.tex").write_text(doc)
    env = {"PATH": TEX + ":/usr/bin:/bin"}
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "page_full"], cwd=BUILD, env=env, capture_output=True)
    log = (BUILD / "page_full.log").read_text(errors="ignore")
    shutil.copy(BUILD / "page_full.pdf", HERE / "page_full.pdf")
    subprocess.run(["pdftoppm", "-r", "110", "-f", "1", "-l", "2", "-png", str(HERE / "page_full.pdf"), str(HERE / "page_full")],
                   check=True, capture_output=True)
    return log.count("\n!"), len(re.findall(r"Overfull \\hbox", log))


if __name__ == "__main__":
    texs = []
    for task, (short, _) in TASKS.items():
        t = table(task)
        (HERE / f"table_full_{short}.tex").write_text(t)
        texs.append(t)
        print(task, "rows:", t.count("\\\\") - 3)
    errs, over = page(texs)
    print("page errors", errs, "overfull", over)
