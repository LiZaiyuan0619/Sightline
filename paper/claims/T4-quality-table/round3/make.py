"""Quality table, round 3 (main-line session 7): round 2's target layout filled with the measured rows.

The user wanted every group to compare the same decoders, with the baselines in the main text; the fill
runs (runs/2026-11-09/table-fill, 80 cells) measured the missing rows. User decisions DFlash x8 stays in the main
table and is reported as measured; the authors' first-valid-of-8 row goes to the appendix; rows are named AR.

Rows per model (two quality metrics per task and the seconds, round 1's v2 columns):
  lossless       AR, EAGLE-3, DFlash, DDTree, DARTree, Ours
  loop control   the same six, each with the same escape-then-stop rule
  best of 8      AR, DFlash, Ours: eight samples in one batch, the same loop stop and the same rendered selection
Marks: only the seconds are ranked, in every group (bold = fastest, underline = second). Each group writes outputs of
one distribution, so the quality columns differ by noise; round 1 ranked quality in best of 8 because its AR row then had
no loop stop. Missing cells (metrics still running) print a gray dot and are listed on stdout.

Data: runs/2026-11-09/table-fill/figdata/table2_v2.json (dump_table2_v2.py: published rows as runs/2026-11-02/dump_table2.py
reads them, new rows node-normalised). Round 1's published-number checks run on import; the new rows are checked against
the fill reports below.
Run: python3 paper/claims/T4-quality-table/round3/make.py   (writes table_quality.tex, page_r3.pdf/png)
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("t4r1", HERE.parent / "round1/make.py")
r1 = importlib.util.module_from_spec(spec)
sys.modules["t4r1"] = r1
spec.loader.exec_module(r1)
r1.D = json.loads((r1.RUNS / "2026-11-09/table-fill/figdata/table2_v2.json").read_text())["blocks"]
r1.BUILD = Path("/tmp/sd7/t4build")

for (m, t, row, sub, key), want in r1.CHECK.items():          # the published numbers survive the new data file
    assert r1.cell(m, t, row, sub, key)[1] == want, (m, t, row, sub, key)
NEW_CHECK = {   # fill reports runs/2026-11-09/table-fill/fill_report-<block>.md (node-normalised seconds)
    ("OmniSVG-8B", "text-to-SVG", "DFlash", "icon", "s"): "1.29",
    ("OmniSVG-8B", "text-to-SVG", "DFlash", "icon", "FID"): "10.76",
    ("OmniSVG-8B", "text-to-SVG", "AR x8s", "illustration", "CLIP-T2I"): "18.58",
    ("OmniSVG-8B", "image-to-SVG", "DFlash x8", "icon", "s"): "3.87",
    ("OmniSVG-8B", "image-to-SVG", "AR (fast)", "illustration", "DINO-Score"): "0.805",
}
for (m, t, row, sub, key), want in NEW_CHECK.items():
    assert r1.cell(m, t, row, sub, key)[1] == want, (m, t, row, sub, key, r1.cell(m, t, row, sub, key)[1])

GROUPS = [("lossless", [("AR", "AR"), ("EAGLE-3", "EAGLE-3"), ("DFlash", "DFlash"), ("DDTree", "DDTree"),
                                ("DARTree", "DARTree"), ("\\textbf{Ours}", "Ours")]),
          ("loop\\\\control", [("AR", "AR (fast)"), ("EAGLE-3", "EAGLE-3 (fast)"), ("DFlash", "DFlash (fast)"),
                               ("DDTree", "DDTree (fast)"), ("DARTree", "DARTree (fast)"), ("\\textbf{Ours}", "Ours (fast)")]),
          ("best\\\\of 8", [("AR", "AR x8s"), ("DFlash", "DFlash x8"), ("\\textbf{Ours}", "Ours x8")])]
NEW = "\\textcolor{gray}{$\\cdot$}"
CAPTION = ("\\textbf{Speed and quality on the full SArena test sets.} Each subset gives seconds per prompt and two "
           "official metrics, on 6,013 icon and 2,001 illustration prompts. The first two groups compare the original "
           "model (AR) and the four fastest published decoders of Figure~\\ref{fig:main} with ours, lossless and with the "
           "same loop control. Best of 8 decodes eight samples in one batch, where verification runs on chains only, so "
           "this group holds AR, DFlash and ours. Loop control escapes a loop once and then stops, while best of 8 stops "
           "a sample at its first loop. \\textbf{Fastest} in bold, \\underline{second}{} underlined, among the seconds "
           "of a group (Appendix~\\ref{app:full} gives every metric).")


def present(model, task, row, sub, key):
    r = r1.D[f"{model}, {task}"].get(row, {}).get(sub)
    if not r:
        return False
    return r.get("seconds") is not None if key == "s" else bool(r.get("metrics")) and r["metrics"].get(key) is not None


def marks(model, rows, cols):
    """Rank the seconds within one group, on the printed values; ties share a mark."""
    out = {}
    for j, (task, sub, key, _) in enumerate(cols):
        if key != "s":
            continue
        vals = {r: float(r1.cell(model, task, r, sub, key)[1]) for r in rows if present(model, task, r, sub, key)}
        order = sorted(set(vals.values()))
        for r, v in vals.items():
            k = order.index(v)
            if k < 2:
                out[(r, j)] = "bu"[k]
    return out


def table():
    cols = r1.columns(True)
    lead = 2
    lines = ["\\begin{table*}[t]", f"\\caption{{{CAPTION}}}", "\\label{tab:quality}\\label{tab:quality-t2s}\\label{tab:quality-i2s}",
             "\\centering", "\\small", "\\setlength{\\tabcolsep}{2.2pt}",
             "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll*{12}{c}@{}}", r1.header(cols, lead)]
    missing = []
    for mi, (model, setting) in enumerate(r1.MODELS):
        if mi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{14}}{{l}}{{\\textbf{{{model}}}, {setting}}} \\\\")
        for gi, (short, rows) in enumerate(GROUPS):
            if gi:
                lines.append("\\cmidrule(lr){1-14}")
            mk = marks(model, [k for _, k in rows], cols)
            for ri, (name, key) in enumerate(rows):
                g = f"\\multirow{{{len(rows)}}}{{1.05cm}}{{\\centering\\itshape\\shortstack{{{short}}}}}" if ri == 0 else ""
                vals = []
                for j, (t, s, k, _) in enumerate(cols):
                    if present(model, t, key, s, k):
                        vals.append(r1.fmtcell(model, t, key, s, k, mk.get((key, j))))
                    else:
                        vals.append(NEW)
                        missing.append((model, t, key, s, k))
                lines.append(" & ".join([g, name] + vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}"]
    return "\n".join(lines) + "\n", missing


if __name__ == "__main__":
    tex, missing = table()
    (HERE / "table_quality.tex").write_text(tex)
    shutil.rmtree(r1.BUILD, ignore_errors=True)
    r1.BUILD.mkdir(parents=True)
    for n in ["preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"]:
        shutil.copy(r1.V0 / n, r1.BUILD / n)
    (r1.BUILD / "r3.tex").write_text(tex.replace("Figure~\\ref{fig:main}", "Figure~4").replace("Appendix~\\ref{app:full}", "Appendix~C"))
    r1.HERE = HERE
    errs, over = r1.page("r3")
    print("page errors", errs, "overfull", over)
    cells = sorted({(m, t, row) for m, t, row, _, _ in missing})
    print(f"missing cells: {len(missing)} in rows {cells}")
