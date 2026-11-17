"""Quality table, round 1 (page mock-ups show page 2: a table* floats to the page after the one that defines it)

 (night, main-line session 6): merge main.pdf's Tables 1 and 2 into one table.

Why: the two full-SArena tables (text-to-SVG, image-to-SVG) take about 10.5 in of double-column space. The project decision
for several merged versions to look at, and asked whether the uneven baselines across the groups invite attack.

Data: runs/2026-11-02/figdata/table2.json (tidy; 380 cells checked against tex/tables/table2{a,b}.tex on ).
Values are rounded once from the raw numbers here; the five image-to-SVG cells that tex/tables/table2b.tex double-rounded
(0.001 apart, noted ) therefore print the directly rounded value.

Each version keeps two quality metrics and the seconds per subset (text-to-SVG: FID, CLIP-T2I; image-to-SVG: DINO,
LPIPS); the other four metrics stay in the full tables, which move to the appendix unchanged.

  v1  current rows, groups as italic lines (as in Tables 1 and 2 now)
  v2  current rows, groups in a left column, seconds first in every subset
  v3  only the strongest published decoder (DDTree, second in 59 of the 60 settings of Figure 4) next to the original model
  v4  v3's rows in v2's layout

Marks: best bold, second underlined, ties at printed precision share the mark. In the groups whose outputs are the same
(or nearly so) only the seconds are ranked, as in the current tables; in best of 8 every column is ranked.

Run: python3 paper/claims/T4-quality-table/round1/make.py   (writes v*.tex, page mock-ups, _pages.png)
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
DATA = RUNS / "2026-11-02/figdata/table2.json"
TEX = "pdflatex"
BUILD = Path("/tmp/sd6/t4build")

D = json.loads(DATA.read_text())["blocks"]
MODELS = [("OmniSVG-8B", "its released sampling settings"),
          ("InternSVG-8B", "greedy; best of 8 at $T{=}0.6$, top-$p$ 0.9")]
TASKS = [("text-to-SVG", [("FID", "FID", 2, -1), ("CLIP-T2I", "CLIP", 2, +1)]),
         ("image-to-SVG", [("DINO-Score", "DINO", 3, +1), ("LPIPS", "LPIPS", 3, -1)])]
SUBS = [("icon", "SArena-Icon"), ("illustration", "SArena-Illus.")]

# published numbers this script must reproduce (Tables 1 and 2 of main.pdf)
CHECK = {("OmniSVG-8B", "text-to-SVG", "Ours", "icon", "s"): "1.01",
         ("OmniSVG-8B", "text-to-SVG", "Ours x8", "icon", "FID"): "9.85",
         ("InternSVG-8B", "image-to-SVG", "Ours x8", "illustration", "s"): "41.20",
         ("InternSVG-8B", "image-to-SVG", "AR", "illustration", "s"): "43.93"}


def cell(model, task, row, sub, key):
    r = D[f"{model}, {task}"][row][sub]
    if key == "s":
        return r["seconds"], f"{r['seconds']:.2f}", -1
    dec, sign = next((dd, sg) for k, _, dd, sg in TASKS[0][1] + TASKS[1][1] if k == key)
    v = r["metrics"][key]
    return v, f"{v:.{dec}f}", sign


for (m, t, row, sub, key), want in CHECK.items():
    got = cell(m, t, row, sub, "s" if key == "s" else key)[1]
    assert got == want, (m, t, row, sub, key, got, want)


def columns(seconds_first):
    cols = []
    for task, mets in TASKS:
        for sub, _ in SUBS:
            ms = [("s", "s", None, -1)] + [(k, lab, dd, sg) for k, lab, dd, sg in mets] if seconds_first else \
                 [(k, lab, dd, sg) for k, lab, dd, sg in mets] + [("s", "s", None, -1)]
            cols += [(task, sub, k, lab) for k, lab, _, _ in ms]
    return cols


def marks(model, rows, cols, rank_all):
    """rows: list of json row keys in one group. Returns {(row, col_index): 'b'|'u'}."""
    out = {}
    for j, (task, sub, key, _) in enumerate(cols):
        if key != "s" and not rank_all:
            continue
        vals = {r: cell(model, task, r, sub, key) for r in rows}
        printed = sorted({(float(p) * -sg) for _, p, sg in vals.values()})
        for r, (_, p, sg) in vals.items():
            k = printed.index(float(p) * -sg)
            if k == 0:
                out[(r, j)] = "b"
            elif k == 1:
                out[(r, j)] = "u"
    return out


def fmtcell(model, task, row, sub, key, mk):
    s = cell(model, task, row, sub, key)[1]
    return {"b": f"\\textbf{{{s}}}", "u": f"\\underline{{{s}}}"}.get(mk, s)


NAMES = {"AR": "Original", "EAGLE-3": "EAGLE-3", "DDTree": "DDTree", "DARTree": "DARTree", "Ours": "\\textbf{Ours}",
         "DDTree (fast)": "DDTree", "Ours (fast)": "\\textbf{Ours}", "Official 8, first valid": "Authors' pipeline",
         "AR x8": "Original", "Ours x8": "\\textbf{Ours}"}


def groups(model, version):
    lossless = ["AR", "EAGLE-3", "DDTree", "DARTree", "Ours"] if version in ("v1", "v2") else ["AR", "DDTree", "Ours"]
    loop = ["DDTree (fast)", "Ours (fast)"]
    bo8 = (["Official 8, first valid"] if model == "OmniSVG-8B" else []) + ["AR x8", "Ours x8"]
    return [("Same outputs as the original model", "same\\\\outputs", lossless, False),
            ("Same loop control", "loop\\\\control", loop, False),
            ("Best of 8, scored after rendering", "best\\\\of 8", bo8, True)]


def header(cols, lead):
    n = len(cols)
    first = " & " * lead
    h1 = first + " & ".join(f"\\multicolumn{{6}}{{c}}{{{task}}}" for task, _ in TASKS) + " \\\\"
    c1 = "".join(f"\\cmidrule(lr){{{lead + 1 + 6 * i}-{lead + 6 + 6 * i}}}" for i in range(2))
    h2 = first + " & ".join(f"\\multicolumn{{3}}{{c}}{{{lab}}}" for _ in TASKS for _, lab in SUBS) + " \\\\"
    c2 = "".join(f"\\cmidrule(lr){{{lead + 1 + 3 * i}-{lead + 3 + 3 * i}}}" for i in range(4))
    arrow = {"s": "$\\downarrow$", "FID": "$\\downarrow$", "LPIPS": "$\\downarrow$", "CLIP": "$\\uparrow$", "DINO": "$\\uparrow$"}
    h3 = ("Method" if lead == 1 else " & Method") + " & " + " & ".join(f"{lab}{arrow[lab]}" for _, _, _, lab in cols) + " \\\\"
    return "\n".join(["\\toprule", h1, c1, h2, c2, h3, "\\midrule"])


CAPTION = {
    "v1": ("\\textbf{Speed and quality on the full SArena test sets.} Seconds per prompt on one H200; 6,013 icon and "
           "2,001 illustration prompts per task. The first group writes the original model's own outputs, the second "
           "adds the same loop control, and the third keeps the best of eight samples by CLIP (text) or DINO (image). "
           "The authors' pipeline samples eight and keeps the first valid one. Best in bold, second underlined within a "
           "group; all metrics in Appendix Table~A1."),
    "v2": None,
    "v3": ("\\textbf{Speed and quality on the full SArena test sets.} As v1, with the strongest published decoder of "
           "Figure~4 (DDTree, second in 59 of the 60 settings) as the baseline in each group; EAGLE-3 and DARTree are "
           "in Appendix Table~A1."),
}
CAPTION["v2"] = CAPTION["v1"]
CAPTION["v4"] = CAPTION["v3"]


def table(version):
    sf = version in ("v2", "v4")
    cols = columns(sf)
    lead = 2 if version in ("v2", "v4") else 1
    spec = ("l" * lead) + "*{12}{c}"
    lines = [f"\\begin{{table*}}[t]", f"\\caption{{{CAPTION[version]}}}", "\\label{tab:quality}", "\\centering",
             "\\small", "\\setlength{\\tabcolsep}{2.2pt}",
             f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{spec}@{{}}}}", header(cols, lead)]
    for mi, (model, setting) in enumerate(MODELS):
        if mi:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{12 + lead}}}{{l}}{{\\textbf{{{model}}}, {setting}}} \\\\")
        for gi, (gname, gshort, rows, rank_all) in enumerate(groups(model, version)):
            mk = marks(model, rows, cols, rank_all)
            if version in ("v2", "v4"):
                if gi:
                    lines.append(f"\\cmidrule(lr){{1-{12 + lead}}}")
            else:
                lines.append(f"\\multicolumn{{{12 + lead}}}{{l}}{{\\hspace{{0.6em}}\\textit{{{gname}}}}} \\\\")
            for ri, r in enumerate(rows):
                vals = [fmtcell(model, t, r, s, k, mk.get((r, j))) for j, (t, s, k, _) in enumerate(cols)]
                lead_cells = [NAMES[r]]
                if version in ("v2", "v4"):
                    g = f"\\multirow{{{len(rows)}}}{{1.05cm}}{{\\centering\\itshape\\shortstack{{{gshort}}}}}" if ri == 0 else ""
                    lead_cells = [g, NAMES[r]]
                lines.append(" & ".join(lead_cells + vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular*}", "\\end{table*}"]
    return "\n".join(lines) + "\n"


FILL = "\\ph{" + ("This gray text stands in for the two columns of prose under the table, so that the share of the page "
                   "the table takes is visible. It has no content. " * 110) + "}"


def page(version):
    body = (open(V0 / "main.tex").read().split("\\begin{document}")[0]
            + "\\begin{document}\n\\input{" + version + "}\n" + FILL + "\n\\end{document}\n")
    (BUILD / f"page_{version}.tex").write_text(body)
    env = {"PATH": TEX + ":/usr/bin:/bin"}
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"page_{version}"], cwd=BUILD, env=env, capture_output=True)
    log = (BUILD / f"page_{version}.log").read_text(errors="ignore")
    errs = log.count("\n!")
    over = len(re.findall(r"Overfull \\hbox", log))
    shutil.copy(BUILD / f"page_{version}.pdf", HERE / f"page_{version}.pdf")
    subprocess.run(["pdftoppm", "-r", "110", "-f", "2", "-l", "2", "-png", "-singlefile", str(HERE / f"page_{version}.pdf"),
                    str(HERE / f"page_{version}")], check=True, capture_output=True)
    return errs, over


def sheet(names):
    from PIL import Image, ImageDraw
    ims = [Image.open(HERE / f"page_{n}.png").convert("RGB") for n in names]
    w, h = ims[0].size
    pad, lab = 16, 26
    out = Image.new("RGB", (len(ims) * (w + pad) + pad, h + lab + pad), "white")
    d = ImageDraw.Draw(out)
    for i, im in enumerate(ims):
        x = pad + i * (w + pad)
        d.text((x, 6), names[i], fill=(200, 0, 0))
        out.paste(im, (x, lab))
        d.rectangle([x - 1, lab - 1, x + w, lab + h], outline=(160, 160, 160))
    out.save(HERE / "_pages.png")


if __name__ == "__main__":
    shutil.rmtree(BUILD, ignore_errors=True)
    BUILD.mkdir(parents=True)
    for n in ["preamble.tex", "cvpr.sty", "ieeenat_fullname.bst"]:
        shutil.copy(V0 / n, BUILD / n)
    names = ["v1", "v2", "v3", "v4"]
    for v in names:
        tex = table(v)
        (HERE / f"{v}.tex").write_text(tex)
        (BUILD / f"{v}.tex").write_text(tex)
        errs, over = page(v)
        print(v, "errors", errs, "overfull", over)
    sheet(names)
    print("wrote", HERE / "_pages.png")
