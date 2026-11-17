"""B2 (B line): appendix A23 re-issued with the 15 fp32 audits of night (X6). 0 GPU.

The pipeline (runs/2026-11-05/appendix/fp32_audits.py -> make_table_a23.py) globs U/**/audit_fp32*.json only;
the X6 audits were written to $SIGHTLINE_RUNS/r1105/audit/{4b,i2s,lottie,sv1b,sv8b}/<method>/ (two of the three formats the
 reader already handles: 'tree' dicts and image-to-SVG lists). This script
  1. reads U exactly as did (summary copied verbatim) and requires the result to equal
     runs/2026-11-05/appendix/fp32_audits.json, and the table built from it with 's own code to equal the published
     tex/tables/app_a23_fp32_audits.tex byte for byte (saved as app_a23_fp32_audits.1105.tex here);
  2. adds the G2 audits, mapped by the same rules (make_table_a23.MAP + nine G2 prefixes below);
  3. writes the re-issued table. Caption changes: the totals; "--" now says why the one remaining gap is open (SpS:
     its evaluation script has no fp32 audit mode, so it cannot be audited without new code); and the stray closing
     brace of the caption is gone (its last string was not an f-string, so "}}" stayed two braces and the
     caption closed one brace too many -- the published file would not compile).
  python3 runs/2026-11-06/chores/b2/a23_v2.py   -> fp32_audits.{json,md} (this dir) + tex/tables/app_a23_fp32_audits.tex
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import glob
import importlib.util
import json
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
HERE = Path(__file__).resolve().parent
U = f"{RUNS}/"
G2 = f"{RUNS}/"
APP = RUNS / "2026-11-05/appendix"
_s = importlib.util.spec_from_file_location("a23", APP / "make_table_a23.py")
A23 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A23)
B1 = A23.B1

G2_MAP = [("g2/r1105/audit/4b/b16", "4b", "b16"), ("g2/r1105/audit/4b/ours", "4b", "b32"),
          ("g2/r1105/audit/4b/", "4b", None),
          ("g2/r1105/audit/i2s/ours-ets", "i2s", "rule"), ("g2/r1105/audit/sv8b/ours-ets", "sv8b", "rule"),
          ("g2/r1105/audit/sv8b/", "sv8b", None), ("g2/r1105/audit/sv1b/", "sv1b", None),
          ("g2/r1105/audit/lottie/", "lottie", None)]


def summary(d):                       # = runs/2026-11-05/appendix/fp32_audits.py summary, verbatim
    if isinstance(d, list):
        des = {}
        for r in d:
            for k, v in r["designs"].items():
                des.setdefault(k, [0, 0])
                des[k][0] += bool(v.get("identical", v.get("identical_to_ar")))
                des[k][1] += 1
        return len(d), sum(bool(r.get("ar_self")) for r in d), {k: v[0] for k, v in des.items()}, \
            sum(r.get("n_ar", 0) for r in d)
    if "total" in d:
        t = d["total"]
        return t["n"], t["ar_self_consistent"], {f"chain-b{d.get('block_size')}": t["identical_to_ar"]}, \
            sum(r.get("n_ar", 0) for r in d["rows"])
    return d["n"], d["ar_self"], d["identical"], sum(r.get("n_ar", 0) for r in d["rows"])


def collect(root, prefix=""):
    out = []
    for f in sorted(glob.glob(root + "**/audit_fp32*.json", recursive=True)):
        n, ar_self, ident, toks = summary(json.load(open(f)))
        out.append(dict(path=prefix + f[len(root):], n=n, ar_self=ar_self, identical=ident, tokens=toks))
    return out


def table(A, mp, caption):
    cells, skipped = {}, []
    for a in A:
        assert all(v == a["n"] for v in a["identical"].values()), a["path"]
        hit = next(((m, h) for p, m, h in mp if a["path"].startswith(p)), None)
        if hit is None:
            skipped.append(a["path"])
            continue
        model, head = hit
        for design, v in a["identical"].items():
            if head is None and design.startswith(("D0", "D1", "chain")):
                skipped.append(a["path"] + ":" + design)
                continue
            k = (A23.method(design, head), model)
            cells[k] = cells.get(k, 0) + v
    n_checks = sum(a["n"] * len(a["identical"]) for a in A)
    n_tok = sum(a["tokens"] for a in A)
    lines = []
    for r in A23.ROWS:
        lines.append(B1.method_name(r) if r == "Ours" else r)
        lines[-1] += " & " + " & ".join(str(cells[(r, c)]) if (r, c) in cells else ("--" if c in A23.RAN[r] else "n/a")
                                        for c, _ in A23.COLS) + " \\\\"
    head = (" & " + " & ".join(n.split("\\\\")[0] for _, n in A23.COLS) + " \\\\\n"
            "Method & " + " & ".join(n.split("\\\\")[1] for _, n in A23.COLS) + " \\\\")
    body = "\n".join(["\\begin{table*}[t]", caption(len(A), n_checks, n_tok), "\\label{app:fp32}", "\\centering",
                      "\\footnotesize", "\\setlength{\\tabcolsep}{2.2pt}",
                      "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l*{8}{c}@{}}", "\\toprule", head,
                      "\\midrule", *lines, "\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""])
    missing = [(r, c) for r in A23.ROWS for c, _ in A23.COLS if (r, c) not in cells and c in A23.RAN[r]]
    return body, cells, skipped, missing, n_checks, n_tok


def cap_1105(n_a, n_checks, n_tok):   # make_table_a23.main's caption, verbatim (stray brace included)
    return (f"\\caption{{\\textbf{{Token-for-token audits in fp32.}} Prompts re-decoded in fp32 by each method under our "
            "strict verifier and by the model's own autoregressive decoding; a cell counts the prompt $\\times$ draft "
            "design checks, and every one of them gave identical tokens. In all, "
            f"{n_a} audits, {n_checks} checks and {n_tok:,} target tokens, with no mismatch. -- = run but not "
            "audited; n/a = the method does not run on that model. "
            "The verifier is the same code for every method and model.}}")


def cap_1106(n_a, n_checks, n_tok):
    return ("\\caption{\\textbf{Token-for-token audits in fp32.} Prompts re-decoded in fp32 by each method under our "
            "strict verifier and by the model's own autoregressive decoding; a cell counts the prompt $\\times$ draft "
            "design checks, and every one of them gave identical tokens. In all, "
            f"{n_a} audits, {n_checks:,} checks and {n_tok:,} target tokens, with no mismatch. -- = run but not "
            "audited (SpS: its evaluation script has no fp32 audit mode); n/a = the method does not run on that model. "
            "The verifier is the same code for every method and model.}")


def main():
    u = collect(U)
    assert u == json.loads((APP / "fp32_audits.json").read_text()), "U audits differ from the 11-05 list"
    old_body = table(u, A23.MAP, cap_1105)[0]
    assert old_body == (HERE / "app_a23_fp32_audits.1105.tex").read_text(), "11-05 table not reproduced"
    print(f"regression: U audits == 11-05 fp32_audits.json ({len(u)}); 11-05 table reproduced byte for byte")
    g = collect(G2 + "r1105/audit/", prefix="g2/r1105/audit/")
    assert len(g) == 15, len(g)
    A = u + g
    (RUNS / "2026-11-05/appendix/fp32_audits.json").write_text(json.dumps(A, indent=1))
    L = ["| path | rows | AR self | identical per design | tokens |", "|---|---|---|---|---|"]
    for a in A:
        L.append(f"| {a['path']} | {a['n']} | {a['ar_self']} | " +
                 ", ".join(f"{k} {v}" for k, v in a["identical"].items()) + f" | {a['tokens']} |")
    (HERE / "fp32_audits.md").write_text("\n".join(L) + "\n")
    body, cells, skipped, missing, n_checks, n_tok = table(A, G2_MAP + A23.MAP, cap_1106)
    assert body.count("{") == body.count("}"), "unbalanced braces"
    assert missing == [("SpS", "t2")], missing
    B1.write("app_a23_fp32_audits", body)
    new = {k: v for k, v in cells.items() if k not in table(u, A23.MAP, cap_1105)[1]}
    print(f"{len(A)} audits, {n_checks} checks, {n_tok:,} tokens; all identical")
    print("cells filled by the 15 new audits:", sorted(new.items()))
    print("still not audited:", missing, "; skipped (no block info in the run name):", len(skipped))


if __name__ == "__main__":
    main()
