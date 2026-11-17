""" main line, session 11: the six theory tables of appendix B (T3, T5, T6, T7, T8, T9).

Sources (read only):
  T3  runs/2026-11-05/theory/decision_horizon_{b32-6ep,b64-3ep}.md, section B, "target H > 0.2" (parsed)
  T5  logs/11-06-theory-sampling-caliber-report.md, sections 2-4 (numbers copied, each traced below)
  T6  logs/11-06-theory-strict-loop-free-report.md, sections 1, 3, 4 (copied)
  T7  runs/2026-11-05/theory/decision_model.md (parsed: held-out model tokens per round / walk tokens per round)
  T8  logs/11-06-theory-NL-report.md, section 4 (copied)
  T9  runs/2026-11-05/theory/{preregistered_predictions.md, e2_vs_prediction.md}, logs/11-06-theory-A4-report.md section 5,
      logs/11-05-night-theory-exec-summary.md section 0, logs/11-06-theory-gapfill-report.md, docs/writing/04-theory-section-v2.md
      sections 3.4, 4.3, 5.3 (copied). Every registered prediction is listed, held or not (writing guide 2.4).
Writes draft/v0/tables/app_t{3,5,6,7,8,9}.tex.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import re
from pathlib import Path
from sightline.paths import RUNS

ROOT = Path(__file__).resolve().parents[1]
TH = RUNS / "2026-11-05/theory"
OUT_DIR = Path(os.environ.get("SIGHTLINE_TABLES",
                              Path(__file__).resolve().parent / "tables"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR


def md_block(path, heading):
    lines, on = [], False
    for ln in open(path):
        if ln.startswith("### ") or ln.startswith("## "):
            if on:
                break
            on = ln.strip().startswith(heading)
            continue
        if on and ln.startswith("|") and not ln.startswith("|---"):
            lines.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return lines[1:]


def acc(cell):
    return cell.split(" ")[0] if cell != "--" else "--"


def t3():
    names = {"cmd": "Commands", "coord": "Coordinates", "color": "Colors", "other": "Other"}
    bands = ["1-8", "9-16", "17-31", "32-47", "48-63"]
    d = {}
    for head in ("b32-6ep", "b64-3ep"):
        for cls, band, c0, c1, c2 in md_block(TH / f"decision_horizon_{head}.md", "### target H > 0.2"):
            d[(head, cls, band)] = (acc(c0), acc(c1), acc(c2))
    rows = []
    for cls in ("cmd", "coord", "color", "other"):
        for i, b in enumerate(bands):
            r32 = d.get(("b32-6ep", cls, b), ("", "", ""))
            r64 = d.get(("b64-3ep", cls, b), ("--", "--", "--"))
            lab = names[cls] if i == 0 else ""
            rows.append(f"{lab} & {b.replace('-', '--')} & " + " & ".join(r32) + " & " + " & ".join(r64) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    rows.pop()
    cap = (r"\caption{\textbf{Accuracy by class, offset and decisions crossed.} Top-1 accuracy of the draft head at positions "
           r"that are not decisions themselves, by token class, offset in the block and the number $c$ of decisions crossed "
           r"before the position. A decision is a position where the target's own next-token entropy exceeds 0.2 nats. "
           r"OmniSVG-8B text-to-SVG, greedy decoding, loop-free content. Cells with fewer than 30 positions are left out, "
           r"and thresholds of 0.05 and 0.5 nats give the same picture.}")
    body = "\n".join([
        r"\begin{table*}[t]", cap, r"\label{app:t3}", r"\centering", r"\footnotesize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}", r"\toprule",
        r" & & \multicolumn{3}{c}{Head trained on blocks of 32} & \multicolumn{3}{c}{Head trained on blocks of 64} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Class & Offset & $c=0$ & $c=1$ & $c\ge2$ & $c=0$ & $c=1$ & $c\ge2$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    (OUT / "app_t3.tex").write_text(body)


def t5():
    # logs/11-06-theory-sampling-caliber-report.md: section 2 (miss rate, CV), 3 (ceiling and losses), 4 (classes, offsets 1-8)
    R = [("64", "greedy", "7.5", "3.16", "79.00", "41.13", "20.58", "12.23", "12.24", "14", "0.92", "0.45"),
         ("", "official sampling", "9.4", "3.31", "62.12", "36.62", "17.45", "10.71", "10.71", "17", "0.91", "0.40"),
         ("32", "greedy", "7.2", "2.97", "82.91", "25.83", "18.50", "12.35", "12.37", "15", "0.92", "0.46"),
         ("", "official sampling", "7.9", "3.29", "64.61", "24.16", "16.87", "11.58", "11.61", "15", "0.91", "0.43"),
         ("16", "greedy", "6.7", "3.63", "86.04", "14.45", "12.73", "10.37", "10.36", "15", "0.93", "0.50"),
         ("", "official sampling", "8.1", "3.03", "64.30", "13.86", "11.75", "9.63", "9.63", "17", "0.92", "0.54")]
    rows = []
    for i, r in enumerate(R):
        rows.append(" & ".join(r[:2]) + f" & {r[2]}\\% & " + " & ".join(r[3:9]) + f" & {r[9]}\\% & " + " & ".join(r[10:]) + r" \\")
        if i in (1, 3):
            rows.append(r"\addlinespace[2pt]")
    cap = (r"\caption{\textbf{The theory under official sampling.} OmniSVG-8B text-to-SVG, loop-free content, three heads trained "
           r"on blocks of 64, 32 and 16. Miss rate and run-length CV are for one candidate per position. Ceiling to Budget "
           r"are the tokens per round of a 127-node tree as the model gains each loss, the window at the head's own block "
           r"length, the horizon and the node budget, and Measured is the decoder itself. The last three columns give the "
           r"share of positions that are decisions and the accuracy of commands and coordinates at offsets 1 to 8 after two "
           r"or more decisions. Lemma~\ref{lem:runlength} matches a round-by-round replay exactly under both decodings.}")
    body = "\n".join([
        r"\begin{table*}[t]", cap, r"\label{app:t5}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccccc@{}}", r"\toprule",
        r"Block & Decoding & Miss rate & Run CV & Ceiling & Window & Horizon & Budget & Measured & Decisions & Commands & Coordinates \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    (OUT / "app_t5.tex").write_text(body)


def t6():
    # logs/11-06-theory-strict-loop-free-report.md: section 3 (tree), section 1 (k=1 CV), section 4 (offsets 1-8)
    R = [("64", "loop-free", "79.00", "41.13", "20.58", "12.23", "12.24", "3.16", "0.92", "0.45"),
         ("", "strict", "52.03", "34.39", "16.92", "11.42", "11.43", "2.53", "0.88", "0.40"),
         ("32", "loop-free", "82.91", "25.83", "18.50", "12.35", "12.37", "2.97", "0.92", "0.46"),
         ("", "strict", "55.82", "23.49", "16.12", "11.60", "11.62", "2.57", "0.88", "0.41"),
         ("16", "loop-free", "86.04", "14.45", "12.73", "10.37", "10.36", "3.63", "0.93", "0.50"),
         ("", "strict", "56.44", "13.72", "11.66", "9.68", "9.66", "2.96", "0.90", "0.41")]
    rows = []
    for i, r in enumerate(R):
        rows.append(" & ".join(r) + r" \\")
        if i in (1, 3):
            rows.append(r"\addlinespace[2pt]")
    cap = (r"\caption{\textbf{Robustness on strict loop-free content.} Strict loop-free content also leaves out outputs that "
           r"repeat without an exact cycle. Columns as in Table~\ref{app:t5}, greedy decoding. Every direction holds. The "
           r"ceiling falls most, since many of the removed outputs have no miss at all.}")
    body = "\n".join([
        r"\begin{table*}[t]", cap, r"\label{app:t6}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccc@{}}", r"\toprule",
        r"Block & Content & Ceiling & Window & Horizon & Budget & Measured & Run CV & Commands & Coordinates \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    (OUT / "app_t6.tex").write_text(body)


def t7():
    heads = [("b64-3ep", "64"), ("b32-6ep", "32"), ("b16-6ep", "16")]
    text = open(TH / "decision_model.md").read()
    rows = []
    for head, blk in heads:
        sec = text.split(f"## {head}")[1].split("\n## ")[0]
        vals = {}
        for ln in sec.splitlines():
            m = re.match(r"\| (\d+) \| (\d+) \| [\d.]+ \| \*\*([\d.]+)\*\* \| ([\d.]+) \|", ln)
            if m:
                vals[(int(m.group(1)), int(m.group(2)))] = float(m.group(3)) / float(m.group(4))
        for k in (1, 16):
            cells = [f"{100 * vals[(k, W)]:.0f}\\%" if (k, W) in vals else "--" for W in (8, 16, 32, 64)]
            rows.append(f"{blk if k == 1 else ''} & {k} & " + " & ".join(cells) + r" \\")
    cap = (r"\caption{\textbf{A conditionally independent model.} The model keeps the measured accuracy of every position, "
           r"given its class, offset and the decisions crossed, and draws positions independently. It is fitted on half of "
           r"the prompts and evaluated on the other half. Entries give its tokens per round as a share of the ideal value "
           r"under the real labels. OmniSVG-8B text-to-SVG, greedy decoding, loop-free content. The share falls as the "
           r"window grows, because rounds that fill the window are rarer in the model than in the real stream.}")
    body = "\n".join([
        r"\begin{table}[t]", cap, r"\label{app:t7}", r"\centering", r"\footnotesize",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}cccccc@{}}", r"\toprule",
        r"Block & Candidates & $W=8$ & $W=16$ & $W=32$ & $W=64$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    (OUT / "app_t7.tex").write_text(body)


def t8():
    # logs/11-06-theory-NL-report.md section 4
    R = [("OmniSVG-8B, block 16", "text-to-SVG", "Commands", "0.36", "0.91"),
         ("OmniSVG-8B, block 32", "text-to-SVG", "Commands", "0.36", "0.89"),
         ("OmniSVG-8B, block 64", "text-to-SVG", "Commands", "0.36", "0.89"),
         ("OmniSVG-4B, block 16", "text-to-SVG", "Commands", "0.43", "0.93"),
         ("Qwen3-8B, block 16", "GSM8K", "Punctuation", "0.34", "0.34"),
         ("Qwen3-8B, block 16", "MATH-500", "Punctuation", "0.40", "0.48"),
         ("Qwen3-8B, block 16", "HumanEval", "Words", "0.45", "0.34"),
         ("Qwen3-8B, block 16", "MBPP", "Words", "0.49", "0.34"),
         ("Qwen3-8B, block 16", "MT-Bench", "Punctuation", "0.32", "0.40")]
    rows = [" & ".join(r) + r" \\" for r in R]
    rows.insert(4, r"\addlinespace[2pt]")
    cap = (r"\caption{\textbf{Vector graphics against natural language.} For each target and task, the token class that keeps "
           r"the most of its accuracy past two decisions. Share is the class's share of the stream, and retention its accuracy "
           r"after crossing two or more decisions divided by its accuracy when none is crossed. Qwen3-8B uses the released DFlash "
           r"head under official decoding. The SVG rows are loop-free content.}")
    body = "\n".join([
        r"\begin{table}[t]", cap, r"\label{app:t8}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lllcc@{}}", r"\toprule",
        r"Target and head & Task & Class & Share & Retention \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    (OUT / "app_t8.tex").write_text(body)


def t9():
    G, S = "Greedy", "Sampled"
    R = [
        (r"Lemma~\ref{lem:runlength}", G, r"The identity on full-window labels bounds the tokens per round of a shorter window from below",
         r"5 of 6 slice cells on OmniSVG-8B, 5 of 5 on InternSVG-8B", "held"),
        ("", S, r"The identity lies within 3\% below the measured value in at least 4 of 5 cells per head",
         r"8 of 10 and 3 of 3 cells; every slice cell; median error 2.7\% over 13 cells", "held"),
        ("", G, r"A replay of cached drafts matches a tree decoder that verifies a prefix of its block to about 1\%",
         r"2.5\% to 3.5\%", "not met"),
        ("", G, r"Per-cell predictions for a chain decoder, from its measured full-window run",
         r"errors from $-11.4\%$ to $+19.0\%$ over 7 cells", "--"),
        ("", S, r"Per-cell main predictions from cached drafts",
         r"median error 6.9\% over 16 cells, 10 of 16 inside the registered band", "--"),
        ("", "Both", r"Drafting fewer positions is no better than verifying a prefix of the block",
         r"5 of 6 cells greedy, 6 of 6 sampled, two sampled magnitudes outside the registered range", "held in 11 of 12"),
        (r"Proposition~\ref{prop:share}", G, r"The return from doubling the window from 16 to 32 grows as the candidates widen",
         r"$\times1.27\to\times1.78$ and $\times1.28\to\times1.79$ for the two heads, one to 16 candidates", "held"),
        ("", G, r"Window and node budget have increasing differences",
         r"every pair for the three heads", "held"),
        ("", S, r"Window and design have increasing differences", r"5 of 5", "held"),
        ("", S, r"The cost of sampling relative to greedy does not depend on the window (80\% of cells within 6\%)",
         r"5 of 12; 9 of 12 confidence intervals still contain the registered value", "did not hold"),
        ("", S, r"A tree pays more for sampling than a chain", r"4 of 7, every confidence interval overlaps", "undecided"),
        (r"Proposition~\ref{prop:horizon}", G, r"At shared positions the head trained on blocks of 64 pays its extra loss on coordinates",
         r"offset 15: commands $0.999\to0.994$, coordinates $0.991\to0.958$", "held"),
        ("", G, r"The head trained on blocks of 64, cut to a window of 32, is weaker than the head trained on blocks of 32",
         r"tree $-7.9\%$ (predicted $-5.6\%$), chain $-1.2\%$", "held"),
        ("", S, r"The same under official sampling", r"tree $-9.8\%$, chain $-2.3\%$", "held"),
        (r"Proposition~\ref{prop:loop}", "Both", r"Loop control speeds up the original model most and Sightline least",
         r"10 of 10 settings", "held"),
        ("", "Both", r"Point predictions of the speed-up from loop control", r"median error 6.1\% over 16 settings", "--"),
        ("", "Both", r"With loop control on both, the lead of a window of 32 over 16 narrows and remains",
         r"lead kept in 8 of 8 settings, narrowed in 7 of 8", "held"),
    ]
    rows = []
    for i, (st, dec, pred, out, ver) in enumerate(R):
        if st and i:
            rows.append(r"\midrule")
        rows.append(f"{st} & {dec} & {pred} & {out} & {ver}" + r" \\")
    cap = (r"\caption{\textbf{Predictions and outcomes.} Every prediction we registered before reading the corresponding "
           r"results, grouped by the statement it tests. Greedy rows use loop-free content of OmniSVG-8B text-to-SVG, and "
           r"sampled rows the official sampling of the same pair. Predictions that did not hold or remain undecided are listed "
           r"with the rest, and no claim of the paper rests on them.}")
    body = "\n".join([
        r"\begin{table*}[t]", cap, r"\label{app:t9}", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}p{0.10\textwidth}p{0.09\textwidth}p{0.34\textwidth}p{0.28\textwidth}p{0.08\textwidth}@{}}",
        r"\toprule", r"Statement & Decoding & Prediction & Outcome & Verdict \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (OUT / "app_t9.tex").write_text(body)


if __name__ == "__main__":
    for f in (t3, t5, t6, t7, t8, t9):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("app_t[0-9].tex")))
