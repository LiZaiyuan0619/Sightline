""" appendix batch 4: positive results too small for the main text, as tables.

  A10  the three loop rules in full: every target x mode x rule         <- runs/2026-11-04/figdata_b3/l_rules.json
  A16  "first k to finish" best-of-N, three settings                   <- runs/2026-10-15-harvest/bon_decode_report_*-firstk.md
  A17  FID of the chosen outputs, two references                       <- runs/2026-10-16-harvest/bon_fid_t2-bench300-firstk.md
  A19  image-to-SVG across sampling temperature and seed, PAIRED rows  <- $SIGHTLINE_RUNS/r1029/{temp-i2,seeds} + table-1 runs
  A21  pick by one scorer, report every metric                         <- runs/2026-10-22/analysis/scorer_swap.md

A19: the harvest (runs/2026-10-29/{temp,seeds}.md) compares methods on different rows at T 0.6 / 1.0 (DDTree and
DFlash finished 194 / 137 of 300). Here every row set is the intersection across the methods compared; where all rows
exist the numbers must equal the report (asserted). By every run had finished (300 rows each), so the
 report's 194 / 137-row cells were simply stale. InternSVG's seeds: DDTree ran with --no-ar right after Ours'
--time-ar in the same job, so it uses Ours' AR timing on the same prompt (table 1's shared-AR convention).
  python3 runs/2026-11-05/appendix/make_tables_b4.py  -> tex/tables/app_a{10,16,17,19,21}_*.tex
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a script from the package root
import glob
import importlib.util
import json
import re
import statistics as st
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
U = RUNS
_s = importlib.util.spec_from_file_location("b3", RUNS / "2026-11-05/appendix/make_tables_b3.py")
B3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(B3)
marks, cell, write, table, B1 = B3.marks, B3.cell, B3.write, B3.table, B3.B1
CI = re.compile(r"([+-]?\d+\.\d+) \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\]")


def md_table(text, header_start):
    """Rows of the markdown table whose header line starts with `header_start`: {first cell: [cells...]}."""
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(header_start))
    head = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    out = {}
    for ln in lines[i + 2:]:
        if not ln.startswith("|"):
            break
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        out[c[0]] = dict(zip(head, c))
    return out


def ci(s):
    m = CI.fullmatch(s.strip())
    assert m, s
    return tuple(float(x) for x in m.groups())


def fnum(v, nd=4, sign=True):
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("+0." + "0" * nd, "0." + "0" * nd).replace("-0." + "0" * nd, "0." + "0" * nd)


# ---------------------------------------------------------------------------------------------- A10
RULE = {"lossless": "lossless", "rollback": "rollback", "escape-then-stop": "\\textbf{escape then stop}",
        "stop": "stop"}
TGT = {"OmniSVG-8B, text-to-SVG": ("clip", "CLIP"), "InternSVG-8B, text-to-SVG": ("clip", "CLIP"),
       "OmniSVG-8B, image-to-SVG": ("dino", "DINO"), "InternSVG-8B, image-to-SVG": ("dino", "DINO")}


def a10():
    d = json.loads((RUNS / "2026-11-04/figdata_b3/l_rules.json").read_text())["ladder"]
    lines = []
    for tgt, (pm, pname) in TGT.items():
        for mode, mlab in (("greedy", "greedy"), ("sample", "official sampling")):
            rs = {r["rule"]: r for r in d if r["target"] == tgt and r["mode"] == mode}
            assert set(rs) == set(RULE), (tgt, mode, list(rs))
            lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{tgt}, {mlab}}} (quality: {pname})}} \\\\")
            for rule in ("lossless", "rollback", "escape-then-stop", "stop"):
                r = rs[rule]
                assert r["primary"] == pm
                base = (f"{RULE[rule]} & {100 * r['eos'] / r['rows']:.0f} & {100 * (r.get('loops') or 0) / r['rows']:.0f}"
                        f" & {r['mean_len']} & {r['nfe_vs_lossless']:.2f}$\\times$")
                if rule == "lossless":
                    lines.append(base + f" & \\multicolumn{{3}}{{c}}{{reference ({r['quality'][pm][0]:.3f})}} \\\\")
                else:
                    q, lo, hi = r["quality"][pm]
                    lines.append(base + f" & {fnum(q)} & {fnum(lo)} & {fnum(hi)} \\\\")
            lines.append("\\addlinespace[2pt]")
    lines = lines[:-1]
    head = (" & ended & still in & mean & fewer & \\multicolumn{3}{c}{quality vs lossless} \\\\\n"
            "\\cmidrule(lr){6-8}\n"
            "Rule & normally (\\%) & a loop (\\%) & tokens & forwards & $\\Delta$ & 95\\% low & 95\\% high \\\\")
    cap = ("\\caption{\\textbf{The three loop rules in full.} Each rule against the same draft's lossless run on the same "
           "prompts (900; InternSVG-8B image-to-SVG 300): outputs that end normally, outputs still in a loop, mean "
           "length, target forward passes saved, and the change in the primary quality metric with its paired 95\\% "
           "bootstrap interval. We use escape then stop. Outputs still in a loop are counted by a strict periodicity "
           "detector, which finds none under InternSVG-8B text-to-SVG's official sampling.}")
    write("app_a10_loop_rules", table("table*", cap, "app:loop-rules", "l*{7}{c}", head, lines, sep="3.0pt"))


# ---------------------------------------------------------------------------------------------- A16 / A17
ARMS = [("ar1", "original model, 1 sample"), ("ar1s", "\\quad + loop control"),
        ("ar8s", "\\quad + loop control, best of 8"), ("ar8f6", "\\quad + loop control, first 6 of 8"),
        ("ar8f4", "\\quad + loop control, first 4 of 8"), ("sd1s", "\\textbf{Ours} + loop control"),
        ("sd8", "\\quad best of 8"), ("sd8f6", "\\quad first 6 of 8"), ("sd8f4", "\\quad first 4 of 8"),
        ("sd16", "\\quad best of 16"), ("sd16f8", "\\quad first 8 of 16")]
SET16 = [("t2", "bon_decode_report_t2-bench300-firstk.md", "clip Δ", 1, "OmniSVG-8B, text-to-SVG", "CLIP gain"),
         ("i2", "bon_decode_report_i2-bench300-firstk.md", "lpips Δ", -1, "OmniSVG-8B, image-to-SVG", "LPIPS drop"),
         ("isvg", "bon_decode_report_isvg-bench300-T0.6-firstk.md", "clip Δ", 1, "InternSVG-8B, text-to-SVG",
          "CLIP gain")]


def a16():
    T = {}
    for k, f, col, sign, *_ in SET16:
        T[k] = md_table((RUNS / "2026-10-15-harvest" / f).read_text(), "| method |")
    arms = [(a, lab) for a, lab in ARMS if all(a in T[k] for k, *_ in SET16)]
    dropped = [a for a, _ in ARMS if a not in dict(arms)]
    assert float(T["t2"]["sd8"]["t total"]) == 4.84 and float(T["isvg"]["sd8"]["t total"]) == 21.80
    lines = []
    for a, lab in arms:
        cells = []
        for k, f, col, sign, *_ in SET16:
            r = T[k][a]
            q, lo, hi = ci(r[col])
            q, lo, hi = (q, lo, hi) if sign > 0 else (-q, -hi, -lo)
            cells += [f"{float(r['t total']):.2f}", fnum(q, 3), fnum(lo, 3), fnum(hi, 3)]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    head = (" & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{nm}}}" for *_, nm, _ in SET16) + " \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n"
            " & " + " & ".join(f"s & \\multicolumn{{3}}{{c}}{{{q} (95\\%)}}" for *_, q in SET16) + " \\\\")
    cap = ("\\caption{\\textbf{Best of the first $k$ to finish.} $N$ candidates are decoded together and, once $k$ of "
           "them have finished, the rest are dropped and the best finished one is kept. Seconds per prompt (decode, "
           "render and score, one H200) and the quality change against one sample of the original model, with its "
           "paired 95\\% bootstrap interval; 300 prompts per setting.}")
    if dropped:
        cap = cap[:-1] + f" Rows without a run in every setting are left out.}}"
    write("app_a16_first_k", table("table*", cap, "app:first-k", "l*{12}{c}", head, lines, sep="2.2pt"))


def a17():
    T = md_table((RUNS / "2026-10-16-harvest/bon_fid_t2-bench300-firstk.md").read_text(), "| method |")
    arms = [(a, lab) for a, lab in ARMS if a in T]
    vals = {}
    for a, _ in arms:
        ic = [float(x) for x in T[a]["icon FID (white / official)"].split(" / ")]
        il = [float(x) for x in T[a]["illustration FID (white / official)"].split(" / ")]
        vals[a] = [ic[0], il[0], ic[1], il[1]]
    assert vals["ar1"] == [132.78, 169.84, 196.48, 162.49]
    # no best / second marks: at 150 images per subset a few FID points are noise, and ranking noise would read as
    # a result (the white-reference icon column's "best" is an original-model arm by 2 points)
    lines = [f"{lab} & " + " & ".join(f"{vals[a][j]:.2f}" for j in range(4)) + " \\\\" for a, lab in arms]
    head = (" & \\multicolumn{2}{c}{white reference} & \\multicolumn{2}{c}{upstream reference} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Output & icon & illus. & icon & illus. \\\\")
    cap = ("\\caption{\\textbf{FID of the chosen outputs.} OmniSVG-8B text-to-SVG, the 300 prompts of "
           "Table~\\ref{app:first-k}, 150 per subset, so differences of a few points are noise. Our outputs are "
           "rendered on white; the upstream reference images are transparent and the upstream script flattens "
           "them onto black, so the white reference is the comparable one. Lower is better.}")
    write("app_a17_fid", table("table", cap, "app:fid", "l*{4}{c}", head, lines, width="\\columnwidth", sep="2.0pt"))


# ---------------------------------------------------------------------------------------------- A19
def rows(d):
    R = [r for f in sorted(glob.glob(f"{d}/shard-*/results.jsonl")) for r in map(json.loads, open(f)) if r]
    return {r["gidx"]: r for r in R if r["mode"] == "sample" and r["set"] == "bench"}


def tau(R):
    return sum(r["rec_full"][1] for r in R) / sum(r["rec_full"][0] for r in R)


def speed(R):
    return st.mean(r["t_ar"] / r["t_spec"] for r in R)


def a19():
    # temperature (OmniSVG-8B image-to-SVG); T 0.3 = the official setting = the table-1 run
    out_t, M3 = [], [("Ours", "ours/D1-top16-N127"), ("DDTree", "b16/D1-top16-N127"), ("DFlash", "b16/D0-chain")]
    pub_t = {"0.3": {"Ours": (13.12, 6.49), "DDTree": (10.09, 5.42), "DFlash": (7.49, 4.58)}}
    for T in ("0.3", "0.6", "1.0"):
        base = U / "tableA/i2-own-b127" if T == "0.3" else U / f"r1029/temp-i2/{T}"
        R = {m: rows(base / sub) for m, sub in M3}
        common = sorted(set.intersection(*(set(v) for v in R.values())))
        assert all(R[m][g].get("t_ar") for m in R for g in common)
        v = {m: (tau([R[m][g] for g in common]), speed([R[m][g] for g in common])) for m in R}
        if T in pub_t:
            assert len(common) == 300 and all(abs(v[m][0] - pub_t[T][m][0]) < 0.006 and
                                              abs(v[m][1] - pub_t[T][m][1]) < 0.006 for m in R), v
        out_t.append((T, len(common), v))
    # seeds; offset 0 = the table-1 run. Both blocks run Ours then DDTree in ONE job on one node; DDTree reuses the
    # AR that the Ours run timed (image: --ar-cache of Ours' --ar-save; InternSVG: --no-ar after Ours' --time-ar), so
    # DDTree's speed-up uses Ours' t_ar on the same prompt -- table 1's shared-AR convention.
    out_s = []
    pub_s = {("i2", "0"): (6.49, 5.42), ("i2", "1000"): (6.16, 4.95), ("i2", "2000"): (5.95, 5.12)}
    T1 = {"i2": U / "tableA/i2-own-b127", "isvg": U / "tableA/isvg-mix/own"}
    for blk in ("i2", "isvg"):
        for off in ("0", "1000", "2000"):
            if off == "0":
                Ro, Rd = rows(T1[blk] / "ours/D1-top16-N127"), rows(T1[blk] / "b16/D1-top16-N127")
            else:
                Ro = rows(U / f"r1029/seeds/{blk}-{off}/ours/D1-top16-N127")
                Rd = rows(U / f"r1029/seeds/{blk}-{off}/ddtree/D1-top16-N127")
            common = sorted(g for g in set(Ro) & set(Rd) if Ro[g].get("t_ar"))
            if blk == "i2":                                   # the convention changes nothing where both exist
                assert all(abs(Rd[g]["t_ar"] - Ro[g]["t_ar"]) < 1e-9 for g in common if Rd[g].get("t_ar")), off
            vo = (tau([Ro[g] for g in common]), st.mean(Ro[g]["t_ar"] / Ro[g]["t_spec"] for g in common))
            vd = (tau([Rd[g] for g in common]), st.mean(Ro[g]["t_ar"] / Rd[g]["t_spec"] for g in common))
            if (blk, off) in pub_s:
                assert len(common) == 300 and abs(vo[1] - pub_s[(blk, off)][0]) < 0.006 and \
                    abs(vd[1] - pub_s[(blk, off)][1]) < 0.006, (blk, off, len(common), vo, vd)
            out_s.append((blk, off, len(common), {"Ours": vo, "DDTree": vd}))
    lines = []
    for T, n, v in out_t:
        ms = dict(zip([m for m, _ in M3], marks([v[m][1] for m, _ in M3], 2)))
        mt = dict(zip([m for m, _ in M3], marks([v[m][0] for m, _ in M3], 2)))
        lab = f"$T$ {T}" + (" (official)" if T == "0.3" else "")
        lines.append(f"{lab} & {n} & " + " & ".join(cell(v[m][0], mt[m]) for m, _ in M3) + " & "
                     + " & ".join(cell(v[m][1], ms[m], x=True) for m, _ in M3) + " \\\\")
    head = (" & & \\multicolumn{3}{c}{$\\tau$} & \\multicolumn{3}{c}{speed-up} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Temperature & prompts & " + " & ".join(B1.method_name(m) for m, _ in M3) * 1 + " & "
            + " & ".join(B1.method_name(m) for m, _ in M3) + " \\\\")
    cap = ("\\caption{\\textbf{Sampling temperature.} OmniSVG-8B image-to-SVG, MMSVG-Bench, official top-$p$; the "
           "prompts every method finished at that temperature. Speed-up is the mean over prompts against the "
           "original model timed in the same job. Best in bold, second underlined, per row.}")
    write("app_a19_temperature", table("table", cap, "app:temperature", "lr*{6}{c}", head, lines,
                                       width="\\columnwidth", sep="1.4pt"))
    lines, spread = [], {}
    for blk, name in (("i2", "OmniSVG-8B, image-to-SVG"), ("isvg", "InternSVG-8B, text-to-SVG")):
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{name}}}}} \\\\")
        sp = []
        for b, off, n, v in out_s:
            if b != blk:
                continue
            sp.append(v["Ours"][1])
            lines.append(f"{int(off) // 1000 + 1} & {n} & {v['Ours'][0]:.2f} & {v['DDTree'][0]:.2f} & "
                         f"{v['Ours'][1]:.2f}$\\times$ & {v['DDTree'][1]:.2f}$\\times$ & "
                         f"{v['Ours'][1] / v['DDTree'][1]:.2f} \\\\")
        spread[blk] = 100 * (max(sp) - min(sp)) / st.mean(sp)
        lines.append("\\addlinespace[2pt]")
    lines = lines[:-1]
    assert all(v["Ours"][1] > v["DDTree"][1] for *_, v in out_s)
    head = (" & & \\multicolumn{2}{c}{$\\tau$} & \\multicolumn{2}{c}{speed-up} & \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
            "Seed & prompts & \\textbf{Ours} & DDTree & \\textbf{Ours} & DDTree & Ours / DDTree \\\\")
    cap = ("\\caption{\\textbf{Sampling seed.} Each model's official sampling, three seeds, the same 300 MMSVG-Bench "
           "prompts; DDTree runs in the same job as ours and shares its timing of the original model. Our speed-up "
           f"moves by {min(spread.values()):.0f}--{max(spread.values()):.0f}\\% across seeds and stays ahead of DDTree "
           "at every seed.}")
    write("app_a19_seeds", table("table", cap, "app:seeds", "lr*{5}{c}", head, lines, width="\\columnwidth",
                                 sep="2.0pt"))
    return out_t, out_s


# ---------------------------------------------------------------------------------------------- A21
def a21():
    text = (RUNS / "2026-10-22/analysis/scorer_swap.md").read_text()
    blocks = [("text-to-SVG (900 prompts", "| selection | clip |", ["clip", "aesthetic", "hps"],
               ["CLIP", "Aesthetic", "HPS"], [1, 1, 1], "text"),
              ("image-to-SVG (900 prompts", "| selection | dino |", ["dino", "ssim", "lpips", "mse"],
               ["DINO", "SSIM", "LPIPS", "MSE"], [1, 1, -1, -1], "image")]
    for title, hdr, keys, names, signs, tag in blocks:
        sub = text[text.index(title):]
        T = md_table(sub, hdr)
        picks = list(T)
        allpos = True
        lines = []
        for p in picks:
            cells = []
            for k, sg in zip(keys, signs):
                q, lo, hi = ci(T[p][k])
                good = (lo > 0) if sg > 0 else (hi < 0)
                allpos &= good
                cells.append(fnum(q, 4))
            lines.append(p.replace("pick by ", "").replace("clip", "CLIP").replace("aesthetic", "Aesthetic")
                         .replace("hps", "HPS").replace("dino", "DINO").replace("ssim", "SSIM")
                         .replace("lpips", "LPIPS").replace("mse", "MSE") + " & " + " & ".join(cells) + " \\\\")
        head = "Pick by & " + " & ".join(names) + " \\\\"
        task = "OmniSVG-8B text-to-SVG" if tag == "text" else "OmniSVG-8B image-to-SVG"
        cap = (f"\\caption{{\\textbf{{Swapping the scorer, {tag}-to-SVG.}} {task}, 900 prompts, eight of our samples "
               "each: the sample picked by one metric, scored on every metric, minus a random pick. "
               + ("Every 95\\% bootstrap interval is on the improving side" if allpos else
                  "95\\% bootstrap intervals: see the released data")
               + (" (LPIPS and MSE: lower is better)." if tag == "image" else ".") + "}")
        write(f"app_a21_scorer_swap_{tag}", table("table", cap, f"app:scorer-swap-{tag}", "l*{%d}{c}" % len(keys),
                                                  head, lines, width="\\columnwidth", sep="3.0pt"))
        print(tag, "all intervals on the improving side:", allpos)


if __name__ == "__main__":
    a10()
    a16()
    a17()
    t, s = a19()
    for T, n, v in t:
        print("T", T, n, {m: (round(a, 2), round(b, 2)) for m, (a, b) in v.items()})
    a21()
