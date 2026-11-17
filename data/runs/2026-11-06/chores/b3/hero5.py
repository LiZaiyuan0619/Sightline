"""B3 (B line): the data layer of the hero figure for ALL FIVE table-1 blocks. 0 GPU, result files only.

Generalises paper/claims/H2-hero/round2/h2data2.py (OmniSVG-8B text-to-SVG only, hand-listed result dirs) by taking
the rows from table 1's own builder instead of re-listing paths: runs/2026-10-30/tableA_v9.py `load(block, mode)` ends
in v4.block_table(cols, methods, data) with data[method] = (rows, AR dict); we wrap that one function and keep its
inputs. So every block, column, method, row and AR timing below is exactly what table 1 reports.

One clock per block (as h2data2): the reference AR is the one table 1 pairs with "Ours (lossless)" (the AR timed
inline in the `ours` job for the four blocks of v4, the block-wide isvg-i2/ar-* for InternSVG image-to-SVG). Methods
timed against that same AR keep t = t_spec. Methods table 1 times against their own job's AR (SpS, DARTree, REST) are
put on the clock with t = t_spec * F, F = (reference AR s/token) / (own AR s/token) over the column's rows, which
leaves each method's table-1 speed-up unchanged (asserted).

check: every (block, mode, column, method) speed-up recomputed from the captured rows equals
runs/2026-11-02/figdata/table1.json to 1e-9, and again on the shared clock; and the two round-2 numbers of
OmniSVG-8B text-to-SVG / MMSVG-Icon are reproduced (AR 2 / 40 and Ours 21 / 40 finished within 1 s; factors
0.940 / 1.026 / 1.022 for SpS / DARTree / REST).

  python3 runs/2026-11-06/chores/b3/hero5.py      -> hero5.json + hero5.md (this dir)
"""
from __future__ import annotations

import importlib.util
import json
import statistics as st
from pathlib import Path
from sightline.paths import RUNS

ROOT = RUNS.parent            # data/, which holds runs/
HERE = Path(__file__).resolve().parent
T1 = RUNS / "2026-11-02/figdata/table1.json"
REF = "Ours (lossless)"
SHORT = {"OmniSVG-8B, text-to-SVG": "t2", "OmniSVG-4B, text-to-SVG": "4b", "InternSVG-8B, text-to-SVG": "isvg",
         "OmniSVG-8B, image-to-SVG": "i2", "InternSVG-8B, image-to-SVG": "i2s"}
LABEL = {REF: "Ours"}                                   # hero labels (h2data2 uses "Ours")


def _v9():
    spec = importlib.util.spec_from_file_location("v9", RUNS / "2026-10-30/tableA_v9.py")
    v9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v9)
    return v9


def capture(modes=("greedy", "sample")) -> dict:
    """{(block, mode): {"cols": [(label, sets, subtype)], "data": {method: (rows, ar)}}} as table 1 sees them."""
    v9 = _v9()
    seen = {}
    orig = v9.v4.block_table

    def grab(cols, meth, data):
        seen["cols"], seen["data"] = cols, {m: data[m] for m in meth}
        return orig(cols, meth, data)

    v9.v4.block_table = grab
    out = {}
    try:
        for spec in v9.blocks("mix"):
            for mode in modes:
                seen.clear()
                v9.load(spec, mode)
                out[(spec[0], mode)] = {"cols": list(seen["cols"]), "data": seen["data"]}
    finally:
        v9.v4.block_table = orig
    return out


def col_rows(rows, ar, sets, sb):
    sets = (sets,) if isinstance(sets, str) else tuple(sets)
    return {r["gidx"]: r for r in rows if r["set"] in sets and r["subtype"] == sb and (r["mode"], r["gidx"]) in ar}


def _rate(ar, mode, ids):
    return sum(ar[(mode, g)][0] for g in ids) / sum(ar[(mode, g)][1] for g in ids)


def column(cap: dict, block: str, mode: str, j: int) -> dict:
    """One table-1 cell group on one clock: per method the rows, factor, per-row seconds and table speed-up."""
    c = cap[(block, mode)]
    lab, sets, sb = c["cols"][j]
    R_ref, ar_ref = c["data"][REF]
    ref = col_rows(R_ref, ar_ref, sets, sb)
    out = {"column": lab, "mode": mode, "block": block, "methods": {}}
    ar_t = {g: ar_ref[(mode, g)][0] for g in ref}
    out["methods"]["original model"] = {"t": ar_t, "n": {g: ar_ref[(mode, g)][1] for g in ref}, "factor": 1.0,
                                        "rule": "reference"}
    for m, (R, ar) in c["data"].items():
        rows = col_rows(R, ar, sets, sb)
        if not rows:
            continue
        ids = sorted(rows)
        sp = _rate(ar, mode, ids) / (sum(rows[g]["t_spec"] for g in ids) / sum(rows[g]["n_spec"] for g in ids))
        shared = all((mode, g) in ar_ref and ar[(mode, g)] == ar_ref[(mode, g)] for g in ids)
        if shared:
            f, rule = 1.0, "shared"
        else:
            both = [g for g in ids if (mode, g) in ar_ref]
            assert len(both) == len(ids), (block, mode, lab, m, len(both), len(ids))
            f, rule = _rate(ar_ref, mode, ids) / _rate(ar, mode, ids), "own AR, rescaled"
        out["methods"][LABEL.get(m, m)] = {
            "t": {g: rows[g]["t_spec"] * f for g in ids}, "n": {g: rows[g]["n_spec"] for g in ids},
            "factor": f, "rule": rule, "speedup_table": sp, "rows": rows}
    return out


def check(cap: dict | None = None, verbose: bool = True) -> dict:
    cap = cap or capture()
    t1 = json.loads(T1.read_text())
    n_ok = 0
    res = {}
    for blk in t1["blocks"]:
        name, mode = blk["block"], blk["mode"]
        labels = [c[0] for c in blk["columns"]]
        for j, lab in enumerate(labels):
            col = column(cap, name, mode, j)
            assert col["column"] == lab, (name, mode, j, col["column"], lab)
            ref = col["methods"]["original model"]
            for m, v in col["methods"].items():
                if m == "original model":
                    continue
                key = REF if m == "Ours" else m
                want = blk["rows"][key][j]["speedup"]
                assert abs(v["speedup_table"] - want) < 1e-9, (name, mode, lab, m, v["speedup_table"], want)
                ids = sorted(v["t"])
                on_clock = (sum(ref["t"][g] for g in ids) / sum(ref["n"][g] for g in ids)) / \
                           (sum(v["t"][g] for g in ids) / sum(v["n"][g] for g in ids))
                assert abs(on_clock - want) < 1e-9, (name, mode, lab, m, on_clock, want)
                n_ok += 1
            res[(name, mode, lab)] = col
    # the round-2 numbers this layer must keep (h2data2.check / README)
    icon = res[("OmniSVG-8B, text-to-SVG", "greedy", "MMSVG-Icon")]["methods"]
    first40 = range(40)
    fin = {m: sum(icon[m]["t"][g] <= 1.0 for g in first40) for m in ("original model", "Ours")}
    assert fin == {"original model": 2, "Ours": 21}, fin
    fac = {m: round(icon[m]["factor"], 3) for m in ("SpS", "DARTree", "REST")}
    assert fac == {"SpS": 0.940, "DARTree": 1.026, "REST": 1.022}, fac
    if verbose:
        print(f"table-1 speed-ups reproduced: {n_ok} (block x mode x column x method) cells, raw and on the shared clock")
        print("round-2 anchors OK: finished within 1 s of the first 40 =", fin, "; factors", fac)
    return res


def summarise(res: dict) -> dict:
    """Per (block, mode, column): rows common to every decoder, mean seconds per drawing, drawings per minute,
    'drawings finished in the time the original model takes for one' (make.py iso_vals: ratio of mean seconds),
    and the table-1 speed-up."""
    out = {}
    for (name, mode, lab), col in res.items():
        ms = col["methods"]
        common = sorted(set.intersection(*(set(v["t"]) for v in ms.values())))
        if not common:
            continue
        unit = st.mean(ms["original model"]["t"][g] for g in common)
        rec = {"rows_common": len(common), "ar_seconds_mean": unit, "decoders": {}}
        for m, v in ms.items():
            mean_t = st.mean(v["t"][g] for g in common)
            rec["decoders"][m] = {"factor": round(v["factor"], 6), "rule": v["rule"],
                                  "rows": len(v["t"]), "seconds_mean": mean_t,
                                  "drawings_per_minute": 60.0 / mean_t,
                                  "drawings_in_one_ar_drawing": unit / mean_t,
                                  "speedup_table1": v.get("speedup_table", 1.0)}
        out[f"{SHORT[name]}|{mode}|{lab}"] = rec
    return out


def main():
    cap = capture()
    res = check(cap)
    summ = summarise(res)
    (HERE / "hero5.json").write_text(json.dumps(summ, indent=1))
    order = ["original model", "PLD", "REST", "SpS", "EAGLE-3", "DFlash", "DARTree", "DDTree", "Ours"]
    L = ["# Hero data, all five table-1 blocks: drawings finished in the time the original model takes for one",
         "", "Shared clock per block (see hero5.py). Value = mean original-model seconds / mean decoder seconds over the",
         "rows common to all decoders (make.py `iso_vals`). Greedy, every column; the hero's right half used MMSVG-Icon.",
         ""]
    for mode in ("greedy", "sample"):
        L += [f"## {mode}", "", "| block | column | rows | AR s | " + " | ".join(order[1:]) + " |",
              "|---|---|---|---|" + "---|" * (len(order) - 1)]
        for k, rec in summ.items():
            b, md, lab = k.split("|")
            if md != mode:
                continue
            d = rec["decoders"]
            L.append(f"| {b} | {lab} | {rec['rows_common']} | {rec['ar_seconds_mean']:.2f} | " +
                     " | ".join(f"{d[m]['drawings_in_one_ar_drawing']:.2f}" if m in d else "–" for m in order[1:]) + " |")
        L.append("")
    (HERE / "hero5.md").write_text("\n".join(L) + "\n")
    print("wrote", HERE / "hero5.json", "and hero5.md")


if __name__ == "__main__":
    main()
