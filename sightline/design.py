"""H2 step 2: exact offline simulation of candidate-tree designs on
the draft cache written by cache_draft.py. CPU only.

Exactness: the realized continuation (the target's own greedy / seeded-sampling output) does not
depend on the drafter, so at block start t a tree accepts the longest prefix of
tokens[t+1: t+1+k] it contains, and the round commits that prefix plus the next realized token
(clipped at the stream end exactly as the harness clips at EOS / max_new). Rounds walk the
stream the same way the harness does; the per-round accepted counts become the same
StreamRecord bundles as the B3 evaluation (analysis/calibers.py), so tokens per round is directly
comparable with the B3 numbers.

Trees: candidates per block offset come from one or more SOURCES, scored, then a budget-N tree is
grown best-first on the cumulative score (the GLANCE construction: under offset-wise
independence the N best prefix-closed paths). Sources (cache fields in brackets):
  top<b>     the draft's top-b tokens                                   [top_ids/top_lp]
  nb<r>      Chebyshev radius-r window around the best coordinate      [coord_best/nb_lp]
  cross      the 4-neighbourhood of the best coordinate                [nb_lp]
  xy<a>      a x a grid of the top-a x and top-a y coordinate marginals [xy_*]
  copy       the 16 most recent distinct coordinates of the prefix      [copy_*]
Spatial sources fire only where the draft's top-1 is a coordinate (``when=top1``) or wherever
the coordinate mass exceeds 10% (``when=mass``); ``first=m`` limits them to the first m such
offsets of a block. Scores are the draft log-probs.

Usage:
  python3 simulate.py --cache <dir> --check              # chain replay vs the B3 run
  python3 simulate.py --cache <dir> --designs default --out <dir>   # the sweep
  python3 simulate.py --cache <dir> --coverage --out <dir>          # coverage analysis
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so the file runs as a script
from analysis.calibers import (MIN_HEALTHY_ONSET, CaliberStream, StreamRecord,  # noqa: E402
                               caliber_table)
sys.path.insert(0, str(ROOT / "src" / "h2"))
from sightline.svgtok.types import COORD_END, COORD_START, GRID  # noqa: E402

NB_R = 3                                   # cache_draft.NB_R
NB_W = 2 * NB_R + 1
LOG_MASS_GATE = math.log(0.10)


def is_coord(t: int) -> bool:
    return COORD_START <= t <= COORD_END


# ------------------------------------------------------------------ designs

@dataclass(frozen=True)
class Design:
    name: str
    budget: int                            # tree nodes (excluding the committed root token)
    sources: tuple = ("top1",)             # e.g. ("top8", "nb1")
    when: str = "top1"                     # spatial sources fire: "top1" | "mass" | "always"
    first: int = 0                         # >0: spatial sources only on the first m firing offsets
    branch_by_conf: tuple = ()             # D6: ((p_min, b),...) top-b width by top-1 prob
    score: str = "draft"                   # "draft" log-probs | "calib" (D7: fitted hit rates)
    grammar: bool = False                  # D5: drop candidates of illegal type, renormalise
    max_branch: int = 0                    # D5 chain: 1 = keep only the best legal candidate
    copy_period: bool = False              # L: extra branch copying one detected period back
    geo: str = ""                          # G: "exact" | "nb1" geometric rule candidates
    copy_long: int = 0                     # L2: extra period-copy branch of up to this many tokens

    @property
    def is_chain(self) -> bool:
        return (self.sources == ("top1",) and not self.branch_by_conf and not self.grammar
                and not self.geo)


def _src(sources, prefix):
    for s in sources:
        if s.startswith(prefix):
            return s
    return None


def default_designs() -> list[Design]:
    """The first-round sweep."""
    out = [Design("D0-chain", 15)]
    for n in (15, 31, 63, 127):
        for b in (2, 4, 8, 16):
            out.append(Design(f"D1-top{b}-N{n}", n, (f"top{b}",)))
        for sp in ("nb1", "nb2", "cross"):
            out.append(Design(f"D2-{sp}-N{n}", n, ("top1", sp)))
        for b in (2, 4, 8):
            for sp in ("nb1", "nb2"):
                out.append(Design(f"D3-top{b}+{sp}-N{n}", n, (f"top{b}", sp)))
                out.append(Design(f"D3m-top{b}+{sp}-N{n}", n, (f"top{b}", sp), when="mass"))
        for m in (1, 2, 3):
            out.append(Design(f"D4-top4+nb1-first{m}-N{n}", n, ("top4", "nb1"), first=m))
        out.append(Design(f"D6-conf-N{n}", n, ("top16",),
                          branch_by_conf=((0.9, 1), (0.6, 2), (0.3, 4), (0.0, 8))))
        for a in (2, 3, 4):
            out.append(Design(f"D8-top4+xy{a}-N{n}", n, ("top4", f"xy{a}")))
        out.append(Design(f"D9-top4+copy-N{n}", n, ("top4", "copy")))
        out.append(Design(f"D9-top4+nb1+copy-N{n}", n, ("top4", "nb1", "copy")))
    return out


# ------------------------------------------------------------------ D7 calibration

P1_EDGES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 0.95, 1.01)
N_P1 = len(P1_EDGES) - 1
F_TOP, F_NB, F_XY, F_COPY = 0, 32, 32 + 2 * 49, 32 + 2 * 49 + 64
N_FEAT = F_COPY + 16
CALIB = None          # (N_P1, N_FEAT) log hit rates, set by set_calibration


def p1_bucket(top1_lp: float) -> int:
    p = math.exp(top1_lp)
    for b in range(N_P1):
        if p < P1_EDGES[b + 1]:
            return b
    return N_P1 - 1


def set_calibration(path):
    global CALIB
    if path is None:
        CALIB = None
        return
    z = json.loads(Path(path).read_text())
    num, den = np.asarray(z["num"], float), np.asarray(z["den"], float)
    CALIB = np.log((num + 0.5) / (den + 1.0))


def fit_counts(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Hit / opportunity counts per (top-1 confidence bucket, candidate feature) over every
    block start and offset of one cached row. Features: top-k rank (32); offset from the best
    coordinate in the 7x7 window, split by whether the top-1 is a coordinate (2 x 49); x/y grid
    cell (64); copy-list recency (16)."""
    num = np.zeros((N_P1, N_FEAT))
    den = np.zeros((N_P1, N_FEAT))
    tokens = row["tokens"]
    n_gen = len(tokens)
    T, K = row["top_ids"].shape[:2]
    for t in range(T):
        k = int(row["n_off"][t])
        for j in range(k):
            if t + 1 + j >= n_gen:
                break
            y = int(tokens[t + 1 + j])
            b = p1_bucket(float(row["top_lp"][t, j, 0]))
            den[b, F_TOP:F_TOP + 32] += 1
            r = int(row["real_rank"][t, j])
            if 0 <= r < 32:
                num[b, F_TOP + r] += 1
            flag = int(is_coord(int(row["top_ids"][t, j, 0])))
            nbv = ~np.isnan(row["nb_lp"][t, j].astype(np.float32))
            den[b, F_NB + flag * 49:F_NB + flag * 49 + 49] += nbv
            cb = int(row["coord_best"][t, j])
            if is_coord(y) and cb >= 0:
                dx = (y - COORD_START) % GRID - (cb - COORD_START) % GRID
                dy = (y - COORD_START) // GRID - (cb - COORD_START) // GRID
                if abs(dx) <= NB_R and abs(dy) <= NB_R:
                    num[b, F_NB + flag * 49 + (dy + NB_R) * NB_W + (dx + NB_R)] += 1
                xs = row["xy_x"][t, j].tolist()
                ys = row["xy_y"][t, j].tolist()
                yx, yy = (y - COORD_START) % GRID, (y - COORD_START) // GRID
                den[b, F_XY:F_XY + 64] += 1
                if yx in xs and yy in ys:
                    num[b, F_XY + ys.index(yy) * 8 + xs.index(yx)] += 1
            else:
                den[b, F_XY:F_XY + 64] += 1
            cid = row["copy_ids"][t]
            den[b, F_COPY:F_COPY + 16] += cid >= 0
            hit = np.nonzero(cid == y)[0]
            if hit.size:
                num[b, F_COPY + int(hit[0])] += 1
    return num, den


def _fit_file(path):
    return fit_counts(load_row(path))


def extra_designs() -> list[Design]:
    """Loop-period copy branch (L) and larger generic trees."""
    out = [Design("L-chain+copyP", 15, copy_period=True)]
    for n in (63, 127):
        out.append(Design(f"L-top16+copyP-N{n}", n, ("top16",), copy_period=True))
    out += [Design("D1-top32-N127", 127, ("top32",)), Design("D1-top16-N255", 255, ("top16",)),
            Design("D1-top32-N255", 255, ("top32",))]
    return out


def grammar_designs() -> list[Design]:
    """D5 (H1 type constraints) on top of D0, D1 and D3."""
    out = [Design("D5-chain", 15, ("top8",), grammar=True, max_branch=1)]   # best legal token per offset
    for n in (31, 63, 127):
        out.append(Design(f"D5-top4-N{n}", n, ("top4",), grammar=True))
        out.append(Design(f"D5-top8-N{n}", n, ("top8",), grammar=True))
        out.append(Design(f"D5-top4+nb1-N{n}", n, ("top4", "nb1"), grammar=True))
    return out


def calib_designs() -> list[Design]:
    """D7: the same candidate sources, scored by fitted hit rates instead of draft log-probs."""
    out = []
    for n in (15, 31, 63, 127):
        for name, src in (("top4", ("top4",)), ("top8", ("top8",)), ("top4+nb1", ("top4", "nb1")),
                          ("top4+nb2", ("top4", "nb2")), ("top8+nb1", ("top8", "nb1")),
                          ("top4+nb1+copy", ("top4", "nb1", "copy")), ("top4+xy3", ("top4", "xy3"))):
            out.append(Design(f"D7-{name}-N{n}", n, src, score="calib"))
    return out


# ------------------------------------------------------------------ candidates per offset

def offset_candidates(row: dict, t: int, j: int, d: Design, fired: int) -> tuple[list, list, bool]:
    """Candidate (ids, scores) at block start t, offset j (0-based), deduplicated (max score),
    sorted by score descending. Returns (ids, scores, spatial_fired)."""
    top_ids, top_lp = row["top_ids"][t, j], row["top_lp"][t, j]
    best = {}
    calib = CALIB[p1_bucket(float(top_lp[0]))] if d.score == "calib" else None

    def add(tok, lp, feat):
        if lp != lp:                        # NaN: off-grid / pad
            return
        tok = int(tok)
        sc = float(calib[feat]) if calib is not None else float(lp)
        if sc > best.get(tok, -math.inf):
            best[tok] = sc

    b = 1
    tsrc = _src(d.sources, "top")
    if tsrc:
        b = int(tsrc[3:])
    if d.branch_by_conf:
        p1 = math.exp(float(top_lp[0]))
        b = next(w for pmin, w in d.branch_by_conf if p1 >= pmin)
    for q in range(b):
        add(top_ids[q], top_lp[q], F_TOP + q)

    spatial = [s for s in d.sources if s.startswith(("nb", "cross", "xy", "copy"))]
    fire = False
    if spatial:
        if d.when == "top1":
            fire = is_coord(int(top_ids[0]))
        elif d.when == "mass":
            fire = float(row["coord_lse"][t, j]) > LOG_MASS_GATE
        else:
            fire = True
        if d.first and fired >= d.first:
            fire = False
    if fire:
        cb = int(row["coord_best"][t, j])
        nb = row["nb_lp"][t, j]
        f_nb = F_NB + 49 * int(is_coord(int(top_ids[0])))
        for s in spatial:
            if s.startswith("nb"):
                r = int(s[2:])
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        q = (dy + NB_R) * NB_W + (dx + NB_R)
                        add(cb + dy * GRID + dx, nb[q], f_nb + q)
            elif s == "cross":
                for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                    q = (dy + NB_R) * NB_W + (dx + NB_R)
                    add(cb + dy * GRID + dx, nb[q], f_nb + q)
            elif s.startswith("xy"):
                a = int(s[2:])
                xs, ys, g = row["xy_x"][t, j], row["xy_y"][t, j], row["xy_lp"][t, j]
                for yi in range(a):
                    for xi in range(a):
                        add(COORD_START + int(ys[yi]) * GRID + int(xs[xi]), g[yi, xi],
                            F_XY + yi * 8 + xi)
            elif s == "copy":
                for q, tok in enumerate(row["copy_ids"][t]):
                    if tok >= 0:
                        add(tok, row["copy_lp"][t, j, q], F_COPY + q)
    items = sorted(best.items(), key=lambda kv: -kv[1])
    return [k for k, _ in items], [v for _, v in items], fire


def block_candidates(row: dict, t: int, d: Design) -> list:
    """Per-offset candidate lists at block start t; memoized per source config (the budget
    only changes the tree growth, so budgets 15..127 share one candidate build)."""
    key = (d.sources, d.when, d.first, d.branch_by_conf, d.score, t)
    memo = row.setdefault("_memo", {})
    hit = memo.get(key)
    if hit is not None:
        return hit
    cands = []
    fired = 0
    for j in range(int(row["n_off"][t])):
        ids, sc, f = offset_candidates(row, t, j, d, fired)
        fired += f
        cands.append((ids, sc))
    memo[key] = cands
    return cands


# ------------------------------------------------------------------ D5 grammar (H1 type masks)

from sightline.svgtok.types import (ARC_END, ARC_START, CMD_ARGS, CMD_BY_ID, CMD_CLOSE,  # noqa: E402
                          CMD_MOVE, COLOR_END, COLOR_START, EOS)
TYPE_IDX = {"coord": 0, "cmd": 1, "color": 2, "arc": 3, "eos": 4, "other": 5}   # cache type_lse order


def tok_class(t: int) -> str:
    if COORD_START <= t <= COORD_END:
        return "coord"
    if CMD_MOVE <= t <= CMD_CLOSE:
        return "cmd"
    if COLOR_START <= t <= COLOR_END:
        return "color"
    if ARC_START <= t <= ARC_END:
        return "arc"
    if t == EOS:
        return "eos"
    return "other"


def g_feed(state: tuple, t: int) -> tuple:
    """(cmd or None, arg position) after token t; official lenient semantics
    (svgtok.types.SVGGrammar.feed_lenient): colors end a path, escapes are skipped, a new
    command interrupts an unfinished one, a wrong-class token is skipped."""
    cmd, pos = state
    c = tok_class(t)
    if c == "color":
        return (None, 0)
    if c in ("other", "eos"):
        return state
    if cmd is None:
        return (CMD_BY_ID[t], 0) if c == "cmd" else state
    if c == CMD_ARGS[cmd][pos]:
        pos += 1
        return (None, 0) if pos == len(CMD_ARGS[cmd]) else (cmd, pos)
    if c == "cmd":
        return (CMD_BY_ID[t], 0)
    return state


def g_legal(state: tuple) -> tuple:
    cmd, pos = state
    if cmd is None:
        return ("cmd", "color", "eos")
    return (CMD_ARGS[cmd][pos], "color")


def grammar_states(tokens) -> list[tuple]:
    """State after tokens[0..t] (BOS at index 0 is ignored), for every t."""
    out = []
    st = (None, 0)
    for i, t in enumerate(tokens):
        if i > 0:
            st = g_feed(st, int(t))
        out.append(st)
    return out


def filter_legal(ids: list, sc: list, legal: tuple, type_lse_row, renorm: bool):
    keep = [(tok, v) for tok, v in zip(ids, sc) if tok_class(tok) in legal]
    if renorm and keep:
        lse = float(np.logaddexp.reduce([float(type_lse_row[TYPE_IDX[c]]) for c in legal]))
        keep = [(tok, v - lse) for tok, v in keep]
    return [k for k, _ in keep], [v for _, v in keep]


def tree_accept_grammar(row: dict, t: int, d: Design, cands: list, realized: list[int]) -> int:
    """tree_accept with path-dependent candidate lists: a node's children are the candidates of
    the next offset whose class is legal after the path to it (draft scores renormalised over
    the legal classes; calibrated scores are left as they are)."""
    k = len(cands)
    m = min(k, len(realized))
    states = row.get("_gstates")
    if states is None:
        states = row["_gstates"] = grammar_states(row["tokens"])
    width = d.max_branch or 10 ** 9
    memo = {}

    def lists(depth, state):
        key = (depth, state)
        if key not in memo:
            ids, sc = cands[depth]
            memo[key] = filter_legal(ids, sc, g_legal(state), row["type_lse"][t, depth],
                                     d.score == "draft")
        return memo[key]

    s0 = states[t]
    ids0, sc0 = lists(0, s0)
    if not ids0:
        return 0
    heap = [(-sc0[0], 0, 0, -1, True, s0)]          #..., parent on path, parent's state
    deepest = 0
    pops = 0
    while heap and pops < d.budget:
        neg, depth, q, parent, parent_ok, pstate = heapq.heappop(heap)
        me = pops
        pops += 1
        cum = -neg
        ids, sc = lists(depth, pstate)
        tok = ids[q]
        node_ok = parent_ok and depth < m and tok == realized[depth]
        if node_ok and depth + 1 > deepest:
            deepest = depth + 1
        if q + 1 < min(len(ids), width):
            heapq.heappush(heap, (-(cum - sc[q] + sc[q + 1]), depth, q + 1, parent, parent_ok, pstate))
        if depth + 1 < k:
            nstate = g_feed(pstate, tok)
            cids, csc = lists(depth + 1, nstate)
            if cids:
                heapq.heappush(heap, (-(cum + csc[0]), depth + 1, 0, me, node_ok, nstate))
    return min(deepest, m)


def tree_accept(cands: list, budget: int, realized: list[int]) -> int:
    """Best-first budget tree over per-offset candidate lists (sorted by score, descending;
    scores are log-probabilities, <= 0). Lazy (first-child, next-sibling) pushes pop nodes in
    the same score order as pushing every child at once (glance.tree.build_budget_tree), so
    the first `budget` pops are its tree. Returns how deep the realized path got.
    Heap keys (-cum, depth, sibling idx, parent pop idx) are unique, so this and grow_tree pop
    in exactly the same order (the verifier builds its trees with grow_tree)."""
    k = len(cands)
    m = min(k, len(realized))
    if not k or not cands[0][0]:
        return 0
    heap = [(-cands[0][1][0], 0, 0, -1, True)]  # (-cum, depth, sibling, parent idx, parent on path)
    deepest = 0
    pops = 0
    while heap and pops < budget:
        neg, depth, q, parent, parent_ok = heapq.heappop(heap)
        me = pops
        pops += 1
        cum = -neg
        ids, sc = cands[depth]
        node_ok = parent_ok and depth < m and ids[q] == realized[depth]
        if node_ok and depth + 1 > deepest:
            deepest = depth + 1
        if q + 1 < len(ids):
            heapq.heappush(heap, (-(cum - sc[q] + sc[q + 1]), depth, q + 1, parent, parent_ok))
        if depth + 1 < k and cands[depth + 1][0]:
            heapq.heappush(heap, (-(cum + cands[depth + 1][1][0]), depth + 1, 0, me, node_ok))
    return min(deepest, m)


def grow_tree(cands: list, budget: int) -> list[tuple]:
    """The tree itself, in pop order (parents before children):
    [(token, depth 1-based, parent index in this list or -1 for the root, sibling rank)]."""
    k = len(cands)
    if not k or not cands[0][0]:
        return []
    nodes = []
    heap = [(-cands[0][1][0], 0, 0, -1)]
    while heap and len(nodes) < budget:
        neg, depth, q, parent = heapq.heappop(heap)
        me = len(nodes)
        cum = -neg
        ids, sc = cands[depth]
        nodes.append((int(ids[q]), depth + 1, parent, q))
        if q + 1 < len(ids):
            heapq.heappush(heap, (-(cum - sc[q] + sc[q + 1]), depth, q + 1, parent))
        if depth + 1 < k and cands[depth + 1][0]:
            heapq.heappush(heap, (-(cum + cands[depth + 1][1][0]), depth + 1, 0, me))
    return nodes


MAX_PERIOD = 64


def periods(tokens) -> np.ndarray:
    """period[t] = smallest P <= MAX_PERIOD such that tokens[t-P+1..t] == tokens[t-2P+1..t-P]
    (the committed prefix ends with two full copies of a P-token cycle), else 0. Online: uses
    tokens up to t only."""
    tok = np.asarray(tokens, dtype=np.int64)
    n = len(tok)
    out = np.zeros(n, dtype=np.int32)
    for P in range(MAX_PERIOD, 0, -1):          # descending, so the smallest P wins
        if 2 * P > n:
            continue
        eq = np.zeros(n, dtype=bool)
        eq[P:] = tok[P:] == tok[:-P]
        run = np.zeros(n, dtype=np.int32)       # consecutive matches ending at i
        r = 0
        for i in range(n):
            r = r + 1 if eq[i] else 0
            run[i] = r
        hit = run >= P
        out[hit] = P
    return out


def copy_chain_accept(row: dict, t: int, realized: list[int], k: int) -> int:
    per = row.get("_periods")
    if per is None:
        per = row["_periods"] = periods(row["tokens"])
    P = int(per[t])
    if not P:
        return 0
    tokens = row["tokens"]
    chain = []
    for j in range(k):
        src = t + 1 + j - P
        chain.append(int(tokens[src]) if src <= t else chain[j - P])
    L = 0
    while L < min(k, len(realized)) and chain[L] == realized[L]:
        L += 1
    return L


# ------------------------------------------------------------------ G: geometric rules

GEO_PRIOR = None      # {rule: log hit rate}, set in main by geo.fit_priors on the extra rows


def _draft_list(row: dict, t: int, j: int, b: int) -> tuple[list, list]:
    """The draft's top-b at block start t, offset j: (ids, log-probs), deduplicated, sorted."""
    memo = row.setdefault("_dl", {})
    hit = memo.get((t, j, b))
    if hit is None:
        best = {}
        for tok, lp in zip(row["top_ids"][t, j, :b], row["top_lp"][t, j, :b]):
            lp = float(lp)
            if lp == lp and lp > best.get(int(tok), -math.inf):
                best[int(tok)] = lp
        items = sorted(best.items(), key=lambda kv: -kv[1])
        hit = memo[(t, j, b)] = ([k for k, _ in items], [v for _, v in items])
    return hit


R_EDGES = (0, 1, 2, 4, 8, 16, 32)          # draft-rank buckets of a rule candidate (+1: absent)
N_RB = len(R_EDGES)
GEO_CALIB = None      # {rule: (N_P1, N_RB) log hit rates}, set by _init_worker (--geo-calib)


def rank_bucket(r: int) -> int:
    for i in range(N_RB - 1, -1, -1):
        if r >= R_EDGES[i]:
            return i
    return 0


def _rule_scores(row: dict, t: int, j: int, st: tuple, nb: bool) -> dict:
    """{token: score} of the rule candidates at block start t, offset j. Uncalibrated: the
    rule's marginal log hit rate (GEO_PRIOR). Calibrated: the fitted log hit rate of
    (rule, draft top-1 confidence bucket, the candidate's rank in the draft's top-32)."""
    from sightline import geo_rules as geo   # not included: see the module docstring
    if GEO_CALIB is None:
        return geo.rule_candidates(st, GEO_PRIOR, nb)
    pts = geo.rule_points(st)
    if not pts:
        return {}
    ids = row["top_ids"][t, j]
    b1 = p1_bucket(float(row["top_lp"][t, j, 0]))
    pos = row.setdefault("_rank", {}).get((t, j))
    if pos is None:
        pos = row["_rank"][(t, j)] = {int(x): q for q, x in enumerate(ids)}
    out = {}
    for p, rule in pts:
        names = [(p, rule)]
        if nb and rule in ("reflect", "circle", "translate"):
            names += [((p[0] + dx, p[1] + dy), rule + "+nb") for dx, dy in geo.NB8]
        for q, name in names:
            tok = geo.xy_tok(q)
            if tok is None:
                continue
            sc = float(GEO_CALIB[name][b1, rank_bucket(pos.get(tok, 32))])
            if sc > out.get(tok, -math.inf):
                out[tok] = sc
    return out


def fit_geo_counts(row: dict) -> dict:
    """Hit / opportunity counts of every rule candidate along the realized stream, per
    (rule, draft top-1 confidence bucket, draft rank bucket), over every block start and
    offset (the branch state at offset j of a block starting at t is the realized state at
    t + j, the only branch whose candidates can matter)."""
    from sightline import geo_rules as geo   # not included: see the module docstring
    tokens = row["tokens"]
    n = len(tokens)
    states = geo.states_along(tokens)
    names = list(geo.RULES) + [r + "+nb" for r in ("reflect", "circle", "translate")]
    num = {r: np.zeros((N_P1, N_RB)) for r in names}
    den = {r: np.zeros((N_P1, N_RB)) for r in names}
    T, K = row["top_ids"].shape[:2]
    for p in range(1, n):
        pts = geo.rule_points(states[p - 1])
        if not pts:
            continue
        real = int(tokens[p])
        ts = np.arange(max(0, p - K), min(p, T))
        js = p - 1 - ts
        keep = js < row["n_off"][ts]
        ts, js = ts[keep], js[keep]
        if not len(ts):
            continue
        ids = row["top_ids"][ts, js]                       # (m, 32)
        b1 = np.array([p1_bucket(float(x)) for x in row["top_lp"][ts, js, 0]])
        cands = []
        for q, rule in pts:
            cands.append((q, rule))
            if rule in ("reflect", "circle", "translate"):
                cands += [((q[0] + dx, q[1] + dy), rule + "+nb") for dx, dy in geo.NB8]
        seen = set()
        for q, name in cands:
            tok = geo.xy_tok(q)
            if tok is None or (tok, name) in seen:
                continue
            seen.add((tok, name))
            eq = ids == tok
            rank = np.where(eq.any(1), eq.argmax(1), 32)
            rb = np.array([rank_bucket(int(r)) for r in rank])
            np.add.at(den[name], (b1, rb), 1)
            if tok == real:
                np.add.at(num[name], (b1, rb), 1)
    return {"num": {k: v.tolist() for k, v in num.items()}, "den": {k: v.tolist() for k, v in den.items()}}


def _fit_geo_file(path):
    return fit_geo_counts(load_row(path))


def _geo_children(row: dict, t: int, j: int, b: int, st: tuple, nb: bool) -> tuple[list, list]:
    """Children of a node whose branch state is st: draft top-b at offset j merged with the
    rule candidates of that branch (scores: draft log-probs / rule scores; max wins)."""
    ids, sc = _draft_list(row, t, j, b)
    rc = _rule_scores(row, t, j, st, nb)
    if not rc:
        return ids, sc
    merged = dict(zip(ids, sc))
    for tok, s in rc.items():
        if s > merged.get(tok, -math.inf):
            merged[tok] = s
    items = sorted(merged.items(), key=lambda kv: -kv[1])
    return [k for k, _ in items], [v for _, v in items]


def tree_accept_geo(row: dict, t: int, d: Design, realized: list[int]) -> int:
    """tree_accept with branch-dependent candidate lists: every node's children are the draft's
    top-b at the next offset plus the geometric rule candidates computed from that node's own
    branch (committed prefix + branch tokens). Same best-first budget growth and heap keys as
    tree_accept, which it reproduces exactly when no rule fires."""
    from sightline import geo_rules as geo   # not included: see the module docstring
    k = int(row["n_off"][t])
    m = min(k, len(realized))
    if not k:
        return 0
    tsrc = _src(d.sources, "top")
    b = int(tsrc[3:]) if tsrc else 1
    nb = d.geo == "nb1"
    states = row.get("_geo_states")
    if states is None:
        states = row["_geo_states"] = geo.states_along(row["tokens"])
    root = states[t]
    ids0, sc0 = _geo_children(row, t, 0, b, root, nb)
    if not ids0:
        return 0
    lists = [(ids0, sc0, 0.0, root, True)]  # (ids, scores, parent cum, parent state, parent on path)
    heap = [(-sc0[0], 0, 0, -1, 0)]          # (-cum, depth, sibling, parent pop idx, list idx)
    deepest = 0
    pops = 0
    while heap and pops < d.budget:
        neg, depth, q, parent, li = heapq.heappop(heap)
        me = pops
        pops += 1
        ids, sc, pcum, pst, pok = lists[li]
        tok = ids[q]
        cum = pcum + sc[q]
        ok = pok and depth < m and tok == realized[depth]
        if ok and depth + 1 > deepest:
            deepest = depth + 1
        if q + 1 < len(ids):
            heapq.heappush(heap, (-(pcum + sc[q + 1]), depth, q + 1, parent, li))
        if depth + 1 < k:
            cst = geo.feed(pst, tok)
            cids, csc = _geo_children(row, t, depth + 1, b, cst, nb)
            if cids:
                lists.append((cids, csc, cum, cst, ok))
                heapq.heappush(heap, (-(cum + csc[0]), depth + 1, 0, me, len(lists) - 1))
    return min(deepest, m)


def _init_worker(calib, pri, geo_calib=None):
    global GEO_PRIOR, GEO_CALIB
    set_calibration(calib)
    GEO_PRIOR = pri
    GEO_CALIB = None
    if geo_calib is not None:
        z = json.loads(Path(geo_calib).read_text())
        GEO_CALIB = {r: np.log((np.asarray(z["num"][r]) + 0.5) / (np.asarray(z["den"][r]) + 1.0))
                     for r in z["num"]}


def geo_designs() -> list[Design]:
    out = [Design("D0-chain", 15)]
    for n in (15, 31, 63):
        out.append(Design(f"G0-top1+geo-N{n}", n, ("top1",), geo="exact"))
        out.append(Design(f"G0-top1+geonb-N{n}", n, ("top1",), geo="nb1"))
    for n in (31, 63, 127, 255):
        out.append(Design(f"D1-top16-N{n}", n, ("top16",)))
        out.append(Design(f"G1-top16+geo-N{n}", n, ("top16",), geo="exact"))
        out.append(Design(f"G2-top16+geonb-N{n}", n, ("top16",), geo="nb1"))
    out.append(Design("D1-top32-N255", 255, ("top32",)))
    out.append(Design("G1-top32+geo-N255", 255, ("top32",), geo="exact"))
    out.append(Design("G2-top32+geonb-N255", 255, ("top32",), geo="nb1"))
    return out


def copylong_designs() -> list[Design]:
    out = [Design("D0-chain", 15)]
    for c in (32, 64, 128, 256):
        out.append(Design(f"L2-chain+copyLong{c}", 15, copy_long=c))
    for n in (63, 127):
        out.append(Design(f"D1-top16-N{n}", n, ("top16",)))
        for c in (64, 128, 256):
            out.append(Design(f"L2-top16+copyLong{c}-N{n}", n, ("top16",), copy_long=c))
    return out


def accept_length(row: dict, t: int, d: Design, realized: list[int]) -> int:
    """Accepted drafts at block start t under design d (exact, given the realized tokens)."""
    k = int(row["n_off"][t])
    if d.copy_long:
        # one extra branch copying the detected period up to copy_long tokens (beyond the block)
        base = replace(d, copy_long=0)
        return max(accept_length(row, t, base, realized[:k]),
                   copy_chain_accept(row, t, realized, d.copy_long))
    m = min(k, len(realized))
    if d.geo:
        return tree_accept_geo(row, t, d, realized)
    if d.copy_period:
        base = Design(d.name, d.budget, d.sources, d.when, d.first, d.branch_by_conf, d.score,
                      d.grammar, d.max_branch, False)
        return max(accept_length(row, t, base, realized), copy_chain_accept(row, t, realized, k))
    if d.is_chain:
        ids = row["top_ids"][t]
        L = 0
        while L < m and int(ids[L, 0]) == realized[L]:
            L += 1
        return L
    if d.grammar:
        return tree_accept_grammar(row, t, d, block_candidates(row, t, d), realized)
    return tree_accept(block_candidates(row, t, d), d.budget, realized)


# ------------------------------------------------------------------ walk + calibers

def walk(row: dict, d: Design) -> list[int]:
    """Per-round committed counts (accepted + bonus, clipped at the stream end)."""
    tokens = row["tokens"]
    n_gen = len(tokens)
    K = max(row["top_ids"].shape[1], d.copy_long)   # offsets per block (15; 31 for B32 heads)
    out = []
    t = 0
    while t < n_gen - 1:
        realized = [int(x) for x in tokens[t + 1:t + 1 + K]]
        L = accept_length(row, t, d, realized)
        emitted = min(L + 1, n_gen - 1 - t)
        out.append(emitted)
        t += emitted
    return out


def records(acc: list[int], onset: int | None, eos: bool) -> CaliberStream:
    """Same rule as eval_b3.spec_records: a round counts toward `trunc` when the first
    token it commits (generated position t+1 for a round starting at t) lies before the onset."""
    full = StreamRecord(steps=len(acc), emitted=sum(acc))
    if onset is None:
        return CaliberStream(full=full, eos=eos, onset=None, trunc=full, loop=None)
    st = et = sl = el = 0
    t = 0
    for a in acc:
        s = t + 1
        if s < onset:
            st += 1
            et += max(0, min(s + a, onset) - s)
        else:
            sl += 1
            el += a
        t += a
    trunc = StreamRecord(st, et) if onset > MIN_HEALTHY_ONSET else None
    loop = StreamRecord(sl, el) if sl else None
    return CaliberStream(full=full, eos=eos, onset=onset, trunc=trunc, loop=loop)


def load_row(path: Path) -> dict:
    z = np.load(path)
    row = {k: z[k] for k in z.files}
    for k in ("gidx", "ref_onset", "n_prompt"):
        row[k] = int(row[k])
    row["ref_eos"] = bool(row["ref_eos"])
    for k in ("mode", "subtype", "set"):
        row[k] = str(row[k])
    row["onset"] = None if row["ref_onset"] < 0 else row["ref_onset"]
    return row


def _sim_file(args):
    path, designs = args
    row = load_row(path)
    res = {"gidx": row["gidx"], "mode": row["mode"], "subtype": row["subtype"], "set": row["set"],
           "onset": row["onset"], "eos": row["ref_eos"], "acc": {}}
    for d in designs:
        res["acc"][d.name] = walk(row, d)
    if any(d.name == "D0-chain" for d in designs):
        res["ref_acc"] = [int(x) for x in row["ref_acceptance"]]
    return res


def summarize(results: list[dict], names: list[str]) -> dict:
    """design -> group -> caliber table, groups: mode, mode/subtype, mode/set."""
    out = {}
    for name in names:
        groups = defaultdict(list)
        for r in results:
            cs = records(r["acc"][name], r["onset"], r["eos"])
            for g in (r["mode"], f"{r['mode']}/{r['subtype']}", f"{r['mode']}/{r['set']}"):
                groups[g].append(cs)
        out[name] = {g: caliber_table(v) for g, v in groups.items()}
    return out


# ------------------------------------------------------------------ coverage analysis

def coverage_rows(row: dict) -> list[dict]:
    """At every chain rejection: is the realized token covered by each candidate source?"""
    tokens = row["tokens"]
    onset = row["onset"]
    out = []
    t = 0
    n_gen = len(tokens)
    chain = Design("D0-chain", 15)
    K = row["top_ids"].shape[1]
    while t < n_gen - 1:
        realized = [int(x) for x in tokens[t + 1:t + 1 + K]]
        k = int(row["n_off"][t])
        L = accept_length(row, t, chain, realized)
        if L < min(k, len(realized)):
            j = L
            y = realized[j]
            d1 = int(row["top_ids"][t, j, 0])
            cb = int(row["coord_best"][t, j])
            rec = {"pos": t + 1 + j, "healthy": onset is None or t + 1 + j < onset,
                   "y_coord": is_coord(y), "d_coord": is_coord(d1), "depth": j,
                   "p1": round(math.exp(float(row["top_lp"][t, j, 0])), 4),
                   "rank": int(row["real_rank"][t, j]), "subtype": row["subtype"],
                   "mode": row["mode"], "set": row["set"]}
            if is_coord(y):
                yx, yy = (y - COORD_START) % GRID, (y - COORD_START) // GRID
                bx, by = (cb - COORD_START) % GRID, (cb - COORD_START) // GRID
                rec["cheb_best"] = max(abs(yx - bx), abs(yy - by))
                rec["cross"] = abs(yx - bx) + abs(yy - by) <= 1
                for a in (2, 3, 4, 8):
                    rec[f"xy{a}"] = (yx in row["xy_x"][t, j][:a].tolist()
                                     and yy in row["xy_y"][t, j][:a].tolist())
                rec["copy"] = y in row["copy_ids"][t].tolist()
            out.append(rec)
        t += min(L + 1, n_gen - 1 - t)
    return out


def _cov_file(path):
    return coverage_rows(load_row(path))


def coverage_table(recs: list[dict]) -> dict:
    """Rescue rates at chain rejections where the realized token is a coordinate, loop-free part."""
    sel = [r for r in recs if r["healthy"] and r["y_coord"]]
    n = len(sel)
    if not n:
        return {"n": 0}

    def rate(f):
        return round(sum(1 for r in sel if f(r)) / n, 4)

    tab = {"n": n, "n_all_rejections": len([r for r in recs if r["healthy"]]),
           "frac_coord_rejections": round(n / max(1, len([r for r in recs if r["healthy"]])), 4)}
    for b in (2, 4, 8, 16, 32):
        tab[f"top{b}"] = rate(lambda r, b=b: 0 <= r["rank"] < b)
    for rr in (1, 2, 3):
        tab[f"nb{rr}"] = rate(lambda r, rr=rr: r["cheb_best"] <= rr)
    tab["cross"] = rate(lambda r: r["cross"])
    for a in (2, 3, 4, 8):
        tab[f"xy{a}"] = rate(lambda r, a=a: r[f"xy{a}"])
    tab["copy"] = rate(lambda r: r["copy"])
    for b in (4, 8, 16):
        for rr in (1, 2):
            tab[f"top{b}|nb{rr}"] = rate(lambda r, b=b, rr=rr: 0 <= r["rank"] < b or r["cheb_best"] <= rr)
            tab[f"nb{rr}-not-top{b}"] = rate(lambda r, b=b, rr=rr: r["cheb_best"] <= rr and not 0 <= r["rank"] < b)
        tab[f"copy-not-top{b}"] = rate(lambda r, b=b: r["copy"] and not 0 <= r["rank"] < b)
        tab[f"xy3-not-top{b}"] = rate(lambda r, b=b: r["xy3"] and not 0 <= r["rank"] < b)
    return tab


# ------------------------------------------------------------------ main

def list_cache(cache: Path, modes: list[str], sets: list[str] | None) -> list[Path]:
    files = []
    for m in modes:
        files += sorted((cache / m).glob("[0-9][0-9][0-9][0-9].npz"))
    if sets:
        keep = []
        for f in files:
            z = np.load(f)
            if str(z["set"]) in sets:
                keep.append(f)
        files = keep
    return files


def pick_designs(spec: str | None) -> list[Design]:
    pool = {}
    for d in (default_designs() + calib_designs() + grammar_designs() + extra_designs()
              + geo_designs() + copylong_designs()):
        pool.setdefault(d.name, d)           # the sets share D0 / D1 designs: keep one each
    pool = list(pool.values())
    if spec in (None, "default"):
        return default_designs()
    if spec == "calib":
        return [Design("D0-chain", 15)] + calib_designs()
    if spec == "grammar":
        return [Design("D0-chain", 15)] + grammar_designs()
    if spec == "geo":
        return geo_designs()
    if spec == "copylong":
        return copylong_designs()
    if spec == "extra":
        return [Design("D0-chain", 15), Design("D1-top16-N63", 63, ("top16",)),
                Design("D1-top16-N127", 127, ("top16",))] + extra_designs()
    if spec == "all":
        return pool
    names = set(spec.split(","))
    return [d for d in pool if d.name in names]


def print_table(summ: dict, designs: list[Design], modes: list[str]):
    for mode in modes:
        print(f"== {mode}: tau_trunc per-stream  all | extra (select) | bench (report)  "
              f"x chain(all) | tau_all")
        base = summ["D0-chain"][mode]["calibers"]["trunc"]["per_stream"]
        for d in designs:
            g = summ[d.name]
            c = g[mode]["calibers"]
            ex = g.get(f"{mode}/extra", {}).get("calibers", {}).get("trunc", {}).get("per_stream", float("nan"))
            be = g.get(f"{mode}/bench", {}).get("calibers", {}).get("trunc", {}).get("per_stream", float("nan"))
            print(f"  {d.name:<30} {c['trunc']['per_stream']:6.3f} | {ex:6.3f} | {be:6.3f}  "
                  f"x{c['trunc']['per_stream'] / base:5.3f} | {c['all']['per_stream']:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--modes", default="greedy,sample")
    ap.add_argument("--sets", default=None, help="restrict to bench / extra rows")
    ap.add_argument("--check", action="store_true", help="chain replay vs the reference run")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--fit-calib", type=Path, default=None,
                    help="D7: fit hit rates on the selected rows (use --sets extra), write JSON")
    ap.add_argument("--calib", type=Path, default=None, help="D7 calibration JSON (for calib designs)")
    ap.add_argument("--designs", default=None, help="default | calib | all | comma-separated names")
    ap.add_argument("--fit-geo", type=Path, default=None,
                    help="G: fit rule hit rates per (rule, top-1 conf, draft rank) on the selected rows "
                         "(use --sets extra), write JSON")
    ap.add_argument("--geo-calib", type=Path, default=None, help="G: calibrated rule scores (JSON)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    modes = args.modes.split(",")
    files = list_cache(args.cache, modes, args.sets.split(",") if args.sets else None)
    print(f"{len(files)} cached rows under {args.cache}", flush=True)

    if args.fit_calib:
        num = np.zeros((N_P1, N_FEAT))
        den = np.zeros((N_P1, N_FEAT))
        with ProcessPoolExecutor(args.workers) as ex:
            for n_, d_ in ex.map(_fit_file, files, chunksize=4):
                num += n_
                den += d_
        args.fit_calib.parent.mkdir(parents=True, exist_ok=True)
        args.fit_calib.write_text(json.dumps({
            "cache": str(args.cache), "modes": modes, "sets": args.sets, "n_rows": len(files),
            "p1_edges": P1_EDGES, "features": {"top": [F_TOP, 32], "nb": [F_NB, 98],
                                                "xy": [F_XY, 64], "copy": [F_COPY, 16]},
            "num": num.tolist(), "den": den.tolist()}))
        rate = (num + 0.5) / (den + 1)
        print("hit rate, top-1 / top-2 / nb centre(coord top-1) / nb +-1x / copy-0, per p1 bucket:")
        for b in range(N_P1):
            c = F_NB + 49 + 24
            print(f"  p1 [{P1_EDGES[b]:.2f},{P1_EDGES[b+1]:.2f}): n={int(den[b, 0])} "
                  f"{rate[b, 0]:.3f} {rate[b, 1]:.3f} {rate[b, c]:.3f} {rate[b, c + 1]:.3f} "
                  f"{rate[b, F_COPY]:.3f}")
        print("wrote", args.fit_calib)
        return

    if args.fit_geo:
        tot_n, tot_d = {}, {}
        with ProcessPoolExecutor(args.workers) as ex:
            for c in ex.map(_fit_geo_file, files, chunksize=2):
                for r in c["num"]:
                    tot_n[r] = tot_n.get(r, 0) + np.asarray(c["num"][r])
                    tot_d[r] = tot_d.get(r, 0) + np.asarray(c["den"][r])
        args.fit_geo.parent.mkdir(parents=True, exist_ok=True)
        args.fit_geo.write_text(json.dumps({
            "cache": str(args.cache), "modes": modes, "sets": args.sets, "n_rows": len(files),
            "p1_edges": P1_EDGES, "rank_edges": R_EDGES,
            "num": {r: v.tolist() for r, v in tot_n.items()},
            "den": {r: v.tolist() for r, v in tot_d.items()}}))
        print("rule hit rate by draft top-1 confidence (rows) x candidate's draft rank "
              f"(cols {list(R_EDGES)} / absent):")
        for r in tot_n:
            rate = (tot_n[r] + 0.5) / (tot_d[r] + 1)
            print(f"  {r}: n={int(tot_d[r].sum())} hits={int(tot_n[r].sum())}")
            for b in range(N_P1):
                print(f"    p1 [{P1_EDGES[b]:.2f},{P1_EDGES[b+1]:.2f}) " + " ".join(
                    f"{rate[b, c]:.3f}({int(tot_d[r][b, c])})" for c in range(N_RB)))
        print("wrote", args.fit_geo)
        return

    if args.check:
        d0 = [Design("D0-chain", 15)]
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_sim_file, [(f, d0) for f in files], chunksize=4))
        same = sum(r["acc"]["D0-chain"] == r["ref_acc"] for r in res)
        by_mode = defaultdict(lambda: [0, 0])
        for r in res:
            by_mode[r["mode"]][0] += 1
            by_mode[r["mode"]][1] += r["acc"]["D0-chain"] == r["ref_acc"]
        sim = summarize(res, ["D0-chain"])["D0-chain"]
        ref_tabs = defaultdict(list)
        for r in res:
            ref_tabs[r["mode"]].append(records(r["ref_acc"], r["onset"], r["eos"]))
        print(f"chain replay: {same}/{len(res)} rows with identical per-round acceptance "
              f"({dict(by_mode)})")
        for mode, recs in ref_tabs.items():
            ref = caliber_table(recs)["calibers"]
            s_ = sim[mode]["calibers"]
            print(f"  {mode}: tau_trunc sim {s_['trunc']['per_stream']:.3f} vs ref "
                  f"{ref['trunc']['per_stream']:.3f}; tau_all sim {s_['all']['per_stream']:.3f} "
                  f"vs ref {ref['all']['per_stream']:.3f}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            mism = [{"gidx": r["gidx"], "mode": r["mode"], "sim": r["acc"]["D0-chain"],
                     "ref": r["ref_acc"]} for r in res if r["acc"]["D0-chain"] != r["ref_acc"]]
            (args.out / "check.json").write_text(json.dumps(
                {"rows": len(res), "identical": same, "by_mode": dict(by_mode),
                 "mismatches": mism}, indent=1))
        return

    if args.coverage:
        with ProcessPoolExecutor(args.workers) as ex:
            recs = [x for part in ex.map(_cov_file, files, chunksize=4) for x in part]
        tabs = {}
        for mode in modes:
            for sub in ("all", "icon", "illustration"):
                sel = [r for r in recs if r["mode"] == mode and (sub == "all" or r["subtype"] == sub)]
                tabs[f"{mode}/{sub}"] = coverage_table(sel)
        print(json.dumps(tabs, indent=1))
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "coverage.json").write_text(json.dumps(tabs, indent=1))
            with (args.out / "coverage_rows.jsonl").open("w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        return

    designs = pick_designs(args.designs)
    if any(d.score == "calib" for d in designs) and args.calib is None:
        ap.error("calibrated designs need --calib (fit it with --fit-calib on --sets extra)")
    pri = None
    if any(d.geo for d in designs):
        fit_files = list_cache(args.cache, modes, ["extra"])
        pri = geo.fit_priors(np.load(f)["tokens"] for f in fit_files)
        print(f"geo priors (hit rates fit on {len(fit_files)} extra rows): " + ", ".join(
            f"{k} {math.exp(v):.4f}" for k, v in pri.items() if not k.startswith("_")), flush=True)
    print(f"{len(designs)} designs", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(args.calib, pri, args.geo_calib)) as ex:
        res = list(ex.map(_sim_file, [(f, designs) for f in files], chunksize=2))
    summ = summarize(res, [d.name for d in designs])
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summ, indent=1))
        if pri is not None:
            (args.out / "geo_priors.json").write_text(json.dumps(pri, indent=1))
        with (args.out / "per_row.jsonl").open("w") as f:
            for r in res:
                f.write(json.dumps({k: r[k] for k in ("gidx", "mode", "subtype", "set", "onset",
                                                      "eos", "acc")}) + "\n")
        (args.out / "designs.json").write_text(json.dumps(
            [{"name": d.name, "budget": d.budget, "sources": d.sources, "when": d.when,
              "first": d.first, "branch_by_conf": d.branch_by_conf, "score": d.score, "geo": d.geo,
              "copy_long": d.copy_long}
             for d in designs], indent=1))
    print_table(summ, designs, modes)


if __name__ == "__main__":
    main()
