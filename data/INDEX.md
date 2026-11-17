# Data index: the floats of the paper, and the files behind them

The per-setting results are kept under `runs/` in the directory layout the scripts expect, so the
table builders, the analysis scripts and the figure scripts find them unchanged. This table maps
each of them to the paper.

## Floats of the paper

| paper | file | what is in it |
|---|---|---|
| Figures 1 and 4, Appendix F.1 (Tables 14 and 15) | `runs/2026-11-02/figdata/table1.json` | the main results: 10 blocks (5 model-task pairs x 2 decodings) x 9 decoders x 6 benchmark sets, each with speedup, tokens per round and the number of prompts. All tokens counted |
| Appendix F.1 (Table 16) | `runs/2026-11-05/appendix/healthy_tau.json` | the main results counted on loop-free content: tokens per round of the rounds before each output's loop onset |
| Table 1, Appendix F.2 (Tables 18 and 19) | `runs/2026-11-09/table-fill/figdata/table2_v2.json` | the quality table on the full SArena test sets: the lossless, loop-control and best-of-8 groups, with seconds and the quality metrics |
| Figure 2, Appendix B | `examples/tree-gidx150.json` | one real round: the 127-node draft tree, which nodes were accepted |
| Figure 3, Appendix D (Tables 2 to 7) | `runs/2026-11-05/theory/` | the theory measurements: run lengths between misses, the decision horizon, the natural-language control, the loop ceiling, the registered predictions |
| Figure 5a, Appendix F.3 (Table 20) | `runs/2026-11-03/table2_refrows.json` | the other SVG generators, the rows InternSVG reports and the rows we measured |
| Figure 5b, Appendix F.4 (Tables 21 to 23) | `runs/2026-11-04/ext_v4.json` | the generality experiment: six models from four other families, three output formats |
| Figures 5d and 6a, Appendix G.2 (Figure 19) | `runs/2026-11-06/chores/b8/figdata/m2_healthy.json` | tokens per round on loop-free content, per decoder and round, for the mechanism panels |
| Figure 6a | `runs/2026-11-02/caliber/healthy_vs_all.md` | tokens per round under both calibers, per decoder |
| Figures 6c to 6f, Appendix H (Figure 20) | `runs/2026-11-04/figdata_b4/` | training and design: the recipe as a 2 x 2 (`d1_ladder.json`), the block length (`d2_block.json`), the shape of the tree (`d3_tree.json`) |
| Figures 6g to 6j, Appendix J.1 (Figure 22) | `runs/2026-11-04/figdata_b3/` | the loop-control rules (`l_rules.json`) and the best of 8 on 300 prompts (`q1_frontier.json`) |
| Figure 6l, Appendix K.1 (Tables 33 and 34), Figure 23b | `runs/2026-11-04/b2w/b2w.json` | the serving experiment: window against concurrency in SGLang |
| Appendix F.5 (Table 24) | `audits/fp32_audits.json` | every fp32 losslessness audit: prompts checked, outputs identical to the original model |
| Appendix F.5 (Table 25) | `runs/2026-11-15/bf16/bf16_identity_main.md` | token identity in bf16: per decoder and model-task pair, the share of greedy outputs whose tokens equal those of the original model in the same job. Computed from the per-output records of the runs, which are not included |
| Appendix E.2 (Table 9) | `../eval/drafters.py` | no data: the decoders compared, one design name each |
| Appendix E.4 (Table 11) | `node_factors.json` | the per-job node factors behind the timing |

Run outputs keep the figure and table numbers of the day they were written (for example,
`runs/2026-11-04/ext_tables_v4.md` calls the generality table "Table 3"). The table above gives
the numbers of the paper.

## Inputs

| file | what it is |
|---|---|
| `prompts/*.jsonl` | the prompt identifiers of every benchmark set used, as `{gidx, set, subtype, id}`. Benchmark content is not included: these identify the exact selection against your own copy of MMSVG, MMSVG-2M or SArena |
| `descriptions/mmsvg_bench_image_descriptions.jsonl` | the 300 one-sentence descriptions of the MMSVG image-to-SVG inputs, written by a DeepSeek model at temperature 0 (Appendix E.1). MMSVG ships those images without text and the InternSVG image-to-SVG prompt needs one |
| `heads/heads_in_paper.json` | the 61 draft heads the paper reports, each with the blocks it serves and its recorded hyper-parameters (Appendix E.3, Table 10) |

## Calibers

Two ways of counting appear in the paper and both are here. The main results count **all tokens**,
loop segments included. The mechanism and ablation figures count **loop-free content**: the rounds
that start before the output enters a loop. `analysis/calibers.py` implements both; a file's `meta`
block says which one it holds.

## Provenance

The `meta` block of each file names the script that produced it and the protocol it followed.
Absolute paths have been removed; what remains ("table-fill-1109", "r1105/cache-sample/b32-6ep")
is the run directory's own name.
