# Contents

462 files. Where each piece of the paper lives, and what a reader can run without a GPU.

## Runs on a plain CPU, in seconds

| command | what it shows |
|---|---|
| `python3 paper/check_paper_numbers.py` | recomputes 13 headline claims from `data/` and checks each against the paper |
| `python3 paper/appendix_tables_main.py` | regenerates the numbers of 8 appendix tables as LaTeX from `data/` (the paper's captions were edited afterwards) |
| `python3 tests/test_verify.py` and 7 more | the unit tests of the verifier, the logit chain, the tree, the simulator |
| `python3 paper/claims/T4-quality-table/round3/make.py` | regenerates Table 1, with a third group (best of 8) that the paper moved to Appendix F.2 |

With `matplotlib` available, four of the paper's floats redraw from `data/`:
Figure 1 (`paper/claims/H2-hero/round5`), Figure 2 (`MF-method/round4`), Figure 4
(`T1-main-table/round6`) and Table 1 (`T4-quality-table/round3`). The three composite figures
(3, 5 and 6) are here with their data layers, but each aggregates several analyses and expects
the full run outputs under `$SIGHTLINE_RUNS`; what ships is the subset the other floats need.
The last section of this file maps every figure script to the paper.

## The method

| file | what |
|---|---|
| `sightline/verify.py` | block verification: which drafted tokens are kept, and the proof obligation that makes it lossless |
| `sightline/process.py` | the logit chain (all of it in fp32, for the reason in the file) and loop control |
| `sightline/tree_spec.py` | one ancestor-masked target pass verifies a whole candidate tree |
| `sightline/design.py` | tree designs, candidate construction, budget growth, and the offline simulator |
| `sightline/spec.py` | the block draft head: one forward pass fills the window |
| `sightline/batch_spec.py` | N samples sharing one prompt, for the best-of-N tier |
| `sightline/cache_draft.py` | cache a head's block distributions once, then study designs on CPU |
| `sightline/svgtok/` | the SVG token table (197k ids) and the loop detector |
| `sightline/tree_utils.py` | four tree utilities, included verbatim from GLANCE under its MIT licence |

## Everything else

| directory | what |
|---|---|
| `baselines/` | our implementation of PLD, REST, SpS, EAGLE-3, ViSpec, DARTree |
| `eval/drafters.py` | **one table** mapping a design name to a decoder: the eight baselines and ours |
| `eval/eval_text.py`, `eval/eval_image_*.py` | one entry point per target family (merged from 19 near-copies) |
| `eval/best_of_n.py` | the best-of-N tier, all four tasks |
| `eval/records.py`, `merge.py`, `audit_fp32.py` | the result schema, the shard merge, the losslessness audit |
| `eval/metrics/`, `eval/rows/` | the official metrics, and how each benchmark's prompts were selected |
| `train/` | data generation, training-set construction, head training, checkpoint selection, patches |
| `targets/` | loading each target: language-side export, vision-language wrappers, per-target profiles |
| `analysis/` | the computations behind the theory and mechanism figures |
| `configs/` | the settings the paper reports: main table, loop control, best-of-N, decoding, 24 head configs |
| `data/` | per-setting results, prompt identifiers, the generated descriptions, head hyper-parameters |
| `docs/` | the draft-head weights, third-party code, how the experiments were run |
| `env/` | package lists for the seven environments, and why there are seven |

## Not included, and why

| | |
|---|---|
| draft-head weights | 2.3 GB each, mostly the embedding of a 197k vocabulary; to be released on Hugging Face, and `docs/weights.md` trains them |
| upstream repositories | named with their commit and licence in `docs/third_party.md`; our changes are patches in `train/patches/` |
| benchmark content | `data/prompts/` identifies the exact prompts against your own copy |
| generated SVGs and renderings | tens of gigabytes; the measurements taken from them are in `data/` |
| cluster job scripts | scheduler-specific; `docs/running.md` records the shape and cost of the work instead |
| explored directions the paper does not use | a geometric tolerance on coordinates, two further loop actions, a copy branch: their flags remain where they were, and `configs/` holds the settings that ran |

## Figure scripts and the paper's floats

`paper/claims/` keeps one directory per claim and one subdirectory per round of drafting. The
comments inside cite the figure and table numbers of the draft they were written for, which are
not the numbers of the paper. This table gives the paper's numbers. Where the paper
prints a later round than the one included (Figures 3 to 6), the later round changed the layout
and the labels, and Figure 5a's later round also marks the generators that InternSVG reports as
diamonds.

| paper | figure script, and the claim directories it draws from |
|---|---|
| Figure 1 | `H2-hero/round5`, data layer `H2-hero/round4/data4.py` |
| Figure 2 | `MF-method/round4`, data layer `MF-method/round1/mfdata.py` |
| Figure 3 | `TH6-theory-grid/round1`: cells from `TH1-run-length`, `TH2-horizon`, `TH3-loop-limit`, `TH4-nl-control` and `TH5-ceiling-losses` |
| Figure 4 | `T1-main-table/round6` |
| Table 1 | `T4-quality-table/round3` |
| Figure 5 | `EX1-grid/round1` and `round2`: a from `T2-landscape`, b from `T3-generality`, c from `M1-token-classes`, d from the loop-free rounds of `data/runs/2026-11-06/chores/b8/figdata/m2_healthy.json` (the claim of `M2-block-ceiling`), e from `M5-loops-common` |
| Figure 6a to f | `EX2-grid/round1`: a from `M8-lead-sources`, b from `P1-speed-identity`, c from `D5-training`, d to f from `D-group` (`D1-recipe-ladder`, `D2-block-length`, `D3-tree-shape`) |
| Figure 6g to l | `EX3-grid/round3`: g from `L-loop-control`, h and i from `Q1-quality-frontier`, j from `Q2-vlm-judge`, k from `Q3-render-selection`, l from `P5-serving` |
| Figure 7 | `H4-qualitative/round1`, the data layer and the selection of examples |
| Appendix D.2, Figure 16 | `TH1-run-length/round3` (a, b) and `TH4-nl-control/round1` (c) |
| Appendix D.3, Figure 17 | `TH4-nl-control/round1` (a), `TH3-loop-limit` (b) and `M1-token-classes/round3` (c) |
| Appendix F.1, Figure 18 | `T1-main-table/round6` |
| Appendix G.2, Figure 19 | `M2-block-ceiling/round3` |
| Appendix H.4, Figure 20 | `D3-tree-shape/round1` |
| Appendix I.2, Figure 21 | `L-loop-control/round1` |
| Appendix J.1, Figure 22 | `Q1-quality-frontier/round2` |
| Appendix J.2, Figure 23 | `Q2-vlm-judge/round1` (a), `P5-serving/round2` (b) and `P4-oneoff-cost/round2` (c) |

The scripts that set the panels of Figures 16, 17 and 23 side by side, and the layouts of the
examples in Appendix A (Figures 8 to 15), are not included.
