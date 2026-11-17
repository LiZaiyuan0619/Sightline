# Data of the hero figure for all five model-task pairs

Data only, no GPU. Figure 1 (`paper/claims/H2-hero/round5`) reads the main results directly; its
round 4 drew the examples picked here.

| file | what it is |
|---|---|
| `hero5.py` -> `hero5.json`, `hero5.md` | the data layer: five model-task pairs x two decodings x six benchmarks x eight decoders, on the clock of the main results. `hero5.py` takes the rows from the builder of the main results and needs the full run outputs, which are not included |
| `hero/cases.json` | one example prompt per pair, picked by the rule below |
| `hero/frames/`, `hero/finals/` | the metadata of the rendered frames and drawings of those examples (the images are not included) |
| `preview-<pair>/`, `same-clock-<pair>/` | on the first 40 MMSVG-Icon prompts under greedy decoding, how far the original model, DDTree and ours have drawn over time, and how many drawings each has finished by a given time |

**One clock per pair.** The reference is the original model that the main results pair with
our decoder. Decoders timed in other jobs (SpS, DARTree, REST) are put on this clock by the
ratio of the two original models' seconds per token over the same rows, which leaves their
speedups unchanged. `hero5.py` asserts that all 432 cells of the main results, recomputed from
the captured rows, equal `runs/2026-11-02/figdata/table1.json` to within 1e-9.

**The example rule**, applied to each pair:
1. the original model ends normally;
2. the original model, DDTree and ours write the same tokens;
3. of those, the ten prompts whose two speed ratios (original model over ours, DDTree over ours)
   lie closest to their medians, by the sum of the log deviations;
4. the longest of these ten.
