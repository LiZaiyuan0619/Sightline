# Draft-head weights

The weights are not in this repository. One head is about 2.3 GB — almost all of it the embedding
and output projection over a 197,000-token SVG vocabulary. We will release the heads of the paper on
Hugging Face. Until then, this file says which heads the paper reports and how to train each of them
from the configurations included here.

## Which heads the paper reports

`data/heads/heads_in_paper.json` lists all 61, each with the table and block it appears in and the
hyper-parameters the trainer recorded. They were collected from the result directories the paper's
tables read, not from the training directories, so heads that were trained and then dropped are
excluded by construction.

| decoder | heads | what the head is |
|---|---|---|
| Ours | 18 | a block draft head at **block length 32**, DFlash's architecture and recipe |
| DFlash, DDTree | 18 | the same architecture at **block length 16**; both decoders read this one head, differing only in chain against tree |
| EAGLE-3 | 11 | an autoregressive draft head, trained with SpecForge's EAGLE-3 objective |
| DARTree | 11 | a block head plus the autoregressive correction head of its recipe |
| ViSpec | 3 | a vision-conditioned head with the image-compression adaptor (image-to-SVG blocks only) |
| SpS | 1 | no training: the smaller released model of the same family |

44 of the 61 carry their recorded hyper-parameters; the remaining 17 were trained before the
trainer began writing them to disk, and their settings are the defaults of
`train/train_block_head.py` with the configuration named in the same record.

## Training one

```bash
# 1. the target generates its own training data (one sample per prompt, official decoding)
python3 train/data/generate_omnisvg_text.py --class icon --units 0-19 --out $SIGHTLINE_OUT/gen

# 2. build the training set: loops are truncated two periods after onset
python3 train/data/build_trainset_omnisvg.py --data $SIGHTLINE_OUT/gen --out train.jsonl

# 3. train, at block length 32
python3 train/train_block_head.py --task t2svg --data train.jsonl \
    --draft-cfg configs/heads/omnisvg-8b-dflash-draft-b32.json \
    --epochs 6 --out $SIGHTLINE_OUT/head
```

The block-length-16 head of the same target is the same command with `-b16.json`: the two
configurations differ in `block_size` and in nothing else, which is what makes the paper's
comparison a comparison of windows. `configs/heads/` holds 24 such configurations, paired for
every target.

Image-to-SVG heads start from the text-to-SVG head of the same target (`--init <head>`), and the
InternSVG-8B text heads continue a first round of training; both are recorded per head in
`data/heads/heads_in_paper.json`.

## Picking the checkpoint

Training saves every epoch. The paper keeps, for every decoder including ours, the checkpoint with
the highest tokens per round on held-out MMSVG-2M prompts — not the lowest training loss, which is
a different checkpoint often enough to matter.

```bash
python3 train/select_head.py --run $SIGHTLINE_OUT/head --rows <held-out rows> --design D0-chain
```

## Label smoothing on coordinates

`train/patches/specforge-label-smoothing.patch` adds an option that smooths the target
distribution over nearby coordinate tokens. It is **off** in every head the paper reports
(`sls_eps = 0`, visible in the recorded hyper-parameters). The patch is included because it was
present in the trainer that produced the heads.
