# Text-motion alignment

This experiment compares normalized raw MotionJEPA motion frames with frozen JEPA
tokens. Both branches learn independent text and motion alignment Transformers
with symmetric multi-positive InfoNCE. It implements the contrastive part of TMR;
there is no motion reconstruction decoder or VAE loss.

## Environment

Run preparation, training, evaluation, and tests in the `motion-jepa` conda environment:

```bash
conda activate motion-jepa
python -m pip install -r experiment/tmr/requirements.txt
```

The requirements add Transformers 5.1 and its dependencies, including a compatible
Hugging Face Hub version below 2. Existing PyTorch/CUDA packages remain in place.

## Model

The frozen text backbone is the first-generation, text-only pretrained
[`google/t5gemma-2b-2b-ul2`](https://huggingface.co/google/t5gemma-2b-2b-ul2).
`T5GemmaEncoderModel` loads with `is_encoder_decoder=False`; no decoder is
constructed. Standard Hugging Face loading can still download all weight shards
of the published encoder-decoder checkpoint. Accept the Gemma model license and
authenticate with `hf auth login`, or provide an already downloaded local model
directory using `--text-model`.

Preparation must use the same `HF_HOME` as your login. On this server the
authenticated cache is `/data/seokhyeon/hf-cache`, so set
`export HF_HOME=/data/seokhyeon/hf-cache` before running the commands below.

The complete valid encoder token sequence is retained. Each trainable readout
projects its input to 256 dimensions, prepends its own learnable CLS token, adds
sinusoidal positions, and applies six pre-norm Transformer blocks (four heads,
1024-wide FFN, dropout 0.1). The final LayerNorm CLS output is L2-normalized. CLS is
added after the frozen text encoder, not to its tokenizer input.

Raw motion uses `[150,366]` frame features. JEPA uses the checkpoint's temporal
tokens, e.g. `[50,384]` for the current 3-frame patch encoder. For 2D checkpoints,
the spatial axis is averaged and time is retained. `target_encoder` is the default
checkpoint key; `--checkpoint-key encoder` is also supported. Valid-frame masks
are converted using the checkpoint token layout, excluding incomplete temporal
patches. An input with zero valid JEPA tokens is rejected.

## BONES captions and caches

Preparation joins the existing processed BONES index to temporal annotation
filenames. Event boundaries use `floor(seconds * fps)` and half-open intervals.
Every event with at least one overlapping frame contributes a separate caption
candidate, ordered by start/end time and original annotation order. Whitespace is
normalized and identical descriptions are deduplicated across the clip, so repeats
do not increase their sampling weight. Descriptions are never concatenated.
Motion windows and their source-level splits stay unchanged. No-overlap clips are
excluded, without falling back to whole-motion captions.

For the current dataset this retains 191,037 train, 6,605 validation, and 13,067
test clips. Boundary event fragments are included, so a candidate can describe a
partially observed action. Candidates come from overlapping temporal events;
they may describe successive actions, rather than equivalent paraphrases.

Text features are extracted once per distinct candidate, using at most 256 tokens.
Each training access samples one candidate uniformly and loads only its token
sequence. PyTorch/worker seeds make this sampling reproducible across an
epoch-boundary resume. Validation and test have no caption sampling.
Preparation records truncations. Features use lazy BF16-bit `uint16` NumPy memory
maps: packed valid text and JEPA tokens plus sequence offsets/lengths. Training
reads only the requested rows and dynamically pads
text batches. It does not instantiate either frozen backbone.

Metadata records dataset/annotation/catalog hashes, text model/tokenizer revision,
token limit, JEPA checkpoint/key/layout, and normalization statistics. Incomplete
or stale caches fail explicitly; rebuild with `--recompute-features`. Keep the
same `--max-samples-per-split` for all preparation runs sharing a cache root.
Format 2 stores individual candidates. Legacy format-1 caches contain joined-text
features and cannot train with candidate sampling: re-run preparation for both
branches with `--recompute-features`, or use a new cache root. Old checkpoints and
their retrieval scores use the previous caption policy and should remain separate.

## Prepare and compare

Run from the repository root with `motion-jepa` activated. Use the same paired
cache, normalization statistics, seed, and head/training settings for both branches:

```bash
TMR_JEPA_CHECKPOINT=output/mot_patch_base_1d-p3-bs.512-ep.300-nframes150-segmentation-20260930/motion-jepa-patch-1d-p3-ep300.pth.tar
TMR_STATS=dataset/bones-seed-processed-nframes150/stats
TMR_CACHE=output/tmr/cache

python -m experiment.tmr.prepare_cache \
  --input-source raw --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE"

python -m experiment.tmr.prepare_cache \
  --input-source jepa --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE"

python -m experiment.tmr \
  --input-source raw --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --output-root output/tmr/raw/seed-42

python -m experiment.tmr \
  --input-source jepa --jepa-checkpoint "$TMR_JEPA_CHECKPOINT" \
  --stats-path "$TMR_STATS" --cache-root "$TMR_CACHE" \
  --output-root output/tmr/jepa/seed-42
```

The default dataset is `dataset/bones-seed-processed-nframes150`, and annotations
are `dataset/bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl`.
`--dataset-root`, `--annotations-path`, `--text-model`, `--text-revision`, and
`--max-text-length` must agree between preparation and training. For JEPA,
normalization defaults to the checkpoint's pretraining statistics; for raw it
defaults to the processed dataset's statistics. Pass the same `--stats-path`
explicitly for a controlled comparison.

For a small smoke run, prepare **both** branches into a separate cache root with
`--max-samples-per-split 8`, then train with `--epochs 2 --warmup-epochs 0
--batch-size 4 --num-workers 0`. A smoke cache is identified in its manifest and
cannot silently be mistaken for the full dataset. It still needs text model
access, unless a local model is supplied.

## Visualize cached motion-caption pairs

The viewer reads an existing `raw/paired-index.json` or `jepa/paired-index.json`
and its `prepared.json`. It displays the individual candidate captions next to
the original motion, with the source interval and clip duration. Selecting another
motion updates both together. Legacy caches still display their joined caption,
marked as a combined caption in the motion information. Playback, frame stepping,
speed, skeleton, mesh, and foot-contact controls are shared with the root viewer.

```bash
python -m experiment.tmr.visualize_dataset \
  --cache-root "$TMR_CACHE" --input-source raw --split train --port 6006
```

Use `--input-source jepa` to browse the JEPA cache's pairs; these also render the
associated original raw motion because JEPA tokens cannot be decoded to joints.
The viewer does not prepare data, extract text/JEPA features, load backbones, or
require Hugging Face authentication. It needs the original motion files referenced
by the cache. Use `--dataset-root` if that dataset has moved, `--mesh` for the
skinned body, `--limit 0` for all pairs, and `--sample-id` or `--sample-index` for
the initial selection. `python experiment/tmr/visualize_dataset.py` is also supported.

## Training and retrieval

Defaults: AdamW, LR `1e-4`, final LR `1e-6`, weight decay `0.01`, 100 epochs,
five-epoch warmup and cosine decay, batch 256, gradient clipping 1, temperature
0.1, seed 42, and CUDA BF16 autocast. Each motion is paired with one randomly
selected candidate. A sampled caption is positive for every motion that lists it
as a candidate, even if that motion sampled a different caption. Different-caption
clips from the same source with overlapping time intervals are excluded from
negatives. Semantically similar but
different captions remain negatives; no unvalidated text-similarity threshold is
used. A final singleton training batch is skipped because it has no negatives.

Validation and test use complete galleries, never minibatch retrieval. Text queries
and candidates are all unique candidate captions; motion queries and candidates
are all retained clips. Text-to-motion accepts any clip listing the query caption;
motion-to-text accepts any of the clip's candidates. Reported recalls
are fractions, ranks start at one, and score ties use ascending candidate index.
Source-overlap exclusions apply only during training. Similarity computation tiles
both axes to bound memory. Best-model selection uses the average of text-to-motion
and motion-to-text validation R@1; test is evaluated with that selected model.

The output directory contains `config.json`, `provenance.json`, `runtime.json`,
`metrics.csv`, `best.pth.tar`, `latest.pth.tar`, and `summary.json`. Checkpoints
contain alignment heads, optimizer/scheduler, and RNG/loader states, without
backbone weights. Resume an interrupted run by repeating its original command
with `--resume`; model, optimization (including planned epochs), and feature
provenance must match. A new experiment needs a new output directory.

```bash
python -m experiment.tmr.evaluate \
  --checkpoint output/tmr/jepa/seed-42/best.pth.tar --split test \
  --output output/tmr/jepa/seed-42/test-retrieval.json

python -m unittest discover -s tests -p 'test_tmr*.py'
```

Tests use a small local text encoder/tokenizer and synthetic motion/checkpoints,
without Hugging Face downloads. A real pretrained 2B inference check additionally
requires access to the gated model. Cache preparation reports an actionable error
if that access is unavailable.
