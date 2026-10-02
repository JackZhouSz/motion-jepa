# MotionJEPA

MotionJEPA learns motion representations by predicting target embeddings from
masked context embeddings. The repository contains a self-contained BONES-SEED
preprocessing pipeline, the `motion_jepa_366_v1` SOMA30 representation, two
MotionJEPA transformer variants, distributed pretraining, and visualization.

The code does not require the Ardy or Kimodo repositories at runtime.

## Layout

```text
dataset/          preprocessing, fixed-clip dataset, and loader
model/            frame-token and skeletal-temporal transformers
mask/             deterministic structured mask collators
motion_rep/       366-D representation and geometry
skeleton/         BVH parsing, SOMA skeletons, kinematics, and assets
visualization/    viser dataset viewer, skeleton renderer, and skinning
utils/            distributed, logging, scheduler, and tensor helpers
configs/          active pretraining config and `_depr_experiments/` archive
train.py          shared JEPA training loop
main.py           local-device and torchrun entry point
```

Public imports are intentionally top-level:

```python
from dataset import MotionDataset, make_motion_dataset
from mask import MaskCollator1D, MaskCollator2D
from model import MotionTransformer1D, MotionTransformer2D
from motion_rep import MotionJEPAMotionRep
from skeleton import SOMASkeleton30, SOMASkeleton77, parse_bvh_motion
from visualization import MotionJEPADatasetViewer, SOMASkin
```

## Installation

Python 3.10 or newer and PyTorch 2.1 or newer are recommended.

```bash
pip install -r requirements-motion.txt
```

## Dataset preprocessing

The default preprocessing command reads BONES-SEED SOMA Uniform BVH files and
writes independently canonicalized clips as individual NumPy arrays:

```bash
python dataset/preprocess_dataset.py --workers 64
```

For a quick end-to-end check, limit preprocessing to a few source motions per
split:

```bash
python dataset/preprocess_dataset.py \
  --limit 3 \
  --workers 1 \
  --output dataset/bones-seed-processed-preview
```

The default output is `dataset/bones-seed-processed`:

```text
bones-seed-processed/
├── motions/{train,val,test}/*.npy
├── motions/{train,val,test}.json
├── train.txt
├── val.txt
├── test.txt
├── index.json
├── meta.json
├── errors.jsonl
└── stats/{mean,std}.npy
```

Each motion file stores one raw `float32[length,366]` array. Split rows contain
`sample_id,relative_npy_path,fps,actual_length`. Higher source rates are reduced only by exact
fixed-step indexing. Lower or non-divisible rates are reported in
`errors.jsonl`. Complete windows use 50% overlap by
default. After the last complete window, one uncovered tail is retained when
it has at least `--min_frames` frames (90 by default).

The final output directory is created immediately, and each sequence NPY is
saved there as soon as its source motion is converted. Split manifests,
statistics, and metadata are finalized after conversion. Complete output is
reused unless `--overwrite` is supplied; interrupted partial output requires
`--overwrite` to restart.

The loader validates `meta.json`, manifests, every split row, value size, FPS, feature
dimension, finite values, and normalization statistics. Runtime resampling is
not supported. Variable-length tails are normalized over their real frames,
then end-padded with literal zeros; padding is excluded from masks, attention,
targets, and loss.

### 100STYLE preprocessing

The 100STYLE preprocessor uses the included official
[`Frame_Cuts.csv`](dataset/Frame_Cuts.csv), trims each original 60 FPS motion
with stop-exclusive `[START:STOP]` bounds, resamples the trimmed sequence to 30
FPS, and creates complete non-overlapping 90-frame windows. The default split
is content-disjoint: `BR/BW/FR/SR/SW` are training contents and all `FW`
motions form the test set. Validation is intentionally empty.

Run a small end-to-end preview with:

```bash
python dataset/preprocess_100style.py \
  --limit 10 \
  --workers 1 \
  --output dataset/100style-soma77-processed-preview
```

Omit `--limit` to process every `bvh/*_soma77.bvh` file under
`dataset/100STYLE_soma77`. Override the protocol only explicitly with
`--contents ... --test-content ...`; content names, style coverage, source
availability, and every frame-cut range are checked strictly. Metadata records
the CSV SHA256 and trim/resampling convention, so an older split protocol is
not silently reused.

Workers convert and save source motions directly, avoiding transfer of large
feature arrays back to the parent process. Training uses global per-epoch
randomization through `DistributedSampler`; each loader worker lazily opens
only the NPY files selected for its current batch. NPY is the supported
processed dataset format for training and visualization.

Visualize processed clips with:

```bash
python visualize_dataset.py dataset/bones-seed-processed --split train --mesh
```

Visualize the exact mask collator output for any raw or patchified 1D/2D
configuration with:

```bash
python visualize_mask.py \
  --config configs/mjepa_patch_2d_tiny_coarse7.yaml \
  --output output/mask-patch-2d-trajectory-coarse7.png \
  --seed 0 \
  --valid-length 90
```

The renderer automatically uses a timeline for 1D layouts and a
time-by-joint/body-group grid for 2D layouts. Patch timelines annotate both
token indices and their corresponding raw-frame spans.

For 1D multiblock masks (including 1D patches), choose how context candidates
are trimmed to the batch's minimum count:

```yaml
mask:
  context_selection: random  # prefix (default) | random
```

`prefix` keeps the earliest remaining context tokens, matching the original
implementation. `random` uniformly selects without replacement from each
sample's remaining context candidates, then sorts their original time indices.
Both modes keep the same context count and target masks for a given collator
step; this option changes only context trimming. It does not change block
sampling, target exclusion, or guarantee uniform coverage of the full timeline.
2D masks do not use this minimum-count trimming; `random` is rejected for them.

The mode is saved in mask checkpoint state for deterministic resume. Old
checkpoints without this field are interpreted as `prefix`. Exact resume
requires the same mode; switching modes requires a separate training experiment.
The base and nmask2 configs explicitly retain `prefix`; select `random` to opt in.

The active `random_spatial_segment` strategy samples four target masks. Each
mask spans a contiguous 40–60% of the patch timeline and exactly four of the
eight trajectory/body tokens. Targets may overlap, but their union is
constrained to 55–65% and the encoder receives its exact complement. Historical
mask experiments remain under `configs/_depr_experiments/`.

## Model variants

`_1d` uses one token per 366-D frame and temporal transformer attention. It
samples contiguous temporal blocks; target exclusion and context trimming can
leave a non-contiguous set of context tokens.

Patchified `_2d` routes canonical root x/z and heading to a trajectory token,
then appends the configured body groups. Coarse7 therefore produces a
`[patches,8]` grid. Root height, rotation, and velocity stay in the pelvis body
token. Separate trajectory/body patch projections and learned spatial
positions identify the streams; there is no token-type embedding. Encoder
blocks apply temporal attention per spatial token followed by spatial
attention per patch. Non-patch `_2d` retains the original 30-joint routing.

Named factories are available for `tiny`, `small`, `base`, `large`, `huge`,
and `giant`, for example `mot_base_1d` and `mot_base_2d`. There are no
unsuffixed compatibility aliases.

### Temporal RoPE for 1D models

Frame-token and temporal-patch `_1d` encoders and predictors can use rotary
position embeddings instead of additive absolute sinusoidal embeddings:

```yaml
position_encoding:
  temporal: rope  # absolute (default) | rope
  rope_theta: 100.0
  rope_time_scale: 1.0
```

RoPE rotates queries and keys in every attention layer using the original
token times. Frame positions are `frame_index / fps`; patch positions are
`(patch_index * patch_size + (patch_size - 1) / 2) / fps`. Positions are gathered
with the same masks as context and target tokens, so removing target tokens
preserves the elapsed time between visible tokens. Predictor context and target
positions use the same timeline even though their tokens are concatenated.
The EMA target encoder uses the full timeline. Padding remains excluded from
attention and loss.

`rope_time_scale` multiplies these times before rotation; its default retains
seconds. `rope_theta` controls the frequency spectrum. Both must be finite and
positive. RoPE currently supports 1D models only and does not change the
configured input length or masking policy. For a controlled V2 experiment:

```bash
python main.py --config configs/mjepa_patch_1d_base_v2_rope.yaml --devices cuda:0
```

Omitting `position_encoding` retains the existing absolute behavior, including
old checkpoints. Exact resume requires the same positional mode, theta, and
time scale; use a separate training run when changing them. Frozen downstream
encoders reconstruct these settings from the saved pretraining configuration.

## Training

Batch size is per rank. Learning rates are not automatically scaled.

Single GPU:

```bash
python main.py --config configs/mjepa_patch_2d_tiny_coarse7.yaml --devices cuda:0
```

Local multi-GPU:

```bash
python main.py \
  --config configs/mjepa_patch_2d_tiny_coarse7.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`main.py` spawns one process per configured device and propagates child
failures. `--debug` forces one process on the first device.

Standard torchrun launch:

```bash
torchrun --standalone --nproc-per-node=4 main.py \
  --config configs/mjepa_patch_2d_tiny_coarse7.yaml
```

`main.py` honors `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. CUDA training uses
NCCL; CPU integration tests use Gloo. `main_distributed.py` remains available
for Submitit/SLURM launches.

Checkpoints contain unwrapped encoder and predictor weights, the EMA target,
optimizer, optional AMP scaler, all schedules, epoch/global step, mask state,
latest online diagnostics, and per-rank Python/NumPy/PyTorch RNG states. Writes are atomic. Exact resume
requires the same world size and continues from the latest completed epoch:

```yaml
meta:
  load_checkpoint: true
  read_checkpoint: null  # null selects <write_tag>-latest.pth.tar
```

The active config also evaluates a fixed unlabeled validation subset at the
linear-probe cadence. TensorBoard receives RankMe, feature variance, sample
cosine, covariance-spectrum, held-out JEPA, PredictionGain, and trajectory
reliance metrics under `online_metrics/`. Full summaries are appended to
`online-metrics.jsonl`; they are diagnostic and do not select checkpoints. See
`agent/ONLINE_REPRESENTATION_METRICS.md` for definitions and
`agent/MOTION_JEPA_TRAJECTORY.md` for the deferred cross-attention design.

BF16 autocast is controlled by `meta.use_bfloat16`. BF16 does not use a
gradient scaler; optional FP16 training uses `meta.use_float16` and restores
its scaler state.

TensorBoard logging is enabled in the standard configs. Install TensorBoard
and launch it against the training output:

```bash
pip install tensorboard
tensorboard --logdir output
```

Events are written under `<logging.folder>/tensorboard` at `logging.log_freq`.
Loss, iteration time, learning rate, and weight decay are interval averages
accumulated since the previous log event. The CSV continues to store raw values
for every optimization step.

## Linear probing

Extract and cache frozen features from a pretrained EMA target encoder, then
train a single linear style classifier on 100STYLE:

```bash
python -m experiment.linear_probe \
  --checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --dataset-root dataset/100style-soma77-processed \
  --output output/linear-probe/<run>
```

For a 2D encoder, preserve anatomical token identity with the group-aware
linear probe. It averages only over valid time tokens, flattens the spatial
tokens, and trains one biased `nn.Linear` head:

```bash
python -m experiment.linear_probe.train_probe_2d \
  --checkpoint output/<run>/motion-jepa-patch-2d-p3-coarse7-latest.pth.tar \
  --dataset-root dataset/100style-soma77-processed
```

Its default output and feature-cache directory is
`<checkpoint directory>/linear-probe-2d`, separate from the global-mean probe.

`--output` may be omitted; in that case results are written under
`<checkpoint directory>/linear-probe`. Supplying it explicitly selects a
different linear-probe result directory.

The encoder is reconstructed from the checkpoint config and uses the
pretraining mean and standard deviation, not the 100STYLE statistics. Its
selected pooled features are cached as `float32` under
`<output>/features`. The probe is a bias-enabled `nn.Linear` trained with
ordinary cross-entropy and momentum SGD. Use `--overwrite` to rerun the head
while retaining valid feature caches, or `--recompute-features` when the
checkpoint, dataset index, statistics, or extraction setup has changed.

Pooled-feature linear-probe commands and online probes now standardize each
feature channel using **training features only**: `(x - train_mean) /
max(train_std, 1e-6)`, with population standard deviation. Validation and test
reuse those training statistics. Raw feature caches stay unchanged; saved
offline heads include a `standardizer` containing the mean and scale required
at inference. Use `--no-standardize` in the probe/LR-sweep CLI, or
`linear_probe.standardize: false` online, to reproduce the previous protocol.
This feature transformation is additional to BONES input-motion normalization.

For the default validation-free dataset, the probe protocol is fixed in
advance: 50 epochs, SGD with LR 0.3, momentum 0.9, zero weight decay, and cosine
decay. Per-epoch training metrics are written to `metrics.csv`; the epoch-50
head is saved and `FW` test is evaluated exactly once. `summary.json` records
`selection: fixed_last_epoch` and `validation_used: false`.

To monitor representation quality during pretraining, opt a training config into
an in-memory 100STYLE probe:

```yaml
linear_probe:
  enabled: true
  # Probe cadence is independent of logging.checkpoint_freq:
  frequency: 10
  dataset_root: ./dataset/100style-soma77-processed
  epochs: 50
  feature_batch_size: 256
  batch_size: 256
  num_workers: 8
  lr: 0.3
  momentum: 0.9
  weight_decay: 0.0
  seed: 42
  standardize: true
  # Use this for 2D encoders to retain spatial token identity:
  pooling: temporal_mean_spatial_flatten
```

The EMA target encoder is evaluated before the first training epoch, at every
`linear_probe.frequency` epoch, and at the final epoch. When `frequency` is
omitted, it defaults to `logging.checkpoint_freq` for backward compatibility.
This cadence does not create additional named epoch checkpoints. Features
remain in memory rather than being cached per checkpoint. The fixed-protocol
`FW` metrics are diagnostic trajectories under `linear_probe/test_*`. They do
not create `<write_tag>-best-accuracy.pth.tar` and do not select a pretraining
checkpoint; use latest or a named epoch checkpoint for final comparison.
Linear probing is disabled when the section is absent or `enabled: false`.

For BABEL-60 and BABEL-120 online probing, use `linear_probe.datasets` instead
of `dataset_root`. The active `configs/mjepa_patch_1d_base.yaml` evaluates both
datasets before training, every 10 epochs, and at the final epoch. Each probe
fits a fresh 50-epoch SGD linear head on frozen EMA features with multi-label
BCE, selects the head by validation mAP, and records validation metrics under
`linear_probe/babel-60/*` and `linear_probe/babel-120/*`. BABEL has no labeled
test split. The best pretraining checkpoints are saved separately as
`<write_tag>-best-babel-60-map.pth.tar` and
`<write_tag>-best-babel-120-map.pth.tar`; probe results and both best scores
resume from the latest checkpoint. The existing single-dataset `dataset_root`
configuration continues to use the 100STYLE probe.

Both `mjepa_patch_1d_base.yaml` and `mjepa_patch_1d_base_nmask2.yaml` also enable
an attentive probe alongside the standardized linear probe:

```yaml
attentive_probe:
  enabled: true
  frequency: 10
  # datasets defaults to linear_probe.datasets (BABEL-60 and BABEL-120).
  epochs: 50
  feature_batch_size: 256
  batch_size: 256
  num_workers: 8
  lr: 0.0003
  weight_decay: 0.05
  warmup_epochs: 5
  final_lr: 1.0e-6
  gradient_clip: 1.0
  num_heads: 6
  seed: 42
```

The attentive head uses one learned query, one cross-attention block and a
residual MLP (ratio 4, GELU), followed by LayerNorm and classification. It trains
with AdamW (betas 0.9/0.999), warmup + cosine, and unweighted BCE. Inputs are
unpooled frozen EMA tokens; padding is excluded, and motions with no valid token
are rejected. It keeps the existing head's LayerNorm without adding the pooled
linear probe's channel standardization. Tokens are held temporarily in CPU BF16
memory, one dataset at a time; no token caches are written during pretraining.

Both probe types run sequentially on rank 0, preserving the pretraining RNG.
Attentive evaluation has its own frequency (defaulting to the linear frequency),
and also runs before training and at the final epoch. Its metrics use
`attentive_probe/babel-{60,120}/*`; its best pretraining checkpoints are
`<write_tag>-best-attentive-babel-{60,120}-map.pth.tar`. Latest checkpoints store
independent latest/best results for all four evaluations. Attentive probing can
be disabled independently or enabled with its own BABEL `datasets` mapping.

When resuming with a changed probe protocol (including raw → standardized
linear features), the affected best scores are reset and that probe evaluates
the resumed encoder before the next optimization step. Compatible probe scores
are restored. TensorBoard's `feature_standardization` scalar marks the protocol;
historical raw and standardized values should not be treated as one unchanged
evaluation curve. A running process keeps its already-loaded configuration;
these settings apply on the next launch/resume. Attentive probing adds full
head-training time at each evaluation interval.

Sweep every direct-child latest checkpoint with:

```bash
python -m experiment.linear_probe.lr_sweep --device cuda:0
```

LR sweeps and validation-selected reports reject validation-free datasets.
They are retained only for legacy datasets with a genuine non-empty validation
split, preventing `FW` test results from becoming an implicit tuning set.

For an explicitly diagnostic LR-sensitivity check on immutable 2D checkpoints,
use `--diagnostic-no-validation`, list the checkpoints, and retain spatial token
identity with `temporal_mean_spatial_flatten`. This reports every FW result but
does not make it a valid LR-selection criterion:

```bash
python -m experiment.linear_probe.lr_sweep \
  --checkpoints output/<run>/*-ep*.pth.tar \
  --pooling temporal_mean_spatial_flatten \
  --diagnostic-no-validation \
  --lrs 0.1 0.3 0.5 0.7 1.0 \
  --seeds 42 \
  --findings-root findings/<diagnostic-name> \
  --device cuda:0
```

Features are extracted once per checkpoint and reused across learning rates.
Completed probe runs resume from their summaries unless `--overwrite-runs` is
given.

The sweep writes its component artifacts to
`findings/000-100style-classification/linear-probe` by default.

Train matched-size supervised CNN and Transformer baselines directly on raw
100STYLE motion with:

```bash
python -m experiment.linear_probe.train_classifier \
  --model all \
  --dataset-root dataset/100style-soma77-processed \
  --seed 42 \
  --device cuda:0
```

Train the single-layer normalized raw-motion baseline with the same data
loader, optimizer, schedule, metrics, and checkpointing entry point:

```bash
python -m experiment.linear_probe.linear \
  --dataset-root dataset/100style-soma77-processed \
  --device cuda:0
```

This model takes the masked temporal mean of each normalized `[90,366]` window
and applies exactly one `Linear(366, 100)` layer. On the validation-free
content split it saves the final epoch as `classifier-final.pth.tar` and then
evaluates `FW` once. The equivalent shared-runner command is
`python -m experiment.linear_probe.train_classifier --model linear ...`.

The same classifiers can consume frozen frame-token features from a pretrained
1D MotionJEPA target encoder:

```bash
python -m experiment.linear_probe.train_classifier \
  --input-source jepa \
  --jepa-checkpoint output/<run>/motion-jepa-1d-latest.pth.tar \
  --checkpoint-key target_encoder \
  --model all \
  --dataset-root dataset/100style-soma77-processed \
  --seed 42 \
  --device cuda:0
```

Token features are cached once as BF16 under
`<checkpoint directory>/linear-probe/token-features`. Classifier outputs default
to `<checkpoint directory>/linear-probe/classifiers`; both locations can be
overridden. Classifier provenance—including the JEPA model name, checkpoint key
and SHA256, feature dimension, and normalization-statistics hashes—is recorded
once in the final summary. The best checkpoint contains only model weights and
the architecture fields required to reconstruct the classifier; resumable latest
state is removed after successful completion.

Regenerate the combined seed-42 comparison, validation-selected CSV, and plots
after both components finish:

```bash
python -m experiment.linear_probe.report
```

See the [unified 100STYLE findings](findings/000-100style-classification/README.md)
for the current CNN, CLS Transformer, and 15-probe comparison.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers preprocessing, representation parity, data validation,
semantic 2D routing, both encoder/predictor variants, masking and overlap,
checkpoint restoration, launch dispatch, single-process resume, and a real
two-process Gloo DDP run. This checkout exposes one GPU, so NCCL multi-GPU
execution cannot be exercised locally.

## Attribution

MotionJEPA includes modified Apache-2.0 portions and assets derived from NVIDIA
Ardy and Kimodo. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

The JEPA training design is based on Meta's I-JEPA implementation and paper:

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```
