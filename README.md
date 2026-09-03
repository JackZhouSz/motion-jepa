# Motion-JEPA

Motion-JEPA learns motion representations by predicting target embeddings from
masked context embeddings. The repository contains a self-contained BONES-SEED
preprocessing pipeline, the `motion_jepa_366_v1` SOMA30 representation, two
Motion-JEPA transformer variants, distributed pretraining, and visualization.

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
configs/          `_1d` and `_2d` pretraining configurations
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
  --config configs/mjepa_patch_2d_base_fine11.yaml \
  --output output/mask-patch-2d-fine11.png \
  --seed 0 \
  --valid-length 90
```

The renderer automatically uses a timeline for 1D layouts and a
time-by-joint/body-group grid for 2D layouts. Patch timelines annotate both
token indices and their corresponding raw-frame spans.

For coarse7, set `mask.strategy: body_region_segment` to predict one
graph-connected body region over a contiguous time segment. Configure the
temporal extent with `pred_frame_mask_ratio` and the number of connected body
groups with `graph_mask_ratio`. `num_regions` samples that many cell-disjoint
segments and trains on their union; the encoder observes the exact complement
of the target over valid tokens. See
`configs/mjepa_patch_2d_tiny_coarse7_body_region_segment.yaml` for the default
2–3 group, 30–60% temporal setup. Omitting `mask.strategy` preserves the
original multiblock behavior.

Set `mask.strategy: random_body_segment` to sample several separate target
masks, each formed by a contiguous temporal segment over uniformly random
coarse7 body groups. Target masks are cell-disjoint, the encoder observes their
exact complement, and `body_mask_ratio` controls only how many of the seven
groups each target selects; it does not use the skeleton graph. See
`configs/mjepa_patch_2d_tiny_coarse7_part1-3_frame15-25.yaml` for the
four-target, 1–3 group, 15–25% temporal setup. The matching one-group control
is `configs/mjepa_patch_2d_tiny_coarse7_part1_frame15-25.yaml`.

## Model variants

`_1d` uses one token per 366-D frame and temporal transformer attention. Its
context and target masks are contiguous temporal blocks.

`_2d` routes the representation into a `[frames,30 joints]` grid. Each joint
receives its own position, rotation, and velocity fields; root position and
heading go to the root token, while the four contact values go to their
corresponding foot/toe tokens. Encoder blocks apply temporal attention per
joint followed by spatial attention per frame. The predictor retains separate
context and target-query streams, including when their coordinates overlap.

Named factories are available for `tiny`, `small`, `base`, `large`, `huge`,
and `giant`, for example `mot_base_1d` and `mot_base_2d`. There are no
unsuffixed compatibility aliases.

## Training

Batch size is per rank. Learning rates are not automatically scaled.

Single GPU:

```bash
python main.py --fname configs/mjepa_1d.yaml --devices cuda:0
```

Local multi-GPU:

```bash
python main.py \
  --fname configs/mjepa_2d.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

`main.py` spawns one process per configured device and propagates child
failures. `--debug` forces one process on the first device.

Standard torchrun launch:

```bash
torchrun --standalone --nproc-per-node=4 main.py \
  --fname configs/mjepa_1d.yaml
```

`main.py` honors `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. CUDA training uses
NCCL; CPU integration tests use Gloo. `main_distributed.py` remains available
for Submitit/SLURM launches.

Checkpoints contain unwrapped encoder and predictor weights, the EMA target,
optimizer, optional AMP scaler, all schedules, epoch/global step, mask state,
and per-rank Python/NumPy/PyTorch RNG states. Writes are atomic. Exact resume
requires the same world size and continues from the latest completed epoch:

```yaml
meta:
  load_checkpoint: true
  read_checkpoint: null  # null selects <write_tag>-latest.pth.tar
```

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

Sweep every direct-child latest checkpoint with:

```bash
python -m experiment.linear_probe.lr_sweep --device cuda:0
```

LR sweeps and validation-selected reports reject validation-free datasets.
They are retained only for legacy datasets with a genuine non-empty validation
split, preventing `FW` test results from becoming an implicit tuning set.

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

The same classifiers can consume frozen frame-token features from a pretrained
1D Motion-JEPA target encoder:

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

Motion-JEPA includes modified Apache-2.0 portions and assets derived from NVIDIA
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
