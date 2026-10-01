# Deterministic motion reconstruction from JEPA features

This experiment trains a frame-query Transformer decoder to reconstruct raw
motion from a frozen JEPA EMA encoder. It measures how much motion information
the learned representation preserves. Predictor training and future prediction
evaluation are outside this experiment.

Run commands from the repository root in the `motion-jepa` Conda environment.
The default configuration is `experiment/prediction/config.yaml`; `--config`
accepts a YAML with overrides. Relative dataset, checkpoint, cache, and output
paths are resolved against the repository root.

## Baseline

| Setting | Default |
| --- | --- |
| Data | BONES-SEED, `dataset/bones-seed-processed-nframes150` |
| Splits | Existing train / val / test; 191,062 / 6,606 / 13,067 clips |
| Motion | 150 frames at 30 FPS; 366 channels; SOMA30 skeleton |
| JEPA checkpoint | `output/mot_patch_base_1d-p3-bs.512-ep.300-nframes150/motion-jepa-patch-1d-p3-ep300.pth.tar` |
| Features | EMA `target_encoder`, `[B, 50, 384]`, functional LayerNorm |
| Decoder | Hidden 384, 4 pre-norm blocks, 6 heads, FFN 1536, GELU, dropout 0.1 |
| Objective | MSE over all valid normalized frame/channel values |
| Training | Seed 42, 100 epochs, batch 256, AdamW |
| Schedule | LR `3e-4`, 5 warmup epochs, cosine decay to `1e-6` |
| Regularization | Weight decay `0.01`, gradient clipping `1.0` |
| Precision | CUDA BF16 autocast; FP32 on CPU |

The EMA encoder remains in evaluation mode with gradients disabled. Its complete
token sequence receives `F.layer_norm(..., [feature_dim])`, matching JEPA teacher
targets. There is no token pooling or L2 normalization. Motion normalization uses
the original JEPA pretraining statistics for every split.

The decoder uses one query per raw frame and cross-attends to JEPA tokens. Frame
queries and memory tokens carry time positions in seconds; temporal patch tokens
use their patch centers. A 2D token layout also receives a learned spatial
embedding before its grid is flattened for attention. Padded memory tokens and
frames are masked; padded output frames are zero. An incomplete final temporal
patch is excluded from encoder memory, while its valid raw frames still receive
queries and contribute to reconstruction loss. No FK, velocity, or contact loss
is used.

The model interface is
`MotionDecoder.forward(tokens, fps, valid_frames, token_mask=None)`. `fps` is a
`[B]` vector and `valid_frames` is a boolean `[B, T]` mask. Optional boolean
`token_mask` uses `True` for usable memory tokens; it is combined with the layout's
valid token mask. Outputs are normalized `[B, T, 366]` motion. Token shapes follow
the checkpoint's 1D/2D and raw/patch `TokenLayout` rather than assuming the
baseline layout.

## Prepare features and train

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.prediction prepare-cache
conda run --no-capture-output -n motion-jepa python -m experiment.prediction train
```

`prepare-cache` defaults to all three splits; `--splits train val` prepares only
those requested. Training prepares or verifies train/val caches automatically,
and prepares or verifies test features for evaluation after training. Successful
existing caches are reused only when their provenance matches.

Tokens are NumPy memory maps containing BF16 bit patterns as `uint16`, reinterpreted
as BF16 tensors on load. All default splits need approximately **7.54 GiB** for tokens. Each
split records sample IDs, lengths, FPS, layout, feature transform, checkpoint
SHA-256, normalization-statistics SHA-256, and source-dataset fingerprints. A
`completed.json` marker is published only when that split is complete; changed
checkpoints, statistics, layouts, or subset limits require a separate cache root.
Targets are read lazily from the existing normalized `MotionDataset`.

Common options include `--jepa-checkpoint`, `--dataset-root`, `--stats-path`,
`--cache-root`, `--output`, `--device`, `--epochs`, `--batch-size`,
`--cache-batch-size`, `--num-workers`, and `--limit-train/val/test`. A zero limit
uses the entire split. The precision and logging switches are `--no-bfloat16`
and `--no-tensorboard`; they have positive counterparts. Architecture parameters
are configured under the YAML's `decoder` mapping.

For a small real-data pipeline check, keep a separate cache and output directory:

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.prediction train \
  --cache-root output/prediction/cache/smoke \
  --output output/prediction/smoke \
  --epochs 3 --batch-size 4 --cache-batch-size 4 --num-workers 0 \
  --limit-train 16 --limit-val 8 --limit-test 8
```

The default training output is
`output/prediction/baseline-ep300/seed-42/`. It contains:

- `best.pth.tar`, selected by the lowest full validation MSE.
- `latest.pth.tar`, including decoder, optimizer, scheduler, RNG and DataLoader
  generator states, global step, epoch, configuration, statistics, and provenance.
- `config.json`, `metrics.csv`, and `tensorboard/` for loss and LR monitoring.
- `test-metrics.json` and `test-reconstructions.npz` after best-checkpoint evaluation.

JEPA weights are referenced by checkpoint path and fingerprint rather than
duplicated in decoder checkpoints. Resume an interrupted run with the same
configuration and data provenance:

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.prediction train \
  --resume output/prediction/baseline-ep300/seed-42/latest.pth.tar
```

## Evaluate and compare

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.prediction evaluate \
  --checkpoint output/prediction/baseline-ep300/seed-42/best.pth.tar \
  --split test --export-count 16

conda run --no-capture-output -n motion-jepa python -m experiment.prediction visualize \
  --results output/prediction/baseline-ep300/seed-42/test-reconstructions.npz \
  --host 0.0.0.0 --port 8080
```

Evaluation reports overall normalized MSE and MSE for each representation block.
After denormalization it uses the existing representation inverse and skeleton
FK to report MPJPE and root position error in millimeters, rotation error in
degrees, and contact F1. The same metrics are reported for a normalized
zero-output baseline, which corresponds to the training mean in raw feature
space. Every metric excludes padded frames. Evaluation uses the saved training
configuration unless explicitly overridden, and checks feature and
normalization compatibility with the trained decoder.

Exports select fixed, evenly spaced split indices. `--export-count 0` suppresses
exports. The NPZ stores `sample_ids [N]`, `lengths [N]`, scalar `fps`, raw
unnormalized `target_motion` and `reconstructed_motion [N,T,366]`, and
`target_joints` and `reconstructed_joints [N,T,30,3]` in meters.

The Viser viewer reads this export independently of the encoder and GPU. It
decodes motion on CPU, expands SOMA30 to SOMA77, and reuses the existing shaded
skeleton renderer. Original and reconstructed motions appear side by side with
shared sample selection, play/pause, frame stepping, speed, and contact display.
Use `--mesh` to start with the existing SOMA mesh renderer; mesh and skeleton
visibility can also be toggled in the browser. Each browser has independent
playback controls. Stop the server with Ctrl+C.

## Verification

```bash
conda run --no-capture-output -n motion-jepa python -m pytest tests/test_prediction*.py -q
```

Tests cover token layouts, output shapes, valid-frame masking including patch
tails, frozen encoder extraction, cache provenance, reconstruction metrics,
decoder save/reload and resume, and exported-data validation and synchronized
viewer playback. A small real BONES-SEED run should also verify decreasing
training loss and working reconstruction exports before a full experiment.
