# JEPA-conditioned raw-motion generation

This experiment learns a conditional rectified flow from Gaussian noise to raw
motion. Frozen JEPA tokens supply the condition; generation takes place in
normalized 366-channel motion space. The JEPA encoder, predictor, existing
deterministic decoder experiment, and TMR experiment are not trained or modified.

Run commands from the repository root in the `motion-jepa` Conda environment.
`experiment/generation/config.yaml` contains defaults; `--config` accepts YAML
overrides. Relative checkpoint, dataset, cache, and output paths resolve against
the repository root.

## Model and objective

The baseline uses BONES-SEED clips with 150 frames at 30 FPS, SOMA30, and 366 raw
feature channels. Conditions are complete `[B,50,384]` EMA encoder tokens from:

```text
output/mot_patch_base_1d-p3-bs.512-ep.300-nframes150/motion-jepa-patch-1d-p3-ep300.pth.tar
```

The frozen encoder uses its pretraining normalization statistics and evaluation
mode. Tokens receive the same functional LayerNorm as JEPA teacher targets;
there is no pooling or L2 normalization. The experiment reuses the deterministic
decoder's BF16 memmap cache and exact provenance checks. The default cache is
`output/prediction/cache/baseline-ep300`, approximately 7.54 GiB across all three
splits. Changed checkpoints, statistics, layouts, or subset limits require a
separate cache root.

For normalized target motion `x`, standard Gaussian noise `z`, and a uniformly
sampled flow time `t` in `[0,1]`, training uses:

```text
x_t = (1 - t) * z + t * x
target_velocity = x - z
loss = valid-frame/channel MSE(v_theta(x_t, t, JEPA), target_velocity)
```

**Flow loss is velocity prediction error, not final motion reconstruction MSE.**
Sampling starts at `z` and integrates the predicted velocity from `t=0` to `t=1`.
No autoencoder, latent-motion tokenizer, FK loss, velocity consistency loss, or
contact loss is added.

The Transformer receives projected noisy motion per frame, physical frame-time
positions, and a separate flow-time embedding. It cross-attends to JEPA tokens
with patch-center time positions. Spatial identity is retained for 2D layouts.
Default architecture: hidden 384, 4 pre-norm blocks, 6 heads, FFN 1536, GELU,
dropout 0.0. Padded frames/tokens are masked and outputs are zero on padded
frames; valid raw frames in an incomplete final temporal patch still receive
predictions.

Classifier-free guidance drops each entire JEPA condition with probability 0.1
during training, replacing memory with a learned null token. Sampling uses
`v_null + scale * (v_conditioned - v_null)`. The baseline scale 1.0 evaluates only
the conditioned branch. Scale 0.0 uses only the null branch; other scales evaluate
both branches and increase sampling cost.

`MotionFlow.forward(noisy_motion, time, tokens, fps, valid_frames,
token_mask=None, condition_drop=None)` returns `[B,T,366]` flow velocity.
`MotionFlow.sample(...)` returns normalized `[B,K,T,366]` motions. An explicit
`initial_noise` permits repeatable sampling; optional token masks use `True` for
usable tokens and combine with the checkpoint layout's valid-memory mask.

## Subset smoke training

The initial run is a small BONES-SEED subset; full generation training is not
automatically launched as part of verification. Use a separate cache/output so
subset provenance does not conflict with the completed full JEPA cache:

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.generation train \
  --cache-root output/generation/cache/smoke \
  --output output/generation/smoke \
  --limit-train 32 --limit-val 16 --limit-test 16 \
  --epochs 50 --batch-size 32 --cache-batch-size 32 --num-workers 2 \
  --export-count 4
```

Training prepares/verifies train and validation caches automatically. Best-model
evaluation prepares/verifies the configured test cache. Feature extraction can
also be run separately with `prepare-cache`; `--splits train val` restricts it to
the requested splits.

Full-training defaults are seed 42, 100 epochs, batch 256, AdamW, LR `3e-4`,
weight decay `0.01`, gradient clipping `1.0`, and 5 warmup epochs followed by
cosine decay to `1e-6`. CUDA uses BF16 autocast and CPU uses FP32. A separate
generator EMA follows each optimizer step with decay
`min(0.999, (1 + global_step) / (10 + global_step))`.

Every epoch evaluates the generator EMA on the complete configured validation
split, with one fixed per-ID noise/time pair and conditions enabled. The lowest
validation flow MSE selects `best.pth.tar`; EMA weights are also used for
sampling. Validation does not run an iterative sampler on every epoch.

Outputs include `best.pth.tar`, `latest.pth.tar`, `config.json`, `metrics.csv`,
and `tensorboard/`. Checkpoints store live and EMA generator weights, optimizer,
scheduler, epoch/step, RNG/DataLoader generator states, model configuration,
statistics, and source provenance. JEPA weights are referenced rather than
duplicated. Resume an interrupted smoke run with its original settings:

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.generation train \
  --cache-root output/generation/cache/smoke \
  --output output/generation/smoke \
  --limit-train 32 --limit-val 16 --limit-test 16 \
  --epochs 50 --batch-size 32 --cache-batch-size 32 --num-workers 2 \
  --export-count 4 --resume output/generation/smoke/latest.pth.tar
```

Common source/training overrides include `--jepa-checkpoint`, `--dataset-root`,
`--stats-path`, `--cache-root`, `--output`, `--device`, `--epochs`, `--batch-size`,
`--cache-batch-size`, `--num-workers`, `--limit-train/val/test`, `--lr`,
`--condition-dropout`, and `--ema-decay`. Zero split limits use all samples.
Use `--no-bfloat16` or `--no-tensorboard` to disable those options. Configure
architecture under the YAML's `flow` mapping.

## Sampling and evaluation

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.generation sample \
  --checkpoint output/generation/smoke/best.pth.tar \
  --split test --indices 0 1 2 3 --num-samples 4 --steps 32 \
  --guidance-scale 1.0 --evaluation-seed 42

conda run --no-capture-output -n motion-jepa python -m experiment.generation evaluate \
  --checkpoint output/generation/smoke/best.pth.tar --split test
```

Both commands inherit the checkpoint's training settings unless overridden.
Sampling uses 32 fixed Euler steps by default, maintaining integration state in
FP32. Noise seeds are derived from evaluation seed 42, sample ID and draw index
using SHA-256, so batching, subset order, and export count do not change a draw.
`sample` exports requested dataset indices; without `--indices`, it selects fixed
evenly spaced samples up to `--export-count`.

`sample` writes `<split>-samples.npz` and `<split>-samples.json`. Evaluation
writes `<split>-generations.npz`, `<split>-metrics.json`, and
`<split>-protocol.json`, so sampling does not replace evaluation results. Repeated
commands with the same output/split replace their own reports atomically.

Evaluation measures draw 0 on every configured test condition. A fixed bank of
up to 512 conditions receives K=4 draws for Monte Carlo average errors,
oracle best-of-K errors, and conditional diversity. Controls are
`--diagnostic-count`, `--num-samples`, `--steps`, `--guidance-scale`,
`--evaluation-seed`, and `--export-count`. The report records the actual protocol
and chosen IDs; the subset smoke uses its available test conditions.

Reconstruction metrics reuse the deterministic experiment: normalized MSE and
representation-block MSE, denormalized FK MPJPE and root error in millimeters,
rotation error in degrees, and contact F1. Oracle best-of-K selects one whole
draw per clip by normalized MSE and uses that draw for every physical metric.
Conditional diversity reports pairwise root-trajectory distance and root-relative
non-root FK-joint distance in millimeters, excluding padded frames.

The normalized-zero baseline is always included. Add a deterministic decoder
baseline explicitly with:

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.generation evaluate \
  --checkpoint output/generation/smoke/best.pth.tar --split test \
  --deterministic-checkpoint output/prediction/baseline-ep300/seed-42/best.pth.tar
```

The baseline must share JEPA features/layout and normalization conventions.
Higher diversity alone does not establish better motion. Best-of-K improves
mechanically as K increases; compare matching K/steps/guidance and retain ordinary
draw-0 errors. FK metrics follow root position and rotation channels, so they do
not independently assess generated local-position/velocity/heading consistency.
Contact F1 measures the contact channels, not physical foot sliding. This first
version does not claim a FID or calibrated conditional-distribution benchmark.

Full BONES test has 13,067 conditions. At 32 Euler steps, one full draw plus three
additional draws on 512 diagnostic conditions requires 467,296 per-example
network evaluations. K=4 across all test conditions would require 1,672,576.
Guidance scales other than 0 or 1 evaluate both branches. Metrics are accumulated
in batches; only selected motions are retained for export.

## Export and visualization

Exports contain raw, unnormalized data in a pickle-free NPZ:

| Field | Shape |
| --- | --- |
| `sample_ids`, `lengths` | `[N]` |
| `fps` | Scalar |
| `target_motion` | `[N,T,366]` |
| `generated_motion` | `[N,K,T,366]` |
| `target_joints` | `[N,T,30,3]`, meters |
| `generated_joints` | `[N,K,T,30,3]`, meters |
| `noise_seeds` | `[N,K]`, unsigned integers |
| `sampling_json` | Scalar JSON string with sampling/source metadata |

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.generation visualize \
  --results output/generation/smoke/test-generations.npz \
  --host 0.0.0.0 --port 8080
```

The CPU viewer shows the original and selected generated draw side by side.
Sample selection, draw selection, shared frame/playback/speed controls, contact
display, and skeleton/mesh visibility are available; `--mesh` starts in mesh
mode. Changing draw preserves the current frame and pauses playback. Each browser
controls its own selection. Existing SOMA30 decoding, SOMA77 expansion and shaded
rendering are reused without loading JEPA or generator weights. Stop with Ctrl+C.

## Verification

```bash
conda run --no-capture-output -n motion-jepa python -m pytest tests/test_generation*.py -q
```

Checks cover motion/token layouts and masks, flow path/velocity construction,
classifier-free conditioning, exact Euler behavior for simple test fields,
stable per-ID noise, save/reload and resume, whole-draw oracle selection,
conditional diversity, export validation, and synchronized draw-switch playback.
Subset training verifies finite/decreasing flow loss and functional sampling;
it does not establish full-dataset generation quality.

## Verified BONES-SEED smoke run

The implemented baseline completed 50 epochs on train 32 / val 16 / test 16,
including an interruption at epoch 25 and resume from `latest.pth.tar`. With one
optimizer step per epoch, training flow MSE decreased from **1.328683 to
1.008341**. Best EMA validation flow MSE was **2.022911**, selected at epoch 2.
Best-model test draw-0 motion MSE was **1.920008**, versus **0.840543** for the
normalized-zero baseline; best-of-4 MSE was **1.911912**. Generation quality is
still poor at this small training budget.

Artifacts are in `output/generation/baseline-ep300-smoke/seed-42/`; its isolated
cache is `output/generation/cache/baseline-ep300-smoke/`. Training/resume,
TensorBoard/CSV, finite physical metrics, multi-draw exports, the sample CLI,
and CPU skeleton/mesh rendering through an actual Viser server were verified.

## Full training command

The existing default full feature caches were verified for all splits. To start
the default 100-epoch generation run from the repository root:

```bash
conda run --no-capture-output -n motion-jepa python -m experiment.generation prepare-cache
conda run --no-capture-output -n motion-jepa python -m experiment.generation train
```

Completed compatible caches are reused. Best-checkpoint test evaluation and
exports run automatically after training. Full generation training has not been
started by this implementation task.
