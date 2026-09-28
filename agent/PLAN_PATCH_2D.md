# 2D Temporal-Spatial Patchification

> This document describes the earlier body-only patchification work. The active
> trajectory-token layout and online diagnostics are documented in
> `MOTION_JEPA_TRAJECTORY.md` and `ONLINE_REPRESENTATION_METRICS.md`. Historical
> YAML configurations now live under `configs/_depr_experiments/`.

## Token geometry

The separate patchified 2D family keeps `motion_jepa_366_v1` and its lossless
SOMA30 semantic routing. A per-joint `Conv1d(kernel_size=3, stride=3)` maps 90
raw frames to 30 non-overlapping temporal patches. Incomplete tail frames are
dropped and temporal positions use patch centers `(3i + 1) / fps`.

Spatial pooling is selected with `patch.spatial_grouping`:

- `fine11` (default): pelvis; torso; head; left/right upper arm; left/right
  lower arm and hand; left/right leg; left/right foot. The output grid is
  `[B, 30, 11, D]` (330 cells).
- `coarse7`: pelvis; torso; head; left/right arm and hand; left/right leg and
  foot. The output grid is `[B, 30, 7, D]` (210 cells).

Every SOMA30 joint belongs to exactly one group. Each temporal patch is mixed
with a residual graph convolution using only self-loops and parent-child edges
whose endpoints belong to the same group, followed by GELU and hard mean
pooling. There are no cross-group stem edges or stem LayerNorm layers. Learned
group embeddings and axial Transformer attention operate after pooling.

## Interface and compatibility

Both presets use `mot_patch_{size}_2d` and `mot_predictor_patch_{size}_2d`.
Configs select `spatial_grouping: fine11 | coarse7` and
`spatial_pooling: graph_mean`. `PatchMaskCollator2D` samples masks on the pooled
grid and maps each raw valid length to `floor(length / 3)`.

Architecture signatures store the explicit grouping, SOMA30 names, graph
edges/version, pooling operator, and token layout. Resume fails before state
loading when encoder, predictor, checkpoint, or collator geometry differs.
Raw 1D/2D and existing patch 1D checkpoint signatures remain unchanged.

Base configs and outputs are:

- `configs/mjepa_patch_2d_base_fine11.yaml` ->
  `output/mot_patch_base_2d-p3-fine11-bs.32-ep.300`
- `configs/mjepa_patch_2d_base_coarse7.yaml` ->
  `output/mot_patch_base_2d-p3-coarse7-bs.32-ep.300`

## Verification

Tests cover output geometry, complete and disjoint group assignment,
group-local adjacency, tail and padding behavior, target-patch leakage,
multi-mask alignment, Conv1d/GraphConv gradients, strict deterministic resume,
geometry mismatch rejection, and frozen feature/token cache reconstruction.
Raw model and existing patch 1D regressions remain part of the suite.
