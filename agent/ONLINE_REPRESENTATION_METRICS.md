# Online representation-quality diagnostics

The online evaluator uses a fixed, unlabeled subset of the pretraining
validation manifest. It runs at epoch 0, at `linear_probe.frequency`, and at
the final epoch. Metrics are diagnostic only and never select checkpoints.

## Representation metrics

For a feature matrix `Z [N,D]` with singular values `sigma`, RankMe is:

```text
p_i = sigma_i / sum_j sigma_j
RankMe = exp(-sum_i p_i log(p_i))
```

RankMe is computed from uncentered float64 pooled features. A value near one or
a decreasing trajectory can indicate dimensional collapse, but a high value
does not imply good downstream performance.

The evaluator records body, trajectory, and combined sample-pooled RankMe,
mean feature std, and mean off-diagonal sample cosine. It additionally records:

- temporal std after spatially pooling the body tokens;
- joint std over the seven body tokens after temporal pooling;
- trajectory temporal std; and
- centered covariance largest-eigenvalue ratio, effective rank, stabilized
  condition number, and mean absolute off-diagonal covariance.

## Held-out JEPA metrics

EMA targets use the same final layer normalization as training. The evaluator
records Smooth-L1, MSE, predictor-target cosine, and results split by target
mask, spatial token, and same-token distance to visible temporal context.

PredictionGain compares the predictor MSE with a predictor that always emits
the held-out target mean:

```text
PredictionGain = 1 - predictor_mse / trivial_mean_mse
```

When target variance is numerically zero, gain is reported as zero if the
prediction is also exact. Loss decreasing together with rank/std is a collapse
warning; loss decreasing with stable rank/std and increasing gain is the
desired pattern.

The evaluator repeats body-target prediction after removing every remaining
trajectory context token:

```text
trajectory_reliance = normal_body_gain - no_trajectory_body_gain
```

This measures reliance, not causal leakage. Use it together with the metrics
above before adopting the deferred cross-attention design documented in
`MOTION_JEPA_TRAJECTORY.md`.

TensorBoard values use the `online_metrics/` prefix. The complete nested
summary is appended to `online-metrics.jsonl` and stored in the latest
checkpoint as `online_metrics_latest`.
