# MotionJEPA trajectory-token design

## Current decision

Patchified 2D MotionJEPA uses one trajectory token followed by the configured
body-part tokens. The active `coarse7` layout is:

```text
[trajectory, pelvis, torso, head, left_arm_hand, right_arm_hand,
 left_leg_foot, right_leg_foot]
```

The trajectory token receives canonicalized root `x/z` and heading
`cos/sin`. The pelvis token retains root height, root 6D rotation, and root
velocity. All other joint routing remains unchanged. The trajectory and body
stems have separate temporal projections and their outputs share the same
axial transformer.

There is no token-type embedding. The separate input projections and learned
spatial positional embeddings already distinguish trajectory from body
tokens; an additional two-way type embedding can be absorbed into those
learned spatial embeddings.

## JEPA masking

Trajectory is deliberately treated as an ordinary spatial token in the first
version. It can be selected as context or target and receives the same JEPA
loss weight as a body token. The active sampler draws four of eight spatial
tokens and a contiguous 40--60% temporal interval for each of four target
masks. Target masks may overlap; the context is the exact complement of their
55--65% union.

This simple design can leak motion-path information into simultaneous body
prediction. A high `trajectory_reliance` value alone is not proof of leakage:
it must be interpreted with body RankMe/std, body-token held-out loss, and
linear-probe behavior.

## Deferred cross-attention design

Cross-attention is intentionally deferred. Revisit it if body RankMe/std
declines while normal held-out loss improves, or if removing trajectory causes
body PredictionGain to collapse.

The follow-up ablation should compare:

1. a separate temporal trajectory encoder and body encoder;
2. predictor-side trajectory-to-body cross-attention;
3. body-only EMA targets;
4. target-aligned trajectory masking; and
5. downstream late fusion of body and trajectory features.

Existing seven-token checkpoints are architecture-incompatible with this
layout and must fail strict loading rather than being silently migrated.
