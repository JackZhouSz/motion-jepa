"""Graph-connected body-region masking for patchified skeletal motion."""

from __future__ import annotations

from itertools import combinations

import torch

from model.token_layout import TokenLayout

from .collators import (
    _StatefulMaskCollator,
    _block_length,
    _sample_ratio,
    _valid_lengths,
)


COARSE7_GROUP_NAMES = (
    "pelvis",
    "torso",
    "head",
    "left_arm_hand",
    "right_arm_hand",
    "left_leg_foot",
    "right_leg_foot",
)
COARSE7_GRAPH_EDGES = (
    (0, 1),  # pelvis -- torso
    (1, 2),  # torso -- head
    (1, 3),  # torso -- left arm/hand
    (1, 4),  # torso -- right arm/hand
    (0, 5),  # pelvis -- left leg/foot
    (0, 6),  # pelvis -- right leg/foot
)


def _connected_subsets(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
    size: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate every connected node subset of an undirected graph."""
    adjacency = [set() for _ in range(num_nodes)]
    for left, right in edges:
        if not 0 <= left < num_nodes or not 0 <= right < num_nodes or left == right:
            raise ValueError(f"Invalid graph edge ({left}, {right}) for {num_nodes} nodes")
        adjacency[left].add(right)
        adjacency[right].add(left)

    connected = []
    for nodes in combinations(range(num_nodes), size):
        selected = set(nodes)
        visited = {nodes[0]}
        frontier = [nodes[0]]
        while frontier:
            node = frontier.pop()
            unseen = (adjacency[node] & selected) - visited
            visited.update(unseen)
            frontier.extend(unseen)
        if visited == selected:
            connected.append(nodes)
    return tuple(connected)


class PatchBodyRegionSegmentMaskCollator2D(_StatefulMaskCollator):
    """Mask a contiguous time segment over a connected coarse7 body region."""

    def __init__(
        self,
        raw_num_frames: int,
        raw_num_joints: int,
        token_num_joints: int,
        temporal_patch_size: int = 3,
        spatial_grouping: str = "coarse7",
        spatial_pooling: str = "graph_mean",
        pred_frame_mask_ratio: tuple[float, float] = (0.3, 0.6),
        graph_mask_ratio: tuple[float, float] = (2.0 / 7.0, 3.0 / 7.0),
        num_regions: int = 1,
    ) -> None:
        super().__init__()
        if spatial_grouping != "coarse7":
            raise ValueError(
                "body_region_segment requires spatial_grouping='coarse7'"
            )
        if spatial_pooling != "graph_mean":
            raise ValueError(
                "body_region_segment supports only spatial_pooling='graph_mean'"
            )
        if int(raw_num_joints) != 30 or int(token_num_joints) != 7:
            raise ValueError("body_region_segment requires SOMA30 pooled to coarse7")

        self.layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=int(raw_num_frames),
            token_num_frames=int(raw_num_frames) // int(temporal_patch_size),
            temporal_patch_size=int(temporal_patch_size),
            raw_num_joints=int(raw_num_joints),
            token_num_joints=int(token_num_joints),
        )
        self.pred_frame_mask_ratio = tuple(
            float(value) for value in pred_frame_mask_ratio
        )
        self.graph_mask_ratio = tuple(float(value) for value in graph_mask_ratio)
        self.num_regions = int(num_regions)
        if self.num_regions <= 0:
            raise ValueError("num_regions must be positive")
        _sample_ratio(torch.Generator().manual_seed(0), self.pred_frame_mask_ratio)
        _sample_ratio(torch.Generator().manual_seed(0), self.graph_mask_ratio)
        if self.graph_mask_ratio[0] <= 0.0:
            raise ValueError("graph_mask_ratio must select at least one body group")

        minimum_groups = _block_length(7, self.graph_mask_ratio[0])
        maximum_groups = _block_length(7, self.graph_mask_ratio[1])
        self._regions_by_size = {
            size: _connected_subsets(7, COARSE7_GRAPH_EDGES, size)
            for size in range(minimum_groups, maximum_groups + 1)
        }
        if any(not regions for regions in self._regions_by_size.values()):
            raise ValueError("graph_mask_ratio has no connected coarse7 regions")

        self._configuration = {
            "variant": "patch_2d_body_region_segment",
            "raw_num_frames": self.layout.raw_num_frames,
            "token_num_frames": self.layout.token_num_frames,
            "temporal_patch_size": self.layout.temporal_patch_size,
            "raw_num_joints": self.layout.raw_num_joints,
            "token_num_joints": self.layout.token_num_joints,
            "spatial_grouping": str(spatial_grouping),
            "spatial_pooling": str(spatial_pooling),
            "pred_frame_mask_ratio": self.pred_frame_mask_ratio,
            "graph_mask_ratio": self.graph_mask_ratio,
            "num_regions": self.num_regions,
            "graph_edges": COARSE7_GRAPH_EDGES,
        }

    def connected_regions(self, group_count: int) -> tuple[tuple[int, ...], ...]:
        """Return the selectable connected coarse7 regions for one group count."""
        return self._regions_by_size.get(int(group_count), ())

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        generator = torch.Generator().manual_seed(self.step())
        raw_lengths = torch.tensor(
            [
                int(sample[2]) if len(sample) >= 3 else self.layout.raw_num_frames
                for sample in batch
            ],
            dtype=torch.long,
        )
        token_lengths = self.layout.valid_token_lengths(raw_lengths)
        if (token_lengths < 1).any():
            raise ValueError(
                "Every sample must contain at least one complete temporal patch"
            )
        valid_lengths = _valid_lengths(
            [(None, None, int(length)) for length in token_lengths.tolist()],
            self.layout.token_num_frames,
        )
        shortest = min(valid_lengths)
        frame_count = _block_length(
            shortest, _sample_ratio(generator, self.pred_frame_mask_ratio)
        )
        group_count = _block_length(
            self.layout.token_num_joints,
            _sample_ratio(generator, self.graph_mask_ratio),
        )
        regions = self.connected_regions(group_count)
        if not regions:
            raise ValueError(f"No connected coarse7 region has {group_count} groups")

        contexts = []
        targets = []
        for valid_length in valid_lengths:
            target = torch.zeros(
                self.layout.token_num_frames,
                self.layout.token_num_joints,
                dtype=torch.bool,
            )
            cells_per_region = frame_count * group_count
            if self.num_regions * cells_per_region >= valid_length * 7:
                raise ValueError(
                    "body-region union leaves no context cells; reduce num_regions "
                    "or mask ratios"
                )
            for _union_attempt in range(256):
                target.zero_()
                complete = True
                for _region_number in range(self.num_regions):
                    for _region_attempt in range(256):
                        start = int(
                            torch.randint(
                                valid_length - frame_count + 1,
                                (),
                                generator=generator,
                            ).item()
                        )
                        region_index = int(
                            torch.randint(
                                len(regions), (), generator=generator
                            ).item()
                        )
                        region = regions[region_index]
                        proposal = torch.zeros_like(target)
                        proposal[start : start + frame_count, list(region)] = True
                        if not (target & proposal).any():
                            target |= proposal
                            break
                    else:
                        complete = False
                        break
                if complete:
                    break
            else:
                raise ValueError(
                    "Could not sample non-overlapping body regions; reduce "
                    "num_regions or mask ratios"
                )
            valid = (
                torch.arange(self.layout.token_num_frames)[:, None] < valid_length
            ).expand(-1, self.layout.token_num_joints)
            targets.append(target)
            contexts.append(valid & ~target)

        return collated_batch, [torch.stack(contexts)], [torch.stack(targets)]


class PatchRandomBodySegmentMaskCollator2D(_StatefulMaskCollator):
    """Sample several temporal segments over uniformly random coarse7 groups."""

    def __init__(
        self,
        raw_num_frames: int,
        raw_num_joints: int,
        token_num_joints: int,
        temporal_patch_size: int = 3,
        spatial_grouping: str = "coarse7",
        spatial_pooling: str = "graph_mean",
        pred_frame_mask_ratio: tuple[float, float] = (0.15, 0.25),
        body_mask_ratio: tuple[float, float] = (1.0 / 7.0, 3.0 / 7.0),
        npred: int = 4,
    ) -> None:
        super().__init__()
        if spatial_grouping != "coarse7":
            raise ValueError(
                "random_body_segment requires spatial_grouping='coarse7'"
            )
        if spatial_pooling != "graph_mean":
            raise ValueError(
                "random_body_segment supports only spatial_pooling='graph_mean'"
            )
        if int(raw_num_joints) != 30 or int(token_num_joints) != 7:
            raise ValueError("random_body_segment requires SOMA30 pooled to coarse7")

        self.layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=int(raw_num_frames),
            token_num_frames=int(raw_num_frames) // int(temporal_patch_size),
            temporal_patch_size=int(temporal_patch_size),
            raw_num_joints=int(raw_num_joints),
            token_num_joints=int(token_num_joints),
        )
        self.pred_frame_mask_ratio = tuple(
            float(value) for value in pred_frame_mask_ratio
        )
        self.body_mask_ratio = tuple(float(value) for value in body_mask_ratio)
        self.npred = int(npred)
        if self.npred <= 0:
            raise ValueError("npred must be positive")
        _sample_ratio(torch.Generator().manual_seed(0), self.pred_frame_mask_ratio)
        _sample_ratio(torch.Generator().manual_seed(0), self.body_mask_ratio)
        if self.body_mask_ratio[0] <= 0.0:
            raise ValueError("body_mask_ratio must select at least one body group")

        self._configuration = {
            "variant": "patch_2d_random_body_segment",
            "raw_num_frames": self.layout.raw_num_frames,
            "token_num_frames": self.layout.token_num_frames,
            "temporal_patch_size": self.layout.temporal_patch_size,
            "raw_num_joints": self.layout.raw_num_joints,
            "token_num_joints": self.layout.token_num_joints,
            "spatial_grouping": str(spatial_grouping),
            "spatial_pooling": str(spatial_pooling),
            "pred_frame_mask_ratio": self.pred_frame_mask_ratio,
            "body_mask_ratio": self.body_mask_ratio,
            "npred": self.npred,
        }

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        generator = torch.Generator().manual_seed(self.step())
        raw_lengths = torch.tensor(
            [
                int(sample[2]) if len(sample) >= 3 else self.layout.raw_num_frames
                for sample in batch
            ],
            dtype=torch.long,
        )
        token_lengths = self.layout.valid_token_lengths(raw_lengths)
        if (token_lengths < 1).any():
            raise ValueError(
                "Every sample must contain at least one complete temporal patch"
            )
        valid_lengths = _valid_lengths(
            [(None, None, int(length)) for length in token_lengths.tolist()],
            self.layout.token_num_frames,
        )
        shortest = min(valid_lengths)

        # All prediction masks share one sampled shape because the predictor
        # concatenates them on the batch axis and therefore requires an equal
        # target-token count. Their positions and body-group identities remain
        # independently sampled.
        shape = (
            _block_length(
                shortest,
                _sample_ratio(generator, self.pred_frame_mask_ratio),
            ),
            _block_length(
                self.layout.token_num_joints,
                _sample_ratio(generator, self.body_mask_ratio),
            ),
        )
        shapes = [shape] * self.npred

        contexts_by_sample = []
        targets_by_sample = []
        for valid_length in valid_lengths:
            if sum(frames * groups for frames, groups in shapes) >= valid_length * 7:
                raise ValueError(
                    "random-body target masks leave no context cells; reduce "
                    "num_pred_masks or mask ratios"
                )

            for _union_attempt in range(256):
                targets = []
                target_union = torch.zeros(
                    self.layout.token_num_frames,
                    self.layout.token_num_joints,
                    dtype=torch.bool,
                )
                complete = True
                for frame_count, group_count in shapes:
                    for _target_attempt in range(256):
                        start = int(
                            torch.randint(
                                valid_length - frame_count + 1,
                                (),
                                generator=generator,
                            ).item()
                        )
                        groups = torch.randperm(
                            self.layout.token_num_joints, generator=generator
                        )[:group_count]
                        proposal = torch.zeros_like(target_union)
                        proposal[start : start + frame_count, groups] = True
                        if not (target_union & proposal).any():
                            targets.append(proposal)
                            target_union |= proposal
                            break
                    else:
                        complete = False
                        break
                if complete:
                    break
            else:
                raise ValueError(
                    "Could not sample cell-disjoint random-body target masks; "
                    "reduce num_pred_masks or mask ratios"
                )

            valid = (
                torch.arange(self.layout.token_num_frames)[:, None] < valid_length
            ).expand(-1, self.layout.token_num_joints)
            contexts_by_sample.append(valid & ~target_union)
            targets_by_sample.append(targets)

        contexts = [torch.stack(contexts_by_sample)]
        targets = [
            torch.stack([sample[index] for sample in targets_by_sample])
            for index in range(self.npred)
        ]
        return collated_batch, contexts, targets


class PatchRandomSpatialSegmentMaskCollator2D(_StatefulMaskCollator):
    """Mask overlapping temporal rectangles over an integer spatial-token count."""

    def __init__(
        self,
        raw_num_frames: int,
        raw_num_joints: int,
        token_num_joints: int,
        temporal_patch_size: int = 3,
        spatial_grouping: str = "coarse7",
        spatial_pooling: str = "graph_mean",
        pred_frame_mask_ratio: tuple[float, float] = (0.4, 0.6),
        pred_spatial_mask_count: int = 4,
        target_union_ratio: tuple[float, float] = (0.55, 0.65),
        npred: int = 4,
        max_sampling_attempts: int = 2048,
    ) -> None:
        super().__init__()
        if spatial_pooling != "graph_mean":
            raise ValueError(
                "random_spatial_segment supports only spatial_pooling='graph_mean'"
            )
        expected_body_groups = {
            "joint30": 30,
            "fine11": 11,
            "coarse7": 7,
        }
        if spatial_grouping not in expected_body_groups:
            choices = ", ".join(sorted(expected_body_groups))
            raise ValueError(
                f"Unknown spatial_grouping {spatial_grouping!r}; choose one of: {choices}"
            )
        expected_tokens = 1 + expected_body_groups[spatial_grouping]
        if int(raw_num_joints) != 30 or int(token_num_joints) != expected_tokens:
            raise ValueError(
                "random_spatial_segment requires SOMA30 with one trajectory token "
                f"plus {expected_body_groups[spatial_grouping]} body tokens"
            )
        self.layout = TokenLayout(
            kind="2d",
            patchified=True,
            raw_num_frames=int(raw_num_frames),
            token_num_frames=int(raw_num_frames) // int(temporal_patch_size),
            temporal_patch_size=int(temporal_patch_size),
            raw_num_joints=int(raw_num_joints),
            token_num_joints=int(token_num_joints),
        )
        self.spatial_grouping = str(spatial_grouping)
        self.spatial_pooling = str(spatial_pooling)
        self.pred_frame_mask_ratio = tuple(
            float(value) for value in pred_frame_mask_ratio
        )
        self.pred_spatial_mask_count = int(pred_spatial_mask_count)
        self.target_union_ratio = tuple(float(value) for value in target_union_ratio)
        self.npred = int(npred)
        self.max_sampling_attempts = int(max_sampling_attempts)
        _sample_ratio(torch.Generator().manual_seed(0), self.pred_frame_mask_ratio)
        _sample_ratio(torch.Generator().manual_seed(0), self.target_union_ratio)
        if not 0 < self.pred_spatial_mask_count <= self.layout.token_num_joints:
            raise ValueError(
                "pred_spatial_mask_count must be in [1, token_num_joints]"
            )
        if self.npred <= 0 or self.max_sampling_attempts <= 0:
            raise ValueError("npred and max_sampling_attempts must be positive")
        if self.target_union_ratio[0] <= 0.0 or self.target_union_ratio[1] >= 1.0:
            raise ValueError("target_union_ratio must leave both target and context cells")
        self._configuration = {
            "variant": "patch_2d_random_spatial_segment",
            "raw_num_frames": self.layout.raw_num_frames,
            "token_num_frames": self.layout.token_num_frames,
            "temporal_patch_size": self.layout.temporal_patch_size,
            "raw_num_joints": self.layout.raw_num_joints,
            "token_num_joints": self.layout.token_num_joints,
            "spatial_grouping": self.spatial_grouping,
            "spatial_pooling": self.spatial_pooling,
            "pred_frame_mask_ratio": self.pred_frame_mask_ratio,
            "pred_spatial_mask_count": self.pred_spatial_mask_count,
            "target_union_ratio": self.target_union_ratio,
            "npred": self.npred,
            "max_sampling_attempts": self.max_sampling_attempts,
        }

    def _target(
        self,
        valid_length: int,
        frame_count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        start = int(
            torch.randint(
                valid_length - frame_count + 1, (), generator=generator
            ).item()
        )
        spatial = torch.randperm(
            self.layout.token_num_joints, generator=generator
        )[: self.pred_spatial_mask_count]
        target = torch.zeros(
            self.layout.token_num_frames,
            self.layout.token_num_joints,
            dtype=torch.bool,
        )
        target[start : start + frame_count, spatial] = True
        return target

    def __call__(self, batch):
        collated_batch = torch.utils.data.default_collate(batch)
        generator = torch.Generator().manual_seed(self.step())
        raw_lengths = torch.tensor(
            [
                int(sample[2]) if len(sample) >= 3 else self.layout.raw_num_frames
                for sample in batch
            ],
            dtype=torch.long,
        )
        token_lengths = self.layout.valid_token_lengths(raw_lengths)
        if (token_lengths < 1).any():
            raise ValueError(
                "Every sample must contain at least one complete temporal patch"
            )
        valid_lengths = _valid_lengths(
            [(None, None, int(length)) for length in token_lengths.tolist()],
            self.layout.token_num_frames,
        )
        shortest = min(valid_lengths)
        frame_count = _block_length(
            shortest, _sample_ratio(generator, self.pred_frame_mask_ratio)
        )
        target_count = frame_count * self.pred_spatial_mask_count
        if target_count <= 0:
            raise ValueError("Target masks cannot be empty")

        contexts_by_sample: list[torch.Tensor] = []
        targets_by_sample: list[list[torch.Tensor]] = []
        lower, upper = self.target_union_ratio
        for valid_length in valid_lengths:
            valid_cells = valid_length * self.layout.token_num_joints
            sampled_targets = None
            target_union = None
            for _attempt in range(self.max_sampling_attempts):
                candidates = [
                    self._target(valid_length, frame_count, generator)
                    for _ in range(self.npred)
                ]
                union = torch.stack(candidates).any(dim=0)
                union_ratio = float(union.sum()) / float(valid_cells)
                if lower <= union_ratio <= upper:
                    sampled_targets = candidates
                    target_union = union
                    break
            if sampled_targets is None or target_union is None:
                raise ValueError(
                    "Could not sample random spatial target masks within "
                    f"target_union_ratio={self.target_union_ratio}; adjust frame "
                    "ratio, spatial count, or union bounds"
                )
            valid = (
                torch.arange(self.layout.token_num_frames)[:, None] < valid_length
            ).expand(-1, self.layout.token_num_joints)
            context = valid & ~target_union
            if not context.any():
                raise ValueError("Target union leaves no context cells")
            contexts_by_sample.append(context)
            targets_by_sample.append(sampled_targets)

        contexts = [torch.stack(contexts_by_sample)]
        targets = [
            torch.stack([sample[index] for sample in targets_by_sample])
            for index in range(self.npred)
        ]
        return collated_batch, contexts, targets


__all__ = [
    "COARSE7_GRAPH_EDGES",
    "COARSE7_GROUP_NAMES",
    "PatchBodyRegionSegmentMaskCollator2D",
    "PatchRandomBodySegmentMaskCollator2D",
    "PatchRandomSpatialSegmentMaskCollator2D",
]
