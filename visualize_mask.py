#!/usr/bin/env python3
"""Render Motion-JEPA raw or patchified 1D/2D masks as a PNG."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from helper import architecture_signature_from_config
from model import TokenLayout
from skeleton import SOMASkeleton30
from train import _build_mask_collator


BACKGROUND = (248, 249, 251)
GRID = (207, 212, 220)
TEXT = (28, 32, 38)
MUTED = (100, 108, 120)
EMPTY = (230, 233, 238)
PADDED = (198, 203, 211)
CONTEXT_COLORS = ((45, 112, 214), (44, 150, 190), (62, 132, 105))
TARGET_COLORS = (
    (225, 72, 72),
    (235, 145, 38),
    (137, 87, 214),
    (34, 164, 122),
    (207, 74, 151),
    (68, 157, 205),
)


@dataclass(frozen=True)
class SampledMasks:
    model_name: str
    layout: TokenLayout
    valid_raw_frames: int
    valid_token_frames: int
    allow_overlap: bool
    contexts: tuple[np.ndarray, ...]
    targets: tuple[np.ndarray, ...]
    spatial_labels: tuple[str, ...]


def _font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _layout_from_config(config: dict[str, Any]) -> tuple[TokenLayout, dict[str, Any]]:
    architecture = architecture_signature_from_config(config)
    layout = TokenLayout(**architecture["encoder_layout"])
    return layout, architecture


def _mask_to_numpy(
    mask: torch.Tensor,
    *,
    layout: TokenLayout,
    sample_index: int,
) -> np.ndarray:
    if layout.kind == "2d":
        expected = (
            layout.token_num_frames,
            int(layout.token_num_joints),
        )
        selected = mask[sample_index].detach().cpu().numpy().astype(bool)
        if selected.shape != expected:
            raise ValueError(
                f"Expected a 2D mask with shape {expected}, got {selected.shape}"
            )
        return selected
    selected = mask[sample_index].detach().cpu().numpy().astype(np.int64)
    result = np.zeros(layout.token_num_frames, dtype=bool)
    result[selected] = True
    return result


def sample_masks(
    config: dict[str, Any],
    *,
    seed: int = 0,
    valid_length: int | None = None,
    batch_size: int = 1,
    sample_index: int = 0,
    allow_overlap: bool | None = None,
) -> SampledMasks:
    """Sample masks through the same collator construction used by training."""
    config = copy.deepcopy(config)
    data = config["data"]
    raw_num_frames = int(data["num_frames"])
    valid_raw_frames = raw_num_frames if valid_length is None else int(valid_length)
    if not 1 <= valid_raw_frames <= raw_num_frames:
        raise ValueError(f"valid_length must be in [1, {raw_num_frames}]")
    if batch_size <= 0 or not 0 <= sample_index < batch_size:
        raise ValueError("sample_index must select an item in batch_size")
    if allow_overlap is not None:
        if "allow_target_overlap" in config["mask"]:
            config["mask"]["allow_target_overlap"] = bool(allow_overlap)
        else:
            config["mask"]["allow_overlap"] = bool(allow_overlap)

    layout, architecture = _layout_from_config(config)
    collator = _build_mask_collator(config, layout)
    state = collator.state_dict()
    state["counter"] = int(seed) - 1
    collator.load_state_dict(state)
    motion_dim = int(data["motion_dim"])
    fps = float(data.get("fps", 60))
    batch = [
        (torch.zeros(raw_num_frames, motion_dim), torch.tensor(fps), valid_raw_frames)
        for _ in range(batch_size)
    ]
    _, context_masks, target_masks = collator(batch)
    spatial_labels: tuple[str, ...] = ()
    if layout.kind == "2d":
        spatial_patch = architecture.get("spatial_patch")
        if spatial_patch is None:
            spatial_labels = tuple(SOMASkeleton30(load=False).names)
        else:
            spatial_labels = tuple(
                spatial_patch.get("spatial_token_names", spatial_patch["group_names"])
            )
    valid_tokens = int(
        layout.valid_token_lengths(torch.tensor([valid_raw_frames]))[0]
    )
    return SampledMasks(
        model_name=str(config["meta"]["model_name"]),
        layout=layout,
        valid_raw_frames=valid_raw_frames,
        valid_token_frames=valid_tokens,
        allow_overlap=bool(
            config["mask"].get(
                "allow_target_overlap", config["mask"].get("allow_overlap", False)
            )
        ),
        contexts=tuple(
            _mask_to_numpy(mask, layout=layout, sample_index=sample_index)
            for mask in context_masks
        ),
        targets=tuple(
            _mask_to_numpy(mask, layout=layout, sample_index=sample_index)
            for mask in target_masks
        ),
        spatial_labels=spatial_labels,
    )


def mask_statistics(sampled: SampledMasks) -> dict[str, Any]:
    shape = sampled.contexts[0].shape
    valid = np.zeros(shape, dtype=bool)
    valid[: sampled.valid_token_frames] = True
    context_union = np.logical_or.reduce(sampled.contexts)
    target_union = np.logical_or.reduce(sampled.targets)
    target_count = np.stack(sampled.targets).sum(axis=0)
    return {
        "model_name": sampled.model_name,
        "kind": sampled.layout.kind,
        "patchified": sampled.layout.patchified,
        "raw_num_frames": sampled.layout.raw_num_frames,
        "token_num_frames": sampled.layout.token_num_frames,
        "temporal_patch_size": sampled.layout.temporal_patch_size,
        "raw_num_joints": sampled.layout.raw_num_joints,
        "token_num_joints": sampled.layout.token_num_joints,
        "valid_raw_frames": sampled.valid_raw_frames,
        "valid_token_frames": sampled.valid_token_frames,
        "context_cells": [int((mask & valid).sum()) for mask in sampled.contexts],
        "target_cells": [int((mask & valid).sum()) for mask in sampled.targets],
        "target_union_cells": int((target_union & valid).sum()),
        "context_target_overlap_cells": int(
            (context_union & target_union & valid).sum()
        ),
        "target_overlap_cells": int(((target_count >= 2) & valid).sum()),
        "max_target_overlap": int(target_count[valid].max(initial=0)),
    }


def _header(
    draw: ImageDraw.ImageDraw,
    sampled: SampledMasks,
    *,
    seed: int,
    left: int,
) -> None:
    layout = sampled.layout
    geometry = f"tokens={layout.token_num_frames}"
    if layout.kind == "2d":
        geometry += f"x{layout.token_num_joints}"
    draw.text((left, 22), "Motion-JEPA mask coverage", fill=TEXT, font=_font(28, True))
    draw.text(
        (left, 65),
        f"model={sampled.model_name}   layout={layout.kind}   "
        f"patchified={str(layout.patchified).lower()}   {geometry}   "
        f"raw_frames={layout.raw_num_frames}   p={layout.temporal_patch_size}",
        fill=MUTED,
        font=_font(16),
    )
    draw.text(
        (left, 94),
        f"valid_raw={sampled.valid_raw_frames}   "
        f"valid_tokens={sampled.valid_token_frames}   seed={seed}   "
        f"allow_overlap={str(sampled.allow_overlap).lower()}",
        fill=MUTED,
        font=_font(16),
    )


def _time_annotation(layout: TokenLayout, token: int) -> str:
    if layout.temporal_patch_size == 1:
        return str(token)
    start = token * layout.temporal_patch_size
    end = start + layout.temporal_patch_size - 1
    return f"{token}\n[{start}-{end}]"


def _render_1d(sampled: SampledMasks, *, seed: int) -> Image.Image:
    context_union = np.logical_or.reduce(sampled.contexts)
    target_union = np.logical_or.reduce(sampled.targets)
    target_count = np.stack(sampled.targets).sum(axis=0)
    rows: list[tuple[str, np.ndarray, tuple[int, int, int] | None]] = []
    rows.extend(
        (f"ENC {index + 1}", mask, CONTEXT_COLORS[index % len(CONTEXT_COLORS)])
        for index, mask in enumerate(sampled.contexts)
    )
    rows.extend(
        (f"PRED {index + 1}", mask, TARGET_COLORS[index % len(TARGET_COLORS)])
        for index, mask in enumerate(sampled.targets)
    )
    rows.extend(
        (("ENC union", context_union, CONTEXT_COLORS[0]),
         ("PRED union", target_union, (196, 55, 72)),
         ("PRED count", target_count, None))
    )
    width, left, right = 1800, 255, 185
    header, row_height, row_gap, footer = 135, 40, 9, 115
    height = header + len(rows) * (row_height + row_gap) + footer
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _header(draw, sampled, seed=seed, left=left)
    plot_width = width - left - right

    def bounds(token: int) -> tuple[int, int]:
        x0 = left + round(token / sampled.layout.token_num_frames * plot_width)
        x1 = left + round((token + 1) / sampled.layout.token_num_frames * plot_width)
        return x0, max(x0 + 1, x1)

    for row_index, (label, values, color) in enumerate(rows):
        y0 = header + row_index * (row_height + row_gap)
        y1 = y0 + row_height
        draw.text((left - 14, (y0 + y1) // 2), label, fill=TEXT, font=_font(17, True), anchor="rm")
        for token in range(sampled.layout.token_num_frames):
            x0, x1 = bounds(token)
            if token >= sampled.valid_token_frames:
                fill = PADDED
            elif color is not None:
                fill = color if bool(values[token]) else EMPTY
            else:
                count = int(values[token])
                fill = EMPTY if count == 0 else TARGET_COLORS[min(count - 1, 3)]
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=GRID)
        active = int(np.count_nonzero(values[: sampled.valid_token_frames]))
        draw.text(
            (left + plot_width + 12, (y0 + y1) // 2),
            f"{active} cells",
            fill=MUTED,
            font=_font(14),
            anchor="lm",
        )
    axis_y = header + len(rows) * (row_height + row_gap) + 4
    step = max(1, sampled.layout.token_num_frames // 9)
    ticks = list(range(0, sampled.layout.token_num_frames, step))
    if sampled.layout.token_num_frames - 1 not in ticks:
        ticks.append(sampled.layout.token_num_frames - 1)
    for token in ticks:
        x0, x1 = bounds(token)
        draw.text(
            ((x0 + x1) // 2, axis_y),
            _time_annotation(sampled.layout, token),
            fill=MUTED,
            font=_font(12),
            anchor="ma",
            spacing=2,
        )
    return image


def _render_2d(sampled: SampledMasks, *, seed: int) -> Image.Image:
    context_union = np.logical_or.reduce(sampled.contexts)
    target_union = np.logical_or.reduce(sampled.targets)
    target_count = np.stack(sampled.targets).sum(axis=0)
    panels: list[tuple[str, np.ndarray, tuple[int, int, int] | None]] = []
    panels.extend(
        (f"ENC {index + 1}", mask, CONTEXT_COLORS[index % len(CONTEXT_COLORS)])
        for index, mask in enumerate(sampled.contexts)
    )
    panels.extend(
        (f"PRED {index + 1}", mask, TARGET_COLORS[index % len(TARGET_COLORS)])
        for index, mask in enumerate(sampled.targets)
    )
    panels.extend(
        (("ENC union", context_union, CONTEXT_COLORS[0]),
         ("PRED union", target_union, (196, 55, 72)),
         ("PRED count", target_count, None))
    )
    frames = sampled.layout.token_num_frames
    joints = int(sampled.layout.token_num_joints)
    cell_width = max(6, min(80, 1260 // frames))
    cell_height = 14 if joints > 15 else 22
    left, right, header, panel_gap, footer = 235, 150, 135, 48, 105
    plot_width = frames * cell_width
    panel_height = joints * cell_height
    width = left + plot_width + right
    height = header + len(panels) * (panel_height + panel_gap) + footer
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _header(draw, sampled, seed=seed, left=left)

    for panel_index, (label, values, color) in enumerate(panels):
        top = header + panel_index * (panel_height + panel_gap)
        draw.text((left, top - 24), label, fill=TEXT, font=_font(17, True))
        for joint, spatial_label in enumerate(sampled.spatial_labels):
            y0 = top + joint * cell_height
            draw.text(
                (left - 10, y0 + cell_height // 2),
                spatial_label,
                fill=TEXT,
                font=_font(11 if joints > 15 else 13),
                anchor="rm",
            )
            for frame in range(frames):
                x0 = left + frame * cell_width
                if frame >= sampled.valid_token_frames:
                    fill = PADDED
                elif color is not None:
                    fill = color if bool(values[frame, joint]) else EMPTY
                else:
                    count = int(values[frame, joint])
                    fill = EMPTY if count == 0 else TARGET_COLORS[min(count - 1, 3)]
                draw.rectangle(
                    (x0, y0, x0 + cell_width, y0 + cell_height),
                    fill=fill,
                    outline=GRID,
                )
        valid_values = values[: sampled.valid_token_frames]
        draw.text(
            (left + plot_width + 12, top + panel_height // 2),
            f"{int(np.count_nonzero(valid_values))}\ncells",
            fill=MUTED,
            font=_font(13),
            anchor="lm",
        )

    axis_y = header + len(panels) * (panel_height + panel_gap) + 5
    step = max(1, frames // 9)
    ticks = list(range(0, frames, step))
    if frames - 1 not in ticks:
        ticks.append(frames - 1)
    for token in ticks:
        x = left + token * cell_width + cell_width // 2
        draw.text(
            (x, axis_y),
            _time_annotation(sampled.layout, token),
            fill=MUTED,
            font=_font(11),
            anchor="ma",
            spacing=2,
        )
    return image


def render_mask_png(output: Path, sampled: SampledMasks, *, seed: int = 0) -> dict[str, Any]:
    """Render sampled masks and return their machine-readable summary."""
    image = (
        _render_1d(sampled, seed=seed)
        if sampled.layout.kind == "1d"
        else _render_2d(sampled, seed=seed)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return mask_statistics(sampled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mjepa_patch_2d_tiny_coarse7.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/mask.png"))
    parser.add_argument("--seed", type=int, default=0, help="Mask collator step/seed.")
    parser.add_argument("--valid-length", type=int, help="Valid raw frames.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--allow-overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mask.allow_overlap from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sampled = sample_masks(
        config,
        seed=args.seed,
        valid_length=args.valid_length,
        batch_size=args.batch_size,
        sample_index=args.sample_index,
        allow_overlap=args.allow_overlap,
    )
    stats = render_mask_png(args.output, sampled, seed=args.seed)
    print(f"Saved mask visualization to {args.output.resolve()}")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
