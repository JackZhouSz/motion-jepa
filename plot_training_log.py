#!/usr/bin/env python3
"""Plot one or more MotionJEPA CSV training logs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


METRIC_ALIASES = {
    "loss": ("loss",),
    "learning_rate": ("learning_rate", "lr"),
    "weight_decay": ("weight_decay", "wd"),
    "time_ms": ("time_ms", "time (ms)", "time"),
    "mask_a": ("mask-A", "mask_a"),
    "mask_b": ("mask-B", "mask_b"),
}
METRIC_TITLES = {
    "loss": "Training loss",
    "learning_rate": "Learning rate",
    "weight_decay": "Weight decay",
    "time_ms": "Iteration time (ms)",
    "mask_a": "Encoder mask",
    "mask_b": "Predictor mask",
}


@dataclass(frozen=True)
class TrainingLog:
    path: Path
    label: str
    step: np.ndarray
    metrics: dict[str, np.ndarray]


def _first_present(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in fieldnames), None)


def _as_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def load_log(path: Path, label: str | None = None) -> TrainingLog:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f"Training log has no data rows: {path}")

    step_name = _first_present(fieldnames, ("global_step", "iteration", "itr"))
    if step_name is None:
        steps = np.arange(1, len(rows) + 1, dtype=np.float64)
    else:
        steps = np.asarray([_as_float(row.get(step_name)) for row in rows])

    # Appended logs can contain steps from an interrupted run more than once.
    # Keep the last occurrence, which represents the most recent run state.
    last_index: dict[float, int] = {}
    for index, step in enumerate(steps):
        if np.isfinite(step):
            last_index[float(step)] = index
    selected = np.asarray(sorted(last_index.values(), key=lambda index: steps[index]))
    if len(selected) == 0:
        raise ValueError(f"Training log has no valid steps: {path}")

    metrics: dict[str, np.ndarray] = {}
    for metric, aliases in METRIC_ALIASES.items():
        column = _first_present(fieldnames, aliases)
        if column is not None:
            metrics[metric] = np.asarray(
                [_as_float(rows[index].get(column)) for index in selected],
                dtype=np.float64,
            )
    if not metrics:
        raise ValueError(f"No supported metric columns in {path}: {fieldnames}")
    return TrainingLog(
        path=path,
        label=label or path.parent.name or path.stem,
        step=steps[selected],
        metrics=metrics,
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.copy()
    result = np.full_like(values, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0.0), np.ones(window), mode="full")[: len(values)]
    counts = np.convolve(finite.astype(np.float64), np.ones(window), mode="full")[: len(values)]
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def plot_logs(
    logs: list[TrainingLog],
    output: Path,
    smoothing: int,
    title: str | None,
    dpi: int,
) -> None:
    metrics = [
        metric
        for metric in METRIC_ALIASES
        if any(metric in training_log.metrics for training_log in logs)
    ]
    columns = 2 if len(metrics) > 1 else 1
    rows = math.ceil(len(metrics) / columns)
    panel_width, panel_height = 780, 420
    title_height = 55 if title else 20
    image = Image.new(
        "RGB",
        (panel_width * columns, title_height + panel_height * rows),
        "white",
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    palette = (
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    )
    if title:
        draw.text((image.width // 2, 18), title, fill="black", font=font, anchor="mm")

    for metric_index, metric in enumerate(metrics):
        panel_row, panel_column = divmod(metric_index, columns)
        panel_x = panel_column * panel_width
        panel_y = title_height + panel_row * panel_height
        left, top = panel_x + 82, panel_y + 42
        right, bottom = panel_x + panel_width - 25, panel_y + panel_height - 58
        series: list[tuple[TrainingLog, np.ndarray, tuple[int, int, int]]] = []
        for log_index, training_log in enumerate(logs):
            if metric not in training_log.metrics:
                continue
            values = training_log.metrics[metric]
            color = palette[log_index % len(palette)]
            series.append((training_log, values, color))
        finite_x = np.concatenate(
            [item.step[np.isfinite(item.step)] for item, _, _ in series]
        )
        finite_y = np.concatenate(
            [values[np.isfinite(values)] for _, values, _ in series]
        )
        x_min, x_max = float(finite_x.min()), float(finite_x.max())
        y_min, y_max = float(finite_y.min()), float(finite_y.max())
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0
        y_padding = (y_max - y_min) * 0.05
        y_min, y_max = y_min - y_padding, y_max + y_padding

        def points(step: np.ndarray, values: np.ndarray) -> list[tuple[int, int]]:
            valid = np.isfinite(step) & np.isfinite(values)
            return [
                (
                    int(left + (x - x_min) / (x_max - x_min) * (right - left)),
                    int(bottom - (y - y_min) / (y_max - y_min) * (bottom - top)),
                )
                for x, y in zip(step[valid], values[valid])
            ]

        draw.text(
            ((left + right) // 2, panel_y + 14),
            METRIC_TITLES[metric],
            fill="black",
            font=font,
            anchor="mm",
        )
        for tick in range(6):
            fraction = tick / 5
            y = int(bottom - fraction * (bottom - top))
            value = y_min + fraction * (y_max - y_min)
            draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
            draw.text((left - 8, y), f"{value:.3g}", fill=(70, 70, 70), font=font, anchor="rm")
            x = int(left + fraction * (right - left))
            step_value = x_min + fraction * (x_max - x_min)
            draw.line((x, top, x, bottom), fill=(238, 238, 238), width=1)
            draw.text((x, bottom + 10), f"{step_value:.4g}", fill=(70, 70, 70), font=font, anchor="ma")
        draw.rectangle((left, top, right, bottom), outline=(80, 80, 80), width=1)
        draw.text(((left + right) // 2, bottom + 36), "Global step", fill="black", font=font, anchor="mm")

        for training_log, raw_values, color in series:
            values = raw_values
            if metric in {"loss", "time_ms"} and smoothing > 1:
                raw_color = tuple(int(channel + (255 - channel) * 0.72) for channel in color)
                raw_points = points(training_log.step, values)
                if len(raw_points) > 1:
                    draw.line(raw_points, fill=raw_color, width=1)
                values = moving_average(values, smoothing)
            line_points = points(training_log.step, values)
            if len(line_points) > 1:
                draw.line(line_points, fill=color, width=3)
        if len(logs) > 1:
            legend_x, legend_y = left + 8, top + 8
            for training_log, _, color in series:
                draw.line((legend_x, legend_y + 5, legend_x + 22, legend_y + 5), fill=color, width=3)
                draw.text((legend_x + 28, legend_y), training_log.label, fill="black", font=font)
                legend_y += 17

    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep --dpi useful as PNG metadata while preserving deterministic pixel dimensions.
    image.save(output, dpi=(dpi, dpi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="+", help="Training CSV file(s).")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional label for each CSV, in the same order.",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path.")
    parser.add_argument(
        "--smoothing",
        type=int,
        default=100,
        help="Trailing moving-average window for noisy metrics (default: 100).",
    )
    parser.add_argument("--title", help="Optional figure title.")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.csv):
        raise ValueError("--labels must contain exactly one label per CSV file.")
    if args.smoothing < 1:
        raise ValueError("--smoothing must be at least 1.")
    labels = args.labels or [None] * len(args.csv)
    logs = [load_log(path.resolve(), label) for path, label in zip(args.csv, labels)]
    if args.output is not None:
        output = args.output.resolve()
    elif len(args.csv) == 1:
        output = args.csv[0].resolve().with_name(f"{args.csv[0].stem}-plots.png")
    else:
        output = Path("training-log-comparison.png").resolve()
    plot_logs(logs, output, args.smoothing, args.title, args.dpi)
    print(f"Saved training plots to {output}")


if __name__ == "__main__":
    main()
