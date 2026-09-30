"""Fixed-bank, head-free motion retrieval and continuation diagnostics.

The past bank is causal, including the terminal velocity/contact channels.
Full-clip instance retrieval and future transfer deliberately use different
candidate sets: only the former contains the query's own full clip.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dataset.motion_dataset import MotionDataset
from experiment.online_metrics import feature_matrix_metrics
from motion_rep.feet import foot_detect_from_pos_and_vel
from motion_rep.geometry import y_rotation
from motion_rep.reps.motion_jepa_motionrep import MotionJEPAMotionRep as Rep
from skeleton import SOMASkeleton30


_VERSION = 1
_TIE_EPS = 1e-7


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _save_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def positions_from_features(raw: torch.Tensor) -> torch.Tensor:
    """Recover stored global positions without FK/rotation round-trip error."""
    root = raw[..., Rep.ROOT_POSITION]
    body = raw[..., Rep.LOCAL_POSITIONS].reshape(*raw.shape[:-1], 29, 3)
    # Stored body positions already contain root height, but not root X/Z.
    origin = root.clone()
    origin[..., 1] = 0
    return torch.cat((root.unsqueeze(-2), body + origin.unsqueeze(-2)), dim=-2)


def causal_past(raw: torch.Tensor, past_frames: int, fps: int, skeleton=None) -> torch.Tensor:
    """Crop before normalization and repair channels whose forward difference leaks."""
    if raw.ndim != 3 or raw.shape[-1] != Rep.FEATURE_DIM or not 2 <= past_frames <= raw.shape[1]:
        raise ValueError("Expected raw [B,T,366] motion and at least two past frames")
    past = raw[:, :past_frames].clone()
    positions = positions_from_features(past)
    terminal_velocity = float(fps) * (positions[:, -1] - positions[:, -2])
    past[:, -1, Rep.VELOCITIES] = terminal_velocity.flatten(-2)
    past[:, -1:, Rep.FOOT_CONTACTS] = foot_detect_from_pos_and_vel(
        positions[:, -1:], terminal_velocity[:, None], skeleton or SOMASkeleton30(), .15, .10
    )
    return past


def retrieval_metrics(scores: torch.Tensor, positive_indices: torch.Tensor) -> dict[str, float]:
    """Expected R@k/MRR under uniform ordering within ties, not index order."""
    scores = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
    positives = torch.as_tensor(positive_indices, dtype=torch.long, device="cpu")
    if scores.ndim != 2 or scores.shape[0] != len(positives) or not scores.numel():
        raise ValueError("Invalid retrieval score matrix")
    if not torch.isfinite(scores).all() or (positives < 0).any() or (positives >= scores.shape[1]).any():
        raise ValueError("Retrieval scores/positives are invalid")
    positive = scores[torch.arange(len(scores)), positives, None]
    better = (scores > positive + _TIE_EPS).sum(1)
    tied = ((scores - positive).abs() <= _TIE_EPS).sum(1)
    output = {}
    for k in (1, 5):
        output[f"recall_at_{k}"] = float(((k - better).double() / tied).clamp(0, 1).mean())
    output["mrr"] = sum(
        float((1.0 / torch.arange(int(b) + 1, int(b + n) + 1, dtype=torch.float64)).mean())
        for b, n in zip(better, tied)
    ) / len(scores)
    return output


def _select_top1(scores: torch.Tensor, seed: int) -> torch.Tensor:
    """Deterministic uniform tie breaking; never prefer a candidate's ID."""
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for row in scores.cpu():
        candidates = torch.nonzero((row - row.max()).abs() <= _TIE_EPS).flatten()
        if not len(candidates):
            raise ValueError("No finite future-transfer candidate")
        selected.append(candidates[torch.randint(len(candidates), (1,), generator=generator)].item())
    return torch.tensor(selected, dtype=torch.long)


def aligned_future(query: torch.Tensor, donor: torch.Tensor, past_frames: int) -> torch.Tensor:
    """Align donor positions to query's observed boundary root/yaw (never future)."""
    boundary = past_frames - 1
    q_heading = query[:, boundary, Rep.ROOT_HEADING]
    d_heading = donor[:, boundary, Rep.ROOT_HEADING]
    angle = torch.atan2(q_heading[:, 1], q_heading[:, 0]) - torch.atan2(d_heading[:, 1], d_heading[:, 0])
    rotation = y_rotation(angle)
    positions = positions_from_features(donor[:, past_frames:])
    donor_origin = donor[:, boundary, Rep.ROOT_POSITION]
    query_origin = query[:, boundary, Rep.ROOT_POSITION]
    return (positions - donor_origin[:, None, None]) @ rotation.transpose(-2, -1)[:, None] + query_origin[:, None, None]


def trajectory_errors(prediction: torch.Tensor, truth: torch.Tensor, fps: int,
                      horizons: list[float]) -> dict[str, dict[str, float]]:
    """Cumulative root XZ ADE, endpoint FDE, and root-relative joint error."""
    if prediction.shape != truth.shape or prediction.ndim != 4 or prediction.shape[-2:] != (30, 3):
        raise ValueError("Expected equal [B,T,30,3] trajectory tensors")
    root_error = torch.linalg.vector_norm((prediction[:, :, 0] - truth[:, :, 0])[..., [0, 2]], dim=-1)
    p_body = prediction[:, :, 1:] - prediction[:, :, :1]
    t_body = truth[:, :, 1:] - truth[:, :, :1]
    body_error = torch.linalg.vector_norm(p_body - t_body, dim=-1).mean(-1)
    output = {}
    for horizon in horizons:
        count = int(round(horizon * fps))
        if not 1 <= count <= truth.shape[1]:
            raise ValueError("Requested horizon exceeds the available future")
        name = f"horizon_{horizon:g}s".replace(".", "_")
        output[name] = {
            "root_ade_xz": float(root_error[:, :count].mean()),
            "root_fde_xz": float(root_error[:, count - 1].mean()),
            "root_relative_joint_error": float(body_error[:, :count].mean()),
        }
    return output


def representation_metrics(features: torch.Tensor, temporal_std: float, temporal_variance: float) -> dict:
    values = features.double()
    std = values.std(dim=0, correction=0)
    standardized = (values - values.mean(0)) / std.clamp_min(1e-6)
    return {
        "raw": feature_matrix_metrics(values),
        "standardized": feature_matrix_metrics(standardized),
        "temporal_mean_std": float(temporal_std),
        "temporal_mean_variance": float(temporal_variance),
        "near_zero_channels": int((std < 1e-6).sum()),
    }


def tensorboard_metrics(summary: dict) -> dict[str, float]:
    """Explicit scalar groups; provenance/counts never become TB curves."""
    output = {}
    for group in ("retrieval", "future_transfer"):
        def collect(path, value):
            if isinstance(value, dict):
                for key, child in value.items():
                    collect(f"{path}/{key}", child)
            elif (isinstance(value, (int, float)) and math.isfinite(value)
                  and path.rsplit("/", 1)[-1] in {
                      "recall_at_1", "recall_at_5", "mrr", "root_ade_xz", "root_fde_xz",
                      "root_relative_joint_error", "root_ade_xz_absolute_gain", "root_ade_xz_relative_gain",
                      "root_fde_xz_absolute_gain", "root_fde_xz_relative_gain",
                      "root_relative_joint_error_absolute_gain", "root_relative_joint_error_relative_gain",
                  }):
                output[path] = float(value)
        if group == "future_transfer":
            for kind in ("feature", "gains"):
                collect(f"{group}/{kind}", summary.get(group, {}).get(kind, {}))
        else:
            collect(group, summary.get(group, {}))
    geometry = summary.get("representation", {})
    for kind in ("raw", "standardized"):
        values = geometry.get(kind, {})
        for name in ("rankme", "mean_off_diagonal_cosine"):
            if name in values:
                output[f"representation/{kind}/{name}"] = float(values[name])
        if kind == "raw" and "mean_std" in values:
            output["representation/raw/mean_std"] = float(values["mean_std"])
        for name in ("effective_rank", "largest_eigenvalue_ratio"):
            if name in values.get("covariance", {}):
                output[f"representation/{kind}/covariance/{name}"] = float(values["covariance"][name])
    for name in ("temporal_mean_std", "temporal_mean_variance", "near_zero_channels"):
        if name in geometry:
            output[f"representation/{name}"] = float(geometry[name])
    if "elapsed_seconds" in summary:
        output["elapsed_seconds"] = float(summary["elapsed_seconds"])
    return {key: value for key, value in output.items() if math.isfinite(value)}


class MotionOnlineMetrics:
    """One immutable source bank; every evaluation refreshes all EMA features."""

    def __init__(self, training_config: dict, metric_config: dict, *, device: torch.device):
        data = training_config["data"]
        self.device = torch.device(device)
        self.root = Path(data["root_path"]).expanduser().resolve()
        self.output = Path(training_config["logging"]["folder"]) / "online_metrics"
        self.batch_size = int(metric_config.get("batch_size", 128))
        self.seed = int(metric_config.get("seed", 42))
        self.num_queries = int(metric_config.get("num_queries", 128))
        self.num_gallery = int(metric_config.get("num_gallery", 512))
        self.retrieval_gallery_size = int(metric_config.get("retrieval_gallery_size", 512))
        self.past_frames = int(metric_config.get("past_frames", 60))
        self.horizons = [float(x) for x in metric_config.get("horizons_seconds", [.5, 1., 2.])]
        self.calibration_samples = int(metric_config.get("calibration_samples", 512))
        self.num_frames, self.fps = int(data["num_frames"]), int(data["fps"])
        self.patch_size = int(training_config.get("patch", {}).get("temporal_patch_size", 1))
        self.use_bfloat16 = bool(training_config.get("meta", {}).get("use_bfloat16", False))
        if (min(self.batch_size, self.num_queries, self.num_gallery, self.calibration_samples) <= 0
                or not 2 <= self.past_frames <= self.num_frames
                or self.past_frames % self.patch_size or not self.horizons
                or any(not math.isfinite(h) or h <= 0 or abs(h * self.fps - round(h * self.fps)) > 1e-6 for h in self.horizons)
                or self.past_frames + round(max(self.horizons) * self.fps) > self.num_frames
                or not self.num_queries <= self.retrieval_gallery_size <= self.num_queries + self.num_gallery
                or int(data["motion_dim"]) != 366):
            raise ValueError("Invalid fixed-bank motion metric dimensions/counts/horizons")
        self.future_frames = int(round(max(self.horizons) * self.fps))
        self.skeleton = SOMASkeleton30()
        self.dataset = MotionDataset(
            self.root, ["val.txt"], self.num_frames, self.fps, motion_dim=366,
            normalize=True, stats_path=data.get("stats_path"),
        )
        if not bool(data.get("normalize", False)):
            raise ValueError("Motion metrics require the BONES train-stat normalized training input")
        self.mean = torch.from_numpy(self.dataset.mean.copy())
        self.std = torch.from_numpy(self.dataset.std.copy())
        records = json.loads((self.root / "index.json").read_text())
        validation_ids = {entry.sample_id for entry in self.dataset.entries}
        generator = np.random.default_rng(self.seed)
        self.records = self._select_sources(records, "val", self.num_queries + self.num_gallery,
                                            generator, validation_ids)
        calibration = self._select_sources(records, "train", self.calibration_samples,
                                            np.random.default_rng(self.seed + 1))
        self.indices = list(range(len(self.records)))  # Logging/backward-compatible sample count.
        self.retrieval_indices = list(range(self.num_queries)) + list(
            range(self.num_queries, self.retrieval_gallery_size)
        )
        self.raw = self._load_records(self.records)
        self.past = causal_past(self.raw, self.past_frames, self.fps, self.skeleton)
        provenance_files = [self.root / name for name in ("meta.json", "index.json", "val.txt", "train.txt")]
        provenance_files += [self.dataset.stats_root / name for name in ("mean.npy", "std.npy")]
        provenance = {str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path): _file_hash(path)
                      for path in provenance_files}
        selected_files = {record["motion_path"]: _file_hash(self.root / record["motion_path"])
                          for record in self.records + calibration}
        protocol = {
            "version": _VERSION, "seed": self.seed, "past_frames": self.past_frames,
            "num_frames": self.num_frames, "fps": self.fps, "patch_size": self.patch_size,
            "horizons_seconds": self.horizons, "num_queries": self.num_queries,
            "num_gallery": self.num_gallery, "retrieval_gallery_size": self.retrieval_gallery_size,
            "calibration_samples": self.calibration_samples, "use_bfloat16": self.use_bfloat16,
            "records": [{key: r[key] for key in ("id", "source_id", "motion_path", "length", "fps")}
                        | {"actor": r["metadata"]["take_actor"]} for r in self.records],
            "calibration_ids": [r["id"] for r in calibration],
            "future_gallery_indices": list(range(self.num_queries, len(self.records))),
            "retrieval_gallery_indices": self.retrieval_indices,
            "provenance": provenance, "motion_hashes": selected_files,
            "distance": "pooled_l2_cosine", "baseline_scaling": "train_channel_population_std_floor_1e-6",
            "ties": "expected_uniform_retrieval_seeded_uniform_transfer",
            "root_error_axes": "xz", "geometry_standardization": "same_fixed_bank_population_zscore",
            "source_length_policy": "exact_full_num_frames",
            "temporal_variance": "mean_over_samples_channels_of_valid_token_population_variance",
        }
        self.protocol_hash = _json_hash(protocol)
        manifest_path = self.output / "manifest.json"
        baseline_path = self.output / "baselines.pt"
        self.output.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if saved.get("protocol_hash") != self.protocol_hash or saved.get("protocol") != protocol:
                raise ValueError("Motion metric protocol/data changed; use a new output directory")
            if not baseline_path.exists() or saved.get("baseline_sha256") != _file_hash(baseline_path):
                raise ValueError("Motion metric baseline artifact is missing or changed")
            payload = torch.load(baseline_path, map_location="cpu", weights_only=True)
            if payload.get("protocol_hash") != self.protocol_hash:
                raise ValueError("Motion metric baseline protocol mismatch")
            self.baselines = payload["metrics"]
        else:
            calibration_raw = self._load_records(calibration)
            payload = self._build_baselines(causal_past(calibration_raw, self.past_frames, self.fps, self.skeleton))
            payload["protocol_hash"] = self.protocol_hash
            temporary = baseline_path.with_suffix(".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, baseline_path)
            self.baselines = payload["metrics"]
            _save_json({"protocol_hash": self.protocol_hash, "protocol": protocol,
                        "baseline_sha256": _file_hash(baseline_path)}, manifest_path)
            _save_json({"protocol_hash": self.protocol_hash, "metrics": self.baselines}, self.output / "baselines.json")

    def _select_sources(self, records, split, count, generator, allowed_ids=None):
        groups = {}
        for record in records:
            meta = record.get("metadata", {})
            if (record.get("split") != split or str(meta.get("is_mirror")).lower() != "false"
                    or not meta.get("take_actor") or not record.get("source_id")
                    or int(record["length"]) != self.num_frames
                    or float(record["fps"]) != self.fps
                    or (allowed_ids is not None and record["id"] not in allowed_ids)):
                continue
            groups.setdefault(record["source_id"], []).append(record)
        if len(groups) < count:
            raise ValueError(f"Need {count} eligible non-mirrored {split} sources; found {len(groups)}")
        source_ids = sorted(groups)
        selected = []
        for index in generator.permutation(len(source_ids))[:count]:
            clips = sorted(groups[source_ids[index]], key=lambda row: row["id"])
            selected.append(clips[int(generator.integers(len(clips)))])
        return selected

    def _load_records(self, records):
        values = []
        for record in records:
            value = np.load(self.root / record["motion_path"], allow_pickle=False)
            if (value.dtype != np.float32 or value.shape != (record["length"], 366)
                    or not np.isfinite(value).all() or len(value) > self.num_frames):
                raise ValueError(f"Invalid fixed-bank motion: {record['id']}")
            padded = torch.zeros(self.num_frames, 366)
            padded[:len(value)] = torch.from_numpy(value.copy())
            values.append(padded)
        return torch.stack(values)

    def _descriptors(self, past):
        positions = positions_from_features(past)
        root = positions[:, :, :1]
        heading = past[:, -1, Rep.ROOT_HEADING]
        rotate = y_rotation(-torch.atan2(heading[:, 1], heading[:, 0]))
        body = (positions - root) @ rotate.transpose(-2, -1)[:, None]
        velocities = past[:, -1, Rep.VELOCITIES].reshape(-1, 30, 3) @ rotate.transpose(-2, -1)
        pose_velocity = torch.cat((body[:, -1, 1:].flatten(1), velocities.flatten(1)), dim=1)
        # Ten evenly spaced poses from the observed last second; fixed for all runs.
        start = max(0, self.past_frames - self.fps)
        times = torch.linspace(start, self.past_frames - 1, 10).round().long()
        past_pose = body[:, times, 1:].flatten(1)
        return {"pose_velocity": pose_velocity, "past_pose": past_pose}

    def _transfer(self, donor_indices):
        query = self.raw[:self.num_queries]
        donor = self.raw[donor_indices]
        prediction = aligned_future(query, donor, self.past_frames)[:, :self.future_frames]
        truth = positions_from_features(query[:, self.past_frames:self.past_frames + self.future_frames])
        return trajectory_errors(prediction, truth, self.fps, self.horizons)

    def _build_baselines(self, calibration_past):
        calibration = self._descriptors(calibration_past)
        descriptors = self._descriptors(self.past)
        scalings, metrics, donors = {}, {}, {}
        for name, values in descriptors.items():
            mean = calibration[name].double().mean(0)
            std = calibration[name].double().std(0, correction=0).clamp_min(1e-6)
            normalized = (values.double() - mean) / std
            score = -torch.cdist(normalized[:self.num_queries], normalized[self.num_queries:]).square()
            chosen = _select_top1(score, self.seed) + self.num_queries
            metrics[name] = self._transfer(chosen)
            scalings[name] = {"mean": mean, "std": std}
            donors[name] = [self.records[index]["id"] for index in chosen.tolist()]
        past_positions = positions_from_features(self.past[:self.num_queries])
        terminal_velocity = (past_positions[:, -1] - past_positions[:, -2]) * self.fps
        times = torch.arange(1, self.future_frames + 1).float() / self.fps
        prediction = past_positions[:, -1:, :, :] + terminal_velocity[:, None] * times[None, :, None, None]
        truth = positions_from_features(self.raw[:self.num_queries, self.past_frames:self.past_frames + self.future_frames])
        metrics["constant_velocity"] = trajectory_errors(prediction, truth, self.fps, self.horizons)
        return {"metrics": metrics, "scaling": scalings, "donor_ids": donors}

    @torch.inference_mode()
    def _encode(self, encoder, values, lengths):
        pooled, variation, variances = [], [], []
        for start in range(0, len(values), self.batch_size):
            raw = values[start:start + self.batch_size]
            length = lengths[start:start + self.batch_size].to(self.device)
            motion = torch.zeros(len(raw), self.num_frames, 366)
            normalized = (raw - self.mean) / self.std
            motion[:, :raw.shape[1]] = normalized
            active = torch.arange(self.num_frames)[None] < length.cpu()[:, None]
            motion *= active[..., None]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.use_bfloat16 and self.device.type in {"cpu", "cuda"}):
                features = encoder(motion.to(self.device),
                                   torch.full((len(raw),), self.fps, device=self.device, dtype=torch.float32),
                                   valid_frames=active.to(self.device))
            if features.ndim != 3:
                raise ValueError("MotionOnlineMetrics requires 1D token features [B,T,D]")
            token_lengths = encoder.token_layout.valid_token_lengths(length).cpu()
            for sample, count in zip(features.float().cpu(), token_lengths.tolist()):
                if count < 1:
                    raise ValueError("Cannot pool an empty motion")
                sample = sample[:count]
                pooled.append(sample.mean(0))
                variation.append(float(sample.std(0, correction=0).mean()))
                variances.append(float(sample.var(0, correction=0).mean()))
        return torch.stack(pooled), sum(variation) / len(variation), sum(variances) / len(variances)

    @torch.inference_mode()
    def evaluate(self, encoder) -> dict:
        if encoder.training or encoder.token_layout.kind != "1d":
            raise ValueError("MotionOnlineMetrics requires an eval-mode 1D EMA encoder")
        if (encoder.token_layout.raw_num_frames != self.num_frames
                or encoder.token_layout.temporal_patch_size != self.patch_size):
            raise ValueError("Encoder token layout differs from the fixed metric protocol")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        full, temporal_std, temporal_variance = self._encode(
            encoder, self.raw, torch.tensor([r["length"] for r in self.records]))
        past, _, _ = self._encode(encoder, self.past, torch.full((len(self.records),), self.past_frames))
        full_normalized, past_normalized = F.normalize(full, dim=1), F.normalize(past, dim=1)
        retrieval = retrieval_metrics(past_normalized[:self.num_queries] @ full_normalized[self.retrieval_indices].T,
                                      torch.arange(self.num_queries))
        scores = past_normalized[:self.num_queries] @ past_normalized[self.num_queries:].T
        donors = _select_top1(scores, self.seed) + self.num_queries
        feature_transfer = self._transfer(donors)
        gains = {}
        for baseline, by_horizon in self.baselines.items():
            gains[baseline] = {}
            for horizon, values in by_horizon.items():
                gains[baseline][horizon] = {}
                for metric, reference in values.items():
                    error = feature_transfer[horizon][metric]
                    gains[baseline][horizon][metric + "_absolute_gain"] = reference - error
                    gains[baseline][horizon][metric + "_relative_gain"] = (reference - error) / reference if reference > 1e-12 else None
        representation = representation_metrics(full, temporal_std, temporal_variance)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "protocol_hash": self.protocol_hash,
            "retrieval": retrieval,
            "future_transfer": {"feature": feature_transfer, "gains": gains},
            "representation": representation,
            "elapsed_seconds": time.perf_counter() - start,
            "metadata": {
                "num_sources": len(self.records), "num_queries": self.num_queries,
                "future_candidate_count": self.num_gallery, "retrieval_candidate_count": self.retrieval_gallery_size,
                "feature_dim": int(full.shape[1]), "geometry_rank_ceiling": min(len(full) - 1, full.shape[1]),
                "future_donor_ids": [self.records[index]["id"] for index in donors.tolist()],
                "position_units": "meters", "retrieval_task": "past_to_own_full_clip",
                "geometry_normalization": "same_bank_population_zscore", "std_floor": 1e-6,
                "baseline_artifact": str(self.output / "baselines.json"),
            },
        }


__all__ = ["MotionOnlineMetrics", "causal_past", "positions_from_features", "aligned_future",
           "trajectory_errors", "retrieval_metrics", "representation_metrics", "tensorboard_metrics"]
