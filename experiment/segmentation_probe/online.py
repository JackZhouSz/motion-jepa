"""Resumable linear frame probing of frozen EMA MotionJEPA tokens.

Feature extraction happens once per encoder checkpoint. Heads are always trained
and evaluated in FP32; only encoder extraction and the CPU token cache use BF16.
Every validation pass covers the complete split and one 120-class head supplies
both vocabularies' scores at the epoch selected by 120-class frame mAP.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.babel_segmentation import BabelSegmentationDataset
from helper import position_encoding_from_model
from experiment.linear_probe.features import (
    PROJECT_ROOT, _atomic_json_save, _atomic_torch_save, _seed_all,
    _sha256_file, resolve_pretraining_stats,
)
from .metrics import FrameMetricAccumulator
from .model import FrameLinearProbe, complete_patch_frame_mask


logger = logging.getLogger(__name__)
FORMAT_VERSION = 1
DEFAULT_DATASET = "dataset/babel-segmentation-120-processed-nframes30-150"


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _encoder_digest(encoder) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _matches_provenance(actual, expected: dict) -> bool:
    if not isinstance(actual, dict):
        return False
    # Existing caches predate configurable positions and always used absolute PE.
    normalized = {"position_encoding": {"temporal": "absolute"}, **actual}
    return normalized == expected


def _cache_digest(cache: dict) -> str:
    """Hash tensor bytes in bounded chunks, including labels and ordering."""
    digest = hashlib.sha256()
    for name in ("features", "labels", "supervised", "token_valid"):
        tensor = cache[name].detach().contiguous().cpu()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
        for begin in range(0, len(data), 8 * 1024 * 1024):
            digest.update(data[begin:begin + 8 * 1024 * 1024])
    digest.update(json.dumps({"sample_ids": cache["sample_ids"], "counts": cache["counts"]},
                             sort_keys=True).encode())
    return digest.hexdigest()


def _capture_rng() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_token_standardizer(cache: dict, *, batch_size: int = 256, epsilon: float = 1e-6) -> dict:
    """Merge FP64 moments over all complete train tokens, without a large cast."""
    count, mean, squared_deviation = 0, None, None
    for begin in range(0, len(cache["features"]), batch_size):
        values = cache["features"][begin:begin + batch_size]
        valid = cache["token_valid"][begin:begin + batch_size]
        values = values[valid].double()
        if not len(values):
            continue
        if not torch.isfinite(values).all():
            raise ValueError("Train token features must be finite")
        batch_mean = values.mean(0)
        batch_m2 = (values - batch_mean).square().sum(0)
        if mean is None:
            mean, squared_deviation = batch_mean, batch_m2
            count = len(values)
        else:
            total = count + len(values)
            delta = batch_mean - mean
            squared_deviation += batch_m2 + delta.square() * (count * len(values) / total)
            mean += delta * (len(values) / total)
            count = total
    if count == 0:
        raise ValueError("No complete train tokens available for standardization")
    return {"mean": mean.float(),
            "scale": (squared_deviation / count).clamp_min(0).sqrt().clamp_min(epsilon).float(),
            "epsilon": epsilon, "fit_split": "train", "fit_tokens": count,
            "fit_policy": "all_complete_raw_valid_tokens", "correction": 0}


def _batch(cache: dict, indices: torch.Tensor, normalizer: dict, device):
    features = cache["features"][indices].to(device=device, dtype=torch.float32)
    features = (features - normalizer["mean"]) / normalizer["scale"]
    return (features, cache["labels"][indices].to(device=device, dtype=torch.float32),
            cache["supervised"][indices].to(device=device))


def train_head_epoch(head, cache, normalizer, optimizer, *, device, batch_size, generator) -> float:
    """Shuffle windows, retaining their valid supervised frame weights."""
    head.train()
    order = torch.randperm(len(cache["features"]), generator=generator)
    loss_sum, frame_count = 0.0, 0
    for begin in range(0, len(order), batch_size):
        features, labels, supervised = _batch(cache, order[begin:begin + batch_size], normalizer, device)
        count = int(supervised.sum())
        if count == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        # Explicitly override an enclosing training autocast context.
        with torch.autocast(device_type=device.type, enabled=False):
            logits = head(features)
            loss = F.binary_cross_entropy_with_logits(logits[supervised], labels[supervised])
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite segmentation probe loss")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * count
        frame_count += count
    if frame_count == 0:
        raise ValueError("No supervised train frames")
    return loss_sum / frame_count


@torch.no_grad()
def evaluate_head(head, cache, normalizer, *, device, batch_size, class_indices_60) -> dict:
    head.eval()
    accumulator = FrameMetricAccumulator(head.num_classes, class_indices_60, device=device)
    for begin in range(0, len(cache["features"]), batch_size):
        indices = torch.arange(begin, min(begin + batch_size, len(cache["features"])))
        features, labels, supervised = _batch(cache, indices, normalizer, device)
        with torch.autocast(device_type=device.type, enabled=False):
            accumulator.update(head(features), labels, supervised)
    return accumulator.compute()


class OnlineSegmentationProbe:
    """Fresh token-linear BABEL head, with on-disk epoch-boundary resume."""

    def __init__(self, training_config: dict, probe_config: dict, *, device):
        self.device = torch.device(device)
        self.epochs = int(probe_config.get("epochs", 50))
        self.learning_rate = float(probe_config.get("lr", 0.3))
        self.momentum = float(probe_config.get("momentum", 0.9))
        self.weight_decay = float(probe_config.get("weight_decay", 0.0))
        self.batch_size = int(probe_config.get("batch_size", 256))
        self.feature_batch_size = int(probe_config.get("feature_batch_size", 256))
        self.num_workers = int(probe_config.get("num_workers", 0))
        self.seed = int(probe_config.get("seed", 42))
        if min(self.epochs, self.batch_size, self.feature_batch_size) <= 0 or self.num_workers < 0:
            raise ValueError("Probe epochs/batches must be positive and workers nonnegative")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1 or self.weight_decay < 0:
            raise ValueError("Invalid segmentation SGD hyperparameters")
        if not bool(probe_config.get("standardize", True)):
            raise ValueError("The segmentation protocol requires train-token standardization")
        data = training_config["data"]
        self.num_frames, self.fps = int(data["num_frames"]), int(data["fps"])
        self.use_bfloat16 = bool(training_config.get("meta", {}).get("use_bfloat16", False))
        root = Path(str(probe_config.get("dataset_root", DEFAULT_DATASET))).expanduser()
        self.dataset_root = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
        self.stats_root = resolve_pretraining_stats(training_config, None)
        output = Path(str(probe_config.get("output_root",
            Path(training_config["logging"]["folder"]) / "segmentation_probe"))).expanduser()
        self.output_root = (output if output.is_absolute() else PROJECT_ROOT / output).resolve()
        self.datasets = {
            split: BabelSegmentationDataset(self.dataset_root, split,
                num_frames=self.num_frames, fps=self.fps, motion_dim=int(data["motion_dim"]),
                stats_root=self.stats_root, normalize=True)
            for split in ("train", "val")
        }
        self.class_names = tuple(self.datasets["train"].class_names)
        self.class_indices_60 = tuple(self.datasets["train"].class_indices_60)
        if len(self.class_names) != 120 or len(self.class_indices_60) != 60:
            raise ValueError("Segmentation protocol requires 120 classes and a named 60-class subset")
        if any(not len(dataset) for dataset in self.datasets.values()):
            raise ValueError("Segmentation requires nonempty train and validation splits")
        if tuple(self.datasets["val"].class_names) != self.class_names:
            raise ValueError("Train/validation class vocabularies differ")
        # Index/metadata include annotation/source provenance from preprocessing.
        dataset_hashes = {name: _sha256_file(self.dataset_root / name)
            for name in ("meta.json", "index.json", "class-index.json", "train.txt", "val.txt")
            if (self.dataset_root / name).is_file()}
        if not dataset_hashes:
            raise ValueError("Segmentation dataset has no provenance metadata")
        self.protocol = {
            "format_version": FORMAT_VERSION,
            "task": "frame_multilabel_segmentation", "head": "token_local_phase_affine",
            "ap": "threshold_grouped_noninterpolated_v1", "selection": "babel-120/frame_map",
            "classes": self.class_names, "class_indices_60": self.class_indices_60,
            "supervision": "positive_120_frames_and_complete_raw_patches",
            "subset_frame_mask": "same_as_120", "standardization": "all_complete_train_tokens_zscore",
            "dataset_root": str(self.dataset_root), "dataset_hashes": dataset_hashes,
            "stats_hashes": {name: _sha256_file(self.stats_root / name) for name in ("mean.npy", "std.npy")},
            "epochs": self.epochs, "lr": self.learning_rate, "momentum": self.momentum,
            "weight_decay": self.weight_decay, "batch_size": self.batch_size,
            "feature_batch_size": self.feature_batch_size, "seed": self.seed,
            "num_frames": self.num_frames, "fps": self.fps, "motion_dim": int(data["motion_dim"]),
            "encoder_bfloat16": self.use_bfloat16, "cache_dtype": "bfloat16", "head_dtype": "float32",
            "scheduler": "CosineAnnealingLR_eta_min_0", "encoder": training_config.get("meta", {}).get("model_name"),
            "patch": training_config.get("patch", {}),
        }
        self.protocol_hash = _digest(self.protocol)

    @torch.no_grad()
    def _extract(self, encoder, dataset) -> dict:
        layout = encoder.token_layout
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(dataset, batch_size=self.feature_batch_size, shuffle=False,
                            num_workers=self.num_workers, generator=generator,
                            pin_memory=self.device.type == "cuda")
        n, tokens, dim = len(dataset), layout.token_num_frames, int(encoder.embed_dim)
        frames = tokens * layout.temporal_patch_size
        cache = {"features": torch.empty(n, tokens, dim, dtype=torch.bfloat16),
                 "labels": torch.zeros(n, frames, len(self.class_names), dtype=torch.bool),
                 "supervised": torch.zeros(n, frames, dtype=torch.bool),
                 "token_valid": torch.zeros(n, tokens, dtype=torch.bool),
                 "sample_ids": []}
        offset, raw_frames, annotated_before_patch = 0, 0, 0
        for motion, fps, length, labels, supervised, sample_ids in loader:
            size = len(motion)
            length = length.long()
            valid_frames = torch.arange(self.num_frames)[None] < length[:, None]
            token_valid = layout.valid_token_mask(valid_frames)
            if not token_valid.any(dim=1).all():
                raise ValueError("Every segmentation window must contain a complete encoder patch")
            mask = complete_patch_frame_mask(length, layout)
            labels = labels[:, :frames]
            supervised = supervised[:, :frames].bool() & labels.bool().any(-1)
            annotated_before_patch += int(supervised.sum())
            supervised &= mask
            raw_frames += int(length.sum())
            autocast = (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                        if self.use_bfloat16 else nullcontext())
            with autocast:
                features = encoder(motion.to(self.device, dtype=torch.float32),
                                   fps.to(self.device, dtype=torch.float32),
                                   valid_frames=valid_frames.to(self.device))
            if features.shape != (size, tokens, dim) or not torch.isfinite(features).all():
                raise ValueError("Expected finite 1D encoder tokens [batch, tokens, channels]")
            cache["features"][offset:offset + size] = features.detach().cpu().to(torch.bfloat16)
            cache["token_valid"][offset:offset + size] = token_valid
            cache["labels"][offset:offset + size] = labels.bool()
            cache["supervised"][offset:offset + size] = supervised
            cache["sample_ids"].extend(sample_ids)
            offset += size
        if offset != n or not cache["supervised"].any():
            raise ValueError("Feature extraction produced an empty or incomplete segmentation split")
        cache["counts"] = {"windows": n, "raw_frames": raw_frames,
                           "complete_patch_frames": int(cache["token_valid"].sum()) * layout.temporal_patch_size,
                           "supervised_frames": int(cache["supervised"].sum()),
                           "annotated_frames_before_patch_filter": annotated_before_patch}
        return cache

    def evaluate(self, encoder, *, pretrain_epoch: int) -> dict:
        if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
            raise ValueError("Segmentation probing requires a frozen eval-mode encoder")
        if encoder.token_layout.kind != "1d":
            raise ValueError("Token-local segmentation currently requires a 1D encoder")
        if encoder.token_layout.raw_num_frames != self.num_frames:
            raise ValueError("Encoder and segmentation dataset frame geometry differ")
        rng = _capture_rng()
        try:
            return self._evaluate(encoder, pretrain_epoch=int(pretrain_epoch))
        finally:
            _restore_rng(rng)

    def _validate_cache(self, cache, dataset, encoder, checksum):
        layout = encoder.token_layout
        n, tokens, dim = len(dataset), layout.token_num_frames, int(encoder.embed_dim)
        frames = tokens * layout.temporal_patch_size
        expected = {
            "features": ((n, tokens, dim), torch.bfloat16),
            "labels": ((n, frames, len(self.class_names)), torch.bool),
            "supervised": ((n, frames), torch.bool),
            "token_valid": ((n, tokens), torch.bool),
        }
        for name, (shape, dtype) in expected.items():
            value = cache.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"Invalid segmentation cache {name} shape/dtype")
        if not torch.isfinite(cache["features"]).all():
            raise ValueError("Non-finite segmentation cache features")
        if hasattr(dataset, "entries"):
            sample_ids = [entry.sample_id for entry in dataset.entries]
            lengths = torch.tensor([entry.length for entry in dataset.entries])
        else:
            samples = [dataset[index] for index in range(n)]
            sample_ids = [sample[5] for sample in samples]
            lengths = torch.tensor([sample[2] for sample in samples])
        if cache.get("sample_ids") != sample_ids:
            raise ValueError("Segmentation cache sample order differs from dataset")
        valid_tokens = torch.arange(tokens)[None] < layout.valid_token_lengths(lengths)[:, None]
        valid_frames = complete_patch_frame_mask(lengths, layout)
        supervised = cache["labels"].any(-1) & valid_frames
        if not torch.equal(cache["token_valid"], valid_tokens):
            raise ValueError("Segmentation cache token validity differs from raw lengths")
        if not torch.equal(cache["supervised"], supervised):
            raise ValueError("Segmentation cache supervision differs from labels/complete patches")
        padding = torch.arange(frames)[None] >= lengths[:, None]
        if cache["labels"][padding].any():
            raise ValueError("Segmentation cache contains labels in raw padding")
        counts = {
            "windows": n, "raw_frames": int(lengths.sum()),
            "complete_patch_frames": int(valid_tokens.sum()) * layout.temporal_patch_size,
            "supervised_frames": int(supervised.sum()),
            "annotated_frames_before_patch_filter": int(cache["labels"].any(-1).sum()),
        }
        if cache.get("counts") != counts:
            raise ValueError("Segmentation cache coverage counts are inconsistent")
        if not isinstance(checksum, str) or _cache_digest(cache) != checksum:
            raise ValueError("Segmentation cache checksum mismatch")

    def _evaluate(self, encoder, *, pretrain_epoch: int) -> dict:
        _sync(self.device)
        start = time.perf_counter()
        output = self.output_root / f"epoch-{pretrain_epoch:04d}"
        output.mkdir(parents=True, exist_ok=True)
        provenance = {"protocol_hash": self.protocol_hash,
                      "target_encoder_sha256": _encoder_digest(encoder),
                      "pretrain_epoch": pretrain_epoch,
                      "token_layout": encoder.token_layout.signature(),
                      "position_encoding": position_encoding_from_model(encoder),
                      "feature_dim": int(encoder.embed_dim)}
        provenance_path = output / "provenance.json"
        if provenance_path.exists():
            saved = json.loads(provenance_path.read_text())
            if not _matches_provenance(saved, provenance):
                raise ValueError(f"Segmentation cache/result provenance mismatch: {output}")
        else:
            _atomic_json_save(provenance, provenance_path)
            _atomic_json_save(self.protocol, output / "config.json")
        if (output / "summary.json").is_file():
            summary = json.loads((output / "summary.json").read_text())
            if not _matches_provenance(summary.get("provenance"), provenance):
                raise ValueError("Completed segmentation summary has mismatched provenance")
            return summary

        caches = {}
        extraction_start = time.perf_counter()
        for split, dataset in self.datasets.items():
            path = output / f"{split}-tokens.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if not _matches_provenance(payload.get("provenance"), provenance) or payload.get("split") != split:
                    raise ValueError(f"Wrong segmentation token cache: {path}")
                caches[split] = payload["cache"]
                self._validate_cache(caches[split], dataset, encoder, payload.get("cache_sha256"))
            else:
                caches[split] = self._extract(encoder, dataset)
                _atomic_torch_save({"provenance": provenance, "split": split,
                                    "cache": caches[split],
                                    "cache_sha256": _cache_digest(caches[split])}, path)
        _sync(self.device)
        extraction_seconds = time.perf_counter() - extraction_start
        return self._fit(encoder, caches, output, provenance,
                         extraction_seconds=extraction_seconds, started=start)

    def _fit(self, encoder, caches, output, provenance, *, extraction_seconds, started):
        _seed_all(self.seed)
        head = FrameLinearProbe(int(encoder.embed_dim), encoder.token_layout.temporal_patch_size,
                                len(self.class_names)).to(self.device, dtype=torch.float32)
        optimizer = torch.optim.SGD(head.parameters(), lr=self.learning_rate,
                                    momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=0)
        generator = torch.Generator().manual_seed(self.seed)
        latest_path = output / "latest.pth.tar"
        history, first_epoch, best_epoch = [], 1, None
        best_metrics, best_score, elapsed_before = None, float("-inf"), 0.0
        if latest_path.exists():
            checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
            if not _matches_provenance(checkpoint.get("provenance"), provenance):
                raise ValueError("Segmentation head resume provenance mismatch")
            head.load_state_dict(checkpoint["head"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.device)
            scheduler.load_state_dict(checkpoint["scheduler"])
            generator.set_state(checkpoint["generator_state"])
            _restore_rng(checkpoint["rng_state"])
            normalizer = checkpoint["normalizer"]
            history, first_epoch = checkpoint["history"], checkpoint["next_epoch"]
            best_epoch, best_score = checkpoint["best_epoch"], checkpoint["best_score"]
            best_metrics = checkpoint["best_val"]
            elapsed_before = checkpoint["head_seconds"]
        else:
            normalizer = fit_token_standardizer(caches["train"], batch_size=self.batch_size)
        normalizer_device = {**normalizer, "mean": normalizer["mean"].to(self.device),
                             "scale": normalizer["scale"].to(self.device)}
        training_start = time.perf_counter()
        for epoch in range(first_epoch, self.epochs + 1):
            epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_bce = train_head_epoch(head, caches["train"], normalizer_device, optimizer,
                device=self.device, batch_size=self.batch_size, generator=generator)
            metrics = evaluate_head(head, caches["val"], normalizer_device,
                device=self.device, batch_size=self.batch_size,
                class_indices_60=self.class_indices_60)
            score = metrics["babel-120"]["frame_map"]
            if score is None:
                raise ValueError("Validation has no positive 120-class annotations")
            improved = score > best_score
            if improved:
                best_score, best_epoch, best_metrics = score, epoch, metrics
            scheduler.step()
            _sync(self.device)
            history.append({"epoch": epoch, "lr": lr, "train_bce": train_bce,
                            "val": metrics, "seconds": time.perf_counter() - epoch_start})
            checkpoint = {"format_version": FORMAT_VERSION, "provenance": provenance,
                "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng_state": _capture_rng(), "generator_state": generator.get_state(),
                "normalizer": normalizer, "next_epoch": epoch + 1,
                "best_epoch": best_epoch, "best_score": best_score, "best_val": best_metrics,
                "history": history, "head_seconds": elapsed_before + time.perf_counter() - training_start}
            if improved:
                _atomic_torch_save(checkpoint, output / "best.pth.tar")
            _atomic_torch_save(checkpoint, latest_path)
            _atomic_json_save(history, output / "history.json")
            logger.info("segmentation pretrain_epoch=%d head_epoch=%d/%d val_frame_mAP120=%.4f "
                        "val_frame_mAP60=%s", provenance["pretrain_epoch"], epoch, self.epochs,
                        score, metrics["babel-60"]["frame_map"])
        if best_epoch is None:
            raise RuntimeError("No segmentation head was trained or restored")
        if any(parameter.grad is not None for parameter in encoder.parameters()):
            raise RuntimeError("Segmentation probe accumulated encoder gradients")
        _sync(self.device)
        summary = {"pretrain_epoch": provenance["pretrain_epoch"], "best_epoch": best_epoch,
            "best_val": best_metrics, "protocol_hash": self.protocol_hash,
            "provenance": provenance, "selection": "babel-120/frame_map",
            "standardization": "train_token_zscore", "test": None,
            "split_counts": {split: cache["counts"] for split, cache in caches.items()},
            "best_head_path": str(output / "best.pth.tar"),
            "latest_head_path": str(latest_path), "history_path": str(output / "history.json"),
            "timings": {"feature_extraction_or_cache_load_seconds": extraction_seconds,
                        "head_seconds": elapsed_before + time.perf_counter() - training_start,
                        "evaluation_call_seconds": time.perf_counter() - started}}
        _atomic_json_save(summary, output / "summary.json")
        return summary
