"""Reproducible EMA flow sampling, held-out metrics and portable draw exports."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

from experiment.linear_probe.features import _atomic_json_save, _sha256_file, _torch_load_checkpoint, resolve_device
from experiment.prediction.data import FEATURE_TRANSFORM, PredictionDataset, _split_metadata, prepare_caches
from experiment.prediction.train import autocast, make_loader, move_batch
from model.token_layout import TokenLayout
from motion_rep import MotionJEPAMotionRep

from .metrics import MultiSampleMetrics, ReconstructionMetrics
from .model import FlowConfig, MotionFlow
from .sampling import seed_for_sample, seeded_noise
from .settings import load_checkpoint_config


def _checked_checkpoint(checkpoint: str | Path) -> dict:
    saved = _torch_load_checkpoint(Path(checkpoint).expanduser().resolve())
    if saved.get("format_version") != 1 or saved.get("kind") != "motion_flow":
        raise ValueError("Unsupported motion flow checkpoint")
    required = ("ema", "flow_config", "config", "model_info", "token_layout", "mean", "std",
                "feature_transform", "provenance")
    if any(key not in saved for key in required):
        raise ValueError("Motion flow checkpoint lacks required inference metadata")
    if saved["feature_transform"] != FEATURE_TRANSFORM:
        raise ValueError("Unsupported motion flow feature transform")
    return saved


def _generator_from_saved(saved: dict, device: torch.device):
    signature = dict(saved["token_layout"])
    for key in ("spatial_token_names", "trajectory_fields"):
        if signature.get(key) is not None:
            signature[key] = tuple(signature[key])
    model = MotionFlow(
        int(saved["model_info"]["feature_dim"]), TokenLayout(**signature),
        int(saved["model_info"]["motion_dim"]), FlowConfig(**saved["flow_config"]),
    )
    model.load_state_dict(saved["ema"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_generator(checkpoint: str | Path, device: torch.device):
    """Load the EMA velocity field, independent of any target motion."""
    saved = _checked_checkpoint(checkpoint)
    return _generator_from_saved(saved, torch.device(device)), saved


def _inputs(checkpoint, split, config_path, overrides):
    checkpoint = Path(checkpoint).expanduser().resolve()
    saved = _checked_checkpoint(checkpoint)
    config = load_checkpoint_config(saved["config"], config_path, overrides)
    if config["flow"] != saved["flow_config"]:
        raise ValueError("Evaluation flow architecture cannot differ from its checkpoint")
    training_config = load_checkpoint_config(saved["config"])
    for key in ("jepa_checkpoint", "dataset_root", "stats_path"):
        if config[key] != training_config[key]:
            raise ValueError(f"Evaluation requires the trained {key}")
    device = resolve_device(str(config["device"]))
    # Inference autocast is independent of the frozen encoder convention that
    # produced the training cache. Precision overrides must not invalidate it.
    cache_config = {**config, "use_bfloat16": bool(saved["config"].get("use_bfloat16", True))}
    prepare_caches(cache_config, splits=(split,))
    dataset = PredictionDataset(cache_config, split)
    if dataset.token_layout.signature() != saved["token_layout"] or dataset.model_info != saved["model_info"]:
        raise ValueError("Evaluation feature layout differs from the trained generator")
    if (not torch.equal(dataset.mean.cpu(), saved["mean"].cpu())
            or not torch.equal(dataset.std.cpu(), saved["std"].cpu())):
        raise ValueError("Evaluation normalization statistics differ from training")
    # Verify the train manifest/source without requiring its feature cache in
    # a newly selected cache directory or extracting train features at inference.
    _, _, current_train = _split_metadata(
        cache_config, "train", dataset.provenance["source_model_config"], dataset.model_info,
        Path(dataset.provenance["checkpoint_path"]), Path(dataset.provenance["stats_root"]),
    )
    if current_train["provenance"] != saved["provenance"]["train"]:
        raise ValueError("Source checkpoint or training dataset changed since flow training")
    return checkpoint, saved, config, device, dataset, _generator_from_saved(saved, device)


def _deterministic_baseline(config: dict, saved: dict, device: torch.device):
    path = config.get("deterministic_checkpoint")
    if path is None:
        return None
    from experiment.prediction.evaluate import load_decoder

    baseline, metadata = load_decoder(path, device)
    def source_provenance(provenance):
        return {key: value for key, value in provenance.items() if key not in {"split", "limit", "num_samples"}}

    if (metadata["model_info"] != saved["model_info"]
            or metadata["token_layout"] != saved["token_layout"]
            or metadata["feature_transform"] != saved["feature_transform"]
            or source_provenance(metadata["provenance"]["train"]) != source_provenance(saved["provenance"]["train"])
            or not torch.equal(metadata["mean"].cpu(), saved["mean"].cpu())
            or not torch.equal(metadata["std"].cpu(), saved["std"].cpu())):
        raise ValueError("Explicit deterministic baseline is incompatible with flow training provenance")
    return baseline.requires_grad_(False)


def _draw(model, batch: dict, config: dict, draw_index: int) -> torch.Tensor:
    noise = seeded_noise(
        batch["sample_id"], int(config["evaluation_seed"]), draw_index,
        tuple(batch["valid_frames"].shape[1:]) + (int(model.motion_dim),),
        batch["tokens"].device, stream="noise",
    )
    generated = model.sample(
        batch["tokens"], batch["fps"], batch["valid_frames"], num_samples=1,
        steps=int(config["steps"]), guidance_scale=float(config["guidance_scale"]),
        initial_noise=noise, use_bfloat16=bool(config["use_bfloat16"]),
    )
    expected = (len(batch["sample_id"]), 1, batch["valid_frames"].shape[1], int(model.motion_dim))
    if generated.shape != expected:
        raise ValueError("Generator sampling returned an unexpected motion shape")
    return generated[:, 0].float()


def _even_indices(count: int, requested: int) -> list[int]:
    selected = min(count, requested)
    return np.linspace(0, count - 1, selected, dtype=int).tolist() if selected else []


def _protocol(checkpoint: Path, saved: dict, config: dict, split: str, device: torch.device) -> dict:
    branches = 1 if float(config["guidance_scale"]) in {0., 1.} else 2
    return {
        "version": 1, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256_file(checkpoint),
        "weights": "ema", "split": split, "evaluation_seed": int(config["evaluation_seed"]),
        "steps": int(config["steps"]), "guidance_scale": float(config["guidance_scale"]),
        "num_samples": int(config["num_samples"]), "sampler": "euler_rectified_flow",
        "noise_protocol": "sha256_per_sample_id_draw_index_noise_stream",
        "feature_transform": saved["feature_transform"],
        "single_sample_draw_index": 0,
        "oracle_selection": "one_whole_draw_by_valid_frame_normalized_366d_mse",
        "diversity": "within_condition_pairwise_fk_root_relative_nonroot_and_root_3d",
        "use_bfloat16": bool(config["use_bfloat16"]),
        "device": str(device),
        "precision": {"state_and_integration": "float32", "condition_cache": "bfloat16",
                      "velocity_network": "bfloat16_autocast" if config["use_bfloat16"] and device.type == "cuda" else "float32"},
        "encoder_extraction_precision": {
            "use_bfloat16": bool(saved["config"].get("use_bfloat16", True)),
            "policy": saved["provenance"]["train"].get("encoder_precision_policy"),
        },
        "cfg_branches": branches,
        "network_evaluations_per_trajectory": int(config["steps"]) * branches,
        "timing_scope": "checkpoint_config_cache_setup_sampling_metrics_and_export",
    }


def _export(rows: list[dict], dataset: PredictionDataset, config: dict,
            protocol: dict, path: Path) -> str | None:
    if not rows:
        return None
    mean, std = dataset.mean.cpu(), dataset.std.cpu()
    target = torch.stack([row["target"] for row in rows]).float() * std + mean
    generated = torch.stack([row["generated"] for row in rows]).float() * std + mean
    lengths = np.asarray([row["length"] for row in rows], dtype=np.int64)
    valid = torch.arange(target.shape[1])[None] < torch.from_numpy(lengths)[:, None]
    target = target.masked_fill(~valid[..., None], 0)
    generated = generated.masked_fill(~valid[:, None, :, None], 0)
    representation = MotionJEPAMotionRep(fps=int(dataset.model_info["fps"]))
    target_joints = representation.inverse(target)["posed_joints"]
    generated_joints = representation.inverse(generated.flatten(0, 1))["posed_joints"]
    generated_joints = generated_joints.reshape(*generated.shape[:3], 30, 3)
    target_joints = target_joints.masked_fill(~valid[:, :, None, None], 0)
    generated_joints = generated_joints.masked_fill(~valid[:, None, :, None, None], 0)
    noise_seeds = np.asarray([
        [seed_for_sample(int(config["evaluation_seed"]), row["sample_id"], draw, stream="noise")
         for draw in range(int(config["num_samples"]))] for row in rows
    ], dtype=np.uint64)
    sampling = {**protocol, "indices": [row["index"] for row in rows],
                "exported_conditions": len(rows), "condition_provenance": dataset.provenance}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, sample_ids=np.asarray([row["sample_id"] for row in rows]), lengths=lengths,
            fps=np.asarray(dataset.model_info["fps"], dtype=np.int64),
            target_motion=target.numpy(), generated_motion=generated.numpy(),
            target_joints=target_joints.numpy(), generated_joints=generated_joints.numpy(),
            noise_seeds=noise_seeds, sampling_json=np.asarray(json.dumps(sampling, sort_keys=True)),
        )
    temporary.replace(path)
    return str(path)


@torch.inference_mode()
def evaluate_checkpoint(checkpoint: str | Path, *, split: str = "test",
                        config_path: str | Path | None = None, overrides: dict | None = None) -> dict:
    started = time.perf_counter()
    checkpoint, saved, config, device, dataset, model = _inputs(
        checkpoint, split, config_path, overrides
    )
    baseline_model = _deterministic_baseline(config, saved, device)
    fps = int(dataset.model_info["fps"])
    single = ReconstructionMetrics(dataset.mean, dataset.std, fps)
    zero = ReconstructionMetrics(dataset.mean, dataset.std, fps)
    baseline = ReconstructionMetrics(dataset.mean, dataset.std, fps) if baseline_model is not None else None
    diagnostic_indices = _even_indices(len(dataset), min(int(config["diagnostic_count"]), 512))
    selected_set = set(diagnostic_indices)
    draw_zero = {}
    offset = 0
    loader = make_loader(dataset, config, device, generator=torch.Generator().manual_seed(0))
    full_sampling_calls = len(loader)
    for batch in tqdm(loader, desc=f"Generate {split} draw 0", mininterval=10):
        batch = move_batch(batch, device)
        generated = _draw(model, batch, config, 0)
        target, active = batch["motion"].float(), batch["valid_frames"]
        single.update(generated, target, active)
        zero.update(torch.zeros_like(generated), target, active)
        if baseline_model is not None:
            with autocast(device, bool(config["use_bfloat16"])):
                deterministic = baseline_model(batch["tokens"], batch["fps"], active)
            baseline.update(deterministic.float(), target, active)
        for row in range(len(generated)):
            if offset + row in selected_set:
                draw_zero[offset + row] = generated[row].cpu()
        offset += len(generated)
    if offset != len(dataset):
        raise RuntimeError("Single-draw generation did not cover the requested split")

    diagnostic = MultiSampleMetrics(dataset.mean, dataset.std, fps)
    exports = []
    export_positions = set(_even_indices(len(diagnostic_indices), int(config["export_count"])))
    offset = 0
    loader = make_loader(Subset(dataset, diagnostic_indices), config, device,
                         generator=torch.Generator().manual_seed(0))
    diagnostic_sampling_calls = len(loader) * (int(config["num_samples"]) - 1)
    for batch in tqdm(loader, desc=f"Generate {split} diagnostic draws", mininterval=10):
        batch = move_batch(batch, device)
        indices = diagnostic_indices[offset:offset + len(batch["sample_id"])]
        draws = [torch.stack([draw_zero.pop(index) for index in indices]).to(device)]
        for draw in range(1, int(config["num_samples"])):
            draws.append(_draw(model, batch, config, draw))
        generated = torch.stack(draws, dim=1)
        diagnostic.update(generated, batch["motion"].float(), batch["valid_frames"])
        for row, index in enumerate(indices):
            if offset + row in export_positions:
                exports.append({"index": index, "sample_id": batch["sample_id"][row],
                                "length": int(batch["length"][row]),
                                "target": batch["motion"][row].float().cpu(),
                                "generated": generated[row].cpu()})
        offset += len(indices)
    protocol = _protocol(checkpoint, saved, config, split, device)
    protocol.update(diagnostic_indices=diagnostic_indices,
                    diagnostic_sample_ids=[dataset.entries[index].sample_id for index in diagnostic_indices],
                    diagnostic_count=len(diagnostic_indices), full_split_samples=len(dataset),
                    diagnostic_selection="evenly_spaced_dataset_indices")
    trajectories = len(dataset) + len(diagnostic_indices) * (int(config["num_samples"]) - 1)
    protocol.update(total_generated_trajectories=trajectories,
                    total_per_example_network_evaluations=trajectories * protocol["network_evaluations_per_trajectory"],
                    sampling_batch_calls=full_sampling_calls + diagnostic_sampling_calls,
                    network_forward_batch_calls=(full_sampling_calls + diagnostic_sampling_calls)
                    * protocol["network_evaluations_per_trajectory"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    result = {"split": split, "samples": len(dataset), "checkpoint": str(checkpoint),
              "single_sample": single.compute(), "zero_output_baseline": zero.compute(),
              **diagnostic.compute(), "diagnostic_samples": len(diagnostic_indices),
              "num_samples": int(config["num_samples"]), "protocol": protocol,
              "feature_transform": saved["feature_transform"], "provenance": dataset.provenance,
              "training_provenance": saved["provenance"],
              "exports": _export(exports, dataset, config, protocol, output / f"{split}-generations.npz")}
    if baseline is not None:
        result["deterministic_baseline"] = {"checkpoint": config["deterministic_checkpoint"],
                                            **baseline.compute()}
    result["elapsed_seconds"] = time.perf_counter() - started
    _atomic_json_save(protocol, output / f"{split}-protocol.json")
    _atomic_json_save(result, output / f"{split}-metrics.json")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"provenance", "training_provenance", "protocol"}}, indent=2), flush=True)
    return result


@torch.inference_mode()
def sample_checkpoint(checkpoint: str | Path, *, split: str = "test", indices=None,
                      config_path: str | Path | None = None, overrides: dict | None = None) -> dict:
    started = time.perf_counter()
    checkpoint, saved, config, device, dataset, model = _inputs(
        checkpoint, split, config_path, overrides
    )
    if indices is None:
        indices = _even_indices(len(dataset), int(config["export_count"]))
    else:
        indices = list(indices)
    if (not indices or any(isinstance(index, bool) or not isinstance(index, int)
                           or not 0 <= index < len(dataset) for index in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("Sample indices must be nonempty, unique valid dataset indices")
    rows, offset = [], 0
    loader = make_loader(Subset(dataset, indices), config, device,
                         generator=torch.Generator().manual_seed(0))
    sampling_calls = len(loader) * int(config["num_samples"])
    for batch in tqdm(loader, desc=f"Sample {split}", mininterval=10):
        batch = move_batch(batch, device)
        draws = [_draw(model, batch, config, draw) for draw in range(int(config["num_samples"]))]
        generated = torch.stack(draws, dim=1)
        for row in range(len(generated)):
            rows.append({"index": indices[offset + row], "sample_id": batch["sample_id"][row],
                         "length": int(batch["length"][row]), "target": batch["motion"][row].float().cpu(),
                         "generated": generated[row].cpu()})
        offset += len(generated)
    protocol = _protocol(checkpoint, saved, config, split, device)
    trajectories = len(indices) * int(config["num_samples"])
    protocol.update(indices=indices, total_generated_trajectories=trajectories,
                    total_per_example_network_evaluations=trajectories * protocol["network_evaluations_per_trajectory"],
                    sampling_batch_calls=sampling_calls,
                    network_forward_batch_calls=sampling_calls * protocol["network_evaluations_per_trajectory"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    result = {"split": split, "samples": len(indices), "num_samples": int(config["num_samples"]),
              "checkpoint": str(checkpoint), "protocol": protocol, "provenance": dataset.provenance,
              "exports": _export(rows, dataset, config, protocol, output / f"{split}-samples.npz"),
              "elapsed_seconds": time.perf_counter() - started}
    _atomic_json_save(result, output / f"{split}-samples.json")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"protocol", "provenance"}}, indent=2), flush=True)
    return result


__all__ = ["load_generator", "evaluate_checkpoint", "sample_checkpoint"]
