"""Held-out raw-motion reconstruction metrics and portable comparison exports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from experiment.linear_probe.features import _atomic_json_save, _torch_load_checkpoint, resolve_device
from motion_rep import MotionJEPAMotionRep
from model.token_layout import TokenLayout
from .data import PredictionDataset, prepare_caches
from .metrics import ReconstructionMetrics
from .model import DecoderConfig, MotionDecoder
from .settings import load_config
from .train import FORMAT_VERSION, autocast, make_loader, move_batch


def load_decoder(checkpoint: str | Path, device: torch.device):
    saved = _torch_load_checkpoint(Path(checkpoint).expanduser().resolve())
    if saved.get("format_version") != FORMAT_VERSION or saved.get("kind") != "motion_decoder":
        raise ValueError("Unsupported motion decoder checkpoint")
    signature = dict(saved["token_layout"])
    for key in ("spatial_token_names", "trajectory_fields"):
        if signature.get(key) is not None:
            signature[key] = tuple(signature[key])
    model = MotionDecoder(int(saved["model_info"]["feature_dim"]), TokenLayout(**signature),
                          int(saved["model_info"]["motion_dim"]), DecoderConfig(**saved["decoder_config"]))
    model.load_state_dict(saved["decoder"], strict=True)
    return model.to(device).eval(), saved


@torch.inference_mode()
def evaluate_checkpoint(checkpoint: str | Path, *, split: str = "test",
                        config_path: str | Path | None = None, overrides: dict | None = None) -> dict:
    checkpoint = Path(checkpoint).expanduser().resolve()
    saved_config = _torch_load_checkpoint(checkpoint)["config"]
    requested = {}
    if config_path is not None:
        requested = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(requested, dict):
            raise ValueError("Evaluation config must be a YAML mapping")
    config = load_config(None, {**saved_config, **requested, **(overrides or {})})
    device = resolve_device(str(config["device"]))
    model, saved = load_decoder(checkpoint, device)
    if config["decoder"] != saved["decoder_config"]:
        raise ValueError("Evaluation decoder architecture cannot differ from its checkpoint")
    prepare_caches(config, splits=(split,))
    dataset = PredictionDataset(config, split)
    if dataset.token_layout.signature() != saved["token_layout"] or dataset.model_info != saved["model_info"]:
        raise ValueError("Evaluation feature layout differs from the trained decoder")
    if not torch.equal(dataset.mean.cpu(), saved["mean"].cpu()) or not torch.equal(dataset.std.cpu(), saved["std"].cpu()):
        raise ValueError("Evaluation normalization statistics differ from training")
    # Test has distinct split provenance, but its encoder/feature convention must match train.
    train_config = load_config(None, saved_config)
    trained_dataset = PredictionDataset(config, "train")
    if trained_dataset.provenance != saved["provenance"]["train"]:
        raise ValueError("Source checkpoint or training dataset changed since decoder training")
    for key in ("jepa_checkpoint", "dataset_root", "stats_path"):
        if config[key] != train_config[key]:
            raise ValueError(f"Evaluation requires the trained {key}")
    actual = ReconstructionMetrics(dataset.mean, dataset.std, int(dataset.model_info["fps"]))
    baseline = ReconstructionMetrics(dataset.mean, dataset.std, int(dataset.model_info["fps"]))
    selected = []
    export_count = min(int(config["export_count"]), len(dataset))
    export_indices = set(np.linspace(0, len(dataset) - 1, export_count, dtype=int).tolist()) if export_count else set()
    offset = 0
    for batch in tqdm(make_loader(dataset, config, device, generator=torch.Generator().manual_seed(0)),
                      desc=f"Evaluate {split}", mininterval=10):
        batch = move_batch(batch, device)
        with autocast(device, bool(config["use_bfloat16"])):
            prediction = model(batch["tokens"], batch["fps"], batch["valid_frames"])
        prediction, target = prediction.float(), batch["motion"].float()
        actual.update(prediction, target, batch["valid_frames"])
        baseline.update(torch.zeros_like(prediction), target, batch["valid_frames"])
        for row in range(len(prediction)):
            if offset + row in export_indices:
                selected.append((batch["sample_id"][row], int(batch["length"][row]),
                                 target[row].cpu(), prediction[row].cpu()))
        offset += len(prediction)
    result = {"split": split, "samples": len(dataset), "checkpoint": str(checkpoint),
              "reconstruction": actual.compute(), "zero_output_baseline": baseline.compute(),
              "feature_transform": saved["feature_transform"], "provenance": dataset.provenance}
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if selected:
        mean, std = dataset.mean.cpu(), dataset.std.cpu()
        target_raw = torch.stack([item[2] for item in selected]) * std + mean
        reconstructed_raw = torch.stack([item[3] for item in selected]) * std + mean
        lengths = np.asarray([item[1] for item in selected], dtype=np.int64)
        valid = torch.arange(target_raw.shape[1])[None, :] < torch.from_numpy(lengths)[:, None]
        target_raw = target_raw.masked_fill(~valid[..., None], 0)
        reconstructed_raw = reconstructed_raw.masked_fill(~valid[..., None], 0)
        rep = MotionJEPAMotionRep(fps=int(dataset.model_info["fps"]))
        target_joints = rep.inverse(target_raw)["posed_joints"].numpy()
        reconstructed_joints = rep.inverse(reconstructed_raw)["posed_joints"].numpy()
        export_path = output / f"{split}-reconstructions.npz"
        temporary = export_path.with_name(export_path.name + ".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, sample_ids=np.asarray([item[0] for item in selected]), lengths=lengths,
                                fps=np.asarray(dataset.model_info["fps"], dtype=np.int64),
                                target_motion=target_raw.numpy(), reconstructed_motion=reconstructed_raw.numpy(),
                                target_joints=target_joints, reconstructed_joints=reconstructed_joints)
        temporary.replace(export_path)
        result["exports"] = str(export_path)
    _atomic_json_save(result, output / f"{split}-metrics.json")
    print(json.dumps({key: value for key, value in result.items() if key != "provenance"}, indent=2), flush=True)
    return result
