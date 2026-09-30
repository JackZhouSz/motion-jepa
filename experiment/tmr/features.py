"""Streaming frozen-backbone caches; training reads only completed cache files."""

from __future__ import annotations

import gc
import json
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import MotionDataset
from experiment.linear_probe.features import (
    _atomic_json_save, _sha256_file, load_frozen_encoder,
    resolve_pretraining_stats,
)
from .utils import resolve_device
from .dataset import (
    DEFAULT_ANNOTATIONS_PATH, DEFAULT_CACHE_ROOT, DEFAULT_DATASET_ROOT,
    DEFAULT_TEXT_MODEL, SPLITS, PreparedPairDataset, RaggedTokenBank,
    build_paired_index, json_digest,
)


CACHE_FORMAT_VERSION = 2


def _model_source_signature(model: str) -> dict[str, Any]:
    path = Path(model).expanduser()
    if not path.is_dir():
        return {"model": model, "local_files": None}
    path = path.resolve()
    files = sorted(
        candidate for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix in {".json", ".safetensors", ".bin", ".model"}
    )
    return {"model": str(path), "local_files": {file.name: _sha256_file(file) for file in files}}


def _text_signature(paired, model, revision, max_length):
    return {
        "format_version": CACHE_FORMAT_VERSION, "dtype": "bfloat16_uint16",
        "dataset": paired["provenance"], "catalog_sha256": paired["catalog_sha256"],
        "model_source": _model_source_signature(str(model)),
        "text_revision": revision, "max_text_length": int(max_length),
        "feature": "encoder_last_hidden_state_valid_tokens",
    }


def _stats_signature(root: Path):
    return {
        "stats_root": str(root.resolve()),
        "stats_mean_sha256": _sha256_file(root / "mean.npy"),
        "stats_std_sha256": _sha256_file(root / "std.npy"),
    }


def _motion_signature(paired, stats, jepa_source):
    return {
        "format_version": CACHE_FORMAT_VERSION, "dtype": "bfloat16_uint16",
        "dataset": paired["provenance"], "paired_index_sha256": paired["paired_index_sha256"],
        **stats, "jepa_source": jepa_source,
        "pooling": "spatial_mean_preserve_temporal",
    }


def _existing_bank(path, signature, recompute):
    if recompute:
        return None
    if path.exists():
        return RaggedTokenBank(path, signature)
    return None


def write_ragged_bank(path, keys, lengths, feature_dim, signature, batches, extra=None):
    """Write bounded batches into a BF16 memmap and publish completion last."""
    path = Path(path)
    if len(keys) != len(lengths) or any(int(length) < 1 for length in lengths):
        raise ValueError("Token-bank keys and strictly positive lengths must agree")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    offsets = np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))
    values = np.lib.format.open_memmap(
        temporary / "values.npy", mode="w+", dtype=np.uint16,
        shape=(int(offsets[-1]), int(feature_dim)),
    )
    np.save(temporary / "offsets.npy", offsets, allow_pickle=False)
    cursor = 0
    try:
        for sequences in batches:
            for sequence in sequences:
                expected = (int(lengths[cursor]), int(feature_dim))
                if tuple(sequence.shape) != expected or not torch.isfinite(sequence).all():
                    raise ValueError(f"Invalid token-cache output: expected {expected}, got {tuple(sequence.shape)}")
                bits = sequence.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy()
                values[int(offsets[cursor]):int(offsets[cursor + 1])] = bits
                cursor += 1
        if cursor != len(keys):
            raise ValueError("Token extraction did not cover every cache row")
        values.flush()
        del values
        marker = {
            "signature": signature, "keys": list(keys), "feature_dim": int(feature_dim),
            "num_tokens": int(offsets[-1]), "max_length": max(lengths, default=0),
            "model_info": extra or {},
        }
        _atomic_json_save(marker, temporary / "complete.json")
        if path.exists():
            shutil.rmtree(path)
        os.replace(temporary, path)
    except BaseException:
        # A failed extraction must never look complete to a training reader.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return RaggedTokenBank(path, signature)


def load_text_backbone(model_name, revision, device):
    """Instantiate only the pretrained bidirectional T5Gemma encoder."""
    # Deliberately deferred: cache-only training has no Transformers dependency.
    try:
        from transformers import AutoConfig, AutoTokenizer, T5GemmaEncoderModel
    except ImportError as error:
        raise RuntimeError(
            f"Cannot extract {model_name!r}: install the text dependencies with "
            "pip install -r experiment/tmr/requirements.txt"
        ) from error
    try:
        config = AutoConfig.from_pretrained(model_name, revision=revision)
        if config.model_type != "t5gemma":
            raise ValueError(f"Text model {model_name!r} must use model_type='t5gemma', got {config.model_type!r}")
        config.is_encoder_decoder = False
        resolved_revision = getattr(config, "_commit_hash", None) or revision
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision)
        model, loading = T5GemmaEncoderModel.from_pretrained(
            model_name, revision=resolved_revision, config=config,
            dtype=torch.bfloat16, output_loading_info=True,
        )
    except OSError as error:
        raise RuntimeError(
            f"Cannot load text model {model_name!r}. Check the model ID/revision or local files. "
            "For Google's gated Gemma weights, accept the model's Hugging Face access terms "
            "and authenticate with hf auth login before running prepare_cache."
        ) from error
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Text encoder weights did not load completely: {loading}")
    model.to(device).eval().requires_grad_(False)
    if hasattr(model, "decoder") or hasattr(model, "lm_head"):
        raise ValueError("Text feature extraction must load the encoder only")
    tokenizer_hash = json_digest({
        "backend": tokenizer.backend_tokenizer.to_str(),
        "special_tokens": {key: str(value) for key, value in tokenizer.special_tokens_map.items()},
    })
    return model, tokenizer, {
        "model": str(model_name), "revision": revision,
        "resolved_revision": getattr(config, "_commit_hash", None),
        "tokenizer_sha256": tokenizer_hash,
        "feature_dim": int(config.encoder.hidden_size), "dtype": "bfloat16",
        "model_class": "T5GemmaEncoderModel",
    }


def _prepare_text(path, paired, *, model_name, revision, max_length, device, batch_size, recompute):
    signature = _text_signature(paired, model_name, revision, max_length)
    existing = _existing_bank(path, signature, recompute)
    if existing is not None:
        return existing
    model, tokenizer, info = load_text_backbone(model_name, revision, device)
    keys = list(paired["catalog"])
    texts = [paired["catalog"][key] for key in keys]
    lengths, truncated = [], 0
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(texts[start:start + batch_size], padding=False, truncation=False)
        sizes = [len(ids) for ids in tokenized["input_ids"]]
        lengths.extend(min(size, max_length) for size in sizes)
        truncated += sum(size > max_length for size in sizes)
    if not keys or any(length < 1 for length in lengths):
        raise ValueError("Text cache requires nonempty captions and valid token sequences")
    info.update(max_text_length=max_length, truncated_captions=truncated)

    def batches():
        with torch.inference_mode():
            for start in tqdm(range(0, len(texts), batch_size), desc="Cache T5Gemma tokens"):
                encoded = tokenizer(
                    texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
                output = model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].bool()
                yield [output[row, mask[row]] for row in range(len(output))]
    try:
        return write_ragged_bank(path, keys, lengths, info["feature_dim"], signature, batches(), info)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


class _MotionRows(Dataset):
    def __init__(self, base, records):
        self.base, self.records = base, records
        rows = {entry.sample_id: index for index, entry in enumerate(base.entries)}
        self.rows = [rows[record["sample_id"]] for record in records]
        for row, record in zip(self.rows, records):
            entry = base.entries[row]
            if entry.length != record["length"] or entry.fps != record["fps"] or entry.path != base.root_path / record["motion_path"]:
                raise ValueError(f"Paired interval/path disagrees with raw split index: {record['sample_id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.base[self.rows[index]]


def _raw_dataset(root, split, paired, stats_root):
    provenance = paired["provenance"]
    return MotionDataset(
        root, f"{split}.txt", provenance["num_frames"], provenance["fps"],
        motion_dim=provenance["raw_motion_dim"], normalize=True, stats_path=stats_root,
    )


def _prepare_motion(path, paired, records, base, encoder, source, stats, *, device, batch_size, num_workers, recompute):
    signature = _motion_signature(paired, stats, source)
    existing = _existing_bank(path, signature, recompute)
    if existing is not None:
        return existing
    lengths = [int(encoder.token_layout.valid_token_lengths(torch.tensor(record["length"]))) for record in records]
    if any(length < 1 for length in lengths):
        raise ValueError("A motion clip has zero valid JEPA tokens")
    loader = DataLoader(
        _MotionRows(base, records), batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=device.type == "cuda",
    )

    def batches():
        with torch.inference_mode():
            for motion, fps, length in tqdm(loader, desc="Cache JEPA tokens"):
                motion = motion.to(device=device, dtype=torch.float32)
                fps = fps.to(device=device, dtype=torch.float32)
                length = length.to(device=device, dtype=torch.long)
                active = torch.arange(motion.shape[1], device=device)[None] < length[:, None]
                amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with amp:
                    encoded = encoder(motion, fps, valid_frames=active)
                if encoded.ndim == 4:
                    encoded = encoded.mean(dim=2)
                if encoded.ndim != 3:
                    raise ValueError("JEPA must return [B,T,D] or [B,T,J,D]")
                token_lengths = encoder.token_layout.valid_token_lengths(length)
                yield [encoded[row, :int(token_lengths[row])] for row in range(len(encoded))]
    return write_ragged_bank(
        path, [record["sample_id"] for record in records], lengths,
        source["model_info"]["feature_dim"], signature, batches(), source["model_info"],
    )


def prepare_caches(
    *, dataset_root=DEFAULT_DATASET_ROOT, annotations_path=DEFAULT_ANNOTATIONS_PATH,
    cache_root=DEFAULT_CACHE_ROOT, input_source="raw", jepa_checkpoint=None,
    checkpoint_key="target_encoder", stats_path=None, text_model=DEFAULT_TEXT_MODEL,
    text_revision=None, max_text_length=256, device="auto", feature_batch_size=64,
    text_batch_size=8, num_workers=0, recompute_features=False, max_samples_per_split=None,
):
    if input_source not in {"raw", "jepa"}:
        raise ValueError("input_source must be raw or jepa")
    if min(max_text_length, feature_batch_size, text_batch_size) < 1 or num_workers < 0:
        raise ValueError("Lengths/batch sizes must be positive and num_workers nonnegative")
    if checkpoint_key not in {"encoder", "target_encoder"}:
        raise ValueError("checkpoint_key must be encoder or target_encoder")
    if input_source == "jepa" and jepa_checkpoint is None:
        raise ValueError("JEPA input requires --jepa-checkpoint")
    root, cache = Path(dataset_root).expanduser().resolve(), Path(cache_root).expanduser().resolve()
    target_device = resolve_device(str(device))
    paired = build_paired_index(root, annotations_path, max_samples_per_split)
    if not paired["split_counts"]["train"]:
        raise ValueError("TMR has no captioned training samples")
    encoder, source = None, None
    if input_source == "jepa":
        checkpoint = Path(jepa_checkpoint).expanduser().resolve()
        # Keep JEPA on CPU until the text model has finished and released VRAM.
        encoder, config, info = load_frozen_encoder(checkpoint, checkpoint_key, torch.device("cpu"))
        for key, expected in (("num_frames", paired["provenance"]["num_frames"]), ("fps", paired["provenance"]["fps"]), ("motion_dim", paired["provenance"]["raw_motion_dim"])):
            if info[key] != expected:
                raise ValueError(f"JEPA checkpoint {key}={info[key]} does not match dataset {expected}")
        stats_root = resolve_pretraining_stats(config, Path(stats_path) if stats_path is not None else None)
        source = {
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_key": checkpoint_key, "layout": encoder.token_layout.signature(),
            "model_info": info,
        }
    else:
        stats_root = Path(stats_path).expanduser().resolve() if stats_path is not None else root / "stats"
    stats = _stats_signature(stats_root)
    raw_datasets = {split: _raw_dataset(root, split, paired, stats_root) for split in SPLITS}
    for split, base in raw_datasets.items():
        records = [record for record in paired["records"] if record["split"] == split]
        _MotionRows(base, records)
    if encoder is not None:
        lengths = encoder.token_layout.valid_token_lengths(torch.tensor([record["length"] for record in paired["records"]]))
        if (lengths < 1).any():
            raise ValueError("A motion clip has zero valid JEPA tokens")
    # Validate motion/statistics before an expensive text-model download or run.
    text_bank = _prepare_text(
        cache / "text", paired, model_name=str(text_model), revision=text_revision,
        max_length=max_text_length, device=target_device,
        batch_size=text_batch_size, recompute=recompute_features,
    )
    if encoder is not None:
        encoder.to(target_device)
    for split in SPLITS:
        base = raw_datasets[split]
        if encoder is not None:
            _prepare_motion(
                cache / "jepa" / split, paired,
                [record for record in paired["records"] if record["split"] == split],
                base, encoder, source, stats, device=target_device,
                batch_size=feature_batch_size, num_workers=num_workers, recompute=recompute_features,
            )
    metadata = {
        "format_version": CACHE_FORMAT_VERSION, "input_source": input_source,
        "motion_dim": paired["provenance"]["raw_motion_dim"] if source is None else source["model_info"]["feature_dim"],
        "motion_num_tokens": paired["provenance"]["num_frames"] if source is None else source["model_info"]["token_num_frames"],
        "text_dim": text_bank.metadata["feature_dim"],
        "text_source": text_bank.metadata["model_info"], "text_signature": text_bank.metadata["signature"],
        "jepa_source": source, **stats,
        "paired_index_sha256": paired["paired_index_sha256"], "catalog_sha256": paired["catalog_sha256"],
        "split_counts": paired["split_counts"], "filtered_counts": paired["filtered_counts"],
        "provenance": paired["provenance"],
    }
    directory = cache / input_source
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(paired, directory / "paired-index.json")
    _atomic_json_save(metadata, directory / "prepared.json")
    return metadata


def load_prepared_datasets(
    dataset_root, annotations_path, cache_root, input_source,
    jepa_checkpoint=None, checkpoint_key="target_encoder", stats_path=None,
    text_model=DEFAULT_TEXT_MODEL, text_revision=None, max_text_length=256,
):
    """Validate provenance without instantiating or deserializing either backbone."""
    root, cache = Path(dataset_root).expanduser().resolve(), Path(cache_root).expanduser().resolve()
    if input_source not in {"raw", "jepa"}:
        raise ValueError("input_source must be raw or jepa")
    marker = cache / input_source / "prepared.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Prepared TMR cache is missing: {marker}; run prepare_cache")
    metadata = json.loads(marker.read_text())
    if metadata.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported prepared TMR cache format; rebuild with prepare_cache "
            "--recompute-features to cache individual caption candidates"
        )
    paired = build_paired_index(root, annotations_path, metadata["provenance"]["max_samples_per_split"])
    for key in ("paired_index_sha256", "catalog_sha256", "provenance", "split_counts", "filtered_counts"):
        if metadata[key] != paired[key]:
            raise ValueError("Prepared TMR cache metadata is stale; use --recompute-features")
    signature = _text_signature(paired, text_model, text_revision, max_text_length)
    text_bank = RaggedTokenBank(cache / "text", signature)
    if metadata["text_signature"] != signature or metadata["text_source"] != text_bank.metadata["model_info"]:
        raise ValueError("Prepared text cache metadata is stale; use --recompute-features")
    stats_root = Path(stats_path).expanduser().resolve() if stats_path is not None else (
        root / "stats" if input_source == "raw" else Path(metadata["stats_root"])
    )
    stats = _stats_signature(stats_root)
    if any(metadata[key] != value for key, value in stats.items()):
        raise ValueError("Prepared normalization statistics are stale; use --recompute-features")
    source = metadata["jepa_source"]
    if input_source == "jepa":
        if source is None:
            raise ValueError("JEPA cache is missing checkpoint provenance")
        checkpoint = Path(jepa_checkpoint or source["checkpoint_path"]).expanduser().resolve()
        if _sha256_file(checkpoint) != source["checkpoint_sha256"] or checkpoint_key != source["checkpoint_key"]:
            raise ValueError("JEPA checkpoint cache is stale; use --recompute-features")
    datasets = {}
    for split in SPLITS:
        records = [record for record in paired["records"] if record["split"] == split]
        if input_source == "raw":
            datasets[split] = PreparedPairDataset(
                records, text_bank, raw_dataset=_raw_dataset(root, split, paired, stats_root),
                sample_captions=split == "train",
            )
        else:
            bank = RaggedTokenBank(cache / "jepa" / split, _motion_signature(paired, stats, source))
            datasets[split] = PreparedPairDataset(records, text_bank, motion_bank=bank, sample_captions=split == "train")
    return datasets, metadata


__all__ = ["prepare_caches", "load_prepared_datasets", "load_text_backbone", "write_ragged_bank"]
