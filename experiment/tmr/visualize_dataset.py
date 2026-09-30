"""Browse existing TMR cache pairs in viser without extracting any features."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Support both `python -m experiment.tmr.visualize_dataset` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment.tmr.dataset import DEFAULT_CACHE_ROOT, SPLITS, json_digest
from visualization import MotionEntry, MotionJEPADatasetViewer, read_dataset_fps
from visualization.dataset_viewer import ViewerSession


@dataclass(frozen=True, kw_only=True)
class TMRMotionEntry(MotionEntry):
    source_id: str
    start_frame: int
    end_frame: int
    fps: int
    caption_id: str
    captions: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return self.id


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prepared TMR cache metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_cached_entries(
    cache_root: Path,
    *,
    input_source: str = "raw",
    split: str = "train",
    limit: int = 500,
    dataset_root: Path | None = None,
) -> tuple[Path, list[TMRMotionEntry]]:
    """Read saved captions/locators; render the original unnormalized motions.

    JEPA tokens cannot be inverted to joints, so their paired raw files are used
    too. Text/token arrays, backbones, annotations, and statistics are not loaded.
    """
    if input_source not in ("raw", "jepa") or split not in SPLITS:
        raise ValueError("Expected input_source raw/jepa and split train/val/test")
    directory = Path(cache_root).expanduser().resolve() / input_source
    prepared = _read_json(directory / "prepared.json")
    paired = _read_json(directory / "paired-index.json")
    if prepared.get("format_version") not in (1, 2) or prepared.get("input_source") != input_source:
        raise ValueError(f"Unsupported or mismatched prepared TMR cache: {directory}")
    for field, data in (("paired_index_sha256", "records"), ("catalog_sha256", "catalog")):
        digest = json_digest(paired[data])
        if paired.get(field) != digest or prepared.get(field) != digest:
            raise ValueError(f"TMR cache {data} does not match its manifest: {directory}")
    if paired["provenance"] != prepared["provenance"]:
        raise ValueError(f"TMR cache provenance does not match its manifest: {directory}")
    root = Path(dataset_root or paired["provenance"]["dataset_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Cached motion dataset is unavailable: {root}; use --dataset-root")
    fps = read_dataset_fps(root)
    metadata = _read_json(root / "meta.json")
    if metadata.get("representation") != "motion_jepa_366_v1" or metadata.get("motion_dim") != 366:
        raise ValueError(f"Visualization requires the original 366D MotionJEPA dataset: {root}")
    if fps != paired["provenance"]["fps"]:
        raise ValueError(f"Dataset FPS differs from the saved TMR cache: {root}")

    records = [record for record in paired["records"] if record["split"] == split]
    if limit > 0:
        records = records[:limit]
    entries: list[TMRMotionEntry] = []
    seen: set[str] = set()
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate cached sample id: {sample_id}")
        seen.add(sample_id)
        relative = Path(record["motion_path"])
        # Original datasets may use symlinked motion files (e.g. smoke subsets).
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a relative cached motion path: {relative}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cached motion is unavailable: {path}; check --dataset-root")
        length = int(record["length"])
        start, end = int(record["start_frame"]), int(record["end_frame"])
        if length <= 0 or start < 0 or end - start != length or record["fps"] != fps:
            raise ValueError(f"Invalid cached motion interval or FPS: {sample_id}")
        caption_id = record["caption_id"]
        caption = paired["catalog"].get(caption_id)
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Missing cached caption for {sample_id}: {caption_id}")
        captions = ()
        if "caption_ids" in record:
            ids = record["caption_ids"]
            if not ids or ids[0] != caption_id or len(set(ids)) != len(ids):
                raise ValueError(f"Invalid cached caption candidates: {sample_id}")
            captions = tuple(paired["catalog"].get(value) for value in ids)
            if any(not isinstance(value, str) or not value.strip() for value in captions):
                raise ValueError(f"Missing cached caption candidate for {sample_id}")
        elif prepared["format_version"] == 2:
            raise ValueError(f"Missing cached caption candidates: {sample_id}")
        entries.append(TMRMotionEntry(
            id=sample_id, path=path, actual_length=length, caption=caption,
            source_id=record["source_id"], start_frame=start, end_frame=end,
            fps=fps, caption_id=caption_id, captions=captions,
        ))
    if not entries:
        raise FileNotFoundError(f"No cached TMR pairs for split {split}: {directory}")
    return root, entries


def _markdown_text(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#!|])", r"\\\1", value)


class TMRDatasetViewer(MotionJEPADatasetViewer):
    def __init__(
        self,
        cache_root: Path,
        *,
        input_source: str = "raw",
        split: str = "train",
        limit: int = 500,
        dataset_root: Path | None = None,
        sample_index: int = 0,
        sample_id: str | None = None,
        host: str = "0.0.0.0",
        port: int = 6006,
        mesh: bool = False,
    ):
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.input_source = input_source
        self.split = split
        root, entries = load_cached_entries(
            self.cache_root, input_source=input_source, split=split,
            limit=limit, dataset_root=dataset_root,
        )
        super().__init__(
            root, split=split, limit=0, entries=entries,
            sample_index=sample_index, sample_id=sample_id, host=host,
            port=port, mesh=mesh, normalized=False, label="TMR Text-Motion Viewer",
        )

    def _update_pair_info(self, session: ViewerSession) -> None:
        entry = self.entries[session.entry_index]
        gui = session.gui
        if gui is None:
            return
        gui["info"].content = (
            f"**Motion {session.entry_index + 1} / {len(self.entries)}** · {self.split}\n\n"
            f"**Clip:** {_markdown_text(entry.id)}\n\n"
            f"**Source:** {_markdown_text(entry.source_id)}\n\n"
            f"**Source interval:** {entry.start_frame / entry.fps:.2f}–"
            f"{entry.end_frame / entry.fps:.2f} s "
            f"({entry.actual_length} frames, {entry.actual_length / entry.fps:.2f} s)"
        )
        if not entry.captions:
            gui["info"].content += "\n\n**Caption format:** Combined caption (legacy cache)"
        if "caption" in gui:
            gui["caption"].content = (
                f"**Caption candidates ({len(entry.captions)})**\n\n"
                + "\n\n".join(f"{index}. {_markdown_text(text)}" for index, text in enumerate(entry.captions, 1))
                + "\n\nOne candidate is sampled for each training pair."
                if entry.captions else _markdown_text(entry.caption)
            )

    def _load_entry(self, session: ViewerSession, index: int) -> None:
        super()._load_entry(session, index)
        self._update_pair_info(session)

    def _on_connect(self, client) -> None:
        super()._on_connect(client)
        session = self.sessions[client.client_id]
        with client.gui.add_folder("Text", order=-1, expand_by_default=True):
            session.gui["caption"] = client.gui.add_markdown("")
        with client.gui.add_folder("Samples", order=-2, expand_by_default=True):
            previous = client.gui.add_button("Previous motion")
            next_ = client.gui.add_button("Next motion")
        self._update_pair_info(session)

        @previous.on_click
        def _(_event):
            index = (session.entry_index - 1) % len(self.entries)
            session.gui["dropdown"].value = self.entries[index].label

        @next_.on_click
        def _(_event):
            index = (session.entry_index + 1) % len(self.entries)
            session.gui["dropdown"].value = self.entries[index].label

    def close(self) -> None:
        if not self.closed:
            super().close()
            self.server.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cached TMR motion-caption pairs with viser.")
    parser.add_argument("input", nargs="?", type=Path, help="Existing TMR cache root")
    parser.add_argument("--cache-root", type=Path, help="Alternative to the positional cache root")
    parser.add_argument("--input-source", choices=("raw", "jepa"), default="raw")
    parser.add_argument("--dataset-root", type=Path, help="Override the original motion dataset location")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--sample-index", "--sample_index", type=int, default=0)
    parser.add_argument("--sample-id", "--sample_id")
    parser.add_argument("--limit", type=int, default=500, help="Maximum pairs to browse; 0 means all")
    parser.add_argument("--host", default=os.environ.get("SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", "6006")))
    parser.add_argument("--mesh", action="store_true")
    args = parser.parse_args(argv)
    if args.input is not None and args.cache_root is not None:
        parser.error("Pass the cache root either positionally or with --cache-root")
    args.cache_root = args.cache_root or args.input or DEFAULT_CACHE_ROOT
    return args


def main() -> None:
    args = parse_args()
    viewer = TMRDatasetViewer(
        args.cache_root, input_source=args.input_source, split=args.split,
        limit=args.limit, dataset_root=args.dataset_root,
        sample_index=args.sample_index, sample_id=args.sample_id,
        host=args.host, port=args.port, mesh=args.mesh,
    )
    print(f"Found {len(viewer.entries)} cached {args.split} motion-caption pair(s) in {viewer.cache_root}")
    host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Open http://{host}:{viewer.server.get_port()}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
