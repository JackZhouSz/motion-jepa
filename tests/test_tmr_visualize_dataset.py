"""Cached caption-to-motion joins and viewer selection without text inference."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from _npy_fixture import write_npy_dataset
from experiment.tmr.dataset import json_digest
from experiment.tmr.visualize_dataset import TMRDatasetViewer, load_cached_entries
from visualization import load_motion
from visualization.dataset_viewer import ViewerSession


class TMRVisualizationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "dataset"
        self.cache = Path(self.directory.name) / "cache"
        features = np.load(
            Path(__file__).parent / "assets/motion_jepa_golden.npz", allow_pickle=False,
        )["features"].astype(np.float32)
        write_npy_dataset(self.root, [features, features], num_frames=len(features))
        self.paired = {"records": [], "catalog": {}, "provenance": {
            "dataset_root": str(self.root), "fps": 60,
        }}
        for index, caption in enumerate(("Walk forward. Turn left.", "Raise the right hand.")):
            caption_id = hashlib.sha256(caption.encode()).hexdigest()
            self.paired["catalog"][caption_id] = caption
            self.paired["records"].append({
                "sample_id": f"clip-{index}", "source_id": "source-motion",
                "motion_path": f"motions/train/sample-{index}.npy", "split": "train",
                "length": len(features), "fps": 60,
                "start_frame": index * len(features), "end_frame": (index + 1) * len(features),
                "caption_id": caption_id,
            })
        # Cached split selection must take precedence over the original split file.
        self.paired["records"][1]["split"] = "val"
        self._save_cache()

    def _save_cache(self):
        self.paired["paired_index_sha256"] = json_digest(self.paired["records"])
        self.paired["catalog_sha256"] = json_digest(self.paired["catalog"])
        for source in ("raw", "jepa"):
            directory = self.cache / source
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "paired-index.json").write_text(json.dumps(self.paired))
            (directory / "prepared.json").write_text(json.dumps({
                "format_version": 1, "input_source": source,
                "paired_index_sha256": self.paired["paired_index_sha256"],
                "catalog_sha256": self.paired["catalog_sha256"],
                "provenance": self.paired["provenance"],
            }))

    def test_both_sources_use_cached_caption_and_original_motion(self):
        for source in ("raw", "jepa"):
            root, entries = load_cached_entries(self.cache, input_source=source, split="val")
            self.assertEqual(root, self.root)
            self.assertEqual([entry.id for entry in entries], ["clip-1"])
            self.assertEqual(entries[0].caption, "Raise the right hand.")
            self.assertEqual(entries[0].start_frame, 8)
            decoded, fps = load_motion(entries[0], fps=entries[0].fps)
            self.assertEqual(decoded["posed_joints"].shape, (8, 77, 3))
            self.assertEqual(fps, 60)
        # There are deliberately no annotation, token cache, stats or model files.

    def test_limit_empty_split_and_relocated_dataset(self):
        self.paired["records"][1]["split"] = "train"
        self._save_cache()
        self.assertEqual(len(load_cached_entries(self.cache, limit=1)[1]), 1)
        self.assertEqual(len(load_cached_entries(self.cache, limit=0)[1]), 2)
        with self.assertRaisesRegex(FileNotFoundError, "split test"):
            load_cached_entries(self.cache, split="test")
        relocated = self.root.with_name("relocated")
        self.root.rename(relocated)
        with self.assertRaisesRegex(FileNotFoundError, "--dataset-root"):
            load_cached_entries(self.cache)
        root, entries = load_cached_entries(self.cache, dataset_root=relocated)
        self.assertEqual(root, relocated)
        self.assertTrue(entries[0].path.is_file())

    def test_rejects_stale_catalog_and_missing_caption(self):
        path = self.cache / "raw/paired-index.json"
        changed = json.loads(path.read_text())
        caption_id = changed["records"][0]["caption_id"]
        changed["catalog"][caption_id] = "Incorrect caption."
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "catalog does not match"):
            load_cached_entries(self.cache)
        del self.paired["catalog"][caption_id]
        self._save_cache()
        with self.assertRaisesRegex(ValueError, "Missing cached caption"):
            load_cached_entries(self.cache)

    def test_relative_path_validation_allows_symlinked_subsets(self):
        original = self.root / "motions/train/sample-0.npy"
        target = self.root.parent / "original.npy"
        original.rename(target)
        original.symlink_to(target)
        self.assertTrue(load_cached_entries(self.cache)[1][0].path.is_file())
        self.paired["records"][0]["motion_path"] = "../original.npy"
        self._save_cache()
        with self.assertRaisesRegex(ValueError, "relative cached motion path"):
            load_cached_entries(self.cache)

    def test_candidate_cache_displays_descriptions_separately(self):
        record = self.paired["records"][0]
        alternative = "A person strolls forward."
        alternative_id = hashlib.sha256(alternative.encode()).hexdigest()
        self.paired["catalog"][alternative_id] = alternative
        record["caption_ids"] = [record["caption_id"], alternative_id]
        self._save_cache()
        path = self.cache / "raw/prepared.json"
        prepared = json.loads(path.read_text())
        prepared["format_version"] = 2
        path.write_text(json.dumps(prepared))
        _, entries = load_cached_entries(self.cache)
        self.assertEqual(entries[0].captions, (entries[0].caption, alternative))
        viewer = TMRDatasetViewer.__new__(TMRDatasetViewer)
        viewer.entries, viewer.split = entries, "train"
        gui = {"info": SimpleNamespace(content=""), "caption": SimpleNamespace(content="")}
        viewer._update_pair_info(ViewerSession(client=Mock(), gui=gui))
        self.assertIn("Caption candidates (2)", gui["caption"].content)
        self.assertIn("1. Walk forward. Turn left.", gui["caption"].content)
        self.assertIn("2. A person strolls forward.", gui["caption"].content)
        self.assertIn("One candidate is sampled", gui["caption"].content)
        self.assertNotIn("legacy cache", gui["info"].content)

    def test_caption_and_interval_follow_motion_selection(self):
        self.paired["records"][1]["split"] = "train"
        self._save_cache()
        _, entries = load_cached_entries(self.cache)
        viewer = TMRDatasetViewer.__new__(TMRDatasetViewer)
        viewer.entries, viewer.fps, viewer.split = entries, 60, "train"
        viewer.normalized, viewer.stats_root, viewer.default_mesh = False, None, False
        gui = {
            "info": SimpleNamespace(content=""), "caption": SimpleNamespace(content=""),
            "frame": SimpleNamespace(max=1, value=0), "fps": SimpleNamespace(value=30),
        }
        session = ViewerSession(client=Mock(), gui=gui)
        with patch("visualization.dataset_viewer.MotionRenderer") as renderer:
            renderer.return_value.length = 8
            viewer._load_entry(session, 0)
            self.assertEqual(gui["caption"].content, entries[0].caption)
            self.assertIn("Combined caption (legacy cache)", gui["info"].content)
            viewer._load_entry(session, 1)
            self.assertEqual(gui["caption"].content, entries[1].caption)
            self.assertIn("Motion 2 / 2", gui["info"].content)
            self.assertIn("0.13–0.27 s", gui["info"].content)
            self.assertEqual(gui["frame"].value, 0)
            self.assertEqual(gui["fps"].value, 60)
            motion = renderer.call_args.args[1]
            self.assertEqual(motion["posed_joints"].shape, (8, 77, 3))


if __name__ == "__main__":
    unittest.main()
