"""Real two-rank CPU training with rank-zero segmentation/motion evaluators."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _npy_fixture import write_npy_dataset
import test_patch_training_smoke as patch_smoke


def _event(directory: Path, kind: str, **values):
    rank = int(os.environ["RANK"])
    with (directory / f"events-rank{rank}.jsonl").open("a") as file:
        file.write(json.dumps({"rank": rank, "kind": kind, **values}) + "\n")


class _Segmentation:
    def __init__(self, config, options, *, device):
        self.directory = Path(config["test_directory"])
        self.failure = config["test_failure"]
        self.protocol_hash = "distributed-segmentation-v1"
        _event(self.directory, "segmentation_init")

    def evaluate(self, encoder, *, pretrain_epoch):
        assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
        _event(self.directory, "segmentation_eval", epoch=pretrain_epoch)
        if self.failure == "segmentation" and pretrain_epoch == 1:
            raise ValueError("injected segmentation failure after one training epoch")
        values = {"frame_map": .2 + .1 * pretrain_epoch, "bce": .4, "micro_f1": .3}
        return {"best_epoch": 1, "protocol_hash": self.protocol_hash,
                "best_val": {"babel-120": values, "babel-60": values}}


class _Motion:
    def __init__(self, config, options, *, device):
        self.directory = Path(config["test_directory"])
        self.failure = config["test_failure"]
        self.protocol_hash = "distributed-motion-v1"
        self.calls = 0
        _event(self.directory, "motion_init")

    def evaluate(self, encoder):
        assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
        epoch = self.calls
        self.calls += 1
        _event(self.directory, "motion_eval", epoch=epoch)
        if self.failure == "motion" and epoch == 1:
            raise ValueError("injected motion failure after one training epoch")
        return {"protocol_hash": self.protocol_hash, "retrieval": {"recall_at_1": .1 * epoch},
                "elapsed_seconds": .01}


def _worker(rank: int, config: dict, port: int):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    import train
    from utils.distributed import cleanup_distributed

    original_save = train._save_checkpoint
    original_seed = train._seed_all
    saves = []

    def stagger_rank_one(*args, **kwargs):
        original_seed(*args, **kwargs)
        if rank == 1:
            # Force the realistic scheduling order in which rank zero has
            # created the directory before rank one reaches output setup.
            # This catches a racy per-rank "directory already exists" guard.
            deadline = time.monotonic() + 2
            while not Path(config["logging"]["folder"]).exists() and time.monotonic() < deadline:
                time.sleep(.005)

    def record_save(path, **kwargs):
        saves.append(copy.deepcopy({
            "next_epoch": kwargs["next_epoch"], "global_step": kwargs["global_step"],
            "segmentation": kwargs["segmentation_probe_state"],
            "motion": kwargs["online_metrics_latest"],
        }))
        return original_save(path, **kwargs)

    result = {"rank": rank, "saves": saves}
    try:
        with patch("experiment.segmentation_probe.OnlineSegmentationProbe", _Segmentation), \
             patch("experiment.motion_online_metrics.MotionOnlineMetrics", _Motion), \
             patch("train._seed_all", stagger_rank_one), \
             patch("train._save_checkpoint", record_save):
            result.update(status="ok", training=train.main(config, device="cpu"))
    except Exception as error:
        result.update(status="error", error=f"{type(error).__name__}: {error}")
        print(f"rank {rank}: {traceback.format_exc()}", file=sys.stderr, flush=True)
    finally:
        # Both ranks must reach this barrier even when rank-zero evaluation
        # fails. The parent subprocess timeout makes a collective hang fail.
        (Path(config["test_directory"]) / f"result-rank{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.barrier()
        cleanup_distributed()


def _run_case(directory: Path, failure: str):
    dataset = directory / "dataset"
    write_npy_dataset(dataset, [np.random.default_rng(i).normal(size=(6, 6)).astype(np.float32)
                                for i in range(2)])
    config = patch_smoke.PatchTrainingSmokeTest()._config(dataset, directory / "output")
    config["data"]["batch_size"] = 1
    config["optimization"]["epochs"] = 2
    config["logging"].update(tensorboard=False, checkpoint_freq=1)
    config["linear_probe"] = {"enabled": False}
    config["attentive_probe"] = {"enabled": False}
    config["segmentation_probe"] = {"enabled": True, "frequency": 1}
    config["online_metrics"] = {"enabled": True, "kind": "motion", "frequency": 1}
    config["test_directory"], config["test_failure"] = str(directory), failure
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ)
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--launch", str(path), str(port)],
        cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, timeout=75,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = [json.loads((directory / f"result-rank{rank}.json").read_text()) for rank in range(2)]
    events = []
    for rank in range(2):
        event_path = directory / f"events-rank{rank}.jsonl"
        if event_path.exists():
            events.extend(json.loads(line) for line in event_path.read_text().splitlines())
    checkpoint = torch.load(directory / "output/patch-smoke-latest.pth.tar", map_location="cpu", weights_only=False)
    return results, events, checkpoint


def test_two_cpu_ranks_broadcast_new_evaluations_and_save_consistent_states():
    with tempfile.TemporaryDirectory() as temporary:
        results, events, checkpoint = _run_case(Path(temporary), "none")
    assert all(result["status"] == "ok" for result in results), results
    assert all(result["training"]["global_step"] == 2 for result in results)
    assert {event["rank"] for event in events} == {0}
    for kind in ("segmentation", "motion"):
        assert sum(event["kind"] == f"{kind}_init" for event in events) == 1
        assert [event["epoch"] for event in events if event["kind"] == f"{kind}_eval"] == [0, 1, 2]
    # An untouched checkpoint is persisted before the potentially long epoch-0
    # head fit, followed by its evaluated state and the two trained epochs.
    assert [state["next_epoch"] for state in results[0]["saves"]] == [0, 0, 1, 2]
    assert len(results[1]["saves"]) == 4
    for left, right in zip(results[0]["saves"], results[1]["saves"]):
        assert left["segmentation"] == right["segmentation"]
        assert left["global_step"] == right["global_step"]
        if left["motion"] is None:
            assert right["motion"] is None
            continue
        # Rank zero adds local timing bookkeeping after broadcast; the actual
        # evaluation, protocol and epoch state must be identical on both ranks.
        for name in ("retrieval", "protocol_hash", "pretrain_epoch", "global_step"):
            assert left["motion"][name] == right["motion"][name]
    assert checkpoint["next_epoch"] == checkpoint["global_step"] == 2
    assert checkpoint["world_size"] == 2
    assert len(checkpoint["rng_states"]) == len(checkpoint["mask_states"]) == 2
    assert checkpoint["segmentation_probe_state"]["best_epoch"] == 2
    assert checkpoint["segmentation_probe_state"] == results[0]["saves"][-1]["segmentation"]
    assert checkpoint["online_metrics_latest"]["pretrain_epoch"] == 2


@pytest.mark.parametrize("failure", ["segmentation", "motion"])
def test_rank_zero_evaluation_failure_reaches_both_cpu_ranks_without_deadlock(failure):
    with tempfile.TemporaryDirectory() as temporary:
        results, events, checkpoint = _run_case(Path(temporary), failure)
    assert {event["rank"] for event in events} == {0}
    assert all(result["status"] == "error" for result in results), results
    assert all(f"injected {failure} failure" in result["error"] for result in results)
    assert all("Rank-zero evaluation failed" in result["error"] for result in results)
    assert [event["epoch"] for event in events if event["kind"] == f"{failure}_eval"] == [0, 1]
    assert checkpoint["next_epoch"] == 0
    assert all(len(result["saves"]) == 2 for result in results)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--launch":
        raise SystemExit("This test subprocess expects --launch CONFIG PORT")
    mp.spawn(_worker, args=(json.loads(Path(sys.argv[2]).read_text()), int(sys.argv[3])),
             nprocs=2, join=True)
