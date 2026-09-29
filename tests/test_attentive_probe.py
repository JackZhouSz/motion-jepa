"""Attentive readout masking, frozen-token integration and exact resume."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiment.linear_probe.attentive import AttentiveProbe
from experiment.linear_probe import train_classifier as runner
from experiment.linear_probe import attentive_sweep
from experiment.linear_probe.attentive_sweep import verify_linear_baselines
from experiment.linear_probe.features import load_frozen_encoder, _sha256_file
from model.token_layout import TokenLayout
from test_babel_classifier import _write_babel_dataset, _args
from test_linear_probe import _write_checkpoint
from test_online_babel_probe import _write_babel_subset


class AttentiveProbeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_padding_invariance_and_empty_mask(self):
        model = AttentiveProbe(12, 4, 3).eval()
        x = torch.randn(2, 4, 12)
        active = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        changed = x.masked_fill(~active.unsqueeze(-1), float("nan"))
        torch.testing.assert_close(model(x, active), model(changed, active))
        with self.assertRaisesRegex(ValueError, "valid token"):
            model(x, torch.zeros_like(active))
        model(x, active).sum().backward()
        self.assertIsNotNone(model.query.grad)
        self.assertEqual(runner.MODELS, ("cnn", "transformer"))

    def test_partial_patch_is_excluded(self):
        layout = TokenLayout(kind="1d", patchified=True, raw_num_frames=6,
                             token_num_frames=2, temporal_patch_size=3)
        raw_active = torch.arange(6)[None, :] < torch.tensor([4])[:, None]
        active = layout.valid_token_mask(raw_active)
        self.assertEqual(layout.valid_token_lengths(torch.tensor([4])).tolist(), [1])
        model = AttentiveProbe(12, 2, 3).eval()
        x = torch.randn(1, 2, 12)
        changed = x.clone(); changed[:, 1] = 1e5
        torch.testing.assert_close(model(x, active), model(changed, active))

    def test_frozen_encoder_and_token_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"; _write_babel_dataset(data)
            checkpoint = root / "encoder.pth.tar"
            _write_checkpoint(checkpoint, data / "stats")
            encoder, _, _ = load_frozen_encoder(checkpoint, "target_encoder", torch.device("cpu"))
            before = copy.deepcopy(encoder.state_dict())
            args = _args(data, root / "out", source="jepa", checkpoint=checkpoint)
            prepared = runner._prepare_input(args, device=torch.device("cpu"))
            self.assertEqual(len(prepared.datasets["test"]), 0)
            motion_datasets, _ = runner.build_classification_datasets(
                data, num_frames=4, fps=30, motion_dim=6, stats_root=data / "stats")
            extracted = runner._extract_token_features(encoder, motion_datasets['train'],
                device=torch.device('cpu'), batch_size=4, num_workers=0)
            tokens = extracted['features']
            head = AttentiveProbe(tokens.shape[-1], tokens.shape[1], 60)
            head(tokens.float()).sum().backward()
            self.assertTrue(any(p.grad is not None for p in head.parameters()))
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in encoder.parameters()))
            for key, value in before.items():
                torch.testing.assert_close(value, encoder.state_dict()[key])
            cache = root / "linear-probe/token-features/babel-60/train.pt"
            payload = torch.load(cache, weights_only=False)
            payload['metadata']['checkpoint_sha256'] = 'wrong'
            torch.save(payload, cache)
            with self.assertRaisesRegex(ValueError, "stale"):
                runner._prepare_input(args, device=torch.device("cpu"))

    def test_babel_training_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "data"
            _write_babel_dataset(data)
            checkpoint = root / "encoder.pth.tar"
            _write_checkpoint(checkpoint, data / "stats")
            args = _args(data, root / "full", source="jepa", checkpoint=checkpoint)
            args.model = "attentive"
            args.epochs = 3
            full = runner.run(args)["attentive"]
            args.output_root = root / "resumed"
            original_save = runner._atomic_torch_save
            def interrupt(value, path):
                original_save(value, path)
                if path.name == "classifier-latest.pth.tar" and value.get("next_epoch") == 1:
                    raise RuntimeError("interrupted")
            with mock.patch.object(runner, "_atomic_torch_save", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    runner.run(args)
            args.resume = True
            resumed = runner.run(args)["attentive"]
            self.assertEqual(full["best_val"], resumed["best_val"])
            self.assertIsNone(resumed["test"])
            self.assertEqual(resumed["best_val"]["classes_with_positives"], 2)
            sub = Path("attentive/seed-42")
            for filename in ("classifier-best.pth.tar", "classifier-latest.pth.tar"):
                left = torch.load(root / "full" / sub / filename, weights_only=False)
                right = torch.load(root / "resumed" / sub / filename, weights_only=False)
                for key in left["model"]:
                    torch.testing.assert_close(left["model"][key], right["model"][key], atol=0, rtol=0)
            with (root / "resumed" / sub / "metrics.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual(resumed["best_val"]["mean_average_precision"], max(float(r["val_mean_average_precision"]) for r in rows))
            self.assertEqual(runner.run(args)["attentive"], resumed)

    def test_raw_input_rejected(self):
        args = runner.build_parser().parse_args(["--model", "attentive"])
        with self.assertRaisesRegex(ValueError, "input-source jepa"):
            runner.run(args)

    def test_baseline_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); data = root / "data"
            _write_babel_dataset(data)
            checkpoint = root / "encoder.pth.tar"
            _write_checkpoint(checkpoint, data / "stats")
            args = _args(data, root / "out", source="jepa", checkpoint=checkpoint)
            prepared = runner._prepare_input(args, device=torch.device("cpu"))
            prepared.jepa_source['comparison_dataset_root'] = str(data)
            source = prepared.jepa_source
            signature = {key: source[key] for key in ('checkpoint_sha256','checkpoint_key','stats_mean_sha256','stats_std_sha256')}
            signature['dataset_files'] = {'index.json': _sha256_file(data/'index.json')}
            baseline = dict(dataset='babel-60',encoder_epoch=100,signature=signature,protocol={'pooling':'valid_token_mean'})
            (root/'results').mkdir()
            path=root/'results/result.json'; path.write_text(json.dumps(baseline))
            self.assertEqual(len(verify_linear_baselines(root,checkpoint,100,'babel-60',prepared)),1)
            signature['checkpoint_sha256']='wrong';path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError,'checkpoint/statistics mismatch'):
                verify_linear_baselines(root,checkpoint,100,'babel-60',prepared)
            signature['checkpoint_sha256']=source['checkpoint_sha256']
            signature['dataset_files']['index.json']='wrong';path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError,'dataset mismatch'):
                verify_linear_baselines(root,checkpoint,100,'babel-60',prepared)

    def test_sweep_end_to_end_and_completed_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data60, data120 = root/'babel-60', root/'babel-120'
            _write_babel_subset(data60,60); _write_babel_subset(data120,120)
            checkpoint=root/'ep100.pth.tar'
            _write_checkpoint(checkpoint,data60/'stats')
            state=torch.load(checkpoint,weights_only=False); state['next_epoch']=100
            torch.save(state,checkpoint)
            baseline_root=root/'linear'; (baseline_root/'results').mkdir(parents=True)
            for data in (data60,data120):
                signature=dict(checkpoint_sha256=_sha256_file(checkpoint),checkpoint_key='target_encoder',
                    stats_mean_sha256=_sha256_file(data60/'stats/mean.npy'),stats_std_sha256=_sha256_file(data60/'stats/std.npy'),
                    dataset_files={name:_sha256_file(data/name) for name in ('index.json','meta.json','train.txt','val.txt')})
                baseline=dict(dataset=data.name,encoder_epoch=100,lr=.3,signature=signature,
                    protocol={'pooling':'valid_token_mean'},summary={'best_val':{'mean_average_precision':.5}},
                    history=[{'head_epoch':1,'top1_label_row_accuracy':.5}])
                (baseline_root/'results'/f'{data.name}.json').write_text(json.dumps(baseline))
            args=attentive_sweep.build_parser().parse_args([
                '--checkpoints',str(checkpoint),'--babel-60-root',str(data60),'--babel-120-root',str(data120),
                '--linear-baseline-root',str(baseline_root),'--output-root',str(root/'out'),
                '--epochs','6','--lrs','.0003','--device','cpu','--feature-workers','0'])
            comparison=attentive_sweep.run(args)
            self.assertEqual(len(comparison['mAP_selected']),2)
            self.assertTrue((root/'out/comparison.png').is_file())
            with mock.patch.object(runner,'train_epoch',side_effect=AssertionError('must reuse complete runs')):
                self.assertEqual(attentive_sweep.run(args),comparison)
            args.lrs=[.001]
            with self.assertRaisesRegex(ValueError,'manifest differs'):
                attentive_sweep.run(args)


if __name__ == '__main__':
    unittest.main()
