import unittest
import json
import tempfile
from pathlib import Path

import torch

from experiment.linear_probe.standardization import ChannelStandardizer
from experiment.linear_probe.dataset import MultiLabelIndex
from experiment.linear_probe.train_probe import train_multilabel_probe, _evaluate_multilabel_linear_probe


class ChannelStandardizerTest(unittest.TestCase):
    def test_train_statistics_are_reused_without_validation_leakage(self):
        train = torch.tensor([[1., 10.], [3., 14.]])
        validation = torch.tensor([[101., 22.]])
        scaler = ChannelStandardizer.fit(train)
        torch.testing.assert_close(scaler.transform(train), torch.tensor([[-1., -1.], [1., 1.]]))
        torch.testing.assert_close(scaler.transform(validation), torch.tensor([[99., 5.]]))
        torch.testing.assert_close(scaler.mean, torch.tensor([2., 12.]))
        torch.testing.assert_close(train, torch.tensor([[1., 10.], [3., 14.]]))

    def test_constant_channels_and_invalid_inputs(self):
        scaler = ChannelStandardizer.fit(torch.tensor([[1., 2.], [1., 4.]]))
        transformed = scaler.transform(torch.tensor([[1., 3.]]))
        torch.testing.assert_close(transformed, torch.zeros(1, 2))
        self.assertTrue(torch.isfinite(scaler.transform(torch.tensor([[2., 3.]]))).all())
        with self.assertRaises(ValueError):
            ChannelStandardizer.fit(torch.empty(0, 2))
        with self.assertRaises(ValueError):
            ChannelStandardizer.fit(torch.tensor([[float('nan')]]))
        with self.assertRaises(ValueError):
            scaler.transform(torch.zeros(2, 3))

    def test_saved_head_matches_report_and_optional_output_preserves_results(self):
        ids = [f'motion-{i}' for i in range(8)]
        rows = {name: (i % 2,) for i, name in enumerate(ids)}
        index = MultiLabelIndex(('a', 'b', 'absent'), {'a': 0, 'b': 1, 'absent': 2}, rows, rows, {})
        x = torch.randn(8, 4, generator=torch.Generator().manual_seed(17))
        y = torch.zeros(8, 3)
        y[torch.arange(8), torch.arange(8) % 2] = 1
        caches = {
            'train': {'features': x[:6], 'labels': y[:6], 'sample_ids': ids[:6]},
            'val': {'features': x[6:], 'labels': y[6:], 'sample_ids': ids[6:]},
        }
        kwargs = dict(label_index=index, device=torch.device('cpu'), epochs=3,
                      batch_size=2, learning_rate=.3, momentum=.9, weight_decay=0., seed=42)
        baseline = train_multilabel_probe(caches, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            saved = train_multilabel_probe(caches, output=output, **kwargs)
            self.assertEqual(saved, baseline)
            head = torch.nn.Linear(4, 3)
            checkpoint = torch.load(output/'linear-probe-best.pth.tar', weights_only=True)
            head.load_state_dict(checkpoint['classifier'])
            metrics = _evaluate_multilabel_linear_probe(head, caches['val'],
                label_index=index, device=torch.device('cpu'), batch_size=2)
            self.assertEqual(metrics, saved['best_val'])
            self.assertEqual(metrics['classes_without_positives'], 1)
            self.assertIsNone(saved['test'])
            history = json.loads((output/'metrics.json').read_text())
            self.assertEqual(len(history), 3)


if __name__ == '__main__':
    unittest.main()
