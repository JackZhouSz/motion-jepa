"""Dual online probes: frozen features, schedules, independent bests and resume."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from test_online_babel_probe import _write_babel_subset, _probe_summary, _Writer
from experiment.linear_probe.attentive import AttentiveProbe
from experiment.linear_probe.online import OnlineBabelProbes
from experiment.linear_probe.online_attentive import OnlineAttentiveBabelProbes
from model import MotionPatchTransformer1D
from train import main as train_main, _evaluate_online_probe_preserving_rng


class FakeLinear:
    calls = 0
    scores = ((.4, .5), (.8, .6), (.7, .9), (.6, .7))

    def __init__(self, config, options, *, device):
        self.epochs, self.learning_rate = 2, .3

    def evaluate(self, encoder):
        assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
        torch.rand(17)  # Must not perturb the pretraining RNG stream.
        scores = self.scores[type(self).calls]
        type(self).calls += 1
        return {name: _probe_summary(value) for name, value in zip(('babel-60', 'babel-120'), scores)}


class FakeAttentive(FakeLinear):
    calls = 0
    scores = ((.5, .4), (.6, .9), (.9, .8), (.8, .7))


def config_for(root, *, epochs=3):
    dataset = root/'pretrain'
    write_npy_dataset(dataset, [np.random.default_rng(i).normal(size=(6,6)).astype(np.float32) for i in range(2)])
    return {
        'data': dict(batch_size=2, root_path=str(dataset), meta_files=['train.txt'], num_workers=0,
            pin_mem=False, persistent_workers=False, drop_last=True, num_frames=6, fps=60,
            motion_dim=6, num_joints=30, normalize=False, stats_path=None),
        'patch': {'temporal_patch_size':3},
        'logging': dict(folder=str(root/'output'), write_tag='dual', log_freq=1, checkpoint_freq=2, tensorboard=True),
        'mask': dict(allow_overlap=False, num_enc_masks=1, num_pred_masks=1,
            enc_frame_mask_ratio=[.5,.5], pred_frame_mask_ratio=[.5,.5]),
        'meta': dict(seed=0, load_checkpoint=False, read_checkpoint=None,
            model_name='mot_patch_tiny_1d', predictor_name='mot_predictor_patch_tiny_1d',
            use_bfloat16=False, use_float16=False),
        'optimization': dict(ema=[.9,1.], epochs=epochs, final_lr=1e-5, final_weight_decay=.4,
            ipe_scale=1., lr=1e-3, start_lr=1e-4, warmup=0, weight_decay=.04),
        'linear_probe': dict(enabled=True, frequency=2, standardize=True,
            datasets={'babel-60':'unused-60','babel-120':'unused-120'}),
        'attentive_probe': dict(enabled=True, frequency=2),
    }


class DualOnlineProbeTest(unittest.TestCase):
    def setUp(self):
        FakeLinear.calls = FakeAttentive.calls = 0

    def test_real_attentive_and_standardized_linear_train_frozen_encoder_and_restore_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stats=root/'stats';stats.mkdir()
            np.save(stats/'mean.npy',np.zeros(6,dtype=np.float32))
            np.save(stats/'std.npy',np.ones(6,dtype=np.float32))
            paths={}
            for subset in (60,120):
                data=root/f'babel-{subset}';_write_babel_subset(data,subset);paths[f'babel-{subset}']=str(data)
            config={'data':dict(root_path=str(root),stats_path='stats',num_frames=4,fps=30,motion_dim=6),
                    'meta':{'use_bfloat16':False}}
            options=dict(datasets=paths,epochs=2,warmup_epochs=0,feature_batch_size=2,batch_size=2,num_workers=0,seed=42)
            encoder=MotionPatchTransformer1D(6,4,temporal_patch_size=2,embed_dim=12,depth=1,num_heads=3).eval().requires_grad_(False)
            original={k:v.clone() for k,v in encoder.state_dict().items()}
            heads=[]
            def make_head(*args,**kwargs):
                head=AttentiveProbe(*args,**kwargs);heads.append(head);return head
            for cls in (OnlineBabelProbes,OnlineAttentiveBabelProbes):
                evaluator=cls(config,options,device=torch.device('cpu'))
                rng=torch.get_rng_state().clone()
                with patch('experiment.linear_probe.online_attentive.AttentiveProbe',side_effect=make_head):
                    result=_evaluate_online_probe_preserving_rng(evaluator,encoder)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for subset in (60,120):
                    summary=result[f'babel-{subset}']
                    self.assertEqual(summary['split_counts'],dict(train=4,val=2,test=0))
                    self.assertIsNone(summary['test'])
                    self.assertEqual(summary['best_val']['classes_without_positives'],subset-2)
                    self.assertEqual(summary['standardization'], 'train_channel_zscore' if cls is OnlineBabelProbes else 'none')
            self.assertTrue(all(any(p.grad is not None for p in h.parameters()) for h in heads))
            self.assertFalse(any(p.grad is not None for p in encoder.parameters()))
            for key,value in encoder.state_dict().items():torch.testing.assert_close(value,original[key],rtol=0,atol=0)
            self.assertFalse(list(root.rglob('*.pt')))

    def test_independent_best_checkpoints_period_final_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=config_for(root)
            writer=_Writer()
            with patch('experiment.linear_probe.online.OnlineBabelProbes',FakeLinear), \
                 patch('experiment.linear_probe.online_attentive.OnlineAttentiveBabelProbes',FakeAttentive), \
                 patch('train._make_tensorboard_writer',return_value=writer):
                train_main(config,device='cpu')
                resumed=copy.deepcopy(config);resumed['meta']['load_checkpoint']=True
                resumed['optimization']['epochs']=4
                train_main(resumed,device='cpu')
            self.assertEqual((FakeLinear.calls,FakeAttentive.calls),(4,4))
            output=root/'output'
            latest=torch.load(output/'dual-latest.pth.tar',weights_only=False)
            for kind,key,bests in [('linear_probe','babel_probe_state',(2,3)),('attentive_probe','attentive_probe_state',(3,2))]:
                for name,epoch in zip(('babel-60','babel-120'),bests):
                    self.assertEqual(latest[key][name]['best_epoch'],epoch)
                    self.assertEqual(latest[key][name]['latest']['pretrain_epoch'],4)
                    prefix='attentive-' if kind=='attentive_probe' else ''
                    best=torch.load(output/f'dual-best-{prefix}{name}-map.pth.tar',weights_only=False)
                    self.assertEqual(best['next_epoch'],epoch)
                    steps=[step for tag,_,step in writer.scalars if tag==f'{kind}/{name}/val_mean_average_precision']
                    self.assertEqual(steps,[0,2,3,4])
            # RNG-consuming evaluators must leave actual pretraining weights unchanged.
            control=copy.deepcopy(config);control['logging']['folder']=str(root/'control')
            control['linear_probe']['enabled']=False;control['attentive_probe']['enabled']=False
            with patch('train._make_tensorboard_writer',return_value=None):train_main(control,device='cpu')
            baseline=torch.load(root/'control'/'dual-latest.pth.tar',weights_only=False)
            reference=torch.load(output/'dual-best-babel-120-map.pth.tar',weights_only=False)
            for key in ('encoder','predictor','target_encoder'):
                for name,value in baseline[key].items():torch.testing.assert_close(value,reference[key][name],rtol=0,atol=0)

    def test_legacy_raw_resume_resets_linear_best_and_initializes_attentive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=config_for(root,epochs=1)
            config['linear_probe']['standardize']=False
            config['attentive_probe']['enabled']=False
            with patch('experiment.linear_probe.online.OnlineBabelProbes',FakeLinear), \
                 patch('experiment.linear_probe.online_attentive.OnlineAttentiveBabelProbes',FakeAttentive), \
                 patch('train._make_tensorboard_writer',return_value=None):
                train_main(config,device='cpu')
                path=root/'output'/'dual-latest.pth.tar'
                old=torch.load(path,weights_only=False)
                old.pop('probe_protocols');old.pop('attentive_probe_state')
                old['config']['linear_probe'].pop('standardize')
                for state in old['babel_probe_state'].values():state['best_val_map']=.99
                torch.save(old,path)
                config['meta']['load_checkpoint']=True
                config['linear_probe']['standardize']=True
                config['attentive_probe']['enabled']=True
                # No further optimization required: evaluate the resumed epoch itself.
                train_main(config,device='cpu')
            latest=torch.load(path,weights_only=False)
            self.assertEqual(latest['next_epoch'],1)
            self.assertEqual((FakeLinear.calls,FakeAttentive.calls),(3,1))
            for state in latest['babel_probe_state'].values():
                self.assertEqual(state['best_epoch'],1)
                self.assertLess(state['best_val_map'],.99)
            for state in latest['attentive_probe_state'].values():self.assertEqual(state['latest']['pretrain_epoch'],1)
            for key in ('encoder','predictor','target_encoder'):
                for name,value in old[key].items():torch.testing.assert_close(value,latest[key][name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
