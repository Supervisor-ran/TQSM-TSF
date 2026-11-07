"""Public method smoke tests using only Python stdlib's unittest.

These tests run entirely offline; the integration test replaces only the
Hugging Face text encoder with deterministic local embeddings, not MMMA.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.models.MMMA_NLinear import Model
from src.data_collector.topk_generator import TopKGenerator


class PublicOfflineTests(unittest.TestCase):
    def test_dual_text_forward_and_aux_loss(self):
        torch.manual_seed(42)
        model = Model(dict(seq_len=6, pred_len=2, num_enc_in=1,
            text_encoder_out_dim=8, cross_modality_n_heads=2,
            use_cross_modality=True, use_movement_alignment=True,
            movement_tau=0.07, movement_loss_weight=0.2, use_time_feature=False))
        x = torch.randn(4, 6, 1)
        macro = torch.randn(4, 6, 8)
        micro = torch.randn(4, 6, 8)
        out = model(x_num=x, macro_text=macro, micro_text=micro)
        y = out[0] if isinstance(out, tuple) else out
        self.assertEqual(tuple(y.shape), (4, 2, 1))
        self.assertTrue(bool(torch.isfinite(y).all()))

    def test_single_text_branch_and_ablation(self):
        torch.manual_seed(42)
        for degrade in (False, True):
            with self.subTest(degrade=degrade):
                model = Model(dict(seq_len=6, pred_len=1, num_enc_in=1,
                    text_encoder_out_dim=8, cross_modality_n_heads=2,
                    use_cross_modality=not degrade,
                    use_movement_alignment=False, degrade=degrade,
                    use_time_feature=False))
                out = model(x_num=torch.randn(3, 6, 1),
                    macro_text=torch.randn(3, 6, 8), micro_text=None)
                pred = out[0] if isinstance(out, tuple) else out
                self.assertEqual(tuple(pred.shape), (3, 1, 1))

    def test_balanced_selector_five_plus_five(self):
        generator = TopKGenerator(debug=False)
        records = [dict(scope=s, month='2026-01', abstract=f'{s}{i}',
                        score_full=1.0-i/100)
                   for s in ('macro', 'micro') for i in range(7)]
        selected = generator.select_records(records, k=10,
                                            variant='full', mode='balanced')
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(x['scope'] == 'macro' for x in selected), 5)
        self.assertEqual(sum(x['scope'] == 'micro' for x in selected), 5)

    def test_one_epoch_synthetic_train(self):
        # Mock external model loading ONLY. Use actual MMMA, optimizer,
        # dataloader, metric computation and training/evaluation logic.
        fake_module = types.ModuleType('src.model_util.text_embedding')
        class LocalEmbedding(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
            def forward(self, sequences, device=None):
                return torch.ones(len(sequences), len(sequences[0]), 8, device=device)
        fake_module.TextEmbedding = LocalEmbedding
        n = 35
        numeric = np.linspace(100, 150, n).astype(np.float32)
        dates = [f'{2022 + (m // 12)}-{1 + (m % 12):02d}' for m in range(n)]
        texts = [[f'Event {i}'] for i in range(n)]
        prev = sys.modules.pop('src.model_excutor.seq_prediction_excutor', None)
        import src.model_excutor as executor_package
        previous_attr = getattr(executor_package, 'seq_prediction_excutor', None)
        try:
            with patch.dict(sys.modules,
                            {'src.model_util.text_embedding': fake_module}):
                exe_mod = importlib.import_module('src.model_excutor.seq_prediction_excutor')
                with patch.object(exe_mod, 'load_multimodal_monthly_data',
                                  return_value=(numeric, texts, texts, dates, n)):
                    config = dict(model_name='MMMA_NLinear', x_num_loc='kamakura',
                        seq_len=6, pred_len=1, num_enc_in=1,
                        text_encoder_out_dim=8, use_cross_modality=True,
                        use_movement_alignment=True, cross_modality_n_heads=2,
                        batch_size=6, train_epochs=1, lr=1e-3,
                        save_artifacts=False, seed=42, val_ratio=0.1,
                        test_ratio=0.1, timeenc=1, freq='M',
                        pooling_mode='sum', optimizer_name='adamw')
                    exe = exe_mod.SequenceExecutor(config)
                    exe.train('test_synthetic_stdlib')
                    self.assertEqual(len(exe.training_log), 1)
                    self.assertTrue(exe.best_val_metrics['had_batch'])
                    self.assertTrue(exe.best_test_metrics['had_batch'])
        finally:
            sys.modules.pop('src.model_excutor.seq_prediction_excutor', None)
            if prev is not None:
                sys.modules['src.model_excutor.seq_prediction_excutor'] = prev
            if previous_attr is not None:
                executor_package.seq_prediction_excutor = previous_attr
            elif hasattr(executor_package, 'seq_prediction_excutor'):
                delattr(executor_package, 'seq_prediction_excutor')


if __name__ == '__main__':
    unittest.main(verbosity=2)
