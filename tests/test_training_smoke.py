"""Data-free integration smoke for training logic, with local deterministic text embeddings.

Does not replace a research reproduction with genuine source data and HF encoders.
"""
import sys
import types
from unittest.mock import patch
import numpy as np
import pytest
import torch


def test_one_epoch_synthetic_training(tmp_path, monkeypatch):
    # Fake only the HF encoder component; real MMMA, dataloader and optimizer run.
    fake_embedding_module = types.ModuleType('src.model_util.text_embedding')
    class LocalEmbedding(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
        def forward(self, sequences, device=None):
            return torch.ones(len(sequences), len(sequences[0]), 8, device=device)
    fake_embedding_module.TextEmbedding = LocalEmbedding
    with patch.dict(sys.modules, {'src.model_util.text_embedding': fake_embedding_module}):
        # Avoid module caching when test suite is repeated.
        sys.modules.pop('src.model_excutor.seq_prediction_excutor', None)
        from src.model_excutor import seq_prediction_excutor as executor_module
        n = 35
        numeric = np.linspace(100, 150, n).astype(np.float32)
        dates = [f"{2022 + (m // 12)}-{1 + (m % 12):02d}" for m in range(n)]
        text = [[f"Event {i}"] for i in range(n)]
        monkeypatch.setattr(executor_module, 'load_multimodal_monthly_data',
            lambda args: (numeric, text, text, dates, n))
        config = dict(model_name='MMMA_NLinear', x_num_loc='kamakura',
                      seq_len=6, pred_len=1, num_enc_in=1, text_encoder_out_dim=8,
                      use_cross_modality=True, use_movement_alignment=True,
                      cross_modality_n_heads=2, batch_size=6, train_epochs=1,
                      lr=1e-3, save_artifacts=False, seed=42,
                      val_ratio=0.1, test_ratio=0.1, timeenc=1, freq='M',
                      pooling_mode='sum', optimizer_name='adamw')
        exe = executor_module.SequenceExecutor(config)
        exe.train('test_synthetic')
        assert len(exe.training_log) == 1
        assert exe.best_val_metrics['had_batch'] is True
        assert exe.best_test_metrics['had_batch'] is True
