"""Offline architecture smoke tests; no downloads or tourism corpus required."""
import torch
from src.models.MMMA_NLinear import Model

def test_dual_text_forward_and_aux_loss():
    model=Model(dict(seq_len=6,pred_len=2,num_enc_in=1,text_encoder_out_dim=8,
        cross_modality_n_heads=2,use_cross_modality=True,use_movement_alignment=True,
        movement_tau=0.07,movement_loss_weight=0.2,use_time_feature=False))
    x=torch.randn(4,6,1)
    macro=torch.randn(4,6,8)
    micro=torch.randn(4,6,8)
    result=model(x_num=x,macro_text=macro,micro_text=micro)
    y=result[0] if isinstance(result,tuple) else result
    assert y.shape==(4,2,1)
    assert torch.isfinite(y).all()

def test_single_text_and_architecture_ablation():
    for degrade in (False,True):
        model=Model(dict(seq_len=6,pred_len=1,num_enc_in=1,text_encoder_out_dim=8,
            cross_modality_n_heads=2,use_cross_modality=not degrade,
            use_movement_alignment=False,degrade=degrade,use_time_feature=False))
        x=torch.randn(3,6,1)
        text=torch.randn(3,6,8)
        out=model(x_num=x,macro_text=text,micro_text=None)
        pred=out[0] if isinstance(out,tuple) else out
        assert pred.shape==(3,1,1)
