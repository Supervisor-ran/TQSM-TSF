"""Paper-method runner. This release contains no comparison-model executors.

Use ``python -m src.run --help`` from the repository root.
"""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import torch
from src.model_excutor.seq_prediction_excutor import SequenceExecutor
from src.model_excutor.simple_monthly_loader import infer_simple_x_num_type, is_simple_monthly_dataset
from src.utils.cleaner import _cleanup_cached_loaders, _reap_dataloader_workers, _hard_cleanup

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def build_parser():
    parser = argparse.ArgumentParser(description="MMMA-NLinear paper-method training")
    parser.add_argument('--model_name', default='MMMA_NLinear', choices=['MMMA_NLinear'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--x_num_type', default='domestic', choices=['domestic','yelp','hotelrec'])
    parser.add_argument('--x_num_loc', default='kamakura')
    parser.add_argument('--num_enc_in', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=6)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=13)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--optimizer_name', default='adamw')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--text_encoder_out_dim', type=int, default=768)
    parser.add_argument('--llm_n_heads', type=int, default=8)
    parser.add_argument('--pooling_mode', default='sum', choices=['sum','mean'])
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--use_time_feature', action='store_true')
    parser.add_argument('--timeenc', type=int, default=1, choices=[0,1])
    parser.add_argument('--time_fusion', default='add', choices=['add','concat'])
    parser.add_argument('--freq', default='M')
    parser.add_argument('--use_movement_alignment', action='store_true')
    parser.add_argument('--movement_tau', type=float, default=0.07)
    parser.add_argument('--movement_loss_weight', type=float, default=0.2)
    parser.add_argument('--use_cross_modality', action='store_true')
    parser.add_argument('--cross_modality_n_heads', type=int, default=6)
    parser.add_argument('--degrade', action='store_true', help='Paper architecture ablation')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--selection_metric', default='mse', choices=['mse','mae','loss'])
    parser.add_argument('--debug_text', action='store_true')
    parser.add_argument('--save_artifacts', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--simple_num_path', default=None)
    parser.add_argument('--simple_text_path', default=None)
    parser.add_argument('--use_external_review_level_encoding', action='store_true')
    parser.add_argument('--external_review_embedding_cache', default=None)
    parser.add_argument('--use_data_ablation', action='store_true')
    parser.add_argument('--ablation_root', default=str(PROJECT_ROOT/'data'/'text_collection'/'topk'))
    parser.add_argument('--ablation_variants', nargs='*', default=None)
    parser.add_argument('--ablation_modes', nargs='*', default=None)
    parser.add_argument('--use_model_ablation', action='store_true')
    return parser

def _settings(args):
    fields = [f"MMMA_NLinear_{args.x_num_loc}_LB{args.seq_len}_Pred{args.pred_len}",
              f"LR{args.lr:g}", f"MA{int(args.use_movement_alignment)}",
              f"Cross{int(args.use_cross_modality)}",f"T{int(args.use_time_feature)}"]
    if args.use_data_ablation: fields.append(f"TXT{args.text_variant}_{args.text_mode}")
    if args.use_model_ablation: fields.append(f"ARCH{args.arch_variant}")
    if args.degrade: fields.append("DEGTextNLinear")
    return '_'.join(fields)

def _data_jobs(args):
    if not args.use_data_ablation: return [('default','default','')]
    if is_simple_monthly_dataset(args):
        v=args.ablation_variants or [];m=args.ablation_modes or []
        if len(v)!=1 or len(m)!=1:
            raise ValueError('External dataset tags require one variant and one mode')
        return [(v[0],m[0],'')]
    cityroot=Path(args.ablation_root)/args.x_num_loc
    if not cityroot.is_dir(): raise FileNotFoundError(cityroot)
    variants=args.ablation_variants or sorted(p.name for p in cityroot.iterdir() if p.is_dir())
    result=[]
    for variant in variants:
        root=cityroot/variant
        if not root.is_dir(): continue
        modes=args.ablation_modes or sorted(p.name for p in root.iterdir() if p.is_dir())
        for mode in modes:
            dirname=root/mode
            if dirname.is_dir() and (any(dirname.glob('*.json')) or any(dirname.glob('*.jsonl'))):
                result.append((variant,mode,str(dirname)))
    if not result: raise RuntimeError(f'No ablation files found under {cityroot}')
    return result

def main(argv=None):
    parser=build_parser(); args=parser.parse_args(argv)
    args.x_num_type=infer_simple_x_num_type(args.x_num_loc, default=args.x_num_type)
    if args.use_model_ablation:
        arch_jobs=[('base',False,False),('cross',True,False),('cross_mov',True,True)]
    else:
        arch_jobs=[('manual',args.use_cross_modality,args.use_movement_alignment)]
    for variant,mode,folder in _data_jobs(args):
        for arch,cross,align in arch_jobs:
            if align and not cross: raise ValueError('Movement alignment requires cross modality for this runner')
            a=argparse.Namespace(**vars(args))
            a.text_variant,a.text_mode,a.text_data_dir=variant,mode,folder
            a.arch_variant=arch
            a.use_cross_modality=cross and not a.degrade
            a.use_movement_alignment=align and not a.degrade
            exe=None
            try:
                exe=SequenceExecutor(vars(a))
                exe.train(_settings(a))
            finally:
                _cleanup_cached_loaders(exe)
                _hard_cleanup(exe)
                _reap_dataloader_workers()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

if __name__ == '__main__': main()
