import numpy as np
import torch
import os
import gc
import multiprocessing as mp
# ================= 工具：DataLoader/子进程的安全清理 ================= #
def _safe_close_loader(loader):
    """尽力关闭 DataLoader 的 worker 进程并清理迭代器引用。"""
    if loader is None:
        return
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
            try:
                loader._iterator = None
            except Exception:
                pass
    except Exception:
        pass


def _cleanup_cached_loaders(obj):
    """关闭对象上可能缓存的 dataloader 字段（如 train_loader/val_loader/test_loader）。"""
    if obj is None:
        return
    possible = [
        "train_loader", "vali_loader", "val_loader", "test_loader",
        "loader", "data_loader", "train_data_loader", "valid_data_loader", "test_data_loader",
    ]
    for name in possible:
        if hasattr(obj, name):
            try:
                _safe_close_loader(getattr(obj, name))
            except Exception:
                pass
            try:
                delattr(obj, name)
            except Exception:
                pass


def _reap_dataloader_workers():
    """兜底：回收当前进程下还活着的 multiprocessing 子进程（通常就是 DataLoader worker）。"""
    try:
        children = mp.active_children()
        for p in children:
            try:
                p.terminate()
            except Exception:
                pass
        for p in children:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
    except Exception:
        pass


def _hard_cleanup(obj):
    """先清 loader，再把重对象移到 CPU 并删除引用，最后强制 GC + 释放显存。"""
    _cleanup_cached_loaders(obj)

    heavy_attrs = [
        'model', 'mlp', 'mlp_proj', 'weight1', 'weight2',
        'llm_model', 'tokenizer', 'st_model', 'finbert_cls', 'finbert_tok',
        'bertopic', 'text_model'
    ]
    try:
        for name in heavy_attrs:
            if hasattr(obj, name):
                inst = getattr(obj, name)
                try:
                    if hasattr(inst, 'to'):
                        inst.to('cpu')
                except Exception:
                    pass
                try:
                    del inst
                except Exception:
                    pass
                try:
                    delattr(obj, name)
                except Exception:
                    pass
    except Exception:
        pass

    # 关闭所有 matplotlib 句柄（以防万一）
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ========================== 数据加载辅助 ========================== #

def _load_array(path):
    """从给定路径加载数组；支持 .npy 和 .npz（若 .npz 则取第一个键）。"""
    if path is None or path == "" or str(path).lower() == "none":
        return None
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.npz'):
        data = np.load(path)
        # 取第一个键
        k = list(data.keys())[0]
        return data[k]
    raise ValueError(f"Unsupported array file type: {path}")


def _maybe_synth_data(args):
    """当未给定数据路径时，生成一份可跑通的合成数据。"""
    T = max(args.seq_len + args.pred_len + 10, 128)
    D = args.enc_in
    X = np.random.randn(T, D).astype(np.float32)
    y = np.random.randn(T, 1).astype(np.float32)
    return X, None, None, y