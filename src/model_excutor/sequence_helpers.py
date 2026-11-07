from __future__ import annotations
import torch
import torchmetrics.functional as MF
from torch.utils.data import DataLoader
from src.dataloader.temporal_dataloader import SlidingWindowStructureDataset, multimodal_collate_fn


def build_optimizer(model_params, optimizer_name="adam", lr=1e-3, weight_decay=0.0, **kwargs):
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )
    elif name == "rmsprop":
        return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def split_time_series(total_length: int, val_ratio: float, test_ratio: float):
    vr = val_ratio or 0.0
    tr = test_ratio or 0.0
    val_len = int(total_length * vr)
    test_len = int(total_length * tr)

    val_len = max(0, min(val_len, total_length))
    test_len = max(0, min(test_len, total_length - val_len))
    if test_len == 0 and total_length >= 1:
        test_len = 1

    train_len = total_length - val_len - test_len
    if train_len < 0:
        train_len = 0
        val_len = max(0, total_length - test_len)

    idx_train = list(range(0, train_len))
    idx_val = list(range(train_len, train_len + val_len))
    idx_test = list(range(train_len + val_len, total_length))

    print(f"train {len(idx_train)}")
    print(f"val {len(idx_val)}")
    print(f"test {len(idx_test)}")
    return {"train": idx_train, "val": idx_val, "test": idx_test}


def prepare_sequence_dataloaders(
    *, model_name: str, X_num, macro_text, micro_text, X_txt, y, x_date,
    batch_size: int, window_size: int, horizon: int, val_ratio: float,
    test_ratio: float, shuffle: bool, freq: str, timeenc: int = 1,
):
    if model_name != "MMMA_NLinear":
        raise ValueError(f"Public release only supports MMMA_NLinear: {model_name}")
    base_len = len(y)
    total_len = base_len - window_size - horizon + 1
    if total_len <= 0:
        raise ValueError(f"Sequence too short: base_len={base_len}, window={window_size}, horizon={horizon}")
    index_map = split_time_series(total_len, val_ratio=val_ratio, test_ratio=test_ratio)
    if not index_map["test"]:
        raise ValueError("test split is empty; cannot build dataloaders safely.")
    end = index_map["test"][-1] + window_size + horizon
    full_dataset = SlidingWindowStructureDataset(
        X_num=X_num[:end] if X_num is not None else None,
        X_text_1=macro_text[:end] if macro_text is not None else None,
        X_text_2=micro_text[:end] if micro_text is not None else None,
        y=y[:end], window_size=window_size, horizon=horizon, stride=1,
        x_date=x_date[:end] if x_date is not None else None, freq=freq, timeenc=timeenc,
    )
    def make_loader(split, shuffle_flag):
        if not index_map[split]: return None
        return DataLoader(torch.utils.data.Subset(full_dataset, index_map[split]),
                          batch_size=batch_size, shuffle=shuffle_flag,
                          collate_fn=multimodal_collate_fn)
    return (index_map, make_loader("train", shuffle),
            make_loader("val", False), make_loader("test", False))


def denorm_preds(preds, trues, y_mean, y_std):
    y_std = y_std.to(preds.device) if isinstance(y_std, torch.Tensor) else y_std
    y_mean = y_mean.to(preds.device) if isinstance(y_mean, torch.Tensor) else y_mean
    preds_denorm = preds * y_std + y_mean
    trues_denorm = trues * y_std + y_mean
    return preds_denorm, trues_denorm

def evaluate_from_tensors(preds_denorm, trues_denorm, avg_loss: float, had_batch: bool):
    if not had_batch:
        return {"had_batch": False, "loss": 0.0, "mse": None, "mae": None, "rmse": None, "mape": None}

    mse = MF.mean_squared_error(preds_denorm, trues_denorm).item()
    mae = MF.mean_absolute_error(preds_denorm, trues_denorm).item()
    rmse = MF.mean_squared_error(preds_denorm, trues_denorm, squared=False).item()
    mape = MF.mean_absolute_percentage_error(preds_denorm, trues_denorm).item()

    return {
        "had_batch": True,
        "loss": avg_loss,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
