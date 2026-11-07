from __future__ import annotations
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from src.pipelines.preview import Preview
from src.pipelines.text_handler import generate_text
from src.model_excutor.simple_monthly_loader import is_simple_monthly_dataset, load_simple_monthly_text_num_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_float_tensor_2d(x):
    import pandas as pd

    if isinstance(x, torch.Tensor):
        t = x.detach().clone().float()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        t = torch.tensor(np.asarray(x), dtype=torch.float32)
    else:
        t = torch.tensor(np.asarray(x), dtype=torch.float32)

    if t.dim() == 1:
        t = t.unsqueeze(-1)
    if t.dim() != 2:
        raise ValueError(f"Expect 2D tensor, got shape={tuple(t.shape)}")
    return t


def clip_head(arr_like, target_len: int):
    if arr_like is None:
        return None
    if hasattr(arr_like, "__len__"):
        L = len(arr_like)
        if L == target_len:
            return arr_like
        if hasattr(arr_like, "iloc"):
            return arr_like.iloc[L - target_len: L]
        return arr_like[-target_len:]
    return arr_like


def align_modalities(
    x_num,
    macro_text,
    micro_text,
    x_date,
):
    """Align numeric and available text streams from the tail.

    Domestic balanced mode has both macro and micro streams, while WSDM
    macro--micro ablations may intentionally activate only one text branch.
    """
    text_streams = [x for x in (macro_text, micro_text) if x is not None]
    if not text_streams:
        raise ValueError("对齐失败：macro_text 和 micro_text 均为空。")

    lengths = [len(x_num), len(x_date)] + [len(x) for x in text_streams]
    min_len = min(lengths)
    if min_len <= 0:
        raise ValueError("对齐失败：存在空模态。")

    x_num = clip_head(x_num, min_len)
    macro_text = clip_head(macro_text, min_len)
    micro_text = clip_head(micro_text, min_len)
    x_date = clip_head(x_date, min_len)

    return x_num, macro_text, micro_text, x_date, min_len


def load_multimodal_monthly_data(args: dict):
    """
    统一加载：
    - 数值序列
    - macro_text
    - micro_text
    - 月索引

    For new_orleans / amsterdam, the source has only one text stream. In that
    case this function returns the real text in ``macro_text`` and returns
    ``micro_text=None``. Downstream models should detect this as single-text
    mode instead of encoding an artificial empty text stream.
    """
    if is_simple_monthly_dataset(args):
        x_num, single_text, x_date, min_len = load_simple_monthly_text_num_data(args)
        return x_num, single_text, None, x_date, min_len

    loc = args.get("x_num_loc").lower()
    num_type = args.get("x_num_type").lower()
    text_data_dir = args.get("text_data_dir", "")
    text_mode = str(args.get("text_mode", "default")).lower()

    pre = Preview(num_type=num_type)
    x_num_series = pre.generate_num(loc)
    x_num = np.asarray(x_num_series, dtype=np.float32).reshape(-1, 1)

    if text_data_dir and text_mode == "mixed":
        # No explicit macro--micro decomposition: all selected records are fed
        # through one text branch.  The MMMA model already supports single-text
        # mode; the second branch remains None.
        macro_text, x_date = generate_text(
            "single",
            loc,
            text_data_dir=text_data_dir,
        )
        micro_text = None

    elif text_data_dir and text_mode == "macro_only":
        macro_text, x_date = generate_text(
            "macro",
            loc,
            text_data_dir=text_data_dir,
        )
        micro_text = None

    elif text_data_dir and text_mode == "micro_only":
        micro_text, x_date = generate_text(
            "micro",
            loc,
            text_data_dir=text_data_dir,
        )
        macro_text = None

    else:
        # Default/balanced mode preserves the original two-branch behavior.
        macro_text, x_date_macro = generate_text(
            "macro",
            loc,
            text_data_dir=text_data_dir if text_data_dir else None,
        )
        micro_text, x_date_micro = generate_text(
            "micro",
            loc,
            text_data_dir=text_data_dir if text_data_dir else None,
        )
        x_date = x_date_macro if len(x_date_macro) >= len(x_date_micro) else x_date_micro

    x_num, macro_text, micro_text, x_date, min_len = align_modalities(
        x_num, macro_text, micro_text, x_date
    )
    return x_num, macro_text, micro_text, x_date, min_len


def compute_reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))))
    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}


class ArtifactManager:
    def __init__(
        self,
        sub_root: str,
        model_name: str,
        city: str,
        text_variant: str,
        text_mode: str,
        settings: str,
    ) -> None:
        self.root = (
            PROJECT_ROOT
            / "res_models"
            / sub_root
            / model_name
            / city
            / text_variant
            / text_mode
            / settings
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / name

    def save_json(self, obj: Any, filename: str):
        with self.path(filename).open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def save_text(self, text: str, filename: str):
        self.path(filename).write_text(text, encoding="utf-8")

    def save_csv_dict(self, obj: dict, filename: str):
        import pandas as pd
        pd.DataFrame(obj).to_csv(self.path(filename), index=False)

    def save_joblib(self, obj: Any, filename: str):
        joblib.dump(obj, self.path(filename))

    def save_torch(self, obj: Any, filename: str):
        torch.save(obj, self.path(filename))


def build_summary_text(summary: dict, cost: dict) -> str:
    return (
        f"settings: {summary.get('settings')}\n"
        f"model_name: {summary.get('model_name')}\n"
        f"x_num_loc: {summary.get('x_num_loc')}\n"
        f"text_variant: {summary.get('text_variant')}\n"
        f"text_mode: {summary.get('text_mode')}\n"
        f"best_epoch: {summary.get('best_epoch')}\n"
        f"best_config: {summary.get('best_config')}\n"
        f"best_val_metrics: {summary.get('best_val_metrics')}\n"
        f"best_test_metrics: {summary.get('best_test_metrics')}\n"
        f"cost: {cost}\n"
    )
