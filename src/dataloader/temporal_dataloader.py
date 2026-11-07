from typing import List, Tuple, Union, Optional, Sequence, Any
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

from src.utils.timefeatures import time_features


def build_time_stamp_features(
    x_date: Sequence[str],
    *,
    timeenc: int = 1,
    freq: str = "M",
) -> torch.Tensor:
    """
    Build time features from raw datetime sequence.

    Args:
        x_date: sequence of date-like strings
        timeenc:
            0 -> manual calendar features
            1 -> use src.utils.timefeatures.time_features
        freq: pandas-like frequency, e.g. "M", "D", "H"

    Returns:
        Tensor[T, K]
    """
    df_stamp = pd.DataFrame({"date": pd.to_datetime(np.asarray(x_date))})

    if timeenc == 0:
        freq_up = str(freq).upper()

        # 对月频数据，day/weekday/hour 没什么意义
        if freq_up.startswith("M"):
            df_stamp["month"] = df_stamp["date"].dt.month
            df_stamp["year"] = df_stamp["date"].dt.year - df_stamp["date"].dt.year.min()
            data_stamp = df_stamp[["month", "year"]].values.astype(np.float32)
        else:
            df_stamp["month"] = df_stamp["date"].dt.month
            df_stamp["day"] = df_stamp["date"].dt.day
            df_stamp["weekday"] = df_stamp["date"].dt.weekday
            df_stamp["hour"] = df_stamp["date"].dt.hour
            data_stamp = df_stamp.drop(columns=["date"]).values.astype(np.float32)

    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0).astype(np.float32)

    else:
        raise ValueError(f"Unsupported timeenc={timeenc}")

    return torch.tensor(data_stamp, dtype=torch.float32)


def _collate_optional_values(values, *, key: str):
    """Collate one optional modality while preserving None semantics."""
    if all(v is None for v in values):
        return None

    if any(v is None for v in values):
        raise ValueError(f"Inconsistent None values in batch for {key}")

    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)

    return list(values)


def multimodal_collate_fn(batch):
    """
    Collate for aligned multimodal time series.

    Supports:
    - numeric tensor windows
    - text windows: List[List[str]]
    - sentiment/numeric auxiliary windows stored in X_text_1 / X_text_2 as Tensor
    - optional time windows in X_time

    Notes:
    - SlidingWindowStructureDataset now returns a dict for x_part, so modality names
      are preserved even when X_text_2 is None and X_time exists.
    - The legacy tuple/list path is kept for backward compatibility.
    """
    x_part, y_part = zip(*batch)

    result = {
        "X_num": None,
        "X_text_1": None,
        "X_text_2": None,
        "X_time": None,
        "y": None,
    }

    # New safe path: keep modality names from Dataset output.
    if isinstance(x_part[0], dict):
        for key in ["X_num", "X_text_1", "X_text_2", "X_time"]:
            values = [x.get(key, None) for x in x_part]
            result[key] = _collate_optional_values(values, key=key)

    # Legacy path: positional mapping. Kept for compatibility with older datasets.
    else:
        modality_keys = ["X_num", "X_text_1", "X_text_2", "X_time"]

        if isinstance(x_part[0], tuple):
            sub_parts = list(zip(*x_part))
        else:
            sub_parts = [x_part]

        for i, part in enumerate(sub_parts):
            key = modality_keys[i] if i < len(modality_keys) else f"X_{i}"
            result[key] = _collate_optional_values(list(part), key=key)

    if isinstance(y_part[0], torch.Tensor):
        result["y"] = torch.stack(y_part)
    else:
        raise TypeError("Unsupported y format, must be tensor")

    return result


class SlidingWindowStructureDataset(Dataset):
    def __init__(
        self,
        X_num: Optional[torch.Tensor] = None,
        X_text_1: Optional[Any] = None,
        X_text_2: Optional[Any] = None,
        y: Optional[torch.Tensor] = None,
        window_size: int = None,
        horizon: int = 1,
        stride: int = 1,
        *,
        x_date: Optional[Sequence[str]] = None,
        freq: str = "M",
        timeenc: int = 1,
    ):
        """
        Dataset for sliding window over numeric / text / senti inputs.

        Supported modality formats:
        - X_num: Tensor[T, D]
        - X_text_1 / X_text_2:
            1) text mode: List[List[str]]
            2) numeric aux mode (e.g. sentiment): Tensor[T, D] or ndarray[T, D]

        y:
        - forecast target Tensor[T, D]

        Returned:
        - input window dict with stable modality keys:
            {
              "X_num": Tensor[window_size, D] or None,
              "X_text_1": text/list/tensor window or None,
              "X_text_2": text/list/tensor window or None,
              "X_time": Tensor[window_size, K] or None,
            }
        - target window: [start + window_size : start + window_size + horizon]
        """
        assert any([X_num is not None, X_text_1 is not None, X_text_2 is not None]), \
            "At least one of X_num, X_text_1, or X_text_2 must be provided"
        assert y is not None, "Target y must be provided"
        assert stride >= 1, "Stride must be >= 1"
        assert window_size is not None and window_size >= 1, "window_size must be >= 1"
        assert horizon >= 1, "horizon must be >= 1"

        self.X_num = X_num
        self.X_text_1 = self._normalize_text_or_tensor(
            self._maybe_to_tensor(X_text_1),
            name="X_text_1",
        )
        self.X_text_2 = self._normalize_text_or_tensor(
            self._maybe_to_tensor(X_text_2),
            name="X_text_2",
        )
        self.y = y
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.freq = freq
        self.timeenc = int(timeenc)

        base_len = len(X_num) if X_num is not None else \
                   len(self.X_text_1) if self.X_text_1 is not None else len(self.X_text_2)

        self.length = (base_len - window_size - horizon) // stride + 1
        if self.length <= 0:
            raise ValueError(
                f"Invalid sliding window setting: base_len={base_len}, "
                f"window_size={window_size}, horizon={horizon}, stride={stride}"
            )

        assert len(y) == base_len, "y must be same length as inputs"
        if X_num is not None:
            assert len(X_num) == base_len

        self._validate_modality(self.X_text_1, "X_text_1", base_len)
        self._validate_modality(self.X_text_2, "X_text_2", base_len)

        self.time_feats: Optional[torch.Tensor] = None
        if x_date is not None:
            if isinstance(x_date, (pd.Index, pd.Series)):
                dates = x_date.values
            else:
                dates = np.asarray(x_date)

            assert len(dates) == base_len, "x_date length must equal inputs length"
            self.time_feats = build_time_stamp_features(
                dates,
                timeenc=self.timeenc,
                freq=self.freq,
            )

    def _maybe_to_tensor(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.float()
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.number):
                return torch.tensor(x, dtype=torch.float32)
            return x.tolist()
        if isinstance(x, pd.Series):
            return x.tolist()
        return x

    def _normalize_text_or_tensor(self, x, name: str):
        """Normalize text modalities while keeping numeric auxiliary tensors unchanged.

        Accepted text formats:
        - List[List[str]]: already one list of texts per timestep.
        - List[str]: one text string per timestep.
        - pandas Series / numpy object array of strings.

        Returned text format is always List[List[str]], so each timestep can be
        consumed by TextEmbedding as one monthly text list. Empty or missing text
        becomes an empty list for that timestep.
        """
        if x is None or isinstance(x, torch.Tensor):
            return x

        if isinstance(x, pd.Series):
            x = x.tolist()
        elif isinstance(x, np.ndarray):
            x = x.tolist()

        if not hasattr(x, "__len__"):
            raise TypeError(
                f"{name} must be a tensor, sequence of strings, or sequence of List[str]."
            )

        normalized = []
        for i, item in enumerate(x):
            if item is None:
                normalized.append([])
                continue

            if isinstance(item, str):
                txt = item.strip()
                normalized.append([txt] if txt else [])
                continue

            if isinstance(item, np.ndarray):
                item = item.tolist()

            if isinstance(item, (list, tuple)):
                texts = []
                for t in item:
                    if t is None:
                        continue
                    if isinstance(t, (list, tuple, np.ndarray)):
                        # Defensive flattening for accidentally nested text lists.
                        for tt in list(t):
                            if tt is None:
                                continue
                            st = str(tt).strip()
                            if st:
                                texts.append(st)
                    else:
                        st = str(t).strip()
                        if st:
                            texts.append(st)
                normalized.append(texts)
                continue

            # Last-resort compatibility for scalar non-string values.
            txt = str(item).strip()
            normalized.append([txt] if txt else [])

        return normalized

    def _validate_modality(self, x, name: str, base_len: int):
        if x is None:
            return

        assert len(x) == base_len, f"{name} must have same length as base input"

        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                raise ValueError(f"{name} tensor must be 2D [T, D], got shape={tuple(x.shape)}")
            return

        for i, tokens in enumerate(x):
            assert isinstance(tokens, list) and all(isinstance(t, str) for t in tokens), \
                f"{name}[{i}] is not List[str]"

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.window_size
        target_end = end + self.horizon

        x_num_window = self.X_num[start:end] if self.X_num is not None else None
        x_text_1_window = self.X_text_1[start:end] if self.X_text_1 is not None else None
        x_text_2_window = self.X_text_2[start:end] if self.X_text_2 is not None else None
        x_time_window = self.time_feats[start:end] if self.time_feats is not None else None

        y_window = self.y[end:target_end]

        # Always return a dict to preserve modality identity.
        # This prevents X_time from being shifted into X_text_2 when X_text_2 is None.
        return {
            "X_num": x_num_window,
            "X_text_1": x_text_1_window,
            "X_text_2": x_text_2_window,
            "X_time": x_time_window,
        }, y_window


class SlidingWindowForecastDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        window_size: int,
        horizon: int = 1,
        X_add: torch.Tensor = None,
        prev_output: torch.Tensor = None,
        *,
        x_date: Optional[Sequence[str]] = None,
        freq: str = "M",
    ):
        self.X = X
        self.X_add = X_add
        self.y = y
        self.prev_output = prev_output
        self.window_size = window_size
        self.horizon = horizon
        self.freq = freq

        if X is not None:
            self.length = len(X) - window_size - horizon + 1
        elif X_add is not None:
            self.length = len(X_add) - window_size - horizon + 1
        else:
            raise RuntimeError("X or X_add must be provided")

        if X is not None:
            if X_add is not None:
                assert len(X) == len(X_add), "X and X_add must have the same length"
            assert len(y) == len(X), "X and y must have the same length"
            if prev_output is not None:
                assert len(prev_output) == len(X), "prev_output and X must have the same length"

        self.time_feats: Optional[torch.Tensor] = None
        base_len = len(X) if X is not None else len(X_add)
        if x_date is not None:
            if isinstance(x_date, (pd.Index, pd.Series)):
                dates = pd.to_datetime(x_date.values)
            else:
                dates = pd.to_datetime(np.asarray(x_date))
            assert len(dates) == base_len, "x_date length must equal inputs length"
            tf = time_features(dates, freq=self.freq).transpose(1, 0)
            self.time_feats = torch.tensor(tf, dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        target_t = idx + self.window_size + self.horizon - 1
        y_target = self.y[target_t]

        if self.X is not None:
            x_window = self.X[idx: idx + self.window_size]
        else:
            x_window = torch.zeros(self.window_size, 1)

        if self.X_add is not None:
            x_add_window = self.X_add[idx: idx + self.window_size]
        else:
            x_add_window = None

        if self.prev_output is not None:
            prev_out = self.prev_output[target_t]
        else:
            prev_out = torch.zeros_like(y_target)

        if self.time_feats is not None:
            x_time_window = self.time_feats[idx: idx + self.window_size]
        else:
            x_time_window = None

        if x_time_window is not None:
            return (x_window, x_add_window, x_time_window), y_target, prev_out
        return (x_window, x_add_window), y_target, prev_out
