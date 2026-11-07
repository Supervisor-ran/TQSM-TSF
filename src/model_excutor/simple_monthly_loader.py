from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


SIMPLE_MONTHLY_DATASETS: dict[tuple[str, str], dict[str, str]] = {
    (
        "yelp",
        "new_orleans",
    ): {
        "num_path": "data_Yelp/num_collection/MSY_airport_arriving_2015_2025_cleaned.csv",
        "text_path": "data_Yelp/text_collection/new_orleans_tourism_reviews_2015_2025_monthly_top200_cleaned.csv",
    },
    (
        "hotelrec",
        "amsterdam",
    ): {
        "num_path": "hotelrec_data/num_collection/Amsterdam_hotel_arriving_tourists_2012_202602_cleaned.csv",
        "text_path": "hotelrec_data/text_collection/hotelrec_amsterdam_2012plus_monthly_date_text_top200.csv",
    },
}


SIMPLE_MONTHLY_LOCS = {loc for _, loc in SIMPLE_MONTHLY_DATASETS.keys()}
SIMPLE_MONTHLY_TYPES = {typ for typ, _ in SIMPLE_MONTHLY_DATASETS.keys()}


def normalize_dataset_key(x_num_type: str | None, x_num_loc: str | None) -> tuple[str, str]:
    """Return normalized dataset key.

    Args:
        x_num_type: Dataset type, such as ``domestic``, ``yelp`` or ``hotelrec``.
        x_num_loc: Location name, such as ``new_orleans`` or ``amsterdam``.

    Returns:
        A normalized ``(x_num_type, x_num_loc)`` tuple.
    """
    typ = str(x_num_type or "").strip().lower()
    loc = str(x_num_loc or "").strip().lower()
    return typ, loc


def infer_simple_x_num_type(x_num_loc: str, default: str = "domestic") -> str:
    """Infer x_num_type from location name for the newly added simple datasets."""
    loc = str(x_num_loc or "").strip().lower()
    if loc == "new_orleans":
        return "yelp"
    if loc == "amsterdam":
        return "hotelrec"
    return default


def is_simple_monthly_dataset(args: dict[str, Any] | Any) -> bool:
    """Return whether args point to the single-text monthly datasets.

    The new datasets do not have macro/micro text streams. They are handled as
    one numerical target stream plus one text stream.
    """
    if isinstance(args, dict):
        x_num_type = args.get("x_num_type")
        x_num_loc = args.get("x_num_loc")
    else:
        x_num_type = getattr(args, "x_num_type", None)
        x_num_loc = getattr(args, "x_num_loc", None)

    typ, loc = normalize_dataset_key(x_num_type, x_num_loc)
    return (typ, loc) in SIMPLE_MONTHLY_DATASETS or loc in SIMPLE_MONTHLY_LOCS or typ in SIMPLE_MONTHLY_TYPES


def _resolve_dataset_config(args: dict[str, Any]) -> tuple[str, str, Path, Path]:
    """Resolve numerical and textual CSV paths for a simple monthly dataset."""
    typ, loc = normalize_dataset_key(args.get("x_num_type"), args.get("x_num_loc"))

    # If loc is enough but x_num_type is still domestic/default, infer it.
    if (typ, loc) not in SIMPLE_MONTHLY_DATASETS:
        inferred_type = infer_simple_x_num_type(loc, default=typ)
        typ = inferred_type

    key = (typ, loc)
    if key not in SIMPLE_MONTHLY_DATASETS:
        supported = ", ".join([f"{t}/{l}" for t, l in sorted(SIMPLE_MONTHLY_DATASETS.keys())])
        raise ValueError(
            f"Unsupported simple monthly dataset: x_num_type={args.get('x_num_type')}, "
            f"x_num_loc={args.get('x_num_loc')}. Supported: {supported}"
        )

    cfg = SIMPLE_MONTHLY_DATASETS[key]

    num_path = Path(args.get("simple_num_path") or cfg["num_path"])
    text_path = Path(args.get("simple_text_path") or cfg["text_path"])

    if not num_path.is_absolute():
        num_path = PROJECT_ROOT / num_path
    if not text_path.is_absolute():
        text_path = PROJECT_ROOT / text_path

    return typ, loc, num_path, text_path


def _to_month_start(series: pd.Series, *, column_name: str) -> pd.Series:
    """Convert date-like values to month-start timestamps.

    The supported raw formats include ``YYYY-MM-01`` and ``YYYY-MM``.
    """
    dt = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    if dt.isna().any():
        bad_values = series[dt.isna()].head(5).tolist()
        raise ValueError(f"Failed to parse {column_name} as month date. Examples: {bad_values}")
    return dt.dt.to_period("M").dt.to_timestamp()


def _read_num_csv(path: Path) -> pd.DataFrame:
    """Read monthly numerical target CSV.

    Expected columns:
        - date: month date, ``YYYY-MM-01`` or parseable date
        - target: numerical target value
    """
    if not path.exists():
        raise FileNotFoundError(f"Numerical CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"date", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Numerical CSV must contain columns {required}, missing={missing}, path={path}")

    out = df[["date", "target"]].copy()
    out["month"] = _to_month_start(out["date"], column_name="date")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["month", "target"])

    if out.empty:
        raise ValueError(f"No valid numerical rows found in: {path}")

    # The provided files are monthly. If duplicates exist, average the target
    # to keep one value per month instead of creating duplicated windows.
    out = (
        out.groupby("month", as_index=False)["target"]
        .mean()
        .sort_values("month")
        .reset_index(drop=True)
    )
    return out


def _read_text_csv(path: Path) -> pd.DataFrame:
    """Read monthly text CSV.

    Expected columns:
        - date: month date, ``YYYY-MM-01`` or ``YYYY-MM``
        - text: text content
    """
    if not path.exists():
        raise FileNotFoundError(f"Text CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"date", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Text CSV must contain columns {required}, missing={missing}, path={path}")

    out = df[["date", "text"]].copy()
    out["month"] = _to_month_start(out["date"], column_name="date")
    out["text"] = out["text"].fillna("").astype(str)

    # Keep every non-empty text under the same month as a list. For the current
    # cleaned files this will usually be one long text string per month.
    out = (
        out.groupby("month", as_index=False)["text"]
        .agg(lambda s: [x for x in s.tolist() if str(x).strip()])
        .rename(columns={"text": "texts"})
        .sort_values("month")
        .reset_index(drop=True)
    )
    out["texts"] = out["texts"].apply(lambda xs: xs if len(xs) > 0 else [""])

    if out.empty:
        raise ValueError(f"No valid text rows found in: {path}")

    return out


def load_simple_monthly_text_num_data(args: dict[str, Any]):
    """Load new single-text monthly tourism datasets for traditional models.

    This loader supports:
        - new_orleans / yelp
        - amsterdam / hotelrec

    Returns:
        x_num: ``np.ndarray`` with shape ``(T, 1)``.
        text_list: ``list[list[str]]`` with length ``T``.
        x_date: list of aligned month strings, ``YYYY-MM-01``.
        min_len: aligned length.
    """
    typ, loc, num_path, text_path = _resolve_dataset_config(args)

    num_df = _read_num_csv(num_path)
    text_df = _read_text_csv(text_path)

    merged = (
        num_df.merge(text_df, on="month", how="inner")
        .sort_values("month")
        .reset_index(drop=True)
    )

    if merged.empty:
        raise ValueError(
            "No overlapping months between numerical and text data. "
            f"num_path={num_path}, text_path={text_path}"
        )

    x_num = merged[["target"]].to_numpy(dtype=np.float32)
    text_list = merged["texts"].tolist()
    x_date = merged["month"].dt.strftime("%Y-%m-%d").tolist()
    min_len = len(merged)

    print(
        f"[simple_monthly_loader] dataset={typ}/{loc}, aligned length={min_len}, "
        f"date range={x_date[0]} to {x_date[-1]}"
    )
    print(f"[simple_monthly_loader] num_path={num_path}")
    print(f"[simple_monthly_loader] text_path={text_path}")

    return x_num, text_list, x_date, min_len
