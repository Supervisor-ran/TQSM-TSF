import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union, Literal

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TextHandler:
    def __init__(self, data_path: str | Path):
        self.data_path: Path = Path(data_path)

    def _iter_records(self, scope_filter: str | None = None):
        files = sorted(list(self.data_path.glob("*.json")) + list(self.data_path.glob("*.jsonl")))
        for file in files:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if scope_filter is not None:
                        if str(obj.get("scope", "")).lower() != scope_filter.lower():
                            continue
                    yield obj

    def extract_senti_score(
        self,
        prefix: str | None = None,
        mode: Literal["day", "month"] = "month",
        agg: Literal["mean", "sum"] = "mean",
        return_numpy: bool = False,
        scope_filter: str | None = None,
    ) -> Union[pd.DataFrame, Tuple[np.ndarray, List[str], List[str]]]:
        raw_scores = defaultdict(list)

        for obj in self._iter_records(scope_filter=scope_filter):
            created_at = obj.get("created_at", "")
            score = obj.get("sentiment_score", None)
            if created_at and score is not None:
                date = created_at[:10]
                raw_scores[date].append(score)

        records = []
        for date in sorted(raw_scores.keys()):
            scores = raw_scores[date]
            value = np.mean(scores) if agg == "mean" else np.sum(scores)
            col_name = f"{prefix}_score" if prefix else "score"
            records.append({"date": date, col_name: value})

        if not records:
            empty = pd.DataFrame(columns=[f"{prefix}_score" if prefix else "score"])
            if return_numpy:
                return empty.to_numpy(), [], list(empty.columns)
            return empty

        df_day = pd.DataFrame(records).set_index("date")

        if mode == "day":
            result = df_day
        else:
            df_day.index = pd.to_datetime(df_day.index)
            if agg == "mean":
                result = df_day.resample("M").mean()
            else:
                result = df_day.resample("M").sum()
            result.index = result.index.strftime("%Y-%m")

        if return_numpy:
            return result.to_numpy(), result.index.tolist(), result.columns.tolist()
        return result

    def extract_full_json_records(self, df: pd.DataFrame, scope_filter: str | None = None) -> List[Tuple[str, str, Dict]]:
        file_cache = {}
        results = []

        for date_str in df.index:
            if len(date_str) == 10:
                month_str = date_str[:7]
            elif len(date_str) == 7:
                month_str = date_str
            else:
                continue

            candidates = [
                self.data_path / f"{month_str.replace('-', '_')}.json",
                self.data_path / f"{month_str.replace('-', '_')}.jsonl",
            ]
            file_path = None
            for c in candidates:
                if c.exists():
                    file_path = c
                    break
            if file_path is None:
                continue

            if month_str not in file_cache:
                with file_path.open("r", encoding="utf-8") as f:
                    rows = []
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            if scope_filter is not None and str(obj.get("scope", "")).lower() != scope_filter.lower():
                                continue
                            rows.append(obj)
                        except json.JSONDecodeError:
                            continue
                    file_cache[month_str] = rows

            records = file_cache[month_str]

            for score_col in df.columns:
                score_val = df.at[date_str, score_col]
                if pd.isna(score_val):
                    continue
                for rec in records:
                    if abs(rec.get("sentiment_score", -999) - score_val) < 1e-6:
                        results.append((date_str, score_col, rec))
                        break

        return results

    def extract_daily_senti_score(self, scope_filter: str | None = None) -> pd.DataFrame:
        day_scores = defaultdict(list)

        for record in self._iter_records(scope_filter=scope_filter):
            created_at = record.get("created_at")
            score = record.get("sentiment_score")
            if created_at and score is not None:
                day_scores[created_at].append(score)

        records = []
        for date, scores in sorted(day_scores.items()):
            row = {"date": date}
            for i, score in enumerate(scores):
                row[f"score_{i + 1}"] = score
            records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.set_index("date", inplace=True)
        return df

    def compute_agg_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        stats = pd.DataFrame({
            "mean": df.mean(axis=1),
            "max": df.max(axis=1),
            "min": df.min(axis=1)
        })
        stats.index.name = "date"
        return stats

    def truncate_missing_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, how='any')

    def extract_text_by_date(
        self,
        mode: str = "day",
        return_format: str = "list",
        scope_filter: str | None = None,
    ) -> Union[Dict[str, List[str]], List[List[str]]]:
        assert mode in ["day", "month"]
        assert return_format in ["dict", "list"]

        raw = defaultdict(list)
        date_set = set()

        for obj in self._iter_records(scope_filter=scope_filter):
            abstract = obj.get("abstract", "")
            created_at = obj.get("created_at", "")
            if not created_at:
                continue
            key = created_at[:10] if mode == "day" else created_at[:7]
            raw[key].append(abstract)
            date_set.add(key)

        if not date_set:
            return {} if return_format == "dict" else []

        fmt = "%Y-%m-%d" if mode == "day" else "%Y-%m"
        min_dt = datetime.strptime(min(date_set), fmt)
        max_dt = datetime.strptime(max(date_set), fmt)

        result_dict = {}
        dt = min_dt
        while dt <= max_dt:
            key = dt.strftime(fmt)
            result_dict[key] = raw.get(key, [])
            if mode == "day":
                dt += timedelta(days=1)
            else:
                year = dt.year + (dt.month // 12)
                month = dt.month % 12 + 1
                dt = datetime(year, month, 1)

        if return_format == "dict":
            return result_dict
        return [result_dict[key] for key in sorted(result_dict.keys())]


def _resolve_text_base_path(path_type: str, loc_name: str, text_data_dir: str | None = None) -> tuple[Path, str | None]:
    """
    返回:
    - base_path
    - scope_filter
    """
    if text_data_dir:
        # topk/ablation directory.  In normal balanced mode, path_type is
        # macro/micro and we keep scope filtering.  For mixed mode, path_type
        # is 'single' and all selected records are consumed as one text stream.
        scope_filter = None if path_type == 'single' else path_type
        return Path(text_data_dir), scope_filter

    if path_type == "macro":
        return PROJECT_ROOT / "data" / "text_collection" / "raw" / "macro" / "global", None
    elif path_type == "micro":
        return PROJECT_ROOT / "data" / "text_collection" / "raw" / "micro" / loc_name, None
    else:
        raise ValueError(f"invalid path_type: {path_type}")


def generate_text(path_type: str, loc_name: str, text_data_dir: str | None = None):
    base_path, scope_filter = _resolve_text_base_path(path_type, loc_name, text_data_dir=text_data_dir)
    text_handler = TextHandler(base_path)

    text_series = text_handler.extract_text_by_date(
        mode="month",
        return_format="list",
        scope_filter=scope_filter,
    )
    senti_df = text_handler.extract_senti_score(
        prefix=f"{loc_name}_{path_type}_sentiment",
        mode="month",
        scope_filter=scope_filter,
    )

    return text_series, senti_df.index.tolist()