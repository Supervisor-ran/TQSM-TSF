from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data_collector.trend_forecaster import RecursiveTrendForecaster


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "text_collection"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: str | Path, records: Sequence[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def list_month_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(list(folder.glob("*.json")) + list(folder.glob("*.jsonl")))


def read_folder_records(folder: str | Path) -> List[dict]:
    records: List[dict] = []
    for file in list_month_files(folder):
        records.extend(load_jsonl(file))
    return records


def month_shift(month_str: str, offset: int) -> str:
    period = pd.Period(month_str, freq="M") + offset
    return str(period)


def cosine_to_unit_interval(cos_value: float) -> float:
    return float((cos_value + 1.0) / 2.0)


class TourismTextScorer:
    VARIANT_MAP = {
        "only_rel": ["rel"],
        "only_novel": ["novel"],
        "only_trend": ["trend"],
        "rel_novel": ["rel", "novel"],
        "rel_trend": ["rel", "trend"],
        "novel_trend": ["novel", "trend"],
        "full": ["rel", "novel", "trend"],
    }

    def __init__(
            self,
            embedding_model_name: str,
            weights: Dict[str, float],
            novelty_lookback_months: int = 3,
            micro_history_window_months: int = 3,
            macro_history_window_months: int = 6,
            micro_trend_horizon_months: int = 3,
            macro_trend_horizon_months: int = 6,
            desc_cache_dir: str | Path | None = None,
            debug: bool = False,
    ) -> None:
        self.embedder = SentenceTransformer(embedding_model_name)
        self.weights = self._normalize_weights(weights)
        self.novelty_lookback_months = novelty_lookback_months
        self.micro_history_window_months = micro_history_window_months
        self.macro_history_window_months = macro_history_window_months
        self.micro_trend_horizon_months = micro_trend_horizon_months
        self.macro_trend_horizon_months = macro_trend_horizon_months
        self.debug = debug
        self.trend_forecaster = RecursiveTrendForecaster(
            cache_dir=desc_cache_dir,
            debug=debug,
        )

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _get_scope_trend_horizon(self, scope: str) -> int:
        return self.macro_trend_horizon_months if str(scope).lower() == "macro" else self.micro_trend_horizon_months

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        keys = ["rel", "novel", "trend"]
        total = sum(float(weights.get(k, 0.0)) for k in keys)
        if total <= 0:
            raise ValueError("weights must contain positive rel/novel/trend values")
        return {k: float(weights.get(k, 0.0)) / total for k in keys}

    def build_task_description(self, target_name_ja: str) -> str:
        return (
            f"この文章が、{target_name_ja}における将来の来訪人数・観光需要・人流変動を予測するために"
            "どれほど有用かを表す観光関連説明文。"
            "イベント、観光政策、移動需要、宿泊価格、地域の話題、外部ショックなど、"
            "来訪人数変動の手がかりになりうる情報を高く評価する。"
        )

    def load_monthly_series(self, spot_name: str, num_type: str = "domestic") -> pd.DataFrame:
        from src.pipelines.preview import Preview

        self._log(f"[debug][load_monthly_series] spot_name={spot_name}, num_type={num_type}")
        pre = Preview(num_type=num_type)
        grouped = pre.generate_monthly_num_df(spot_name)

        self._log(f"[debug][load_monthly_series] type={type(grouped)}")
        self._log(f"[debug][load_monthly_series] shape={grouped.shape}")
        self._log(f"[debug][load_monthly_series] columns={list(grouped.columns)}")
        self._log(f"[debug][load_monthly_series] head=\n{grouped.head(12)}")

        if "month" not in grouped.columns or "value" not in grouped.columns:
            raise ValueError("Preview.generate_monthly_num_df must return columns month/value")

        grouped = grouped[["month", "value"]].copy()
        grouped["month"] = grouped["month"].astype(str)
        grouped["value"] = pd.to_numeric(grouped["value"], errors="coerce")
        grouped = grouped.dropna(subset=["value"]).sort_values("month").reset_index(drop=True)

        self._log(f"[debug][load_monthly_series] cleaned shape={grouped.shape}")
        self._log(f"[debug][load_monthly_series] cleaned=\n{grouped.head(24)}")
        return grouped

    def compute_scores_for_city(
        self,
        target_name_ja: str,
        city_key: str,
        micro_raw_dir: str | Path,
        macro_raw_dir: str | Path,
        out_dir: str | Path,
        num_type: str = "domestic",
    ) -> List[dict]:
        micro_records = read_folder_records(micro_raw_dir)
        macro_records = read_folder_records(macro_raw_dir)
        all_records = [*micro_records, *macro_records]

        self._log(f"[debug][compute_scores_for_city] city={city_key}")
        self._log(f"[debug][compute_scores_for_city] micro_raw_dir={micro_raw_dir}")
        self._log(f"[debug][compute_scores_for_city] macro_raw_dir={macro_raw_dir}")
        self._log(f"[debug][compute_scores_for_city] micro_count={len(micro_records)}")
        self._log(f"[debug][compute_scores_for_city] macro_count={len(macro_records)}")
        self._log(f"[debug][compute_scores_for_city] all_count={len(all_records)}")

        if not all_records:
            print(f"[warn] no candidate texts for {city_key}")
            return []

        all_records = [dict(x) for x in all_records]
        all_records.sort(
            key=lambda x: (
                x.get("month", ""),
                x.get("created_at", ""),
                x.get("scope", ""),
                x.get("source", ""),
                x.get("key", 0),
            )
        )
        for idx, rec in enumerate(all_records, start=1):
            rec["raw_order"] = idx

        series_df = self.load_monthly_series(spot_name=city_key, num_type=num_type)

        texts = [rec.get("abstract", "") for rec in all_records]
        text_emb = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        task_desc = self.build_task_description(target_name_ja)
        task_emb = self.embedder.encode([task_desc], normalize_embeddings=True, convert_to_numpy=True)[0]

        rel_scores = [cosine_to_unit_interval(float(np.dot(vec, task_emb))) for vec in text_emb]
        for rec, score in zip(all_records, rel_scores):
            rec["rel_score"] = score
            rec["task_description"] = task_desc
            rec["city_key"] = city_key
            rec["target_name_ja"] = target_name_ja

        month_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, rec in enumerate(all_records):
            month_to_indices[rec["month"]].append(idx)

        sorted_months = sorted(month_to_indices.keys())
        self._log(f"[debug][compute_scores_for_city] text months={sorted_months}")

        for month in sorted_months:
            current_indices = month_to_indices[month]
            history_indices: list[int] = []
            for back in range(1, self.novelty_lookback_months + 1):
                history_indices.extend(month_to_indices.get(month_shift(month, -back), []))

            self._log(
                f"[debug][novel] month={month} current_count={len(current_indices)} "
                f"history_count={len(history_indices)}"
            )

            if not history_indices:
                for idx in current_indices:
                    all_records[idx]["novel_score"] = 1.0
                    all_records[idx]["novelty_reference_months"] = []
                continue

            history_mat = text_emb[history_indices]
            ref_months = [month_shift(month, -b) for b in range(1, self.novelty_lookback_months + 1)]
            for idx in current_indices:
                sims = history_mat @ text_emb[idx]
                max_sim = max(cosine_to_unit_interval(float(x)) for x in sims.tolist())
                all_records[idx]["novel_score"] = max(0.0, 1.0 - max_sim)
                all_records[idx]["novelty_reference_months"] = ref_months

        trend_desc_emb_map: dict[tuple[str, str], np.ndarray | None] = {}
        trend_meta_map: dict[tuple[str, str], dict] = {}

        month_to_scopes: dict[str, set[str]] = defaultdict(set)
        for rec in all_records:
            month_to_scopes[rec["month"]].add(str(rec.get("scope", "micro")).lower())

        for month in sorted_months:
            for scope in sorted(month_to_scopes[month]):
                lookback = self._get_scope_history_window(scope)
                horizon = self._get_scope_trend_horizon(scope)

                result = self.trend_forecaster.forecast_for_month(
                    series_df=series_df,
                    spot_name_ja=target_name_ja,
                    base_month=month,
                    scope=scope,
                    lookback_window=lookback,
                    horizon=horizon,
                )

                key = (month, scope)
                trend_meta_map[key] = {
                    "trend_available": len(result.predicted_values) > 0,
                    "trend_description": result.description if result.predicted_values else None,
                    "trend_horizon_months": result.horizon,
                    "trend_start_month": result.start_month,
                    "trend_history_window": result.lookback_window,
                    "trend_history_size": len(result.history_values),
                    "trend_history_values": result.history_values,
                    "trend_predicted_values": result.predicted_values,
                }

                self._log(
                    f"[debug][trend] month={month} scope={scope} "
                    f"history={result.history_values} preds={result.predicted_values}"
                )

                if result.predicted_values:
                    trend_desc_emb_map[key] = self.embedder.encode(
                        [result.description],
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )[0]
                else:
                    trend_desc_emb_map[key] = None

        for month, indices in month_to_indices.items():
            for idx in indices:
                scope = str(all_records[idx].get("scope", "micro")).lower()
                key = (month, scope)
                meta = trend_meta_map.get(
                    key,
                    {
                        "trend_available": False,
                        "trend_description": None,
                        "trend_horizon_months": 0,
                        "trend_start_month": None,
                        "trend_history_window": self._get_scope_history_window(scope),
                        "trend_history_size": 0,
                        "trend_history_values": [],
                        "trend_predicted_values": [],
                    },
                )
                trend_emb = trend_desc_emb_map.get(key)

                trend_score = (
                    None
                    if trend_emb is None
                    else cosine_to_unit_interval(float(np.dot(text_emb[idx], trend_emb)))
                )

                all_records[idx]["trend_available"] = meta["trend_available"]
                all_records[idx]["trend_description"] = meta["trend_description"]
                all_records[idx]["trend_horizon_months"] = meta["trend_horizon_months"]
                all_records[idx]["trend_start_month"] = meta["trend_start_month"]
                all_records[idx]["trend_history_window"] = meta["trend_history_window"]
                all_records[idx]["trend_history_size"] = meta["trend_history_size"]
                all_records[idx]["trend_history_values"] = meta["trend_history_values"]
                all_records[idx]["trend_predicted_values"] = meta["trend_predicted_values"]
                all_records[idx]["trend_score"] = trend_score

                self._log(
                    f"[debug][trend assign] month={month} scope={scope} "
                    f"trend_available={meta['trend_available']} trend_score={trend_score}"
                )

        for rec in all_records:
            rec["score_no_filter"] = None
            for variant in self.VARIANT_MAP:
                rec[f"score_{variant}"] = self.compute_variant_score(rec, variant)

        self.save_scored_by_month(records=all_records, out_dir=out_dir)
        self._log(f"[debug][compute_scores_for_city] saved scored records to {out_dir}")
        return all_records

    def compute_variant_score(self, record: dict, variant: str) -> float | None:
        keys = self.VARIANT_MAP[variant]
        available_keys = [k for k in keys if record.get(f"{k}_score") is not None]
        if not available_keys:
            return None

        sub_weights = {k: self.weights[k] for k in available_keys}
        denom = sum(sub_weights.values())
        normalized = {k: v / denom for k, v in sub_weights.items()}

        return float(sum(float(record[f"{k}_score"]) * normalized[k] for k in available_keys))

    def _get_scope_history_window(self, scope: str) -> int:
        return self.macro_history_window_months if str(scope).lower() == "macro" else self.micro_history_window_months

    @staticmethod
    def save_scored_by_month(records: Iterable[dict], out_dir: str | Path) -> None:
        out_dir = ensure_dir(out_dir)
        month_map: dict[str, list[dict]] = defaultdict(list)

        for rec in records:
            month_map[rec["month"]].append(rec)

        for month, items in month_map.items():
            items = sorted(
                items,
                key=lambda x: (
                    x.get("created_at", ""),
                    x.get("scope", ""),
                    x.get("source", ""),
                    x.get("raw_order", 0),
                ),
            )
            filename = out_dir / f"{month.replace('-', '_')}.jsonl"
            dump_jsonl(filename, items)