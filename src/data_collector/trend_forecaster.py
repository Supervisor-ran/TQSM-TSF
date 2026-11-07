from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def month_shift(month_str: str, offset: int) -> str:
    period = pd.Period(month_str, freq='M') + offset
    return str(period)


@dataclass
class TrendForecastResult:
    base_month: str
    start_month: str
    scope: str
    lookback_window: int
    horizon: int
    history_values: List[float]
    predicted_values: List[float]
    description: str


class RecursiveTrendForecaster:
    def __init__(
        self,
        cache_dir: str | Path | None = None,
        debug: bool = False,
    ) -> None:
        self.cache_dir = ensure_dir(
            cache_dir or (Path(__file__).resolve().parents[2] / "data" / "text_collection" / "cache" / "trend_descriptions")
        )
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    @staticmethod
    def collect_history_values(
        series_df: pd.DataFrame,
        month: str,
        window: int,
    ) -> List[float]:
        if series_df.empty:
            return []

        months = series_df["month"].astype(str).tolist()
        values = pd.to_numeric(series_df["value"], errors="coerce").tolist()

        if month not in months:
            return []

        idx = months.index(month)
        start_idx = max(0, idx - window + 1)
        history = [float(v) for v in values[start_idx: idx + 1] if pd.notna(v)]
        return history

    @staticmethod
    def recursive_mean_forecast(
        history_values: List[float],
        lookback_window: int,
        horizon: int,
    ) -> List[float]:
        if not history_values or horizon <= 0:
            return []

        series = [float(x) for x in history_values]
        preds: List[float] = []

        for _ in range(horizon):
            context = series[-lookback_window:] if lookback_window > 0 else series
            pred = float(np.mean(context))
            preds.append(pred)
            series.append(pred)

        return preds

    @staticmethod
    def build_description(
        spot_name_ja: str,
        start_month: str,
        predicted_values: List[float],
        *,
        scope: str,
        lookback_window: int,
        history_values: List[float],
    ) -> str:
        if not predicted_values:
            return f"{spot_name_ja}の{start_month}以降の傾向説明を生成できませんでした。"

        diffs = []
        for i in range(1, len(predicted_values)):
            delta = predicted_values[i] - predicted_values[i - 1]
            direction = "上昇" if delta > 0 else "下降" if delta < 0 else "横ばい"
            diffs.append(
                f"{month_shift(start_month, i - 1)}→{month_shift(start_month, i)}で{direction}({delta:.4f})"
            )

        volatility = float(np.std(predicted_values)) if len(predicted_values) > 1 else 0.0

        return (
            f"{spot_name_ja}について、{scope}系列用の直近{len(history_values)}か月"
            f"（基準窓{lookback_window}か月）から再帰平均で外挿した"
            f"{start_month}以降{len(predicted_values)}か月の推定系列では、"
            f"最小値{min(predicted_values):.4f}、最大値{max(predicted_values):.4f}、"
            f"標準偏差{volatility:.4f}が見込まれる。"
            f"推移は{'、'.join(diffs) if diffs else '単月のため比較不可'}。"
        )

    def forecast_for_month(
        self,
        series_df: pd.DataFrame,
        *,
        spot_name_ja: str,
        base_month: str,
        scope: str,
        lookback_window: int,
        horizon: int,
    ) -> TrendForecastResult:
        cache_file = (
            self.cache_dir
            / spot_name_ja
            / scope
            / f"{base_month.replace('-', '_')}_lb{lookback_window}_hz{horizon}.json"
        )
        ensure_dir(cache_file.parent)

        if cache_file.exists():
            data = pd.read_json(cache_file, typ="series").to_dict()
            return TrendForecastResult(
                base_month=data["base_month"],
                start_month=data["start_month"],
                scope=data["scope"],
                lookback_window=int(data["lookback_window"]),
                horizon=int(data["horizon"]),
                history_values=list(data["history_values"]),
                predicted_values=list(data["predicted_values"]),
                description=str(data["description"]),
            )

        history_values = self.collect_history_values(
            series_df=series_df,
            month=base_month,
            window=lookback_window,
        )
        start_month = month_shift(base_month, 1)

        predicted_values = self.recursive_mean_forecast(
            history_values=history_values,
            lookback_window=lookback_window,
            horizon=horizon,
        )

        description = self.build_description(
            spot_name_ja=spot_name_ja,
            start_month=start_month,
            predicted_values=predicted_values,
            scope=scope,
            lookback_window=lookback_window,
            history_values=history_values,
        )

        result = TrendForecastResult(
            base_month=base_month,
            start_month=start_month,
            scope=scope,
            lookback_window=lookback_window,
            horizon=len(predicted_values),
            history_values=history_values,
            predicted_values=predicted_values,
            description=description,
        )

        pd.Series(
            {
                "base_month": result.base_month,
                "start_month": result.start_month,
                "scope": result.scope,
                "lookback_window": result.lookback_window,
                "horizon": result.horizon,
                "history_values": result.history_values,
                "predicted_values": result.predicted_values,
                "description": result.description,
            }
        ).to_json(cache_file, force_ascii=False)

        self._log(
            f"[debug][trend_forecaster] month={base_month} scope={scope} "
            f"history={history_values} preds={predicted_values}"
        )
        return result