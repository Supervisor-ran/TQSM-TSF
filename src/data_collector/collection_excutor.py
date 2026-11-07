from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.data_collector.scoring_utils import DATA_ROOT, TourismTextScorer, ensure_dir
from src.data_collector.text_collector import TextCollector


RAW_ROOT = DATA_ROOT / 'raw'
SCORED_ROOT = DATA_ROOT / 'scored'


def split_time(start_time: str, end_time: str, step_days: int = 8) -> List[Tuple[str, str]]:
    start = datetime.strptime(start_time, '%Y-%m-%d').date()
    end = datetime.strptime(end_time, '%Y-%m-%d').date()

    result = []
    current_start = start
    while current_start <= end:
        current_end = current_start + timedelta(days=step_days - 1)
        if current_end > end:
            current_end = end
        result.append((current_start.isoformat(), current_end.isoformat()))
        current_start = current_end + timedelta(days=1)
    return result


def collect_scope_texts(
    collector: TextCollector,
    keywords: List[str],
    start_time: str,
    end_time: str,
    scope: str,
    output_name: str,
    target_name: str,
    split_days: int = 8,
) -> List[dict]:
    if scope not in {'micro', 'macro'}:
        raise ValueError("scope must be 'micro' or 'macro'")

    base_path = RAW_ROOT / scope / output_name
    ensure_dir(base_path)

    intervals = split_time(start_time=start_time, end_time=end_time, step_days=split_days)
    results: List[List[dict]] = []
    for sub_start, sub_end in intervals:
        preprocessed = collector.col_and_preprocess_raw(
            start_time=sub_start,
            end_time=sub_end,
            keywords=keywords,
            scope=scope,
            target_name=target_name,
        )
        results.append(preprocessed)

    flattened = [item for chunk in results for item in chunk]
    TextCollector.add_month_key(flattened)
    collector.save_by_month(flattened, base_path)
    return flattened


def run_collection_and_scoring(config: Dict, retry: int = 1, pause_sec: float = 0.0) -> None:
    from time import sleep

    collector = TextCollector()
    scorer = TourismTextScorer(
        embedding_model_name=config["embedding_model_name"],
        weights=config["weights"],
        novelty_lookback_months=config["novelty_lookback_months"],
        micro_history_window_months=config["micro_history_window_months"],
        macro_history_window_months=config["macro_history_window_months"],
        micro_trend_horizon_months=config["micro_trend_horizon_months"],
        macro_trend_horizon_months=config["macro_trend_horizon_months"],
        debug=config.get("debug", False),
    )

    macro_cfg = config['global_macro']
    if macro_cfg.get('keywords'):
        print('[run] global macro collection')
        collect_scope_texts(
            collector=collector,
            keywords=macro_cfg['keywords'],
            start_time=macro_cfg['start_time'],
            end_time=macro_cfg['end_time'],
            scope='macro',
            output_name='global',
            target_name='global_macro',
            split_days=config.get('split_days', 6),
        )
    else:
        print('[skip] global macro keywords are empty')

    for city_key, city_cfg in config['cities'].items():
        kws = city_cfg['micro']['keywords']
        st = city_cfg['micro']['start_time']
        et = city_cfg['micro']['end_time']
        attempt = 0

        while True:
            try:
                if not kws:
                    print(f'[skip] {city_key}:micro keywords are empty')
                else:
                    print(f'[run] {city_key}:micro | {st} ~ {et} | {len(kws)} keywords')
                    collect_scope_texts(
                        collector=collector,
                        keywords=kws,
                        start_time=st,
                        end_time=et,
                        scope='micro',
                        output_name=city_key,
                        target_name=city_cfg.get('target_name_ja', city_key),
                        split_days=config.get('split_days', 6),
                    )

                print(f'[run] scoring for {city_key}')
                scorer.compute_scores_for_city(
                    target_name_ja=city_cfg.get('target_name_ja', city_key),
                    city_key=city_key,
                    micro_raw_dir=RAW_ROOT / 'micro' / city_key,
                    macro_raw_dir=RAW_ROOT / 'macro' / 'global',
                    out_dir=SCORED_ROOT / city_key,
                    num_type=city_cfg.get('num_type', 'domestic'),
                )
                break
            except Exception as e:
                attempt += 1
                if attempt > retry:
                    print(f'[fail] {city_key} -> {e}')
                    break
                print(f'[retry {attempt}/{retry}] {city_key} -> {e}')
                sleep(1.0)
            finally:
                if pause_sec > 0:
                    sleep(pause_sec)
