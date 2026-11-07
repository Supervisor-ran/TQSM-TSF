from __future__ import annotations

from src.data_collector.collection_excutor import run_collection_and_scoring


CONFIG = {
    'embedding_model_name': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'weights': {
        'rel': 0.45,
        'novel': 0.25,
        'trend': 0.30,
    },
    'novelty_lookback_months': 3,
    'micro_history_window_months': 3,
    'macro_history_window_months': 6,
    "micro_trend_horizon_months": 3,
    "macro_trend_horizon_months": 6,
    'split_days': 6,
    'debug': True,
    'global_macro': {
        'keywords': ['観光政策', '国内旅行', '日本経済', '物価', '宿泊料金'],
        'start_time': '2021-01-01',
        'end_time': '2026-04-01',
    },
    'cities': {
        'kamakura': {
            'target_name_ja': '鎌倉市',
            'num_type': 'domestic',
            'micro': {
                'keywords': ['江ノ島', '湘南', '鎌倉大仏', '鎌倉高校前'],
                'start_time': '2021-01-01',
                'end_time': '2026-04-01',
            },
        },
        'yokohama': {
            'target_name_ja': '横浜市中区',
            'num_type': 'domestic',
            'micro': {
                'keywords': ['横浜中区', '赤レンガ倉庫', '中華街', '桜木町', '根岸森林公園', '三渓園'],
                'start_time': '2021-01-01',
                'end_time': '2026-04-01',
            },
        },
    },
}


if __name__ == '__main__':
    run_collection_and_scoring(CONFIG, retry=1, pause_sec=0.0)