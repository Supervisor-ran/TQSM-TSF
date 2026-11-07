from pathlib import Path

from src.data_collector.topk_generator import TopKGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOPK_CONFIG = {
    'city_keys': ['kamakura', 'yokohama'],
    'k': 10,
    'variants': [
        'no_filter',
        'only_rel',
        'only_novel',
        'only_trend',
        'rel_novel',
        'rel_trend',
        'novel_trend',
        'full',
    ],
    'modes': [
        'balanced',
    ],
    'debug': True,
}


def main():
    generator = TopKGenerator(
        scored_root=PROJECT_ROOT / 'data' / 'text_collection' / 'scored',
        debug=TOPK_CONFIG['debug'],
    )
    summary = generator.generate_for_many_cities(
        city_keys=TOPK_CONFIG['city_keys'],
        k=TOPK_CONFIG['k'],
        variants=TOPK_CONFIG['variants'],
        modes=TOPK_CONFIG['modes'],
    )
    print('Top-k generation finished.')
    print(summary)


if __name__ == '__main__':
    main()