from __future__ import annotations

from pathlib import Path

from src.data_collector.scoring_utils import TourismTextScorer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_ROOT = PROJECT_ROOT / "data" / "text_collection"

CONFIG = {
    "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "weights": {
        "rel": 0.45,
        "novel": 0.25,
        "trend": 0.30,
    },
    "novelty_lookback_months": 3,
    "micro_trend_horizon_months": 3,
    "macro_trend_horizon_months": 6,
    "micro_history_window_months": 3,
    "macro_history_window_months": 6,
    "debug": True,
    "scored_dirname": "scored_trend",
    "cities": {
        "kamakura": {
            "target_name_ja": "鎌倉市",
            "num_type": "domestic",
        },
        "yokohama": {
            "target_name_ja": "横浜市中区",
            "num_type": "domestic",
        },
    },
}


def run_rescore_existing(config: dict) -> None:
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

    raw_macro_dir = TEXT_ROOT / "raw" / "macro" / "global"
    if not raw_macro_dir.exists():
        raise FileNotFoundError(f"global macro raw dir not found: {raw_macro_dir}")

    scored_root = TEXT_ROOT / config.get("scored_dirname", "scored_trend")
    scored_root.mkdir(parents=True, exist_ok=True)

    for city_key, city_cfg in config["cities"].items():
        micro_raw_dir = TEXT_ROOT / "raw" / "micro" / city_key
        if not micro_raw_dir.exists():
            print(f"[skip] {city_key}: micro raw dir not found -> {micro_raw_dir}")
            continue

        out_dir = scored_root / city_key

        print(f"[run] rescore existing raw texts for {city_key}")
        print(f"  micro_raw_dir = {micro_raw_dir}")
        print(f"  macro_raw_dir = {raw_macro_dir}")
        print(f"  out_dir = {out_dir}")

        scorer.compute_scores_for_city(
            target_name_ja=city_cfg["target_name_ja"],
            city_key=city_key,
            micro_raw_dir=micro_raw_dir,
            macro_raw_dir=raw_macro_dir,
            out_dir=out_dir,
            num_type=city_cfg.get("num_type", "domestic"),
        )

    print("\nDone.")
    print(f"New scored files are under: {scored_root}")


if __name__ == "__main__":
    run_rescore_existing(CONFIG)