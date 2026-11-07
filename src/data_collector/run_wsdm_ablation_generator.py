from pathlib import Path

from src.data_collector.topk_generator import TopKGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CITY_KEYS = ["kamakura", "yokohama"]
RANDOM_SEED = 42


def main():
    generator = TopKGenerator(
        scored_root=PROJECT_ROOT / "data" / "text_collection" / "scored",
        out_root=PROJECT_ROOT / "data" / "text_collection" / "topk",
        debug=True,
        random_seed=RANDOM_SEED,
    )

    # Full model uses 5 macro + 5 micro = 10 texts/month.
    # Mixed keeps the same TOTAL text budget: pooled TQSM Top-10.
    mixed_summary = generator.generate_for_many_cities(
        city_keys=CITY_KEYS,
        k=10,
        variants=["full"],
        modes=["mixed"],
    )

    # Remove-one-branch ablations keep the surviving branch unchanged:
    # macro_only = TQSM Top-5 macro, micro branch absent
    # micro_only = TQSM Top-5 micro, macro branch absent
    scope_summary = generator.generate_for_many_cities(
        city_keys=CITY_KEYS,
        k=5,
        variants=["full"],
        modes=["macro_only", "micro_only"],
    )

    print("WSDM macro-micro generation finished.")
    print("mixed: full/pooled Top-10")
    print("macro_only: full/TQSM macro Top-5")
    print("micro_only: full/TQSM micro Top-5")
    print({"mixed": mixed_summary, "scope_top5": scope_summary})


if __name__ == "__main__":
    main()
