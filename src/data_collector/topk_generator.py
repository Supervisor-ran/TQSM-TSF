from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / 'data' / 'text_collection'


VALID_VARIANTS = [
    'no_filter',
    'only_rel',
    'only_novel',
    'only_trend',
    'rel_novel',
    'rel_trend',
    'novel_trend',
    'full',
]

VALID_MODES = [
    'balanced',
    'mixed',
    'macro_only',
    'micro_only',
]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: str | Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: str | Path, records: Sequence[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def list_month_files(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(list(folder.glob('*.json')) + list(folder.glob('*.jsonl')))


def read_folder_records(folder: str | Path) -> List[dict]:
    records: List[dict] = []
    for file in list_month_files(folder):
        records.extend(load_jsonl(file))
    return records


class TopKGenerator:
    def __init__(
        self,
        scored_root: str | Path | None = None,
        out_root: str | Path | None = None,
        debug: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.scored_root = Path(scored_root or (DATA_ROOT / 'scored'))
        self.out_root = Path(out_root or (DATA_ROOT / 'topk'))
        self.debug = debug
        self.random_seed = int(random_seed)

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    @staticmethod
    def _month_sort_key(x: dict):
        return (
            x.get('created_at', ''),
            x.get('scope', ''),
            x.get('source', ''),
            x.get('raw_order', 0),
        )

    @staticmethod
    def _score_sort_key(x: dict, variant: str):
        score = x.get(f'score_{variant}')
        if score is None:
            score = float('-inf')
        return (
            float(score),
            -int(x.get('raw_order', 0)),
        )



    def _sort_records(self, records: List[dict], variant: str) -> List[dict]:
        if variant == 'no_filter':
            return sorted(records, key=self._month_sort_key)
        return sorted(records, key=lambda x: self._score_sort_key(x, variant), reverse=True)

    @staticmethod
    def _filter_variant_available(records: List[dict], variant: str) -> List[dict]:
        if variant == 'no_filter':
            return list(records)
        return [x for x in records if x.get(f'score_{variant}') is not None]

    @staticmethod
    def _filter_by_scope(records: List[dict], scope: str) -> List[dict]:
        return [x for x in records if str(x.get('scope', '')).lower() == scope.lower()]

    def _select_mixed(self, records: List[dict], k: int, variant: str) -> List[dict]:
        usable = self._filter_variant_available(records, variant)
        ordered = self._sort_records(usable, variant)
        return ordered[:k]

    def _select_micro_only(self, records: List[dict], k: int, variant: str) -> List[dict]:
        usable = self._filter_variant_available(records, variant)
        usable = self._filter_by_scope(usable, 'micro')
        ordered = self._sort_records(usable, variant)
        return ordered[:k]

    def _select_macro_only(self, records: List[dict], k: int, variant: str) -> List[dict]:
        usable = self._filter_variant_available(records, variant)
        usable = self._filter_by_scope(usable, 'macro')
        ordered = self._sort_records(usable, variant)
        return ordered[:k]

    def _select_balanced(self, records: List[dict], k: int, variant: str) -> List[dict]:
        usable = self._filter_variant_available(records, variant)

        micro = self._filter_by_scope(usable, 'micro')
        macro = self._filter_by_scope(usable, 'macro')

        micro = self._sort_records(micro, variant)
        macro = self._sort_records(macro, variant)

        micro_quota = k // 2
        macro_quota = k // 2

        if k % 2 == 1:
            if len(micro) >= len(macro):
                micro_quota += 1
            else:
                macro_quota += 1

        selected_micro = micro[:micro_quota]
        selected_macro = macro[:macro_quota]

        selected_ids = {id(x) for x in selected_micro + selected_macro}
        selected = selected_micro + selected_macro

        remainder = k - len(selected)
        if remainder > 0:
            leftover = [x for x in (micro + macro) if id(x) not in selected_ids]
            leftover = self._sort_records(leftover, variant)
            selected.extend(leftover[:remainder])

        return self._sort_records(selected, variant)[:k]

    def select_records(
        self,
        records: List[dict],
        k: int,
        variant: str,
        mode: str,
    ) -> List[dict]:
        if variant not in VALID_VARIANTS:
            raise ValueError(f'Invalid variant: {variant}')
        if mode not in VALID_MODES:
            raise ValueError(f'Invalid mode: {mode}')
        if k <= 0:
            raise ValueError('k must be positive')

        if variant == 'no_filter':
            if mode in ('mixed', 'balanced'):
                return sorted(records, key=self._month_sort_key)
            if mode == 'micro_only':
                return sorted(self._filter_by_scope(records, 'micro'), key=self._month_sort_key)
            if mode == 'macro_only':
                return sorted(self._filter_by_scope(records, 'macro'), key=self._month_sort_key)

        if mode == 'mixed':
            return self._select_mixed(records, k, variant)
        if mode == 'micro_only':
            return self._select_micro_only(records, k, variant)
        if mode == 'macro_only':
            return self._select_macro_only(records, k, variant)
        if mode == 'balanced':
            return self._select_balanced(records, k, variant)

        raise ValueError(f'Unknown mode: {mode}')

    def generate_for_city(
        self,
        city_key: str,
        k: int,
        variants: Iterable[str] | None = None,
        modes: Iterable[str] | None = None,
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        scored_dir = self.scored_root / city_key
        if not scored_dir.exists():
            raise FileNotFoundError(f'scored dir not found: {scored_dir}')

        variants = list(variants or VALID_VARIANTS)
        modes = list(modes or VALID_MODES)

        all_records = read_folder_records(scored_dir)
        if not all_records:
            self._log(f'[warn] no scored records for {city_key}')
            return {}

        month_map: Dict[str, List[dict]] = defaultdict(list)
        for rec in all_records:
            month_map[str(rec['month'])].append(rec)

        summary: Dict[str, Dict[str, Dict[str, int]]] = {}

        self._log(f'[debug][topk] city={city_key} total_records={len(all_records)} months={sorted(month_map.keys())}')

        for variant in variants:
            summary[variant] = {}
            for mode in modes:
                summary[variant][mode] = {}
                out_dir = ensure_dir(self.out_root / city_key / variant / mode)

                for month, month_records in sorted(month_map.items()):
                    selected = self.select_records(
                        records=month_records,
                        k=k,
                        variant=variant,
                        mode=mode,
                    )

                    filename = out_dir / f"{month.replace('-', '_')}.jsonl"
                    dump_jsonl(filename, selected)
                    summary[variant][mode][month] = len(selected)

                    micro_cnt = sum(1 for x in selected if x.get('scope') == 'micro')
                    macro_cnt = sum(1 for x in selected if x.get('scope') == 'macro')

                    self._log(
                        f"[debug][topk] city={city_key} month={month} "
                        f"variant={variant} mode={mode} "
                        f"selected={len(selected)} micro={micro_cnt} macro={macro_cnt}"
                    )

        meta_file = ensure_dir(self.out_root / city_key) / f'_summary_top{k}.json'
        meta_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        self._log(f'[debug][topk] summary saved to {meta_file}')
        return summary

    def generate_for_many_cities(
        self,
        city_keys: Iterable[str],
        k: int,
        variants: Iterable[str] | None = None,
        modes: Iterable[str] | None = None,
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, int]]]]:
        all_summary: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
        for city_key in city_keys:
            all_summary[city_key] = self.generate_for_city(
                city_key=city_key,
                k=k,
                variants=variants,
                modes=modes,
            )
        return all_summary