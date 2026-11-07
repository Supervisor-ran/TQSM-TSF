from pathlib import Path
import pandas as pd
import hashlib
from typing import List
import itertools


class Preview:
    def __init__(self, num_type='domestic', text_encoder='cp932', debug: bool = False):
        assert num_type in ('domestic', 'inbound')
        self.base_path = Path(__file__).parent.parent.parent / 'data' / 'num_collection' / num_type
        self.num_raw_path = self.base_path / 'raw'
        self.num_prep_path = self.base_path / 'preprocessed'
        self.encoder = text_encoder
        self.debug = debug

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _get_folder_signature(self, folder: Path, suffixes: List[str]) -> str:
        if not folder.exists():
            raise FileNotFoundError(f"❌ 路径不存在: {folder}")
        files = sorted([
            f for suffix in suffixes
            for f in folder.glob(f"*{suffix}")
        ])
        signature_str = "".join(f"{f.name}:{f.stat().st_mtime}|" for f in files)
        return hashlib.md5(signature_str.encode('utf-8')).hexdigest()

    def _load_cached_df(self, cache_dir: Path, cache_key: str, current_hash: str) -> pd.DataFrame | None:
        cache_path = cache_dir / f"{cache_key}.pkl"
        hash_path = cache_dir / f"{cache_key}.hash"
        if cache_path.exists() and hash_path.exists():
            if hash_path.read_text().strip() == current_hash:
                self._debug_print(f"✅ 使用磁盘缓存：{cache_key}")
                return pd.read_pickle(cache_path)
        return None

    def _save_cached_df(self, df: pd.DataFrame, cache_dir: Path, cache_key: str, current_hash: str):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.pkl"
        hash_path = cache_dir / f"{cache_key}.hash"
        df.to_pickle(cache_path)
        hash_path.write_text(current_hash)

    def integrate_num(self) -> pd.DataFrame:
        suffixes = [".csv", ".xls", ".xlsx"]
        current_hash = self._get_folder_signature(self.num_raw_path, suffixes)
        cache_key = "integrate_num"

        df = self._load_cached_df(self.num_prep_path, cache_key, current_hash)
        if df is not None:
            self._debug_print(f"[debug][integrate_num] shape={df.shape}")
            self._debug_print(f"[debug][integrate_num] columns={list(df.columns)}")
            return df

        files = sorted([
            f for suffix in suffixes
            for f in self.num_raw_path.glob(f"*{suffix}")
        ])
        if not files:
            raise ValueError(f"⚠️ 未找到 CSV 或 Excel 文件：{self.num_raw_path}")

        dfs = []

        def safe_read_csv(filepath):
            encodings = [self.encoder, "utf-8-sig", "utf-8", "cp932", "shift_jis"]
            last_err = None
            for enc in encodings:
                try:
                    return pd.read_csv(filepath, encoding=enc)
                except Exception as e:
                    last_err = e
            raise RuntimeError(f"❌ 文件读取失败: {filepath} -> {last_err}")

        for file in files:
            try:
                if file.suffix == ".csv":
                    df = safe_read_csv(file)
                else:
                    df = pd.read_excel(file)
                df["__source_file__"] = file.name
                self._debug_print(
                    f"[debug][integrate_num] loaded {file.name}: "
                    f"shape={df.shape}, columns={list(df.columns)}"
                )
                dfs.append(df)
            except Exception as e:
                print(f"❌ 文件读取失败: {file.name} -> {e}")

        if not dfs:
            raise ValueError("⚠️ 所有文件读取失败，没有可用数据。")

        merged_df = pd.concat(dfs, ignore_index=True).drop_duplicates()

        self._debug_print(f"[debug][integrate_num] merged shape={merged_df.shape}")
        self._debug_print(f"[debug][integrate_num] merged columns={list(merged_df.columns)}")

        self._save_cached_df(merged_df, self.num_prep_path, cache_key, current_hash)
        self._debug_print("✅ 数值数据缓存已更新")
        return merged_df

    def filter_dataframe(self, df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
        if not conditions:
            return df.copy()

        keys = list(conditions.keys())
        value_combinations = list(itertools.product(*[conditions[k] for k in keys]))

        filtered_df = pd.DataFrame()
        for combo in value_combinations:
            mask = pd.Series([True] * len(df), index=df.index)
            for k, v in zip(keys, combo):
                mask &= (df[k] == v)
            filtered_df = pd.concat([filtered_df, df[mask]])

        out = filtered_df.drop_duplicates()
        self._debug_print(f"[debug][filter_dataframe] conditions={conditions} -> shape={out.shape}")
        return out

    def generate_monthly_num_df(self, spot_name: str) -> pd.DataFrame:
        df = self.integrate_num().copy()

        self._debug_print(f"[debug][generate_monthly_num_df] spot_name={spot_name}")
        self._debug_print(f"[debug][generate_monthly_num_df] raw shape={df.shape}")
        self._debug_print(f"[debug][generate_monthly_num_df] raw columns={list(df.columns)}")

        if spot_name == 'kamakura':
            loc = {
                '都道府県名': ["神奈川県"],
                '地域名称': ["鎌倉市"]
            }
        elif spot_name == 'yokohama':
            loc = {
                '都道府県名': ["神奈川県"],
                '地域名称': ["横浜市中区"]
            }
        else:
            raise ValueError(f'Not valid domestic spot name: {spot_name}')

        filtered = self.filter_dataframe(df, loc).copy()

        self._debug_print(f"[debug][generate_monthly_num_df] filtered shape={filtered.shape}")
        self._debug_print(f"[debug][generate_monthly_num_df] filtered columns={list(filtered.columns)}")
        self._debug_print(f"[debug][generate_monthly_num_df] filtered head=\n{filtered.head(5)}")

        required_cols = ['年', '月', '人数']
        missing = [c for c in required_cols if c not in filtered.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}, 当前列={list(filtered.columns)}")

        filtered['年'] = pd.to_numeric(filtered['年'], errors='coerce')
        filtered['月'] = pd.to_numeric(filtered['月'], errors='coerce')
        filtered['人数'] = pd.to_numeric(filtered['人数'], errors='coerce')

        filtered = filtered.dropna(subset=['年', '月', '人数']).copy()

        self._debug_print(f"[debug][generate_monthly_num_df] after numeric/dropna shape={filtered.shape}")
        self._debug_print(
            f"[debug][generate_monthly_num_df] unique years="
            f"{sorted(filtered['年'].unique().tolist())[:10]}"
        )
        self._debug_print(
            f"[debug][generate_monthly_num_df] unique months="
            f"{sorted(filtered['月'].unique().tolist())[:12]}"
        )

        grouped = (
            filtered.groupby(['年', '月'], as_index=False)['人数']
            .sum()
            .sort_values(['年', '月'])
            .reset_index(drop=True)
        )

        self._debug_print(f"[debug][generate_monthly_num_df] grouped shape={grouped.shape}")
        self._debug_print(f"[debug][generate_monthly_num_df] grouped columns={list(grouped.columns)}")
        self._debug_print(f"[debug][generate_monthly_num_df] grouped head=\n{grouped.head(12)}")

        grouped['month'] = (
            grouped['年'].astype(int).astype(str).str.zfill(4)
            + '-'
            + grouped['月'].astype(int).astype(str).str.zfill(2)
        )
        grouped['value'] = grouped['人数'].astype(float)

        out = grouped[['month', 'value']].copy()

        self._debug_print(f"[debug][generate_monthly_num_df] final monthly df shape={out.shape}")
        self._debug_print(f"[debug][generate_monthly_num_df] final monthly df=\n{out.head(12)}")

        return out

    def generate_num(self, spot_name: str):
        monthly_df = self.generate_monthly_num_df(spot_name)
        return monthly_df['value']