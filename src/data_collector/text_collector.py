from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal

import pandas as pd

from src.model_util.llm_caller import AIAgentClient
from src.model_util.sentiment_scoring import get_sentiment_score
from src.utils.check_text import validate_json_structure


class TextCollector:
    """Collects Japanese tourism-related candidate texts and saves them by month."""

    def __init__(
        self,
        xai_env: str = "XAI_API_KEY",
        gpt_env: str = "OPENAI_API_KEY",
    ) -> None:
        xai_key = os.getenv(xai_env)
        gpt_key = os.getenv(gpt_env)

        missing = [name for name, val in ((xai_env, xai_key), (gpt_env, gpt_key)) if not val]
        if missing:
            raise RuntimeError(f"Missing env var(s): {', '.join(missing)}")

        self.ai_agent = AIAgentClient(xai_key=xai_key, gpt_key=gpt_key)

    def build_domestic_query(self, start_time: str, end_time: str, keywords: List[str]) -> str:
        joined_keywords = ", ".join(keywords)
        return (
            "あなたは日本国内観光向けの時系列テキスト収集エージェントです。\n"
            f"以下のキーワードに関する{start_time}から{end_time}までの日本語情報を検索してください: {joined_keywords}\n"
            "X プラットフォーム (site:x.com) と Yahoo! Japan (site:yahoo.co.jp) を優先してください。"
            "ただし必要であれば、他の日本語ニュース・ブログ・自治体・観光協会サイトも含めて構いません。\n"
            "結果はJSON配列のみで返してください。各要素は以下の形式です。\n"
            "[{\"abstract\": \"情報の要約\", \"created_at\": \"YYYY-MM-DD\", \"source\": \"URLまたは出典\", \"view_count\": \"閲覧数または不明\"}]\n"
            "created_at は必ず日付のみ、abstract は100〜300字程度の日本語で要約してください。"
            "情報が見つからない場合は空配列 [] を返してください。"
        )

    def col_raw(self, start_time: str, end_time: str, keywords: List[str], query: str | None = None):
        if query is None:
            query = self.build_domestic_query(start_time=start_time, end_time=end_time, keywords=keywords)
        answer = self.ai_agent.ask_grok(query)
        return self._parse_answer(answer)

    def col_and_preprocess_raw(
        self,
        start_time: str,
        end_time: str,
        keywords: List[str],
        scope: str,
        target_name: str,
        query: str | None = None,
    ) -> List[dict]:
        answer = self.col_raw(start_time=start_time, end_time=end_time, keywords=keywords, query=query)
        validated = self.give_sentiment_score(answer, start_time, end_time)

        for rec in validated:
            rec["scope"] = scope
            rec["target_name"] = target_name
            rec["keywords"] = keywords
            rec["query_start"] = start_time
            rec["query_end"] = end_time

        return validated

    @staticmethod
    def give_sentiment_score(texts: List[dict], start_time: str, end_time: str) -> List[dict]:
        file_name = f"{start_time}_{end_time}"
        validated_texts: List[dict] = []

        for idx, text in enumerate(texts, start=1):
            res, msg = validate_json_structure(text, file_name, idx)
            if not res:
                print(f"❌ 第 {idx} 条 JSON 被移除：{msg}")
                continue

            try:
                score, prob = get_sentiment_score(text["abstract"])
            except Exception:
                score, prob = None, None

            text["sentiment_score"] = score
            text["label_prob"] = prob
            validated_texts.append(text)

        return validated_texts

    @staticmethod
    def deduplicate(records: List[Dict], existed_rec: List[Dict]) -> List[Dict]:
        seen = set()
        merged: List[Dict] = []

        for rec in existed_rec + records:
            key = (
                (rec.get("abstract") or "").strip(),
                (rec.get("created_at") or "").strip(),
                (rec.get("source") or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)

        merged.sort(key=lambda x: (x.get("created_at", ""), x.get("source", ""), x.get("abstract", "")))
        return merged

    def save_by_month(
        self,
        records: Iterable[dict],
        out_path: str | Path,
        *,
        orient: Literal["split", "records", "index", "table", "columns", "values"] = "records",
    ) -> None:
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        month_map: dict[str, list[dict]] = defaultdict(list)

        for rec in records:
            month = rec["month"]
            month_map[month].append(rec)

        for month, items in month_map.items():
            year, mon = month.split("-")
            filename = out_path / f"{year}_{mon}.jsonl"

            existing = self.load_jsonl(filename) if filename.exists() else []
            all_records = self.deduplicate(items, existing)

            df = pd.DataFrame(all_records)
            filename.write_text(
                df.to_json(orient=orient, force_ascii=False, lines=True),
                encoding="utf-8",
            )

    @staticmethod
    def add_month_key(records: List[dict], date_field: str = "created_at") -> None:
        month_groups: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            rec["month"] = rec[date_field][:7]
            month_groups[rec["month"]].append(rec)

        for month, items in month_groups.items():
            items.sort(key=lambda x: (x.get(date_field, ""), x.get("source", ""), x.get("abstract", "")))
            for i, rec in enumerate(items, 1):
                rec["key"] = i

    @staticmethod
    def load_jsonl(path: str | Path) -> List[dict]:
        path = Path(path)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _parse_answer(self, answer) -> List[dict]:
        if isinstance(answer, list):
            return [self._normalize_record(x) for x in answer if isinstance(x, dict)]

        if isinstance(answer, dict):
            if "results" in answer and isinstance(answer["results"], list):
                return [self._normalize_record(x) for x in answer["results"] if isinstance(x, dict)]
            return [self._normalize_record(answer)]

        if isinstance(answer, str):
            payload = self._extract_json_payload(answer)
            return [self._normalize_record(x) for x in payload if isinstance(x, dict)]

        raise ValueError("Unsupported LLM answer format.")

    @staticmethod
    def _extract_json_payload(text: str) -> List[dict]:
        stripped = text.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
                return data["results"]
        except Exception:
            pass

        match = re.search(r"\[.*\]", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("Cannot parse JSON array from LLM answer.")
        return json.loads(match.group(0))

    @staticmethod
    def _normalize_record(item: dict) -> dict:
        created_at = str(item.get("created_at", "")).strip()[:10]
        return {
            "abstract": str(item.get("abstract", "")).strip(),
            "created_at": created_at,
            "source": str(item.get("source", "")).strip(),
            "view_count": str(item.get("view_count", "不明")).strip() or "不明",
        }
