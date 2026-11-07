#Use Grok
from openai import OpenAI
from typing import List, Dict, Optional
import json
from datetime import datetime


class AIAgentClient:
    def __init__(self, xai_key: str = None, gpt_key: str = None):
        self.xai_key = xai_key
        self.gpt_key = gpt_key
        self.xai_client = OpenAI(api_key=self.xai_key, base_url="https://api.x.ai/v1")
        self.gpt_client = OpenAI(api_key=self.gpt_key)

    def ask_grok(self, query: str, model: str = "grok-3",
                 max_tokens: int = 2000, temperature: float = 0.7) -> List[Dict[str, Optional[str]]]:

        if not self.xai_key:
            raise ValueError("xai_api_key is required")

        try:
            response = self.xai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            answer = response.choices[0].message.content
            try:
                results = json.loads(answer)
                return results if isinstance(results, list) else []
            except json.JSONDecodeError:
                return []
        except Exception as e:
            return [{
                "abstract": "検索中にエラーが発生しました",
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "source": "Grok API",
                "view_count": "0",
                "error": str(e)
            }]

    def ask_openai(self, prompt: str, model: str = "gpt-4", temperature: float = 0.7) -> str:

        if not self.gpt_key:
            raise ValueError("xai_api_key is required")
        try:
            response = self.gpt_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"请求失败: {e}"
