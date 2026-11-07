import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, Dict


_MODEL_NAME = "LoneWolfgang/bert-for-japanese-twitter-sentiment"
_LABELS = ["negative", "neutral", "positive"]

_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
_model.eval()  # switch to inference mode

def get_sentiment_score(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Compute sentiment for Japanese text.

    Returns
    -------
    score : float
        Continuous value in [-1, 1] (neg → pos).
    dist  : dict
        Class-probability mapping {negative, neutral, positive}.
    """
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
    score = float(np.dot(probs, [-1, 0, 1]))
    return score, {k: float(v) for k, v in zip(_LABELS, probs)}