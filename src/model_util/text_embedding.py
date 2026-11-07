# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


# 尽量避免 PyTorch / Transformers 在 CUDA 上使用 SDPA / flash attention。
# Windows + 某些 CUDA / torch / transformers 组合下，BERT attention 容易报：
# RuntimeError: CUDA error: invalid configuration argument
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


class TextEmbedding(nn.Module):
    def __init__(
        self,
        model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
        use_attention: bool = False,
        use_multihead: bool = False,
        num_heads: int = 4,
        pooling_mode: str = "mean",
        max_length: int = 512,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 重点修改：
        # 强制使用 eager attention，避免调用 scaled_dot_product_attention。
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(model_name)
            if hasattr(self.encoder.config, "_attn_implementation"):
                self.encoder.config._attn_implementation = "eager"

        self.use_attention = use_attention
        self.use_multihead = use_multihead
        self.num_heads = num_heads
        self.max_length = int(max_length)

        assert pooling_mode in ["mean", "sum"], "pooling_mode must be 'mean' or 'sum'"
        self.pooling_mode = pooling_mode

        # BERT 只作为 frozen feature extractor 使用。
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        hidden_size = self.encoder.config.hidden_size

        if self.use_attention:
            if self.use_multihead:
                assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
                self.attn = nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    batch_first=True,
                )
            else:
                self.self_attn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1),
                )

    def forward(self, text_list: List[List[List[str]]], device=None) -> torch.Tensor:
        """
        Parameters
        ----------
        text_list : List[B][W][List[str sentences]]
            每个时间步是多句文本，句子未分词，直接是字符串。

        Returns
        -------
        Tensor of shape (B, W, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device

        device = torch.device(device)

        # 确保 encoder 在目标 device 上。
        self.encoder.to(device)
        self.encoder.eval()

        B = len(text_list)
        W = len(text_list[0])

        outputs = []

        for b in range(B):
            window_embs = []

            for t in range(W):
                sentences = text_list[b][t]

                # 防止某个月没有文本导致 tokenizer 报错。
                if sentences is None:
                    sentences = [""]

                if isinstance(sentences, str):
                    sentences = [sentences]

                if len(sentences) == 0:
                    sentences = [""]

                sentences = [str(s) if s is not None else "" for s in sentences]

                encoded = self.tokenizer(
                    sentences,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                encoded = {k: v.to(device) for k, v in encoded.items()}

                # 重点修改：
                # frozen encoder 不需要梯度，否则会浪费显存，也可能触发不稳定 CUDA kernel。
                with torch.no_grad():
                    model_output = self.encoder(**encoded)

                token_embeddings = model_output.last_hidden_state  # (S, L, H)

                attention_mask = encoded["attention_mask"].unsqueeze(-1).to(
                    token_embeddings.dtype
                )

                # 对每个句子做 mean pooling。
                sentence_embs = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(
                    1
                ).clamp(min=1)

                if self.use_attention and len(sentences) > 1:
                    if self.use_multihead:
                        attn_output, _ = self.attn(
                            sentence_embs.unsqueeze(0),
                            sentence_embs.unsqueeze(0),
                            sentence_embs.unsqueeze(0),
                        )
                        pooled = attn_output.mean(1).squeeze(0)
                    else:
                        scores = self.self_attn(sentence_embs)
                        weights = torch.softmax(scores, dim=0)
                        pooled = (weights * sentence_embs).sum(0)
                else:
                    if self.pooling_mode == "mean":
                        pooled = sentence_embs.mean(0)
                    else:
                        pooled = sentence_embs.sum(0)

                window_embs.append(pooled)

            outputs.append(torch.stack(window_embs))

        return torch.stack(outputs)  # (B, W, H)