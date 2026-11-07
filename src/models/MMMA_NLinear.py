# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLPProjector(nn.Module):
    """
    Generic MLP projector:
    [B, T, Din] -> [B, T, Dout]
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SingleDirectionCrossModalAttention(nn.Module):
    """
    Single-direction cross-modal attention:
    query <- numeric branch
    key/value <- text branch

    Input:
        q:  [B, N, L]
        kv: [B, N, L]
    Output:
        attended_q: [B, N, L]
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(q, kv, kv)
        out = self.dropout(out)
        out = self.out_ln(out)
        return out


def _movement_label_from_numeric(x_num: torch.Tensor) -> torch.Tensor:
    """
    Compute numeric movement labels from raw numeric series.

    Args:
        x_num: [B, seq_len, N]

    Returns:
        movement_label: [B, seq_len-1] with values in {-1, 0, 1}
    """
    diff = x_num[:, 1:, :] - x_num[:, :-1, :]
    diff_mean = diff.mean(dim=-1)
    label = torch.zeros_like(diff_mean, dtype=torch.long)

    label = torch.where(diff_mean > 0, torch.ones_like(label), label)
    label = torch.where(diff_mean < 0, -torch.ones_like(label), label)
    return label


def _movement_feature_and_label_from_text(text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute text movement feature and text movement label.

    Args:
        text_emb: [B, seq_len, D]

    Returns:
        movement_feat:  [B, seq_len-1, D]
        movement_label: [B, seq_len-1] in {-1,0,1}
    """
    diff = text_emb[:, 1:, :] - text_emb[:, :-1, :]
    diff_mean = diff.mean(dim=-1)
    label = torch.zeros_like(diff_mean, dtype=torch.long)
    label = torch.where(diff_mean > 0, torch.ones_like(label), label)
    label = torch.where(diff_mean < 0, -torch.ones_like(label), label)
    return diff, label


def _supervised_infonce_from_labels(
    feat: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    Supervised InfoNCE / SupCon-like loss using labels.
    """
    B, T, D = feat.shape
    x = feat.reshape(B * T, D)
    y = labels.reshape(B * T)

    x = F.normalize(x, dim=-1)
    sim = torch.matmul(x, x.t()) / tau
    M = sim.shape[0]

    eye = torch.eye(M, device=sim.device, dtype=torch.bool)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~eye)
    neg_mask = ~eye

    sim_den = sim.masked_fill(~neg_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim_den, dim=1, keepdim=True)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0

    if valid.sum() == 0:
        return torch.zeros((), device=feat.device, dtype=feat.dtype)

    mean_log_prob_pos = (log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1) / pos_count.clamp_min(1))
    loss = -mean_log_prob_pos[valid].mean()
    return loss


class Model(nn.Module):
    """
    MMMA_NLinear
    = Multimodal Movement Alignment NLinear

    Notes:
        The original domestic dataset has two text streams: macro and micro.
        The new Yelp / HotelRec datasets have only one text stream. This model
        therefore automatically switches branch behavior at forward time:

        - dual:   macro_text is not None and micro_text is not None
        - single: exactly one of macro_text / micro_text is not None
        - none:   both are None

        ``degrade=True`` enables the NLinear + Text Early Fusion ablation path:
        text embeddings are projected to the numeric channel dimension and added
        to the numeric input before the plain NLinear head. This path does not
        use movement alignment, contrastive text representation, or cross-modal
        attention.

        No run.py option is required for this switch.

        Text embeddings must match ``text_encoder_out_dim`` exactly. For the
        English Yelp / HotelRec datasets, SequenceExecutor uses a 768-D English
        encoder so that MMMA_NLinear does not rely on padding or truncation.
    """

    def __init__(self, configs, individual: bool = True):
        super().__init__()
        if isinstance(configs, dict):
            configs = SimpleNamespace(**configs)

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.num_enc_in = int(configs.num_enc_in)
        self.text_encoder_out_dim = int(configs.text_encoder_out_dim)
        self.individual = individual
        self.dropout = float(getattr(configs, "dropout", 0.0))
        self.degrade = bool(getattr(configs, "degrade", False))

        self.debug_forward = bool(getattr(configs, "debug_forward", False))
        self.debug_forward_max_calls = int(getattr(configs, "debug_forward_max_calls", 20))
        self._debug_forward_calls = 0

        self.use_movement_alignment = bool(
            getattr(configs, "use_movement_alignment", getattr(configs, "use_alignment_2", False))
        )
        self.movement_tau = float(
            getattr(configs, "movement_tau", getattr(configs, "alignment2_tau", 0.07))
        )
        self.movement_loss_weight = float(
            getattr(configs, "movement_loss_weight", getattr(configs, "alignment2_weight", 0.2))
        )

        self.use_cross_modality = bool(getattr(configs, "use_cross_modality", False))
        self.cross_modality_n_heads = int(getattr(configs, "cross_modality_n_heads", 1))

        self.use_time_feature = bool(getattr(configs, "use_time_feature", False))
        self.timeenc = int(getattr(configs, "timeenc", 1))
        self.time_fusion = str(getattr(configs, "time_fusion", "add")).lower()
        self.time_feat_dim = int(getattr(configs, "time_feat_dim", 0))

        if self.time_fusion not in {"add", "concat"}:
            raise ValueError(f"Unsupported time_fusion={self.time_fusion}")

        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, but got seq_len={self.seq_len}")

        if self.cross_modality_n_heads <= 0:
            raise ValueError(
                f"cross_modality_n_heads must be positive, "
                f"but got cross_modality_n_heads={self.cross_modality_n_heads}"
            )

        if self.seq_len % self.cross_modality_n_heads != 0:
            print(
                f"[MMMA_NLinear] cross_modality_n_heads={self.cross_modality_n_heads} "
                f"cannot divide seq_len={self.seq_len}, so heads is reset to seq_len={self.seq_len}."
            )
            self.cross_modality_n_heads = self.seq_len

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.num_enc_in)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        hidden_dim = max(self.text_encoder_out_dim, 256)

        self.macro_contrastive_projector = _MLPProjector(
            in_dim=self.text_encoder_out_dim,
            out_dim=self.text_encoder_out_dim,
            hidden_dim=hidden_dim,
            dropout=self.dropout,
        )
        self.micro_contrastive_projector = _MLPProjector(
            in_dim=self.text_encoder_out_dim,
            out_dim=self.text_encoder_out_dim,
            hidden_dim=hidden_dim,
            dropout=self.dropout,
        )

        self.macro_to_num_proj = nn.Linear(self.text_encoder_out_dim, self.num_enc_in)
        self.micro_to_num_proj = nn.Linear(self.text_encoder_out_dim, self.num_enc_in)

        self.macro_cross_attn = _SingleDirectionCrossModalAttention(
            embed_dim=self.seq_len,
            num_heads=self.cross_modality_n_heads,
            dropout=self.dropout,
        )
        self.micro_cross_attn = _SingleDirectionCrossModalAttention(
            embed_dim=self.seq_len,
            num_heads=self.cross_modality_n_heads,
            dropout=self.dropout,
        )

        self.macro_beta_raw = nn.Parameter(torch.zeros(1, self.seq_len, 1))
        self.micro_beta_raw = nn.Parameter(torch.zeros(1, self.seq_len, 1))
        self.alpha_raw = nn.Parameter(torch.zeros(1, 1, self.seq_len))

        self.macro_gamma_raw = nn.Parameter(torch.zeros(1, 1, self.seq_len))
        self.micro_gamma_raw = nn.Parameter(torch.zeros(1, 1, self.seq_len))

        if self.use_time_feature:
            if self.time_feat_dim <= 0:
                raise ValueError(
                    f"use_time_feature=True but invalid time_feat_dim={self.time_feat_dim}"
                )

            self.num_time_proj = nn.Sequential(
                nn.Linear(self.time_feat_dim, max(self.time_feat_dim, self.num_enc_in)),
                nn.GELU(),
                nn.Linear(max(self.time_feat_dim, self.num_enc_in), self.num_enc_in),
            )
            self.text_time_proj = nn.Sequential(
                nn.Linear(self.time_feat_dim, max(self.time_feat_dim, self.text_encoder_out_dim)),
                nn.GELU(),
                nn.Linear(max(self.time_feat_dim, self.text_encoder_out_dim), self.text_encoder_out_dim),
            )

            self.num_concat_proj = nn.Sequential(
                nn.Linear(self.num_enc_in + self.time_feat_dim, max(self.num_enc_in + self.time_feat_dim, self.num_enc_in)),
                nn.GELU(),
                nn.Linear(max(self.num_enc_in + self.time_feat_dim, self.num_enc_in), self.num_enc_in),
            )
            self.text_concat_proj = nn.Sequential(
                nn.Linear(
                    self.text_encoder_out_dim + self.time_feat_dim,
                    max(self.text_encoder_out_dim + self.time_feat_dim, self.text_encoder_out_dim),
                ),
                nn.GELU(),
                nn.Linear(
                    max(self.text_encoder_out_dim + self.time_feat_dim, self.text_encoder_out_dim),
                    self.text_encoder_out_dim,
                ),
            )

    def _should_debug(self) -> bool:
        return self.debug_forward and (self._debug_forward_calls < self.debug_forward_max_calls)

    def _dbg_tensor(self, name: str, x: torch.Tensor | None):
        if not self._should_debug():
            return
        if x is None:
            print(f"[debug][forward] {name}=None")
            return
        with torch.no_grad():
            print(
                f"[debug][forward] {name}.shape={tuple(x.shape)} "
                f"norm={float(x.norm().item()):.6f} "
                f"absmean={float(x.abs().mean().item()):.6f} "
                f"mean={float(x.float().mean().item()):.6f} "
                f"std={float(x.float().std().item()):.6f}"
            )

    def _dbg_scalar(self, name: str, x):
        if not self._should_debug():
            return
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                print(f"[debug][forward] {name}={float(x.detach().item()):.6f}")
            else:
                print(f"[debug][forward] {name} tensor shape={tuple(x.shape)}")
        else:
            print(f"[debug][forward] {name}={x}")

    def _dbg_delta(self, name: str, a: torch.Tensor | None, b: torch.Tensor | None):
        if not self._should_debug():
            return
        if a is None or b is None:
            print(f"[debug][forward] {name}=None")
            return
        with torch.no_grad():
            diff = a - b
            print(
                f"[debug][forward] {name}.shape={tuple(diff.shape)} "
                f"norm={float(diff.norm().item()):.6f} "
                f"absmean={float(diff.abs().mean().item()):.6f}"
            )

    def _dbg_header(self):
        if not self._should_debug():
            return
        print("\n" + "=" * 120)
        print(f"[debug][forward] call_index={self._debug_forward_calls + 1}")
        print("=" * 120)

    def _dbg_footer(self):
        if not self._should_debug():
            return
        print("=" * 120 + "\n")

    def _infer_text_branch_mode(
        self,
        macro_text: torch.Tensor | None,
        micro_text: torch.Tensor | None,
    ) -> str:
        has_macro = macro_text is not None
        has_micro = micro_text is not None
        if has_macro and has_micro:
            return "dual"
        if has_macro or has_micro:
            return "single"
        return "none"

    def _fuse_time_feature(
        self,
        x: torch.Tensor | None,
        x_time: torch.Tensor | None,
        branch: str,
    ) -> torch.Tensor | None:
        """
        x:
          - branch='num'  -> [B, L, N]
          - branch='text' -> [B, L, D]
        x_time: [B, L, K]
        """
        if x is None or x_time is None or not self.use_time_feature:
            return x

        if branch == "num":
            if self.time_fusion == "add":
                time_emb = self.num_time_proj(x_time)
                return x + time_emb
            elif self.time_fusion == "concat":
                fused = torch.cat([x, x_time], dim=-1)
                return self.num_concat_proj(fused)

        elif branch == "text":
            if self.time_fusion == "add":
                time_emb = self.text_time_proj(x_time)
                return x + time_emb
            elif self.time_fusion == "concat":
                fused = torch.cat([x, x_time], dim=-1)
                return self.text_concat_proj(fused)

        raise ValueError(f"Unknown branch={branch}")

    def _apply_nlinear(self, x_fused: torch.Tensor) -> torch.Tensor:
        """
        x_fused: [B, N, seq_len]
        out:     [B, N, pred_len]
        """
        if self.individual:
            out = torch.zeros(
                x_fused.size(0),
                x_fused.size(1),
                self.pred_len,
                device=x_fused.device,
                dtype=x_fused.dtype,
            )
            for i in range(x_fused.size(1)):
                out[:, i, :] = self.Linear[i](x_fused[:, i, :])
        else:
            out = self.Linear(x_fused)
        return out

    def _project_text_to_num(
        self,
        text_emb: torch.Tensor | None,
        to_num_proj: nn.Module,
        branch_name: str,
    ) -> torch.Tensor | None:
        """Project a text branch to the numeric channel dimension.

        Args:
            text_emb: [B, seq_len, text_encoder_out_dim] or None.
            to_num_proj: Linear(text_encoder_out_dim -> num_enc_in).
            branch_name: Name used only for debug messages.

        Returns:
            Tensor [B, seq_len, num_enc_in] or None.
        """
        if text_emb is None:
            self._dbg_tensor(f"{branch_name}_early_proj", None)
            return None

        projected = to_num_proj(text_emb)
        self._dbg_tensor(f"{branch_name}_early_proj", projected)
        return projected

    def _forward_degrade_nlinear_text(
        self,
        x_num: torch.Tensor,
        macro_text: torch.Tensor | None,
        micro_text: torch.Tensor | None,
        x_time: torch.Tensor | None,
        text_branch_mode: str,
    ) -> torch.Tensor:
        """NLinear + Text Early Fusion ablation path.

        This path intentionally bypasses movement alignment, contrastive text
        enhancement, and cross-modal attention. Each available text branch is
        projected to the numeric channel dimension and added to the numeric
        input. The resulting fused sequence then goes through the same plain
        NLinear temporal head.
        """
        self._dbg_scalar("degrade", True)
        self._dbg_scalar("text_branch_mode", text_branch_mode)

        x_num_base = x_num
        x_num = self._fuse_time_feature(x_num, x_time, branch="num")
        macro_text = self._fuse_time_feature(macro_text, x_time, branch="text")
        micro_text = self._fuse_time_feature(micro_text, x_time, branch="text")

        self._dbg_tensor("x_num_after_time_fusion", x_num)
        self._dbg_tensor("macro_text_after_time_fusion", macro_text)
        self._dbg_tensor("micro_text_after_time_fusion", micro_text)

        text_signal = torch.zeros_like(x_num)

        macro_signal = self._project_text_to_num(
            macro_text,
            self.macro_to_num_proj,
            branch_name="macro",
        )
        if macro_signal is not None:
            text_signal = text_signal + macro_signal

        micro_signal = self._project_text_to_num(
            micro_text,
            self.micro_to_num_proj,
            branch_name="micro",
        )
        if micro_signal is not None:
            text_signal = text_signal + micro_signal

        self._dbg_tensor("early_text_signal", text_signal)

        # Keep the NLinear denormalization anchor on the original numeric
        # sequence scale. This makes the ablation comparable to the vanilla
        # NLinear centering rule while injecting projected text as an additive
        # residual feature before the temporal linear head.
        seq_last = x_num_base[:, -1:, :].detach()
        x_fused = x_num + text_signal
        x_norm = (x_fused - seq_last).permute(0, 2, 1).contiguous()

        self._dbg_tensor("seq_last", seq_last)
        self._dbg_tensor("x_fused_early", x_fused)
        self._dbg_tensor("x_norm_degrade", x_norm)

        out = self._apply_nlinear(x_norm)
        self._dbg_tensor("out", out)

        dec_out = out.permute(0, 2, 1).contiguous() + seq_last
        self._dbg_tensor("dec_out", dec_out)
        return dec_out

    def _build_text_repr_and_loss(
        self,
        text_emb: torch.Tensor | None,
        projector: nn.Module,
        to_num_proj: nn.Module,
        beta_raw: nn.Parameter,
        numeric_movement_label: torch.Tensor | None,
        branch_name: str = "text",
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Args:
            text_emb: [B, seq_len, D]
            numeric_movement_label: [B, seq_len-1]

        Returns:
            text_kv: [B, N, seq_len] or None
            movement_loss: scalar
        """
        if text_emb is None:
            zero = torch.zeros((), device=beta_raw.device)
            self._dbg_tensor(f"{branch_name}_input", None)
            self._dbg_tensor(f"{branch_name}_contrastive_text", None)
            self._dbg_tensor(f"{branch_name}_text_repr", None)
            self._dbg_tensor(f"{branch_name}_kv", None)
            self._dbg_scalar(f"{branch_name}_movement_loss", zero)
            return None, zero

        self._dbg_tensor(f"{branch_name}_input", text_emb)

        contrastive_text = projector(text_emb)
        self._dbg_tensor(f"{branch_name}_contrastive_text", contrastive_text)

        movement_loss = torch.zeros((), device=text_emb.device, dtype=text_emb.dtype)
        if self.use_movement_alignment and numeric_movement_label is not None:
            text_movement_feat, _ = _movement_feature_and_label_from_text(contrastive_text)
            self._dbg_tensor(f"{branch_name}_movement_feat", text_movement_feat)
            movement_loss = _supervised_infonce_from_labels(
                feat=text_movement_feat,
                labels=numeric_movement_label,
                tau=self.movement_tau,
            )

        beta = torch.sigmoid(beta_raw)
        self._dbg_tensor(f"{branch_name}_beta", beta)

        text_repr = text_emb + beta * contrastive_text
        self._dbg_tensor(f"{branch_name}_text_repr", text_repr)

        text_kv = to_num_proj(text_repr).permute(0, 2, 1).contiguous()
        self._dbg_tensor(f"{branch_name}_kv", text_kv)
        self._dbg_scalar(f"{branch_name}_movement_loss", movement_loss)

        return text_kv, movement_loss

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        macro_text: torch.Tensor | None = None,
        micro_text: torch.Tensor | None = None,
        x_time: torch.Tensor | None = None,
        y_tar: torch.Tensor | None = None,
    ):
        """
        Args:
            x_num:       [B, seq_len, N]
            macro_text:  [B, seq_len, text_dim] or None
            micro_text:  [B, seq_len, text_dim] or None
            x_time:      [B, seq_len, time_feat_dim]
            y_tar: unused

        Returns:
            dec_out or (dec_out, aux_loss)
            dec_out: [B, pred_len, N]
        """
        if x_num is None:
            raise ValueError("x_num cannot be None")

        B, L, N = x_num.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, but got x_num.shape[1]={L}")
        if N != self.num_enc_in:
            raise ValueError(f"Expected num_enc_in={self.num_enc_in}, but got x_num.shape[2]={N}")

        if macro_text is not None:
            if (
                macro_text.shape[0] != B
                or macro_text.shape[1] != self.seq_len
                or macro_text.shape[2] != self.text_encoder_out_dim
            ):
                raise ValueError(
                    f"macro_text must be [B, {self.seq_len}, {self.text_encoder_out_dim}], "
                    f"but got {tuple(macro_text.shape)}. "
                    "For English datasets, use a 768-D encoder such as "
                    "sentence-transformers/all-mpnet-base-v2."
                )

        if micro_text is not None:
            if (
                micro_text.shape[0] != B
                or micro_text.shape[1] != self.seq_len
                or micro_text.shape[2] != self.text_encoder_out_dim
            ):
                raise ValueError(
                    f"micro_text must be [B, {self.seq_len}, {self.text_encoder_out_dim}], "
                    f"but got {tuple(micro_text.shape)}. "
                    "For English datasets, use a 768-D encoder such as "
                    "sentence-transformers/all-mpnet-base-v2."
                )

        if x_time is not None:
            if x_time.shape[1] != self.seq_len:
                raise ValueError(
                    f"x_time must have seq_len={self.seq_len}, but got {tuple(x_time.shape)}"
                )
            if self.use_time_feature and x_time.shape[2] != self.time_feat_dim:
                raise ValueError(
                    f"x_time must have feat_dim={self.time_feat_dim}, but got {tuple(x_time.shape)}"
                )

        text_branch_mode = self._infer_text_branch_mode(macro_text, micro_text)

        self._dbg_header()
        self._dbg_scalar("text_branch_mode", text_branch_mode)
        self._dbg_scalar("degrade", self.degrade)
        self._dbg_tensor("x_num", x_num)
        self._dbg_tensor("macro_text_in", macro_text)
        self._dbg_tensor("micro_text_in", micro_text)
        self._dbg_tensor("x_time", x_time)

        if self.degrade:
            dec_out = self._forward_degrade_nlinear_text(
                x_num=x_num,
                macro_text=macro_text,
                micro_text=micro_text,
                x_time=x_time,
                text_branch_mode=text_branch_mode,
            )
            self._dbg_footer()
            self._debug_forward_calls += 1
            return dec_out

        x_num = self._fuse_time_feature(x_num, x_time, branch="num")
        macro_text = self._fuse_time_feature(macro_text, x_time, branch="text")
        micro_text = self._fuse_time_feature(micro_text, x_time, branch="text")

        self._dbg_tensor("x_num_after_time_fusion", x_num)
        self._dbg_tensor("macro_text_after_time_fusion", macro_text)
        self._dbg_tensor("micro_text_after_time_fusion", micro_text)

        seq_last = x_num[:, -1:, :].detach()
        x_norm = (x_num - seq_last).permute(0, 2, 1)

        self._dbg_tensor("seq_last", seq_last)
        self._dbg_tensor("x_norm", x_norm)

        numeric_movement_label = None
        if self.use_movement_alignment:
            numeric_movement_label = _movement_label_from_numeric(x_num)
            self._dbg_tensor("numeric_movement_label", numeric_movement_label.float())

        macro_kv, macro_move_loss = self._build_text_repr_and_loss(
            text_emb=macro_text,
            projector=self.macro_contrastive_projector,
            to_num_proj=self.macro_to_num_proj,
            beta_raw=self.macro_beta_raw,
            numeric_movement_label=numeric_movement_label,
            branch_name="macro",
        )

        micro_kv, micro_move_loss = self._build_text_repr_and_loss(
            text_emb=micro_text,
            projector=self.micro_contrastive_projector,
            to_num_proj=self.micro_to_num_proj,
            beta_raw=self.micro_beta_raw,
            numeric_movement_label=numeric_movement_label,
            branch_name="micro",
        )

        self._dbg_tensor("residual_init", x_norm)

        macro_delta = torch.zeros_like(x_norm)
        micro_delta = torch.zeros_like(x_norm)

        if self.use_cross_modality:
            if macro_kv is not None:
                macro_attended = self.macro_cross_attn(x_norm, macro_kv)
                self._dbg_tensor("macro_attended", macro_attended)
                self._dbg_delta("macro_attended_minus_xnorm", macro_attended, x_norm)
                macro_delta = macro_attended - x_norm
                self._dbg_tensor("macro_delta", macro_delta)

            if micro_kv is not None:
                micro_attended = self.micro_cross_attn(x_norm, micro_kv)
                self._dbg_tensor("micro_attended", micro_attended)
                self._dbg_delta("micro_attended_minus_xnorm", micro_attended, x_norm)
                micro_delta = micro_attended - x_norm
                self._dbg_tensor("micro_delta", micro_delta)

        macro_gamma = torch.sigmoid(self.macro_gamma_raw)
        micro_gamma = torch.sigmoid(self.micro_gamma_raw)

        self._dbg_tensor("macro_gamma", macro_gamma)
        self._dbg_tensor("micro_gamma", micro_gamma)

        residual_after_two = x_norm + macro_gamma * macro_delta + micro_gamma * micro_delta
        self._dbg_tensor("residual_after_two", residual_after_two)
        self._dbg_delta("residual_after_two_minus_xnorm", residual_after_two, x_norm)

        fused = residual_after_two
        self._dbg_tensor("fused", fused)
        self._dbg_delta("fused_minus_xnorm", fused, x_norm)

        out = self._apply_nlinear(fused)
        self._dbg_tensor("out", out)

        dec_out = out.permute(0, 2, 1).contiguous() + seq_last
        self._dbg_tensor("dec_out", dec_out)

        aux_loss = torch.zeros((), device=x_num.device, dtype=x_num.dtype)
        if self.use_movement_alignment:
            aux_loss = self.movement_loss_weight * (macro_move_loss + micro_move_loss)

        self._dbg_scalar("macro_move_loss", macro_move_loss)
        self._dbg_scalar("micro_move_loss", micro_move_loss)
        self._dbg_scalar("aux_loss", aux_loss)
        self._dbg_footer()

        self._debug_forward_calls += 1
        return (dec_out, aux_loss) if self.use_movement_alignment else dec_out
