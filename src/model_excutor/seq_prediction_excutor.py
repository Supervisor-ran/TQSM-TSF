# -*- coding: utf-8 -*-
from __future__ import annotations
import time
import re
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np

from src.model_util.text_embedding import TextEmbedding
from src.models import MMMA_NLinear
from src.model_excutor.executor_common import (
    set_random_seed, load_multimodal_monthly_data, to_float_tensor_2d,
    ArtifactManager, build_summary_text,
)
from src.model_excutor.sequence_helpers import (
    build_optimizer, split_time_series, prepare_sequence_dataloaders,
    denorm_preds, evaluate_from_tensors,
)
from src.dataloader.temporal_dataloader import build_time_stamp_features

_EXTERNAL_REVIEW_SEP_RE = re.compile(r"\s*\n\s*--- REVIEW ---\s*\n\s*", flags=re.MULTILINE)

def _split_external_review_steps(text_steps):
    if text_steps is None: return None
    out = []
    for step in text_steps:
        if step is None:
            out.append([])
            continue
        raw_items = list(step) if isinstance(step, (list, tuple)) else [step]
        pieces = []
        for raw in raw_items:
            text = str(raw or "").strip()
            if text:
                pieces.extend(p.strip() for p in _EXTERNAL_REVIEW_SEP_RE.split(text) if p.strip())
        out.append(pieces)
    return out

class SequenceExecutor:
    def __init__(self, args: dict):
        if str(args.get("model_name", "MMMA_NLinear")) != "MMMA_NLinear":
            raise ValueError("Only the proposed MMMA_NLinear model is shipped.")
        self.args = args
        self.model_name = "MMMA_NLinear"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.random_seed = args.get("random_seed", args.get("seed", 42))
        set_random_seed(self.random_seed)
        self.text_data_dir = args.get("text_data_dir", "")
        self.text_variant = args.get("text_variant", "default")
        self.text_mode = args.get("text_mode", "default")
        self.external_review_level_encoding = False
        self.external_review_cached_embeddings = False
        self.external_review_embedding_cache_path = None
        self.save_artifacts = bool(args.get("save_artifacts", True))
        self.early_stopping_patience = int(args.get("early_stopping_patience", 5))
        self.selection_metric = args.get("selection_metric", "mse")
        self.training_log = []
        self.summary_dict = {}
        self.cost_dict = {}
        self.best_epoch = None
        self.best_val_metrics = None
        self.best_test_metrics = None
        self.best_ckpt_path = None
        self.iter_interval = args.get("iter_interval", 10)
        self.batch_size = args.get("batch_size", 13)
        self.window_size = args.get("seq_len", 6)
        self.horizon = args.get("pred_len", 1)
        self.val_ratio = args.get("val_ratio", 0.1)
        self.test_ratio = args.get("test_ratio", 0.1)
        self.optimizer_name = args.get("optimizer_name", "adamw")
        self.lr = args.get("lr", 1e-1)
        self.weight_decay = args.get("weight_decay", 1e-4)
        self.optimizer_kwargs = {}
        self.epochs = args.get("train_epochs", 10)
        self._scaler_mean = self._scaler_std = None
        self._y_mean = self._y_std = None
        self.X_txt = None
        self.debug_text = bool(args.get("debug_text", False))
        self._debug_batch_printed = False

        x_num, self.macro_text, self.micro_text, self.x_date, _ = load_multimodal_monthly_data(args)
        self.X_num = to_float_tensor_2d(x_num)
        self.y = to_float_tensor_2d(x_num)
        loc = str(args.get("x_num_loc", "")).strip().lower()
        review_opt_in = bool(args.get("use_external_review_level_encoding", False))
        if review_opt_in and loc not in ("amsterdam", "new_orleans"):
            raise ValueError("Review-level encoding is only supported for Amsterdam/New Orleans")
        self.external_review_level_encoding = review_opt_in and loc in ("amsterdam", "new_orleans")
        if self.external_review_level_encoding:
            self.macro_text = _split_external_review_steps(self.macro_text)
            if self.micro_text is not None:
                self.micro_text = _split_external_review_steps(self.micro_text)

        if self.x_date is not None:
            features = build_time_stamp_features(
                self.x_date, timeenc=int(self.args.get("timeenc", 1)), freq=self.args.get("freq", "M"))
            self.args["time_feat_dim"] = int(features.shape[1])

        review_cache = self.args.get("external_review_embedding_cache")
        if review_cache:
            if not self.external_review_level_encoding or loc not in ("amsterdam", "new_orleans"):
                raise ValueError("Review embedding cache requires review-level encoding on external cities")
            with np.load(str(review_cache), allow_pickle=False) as cache:
                features = np.asarray(cache["features"], dtype=np.float32)
                dates = [str(x) for x in cache["dates"].tolist()]
            expected_dim = int(args.get("text_encoder_out_dim", 768))
            if features.ndim != 2 or features.shape[1] != expected_dim:
                raise ValueError(f"Invalid embedding cache shape {features.shape}; expected dimension {expected_dim}")
            if len(features) != len(self.X_num):
                raise ValueError("Review embedding cache length mismatches numeric series")
            src_months = [str(x)[:7] for x in self.x_date] if self.x_date is not None else []
            cached_months = [str(x)[:7] for x in dates]
            if src_months and src_months != cached_months:
                raise ValueError("Review cache month sequence mismatches source series")
            self.macro_text = torch.tensor(features, dtype=torch.float32)
            self.micro_text = None
            self.external_review_cached_embeddings = True
            self.external_review_embedding_cache_path = str(review_cache)

        self.model = MMMA_NLinear.Model(self.args).to(self.device)
        self.text_encoder = None if self.external_review_cached_embeddings else self._init_text_embedding(loc)
        if self.text_encoder is not None:
            self.text_encoder.to(self.device).eval()
        self._prepare_data()


    def _count_parameters(self):
        n_parameters = sum(p.numel() for p in self.model.parameters())
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return int(n_parameters), int(trainable_parameters)

    def _get_model_file_size_mb(self, path: Path | None) -> float | None:
        if path is None or not path.exists():
            return None
        return float(path.stat().st_size / (1024 * 1024))

    def _get_peak_gpu_memory_mb(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        try:
            return float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))
        except Exception:
            return None

    def _build_cost_dict(
        self,
        fit_seconds: float,
        eval_seconds: float,
        model_path: Path | None = None,
    ) -> dict:
        n_parameters, trainable_parameters = self._count_parameters()
        epochs_trained = len(self.training_log)
        avg_epoch_seconds = None
        if epochs_trained > 0:
            avg_epoch_seconds = float(fit_seconds / epochs_trained)

        return {
            "fit_seconds": float(fit_seconds),
            "eval_seconds": float(eval_seconds),
            "total_seconds": float(fit_seconds + eval_seconds),
            "batch_size": int(self.batch_size),
            "epochs_requested": int(self.epochs),
            "epochs_trained": int(epochs_trained),
            "best_epoch": None if self.best_epoch is None else int(self.best_epoch),
            "avg_epoch_seconds": avg_epoch_seconds,
            "device": str(self.device),
            "train_size": int(len(self.index_map["train"])),
            "val_size": int(len(self.index_map["val"])),
            "test_size": int(len(self.index_map["test"])),
            "n_parameters": n_parameters,
            "trainable_parameters": trainable_parameters,
            "peak_gpu_memory_mb": self._get_peak_gpu_memory_mb(),
            "model_file_size_mb": self._get_model_file_size_mb(model_path),
        }

    def _make_artifact_manager(self, settings: str):
        return ArtifactManager(
            sub_root="deep",
            model_name=self.model_name,
            city=self.args.get("x_num_loc", "unknown"),
            text_variant=self.text_variant,
            text_mode=self.text_mode,
            settings=settings,
        )

    def _save_common_artifacts(
        self,
        settings: str,
        fit_seconds: float,
        eval_seconds: float,
        predictions: dict | None = None,
    ):
        if not self.save_artifacts:
            return

        am = self._make_artifact_manager(settings)
        am.save_json(deepcopy(self.args), "args.json")

        summary = {
            "settings": settings,
            "model_name": self.model_name,
            "x_num_loc": self.args.get("x_num_loc"),
            "text_variant": self.text_variant,
            "text_mode": self.text_mode,
            "best_epoch": self.best_epoch,
            "best_config": None,
            "best_val_metrics": self.best_val_metrics,
            "best_test_metrics": self.best_test_metrics,
            "train_size": len(self.index_map["train"]),
            "val_size": len(self.index_map["val"]),
            "test_size": len(self.index_map["test"]),
        }
        self.summary_dict = summary
        am.save_json(summary, "summary.json")

        model_path = am.path("best_model.pt") if self.best_ckpt_path is not None else None
        cost = self._build_cost_dict(
            fit_seconds=fit_seconds,
            eval_seconds=eval_seconds,
            model_path=model_path,
        )
        self.cost_dict = cost
        am.save_json(cost, "cost.json")
        am.save_json(self.training_log, "training_log.json")

        if predictions is not None:
            am.save_csv_dict(predictions, "predictions.csv")

        am.save_text(build_summary_text(summary, cost), "summary.txt")

    def _init_text_embedding(self, loc_str: str):
        loc = (loc_str or "").strip().lower()
        if loc in ("kamakura", "yokohama", "fujisawa"):
            model_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        elif loc in ("new_orleans", "amsterdam"):
            # English monthly datasets should keep the same 768-D text interface
            # as the original MMMA_NLinear setting, so no padding/truncation
            # adapter is needed inside the model.
            model_name = "sentence-transformers/all-mpnet-base-v2"
        else:
            model_name = "sentence-transformers/all-mpnet-base-v2"

        print(f"🧠 使用文本编码模型: {model_name}")
        if self.external_review_level_encoding:
            print("[SequenceExecutor][external-review-level-mean-v1] review embedding pool=mean")
        return TextEmbedding(
            model_name=model_name,
            use_attention=False,
            num_heads=self.args.get("llm_n_heads"),
            pooling_mode=("mean" if self.external_review_level_encoding else self.args.get("pooling_mode")),
        )

    def _ensure_model_initialized(self):
        with torch.no_grad():
            if self.X_num is None:
                raise RuntimeError("X_num is None")
            B, T, N = 1, self.window_size, self.X_num.shape[-1]
            dummy_x = torch.zeros(B, T, N, device=self.device, dtype=torch.float32)
            dummy_time = None
            if bool(self.args.get("use_time_feature", False)):
                time_dim = int(self.args.get("time_feat_dim", 0))
                if time_dim > 0:
                    dummy_time = torch.zeros(B,T,time_dim,device=self.device,dtype=torch.float32)
            _ = self.model(x_num=dummy_x, macro_text=None, micro_text=None, x_time=dummy_time, y_tar=None)


    def _prepare_data(self):
        lengths = [len(self.X_num), len(self.y)]
        if self.macro_text is not None: lengths.append(len(self.macro_text))
        if self.micro_text is not None: lengths.append(len(self.micro_text))
        base_len = min(lengths)
        self.X_num = self.X_num[:base_len]
        self.y = self.y[:base_len]
        self.macro_text = self.macro_text[:base_len] if self.macro_text is not None else None
        self.micro_text = self.micro_text[:base_len] if self.micro_text is not None else None
        self.x_date = self.x_date[:base_len] if self.x_date is not None else None
        total_len = base_len - self.window_size - self.horizon + 1
        if total_len <= 0:
            raise ValueError(f"Sequence too short: {base_len} observations for L={self.window_size}, H={self.horizon}")
        self.index_map = split_time_series(total_len, self.val_ratio, self.test_ratio)
        train_last_end = (self.index_map["train"][-1] + self.window_size) if self.index_map["train"] else max(self.window_size,1)
        tr_slice = slice(0,train_last_end)
        xnum_tr = self.X_num[tr_slice]
        self._scaler_mean = xnum_tr.mean(dim=0)
        self._scaler_std = xnum_tr.std(dim=0).clamp_min(1e-6)
        self.X_num = (self.X_num - self._scaler_mean) / self._scaler_std
        y_tr = self.y[tr_slice]
        self._y_mean = y_tr.mean(dim=0)
        self._y_std = y_tr.std(dim=0).clamp_min(1e-6)
        self.y = (self.y - self._y_mean) / self._y_std
        (self.index_map, self.train_loader, self.val_loader, self.test_loader) = prepare_sequence_dataloaders(
            model_name=self.model_name, X_num=self.X_num, macro_text=self.macro_text,
            micro_text=self.micro_text, X_txt=self.X_txt, y=self.y, x_date=self.x_date,
            batch_size=self.batch_size, window_size=self.window_size, horizon=self.horizon,
            val_ratio=self.val_ratio, test_ratio=self.test_ratio,
            shuffle=self.args.get("shuffle", False), freq=self.args.get("freq", "M"),
            timeenc=int(self.args.get("timeenc", 1)),
        )


    def _forward_batch(self, batch):
        if not isinstance(batch, dict):
            raise TypeError(f"MMMA_NLinear expected dict batch, got {type(batch)}")
        x_num = batch.get("X_num")
        macro_batch = batch.get("X_text_1")
        micro_batch = batch.get("X_text_2")
        x_time = batch.get("X_time")
        targets = batch["y"].to(self.device)
        if self.external_review_cached_embeddings:
            macro = macro_batch.to(self.device) if macro_batch is not None else None
            micro = micro_batch.to(self.device) if micro_batch is not None else None
        else:
            macro = self.text_encoder(macro_batch, device=self.device) if macro_batch is not None else None
            micro = self.text_encoder(micro_batch, device=self.device) if micro_batch is not None else None
        x_num = x_num.to(self.device) if x_num is not None else None
        x_time = x_time.to(self.device) if x_time is not None else None
        result = self.model(x_num=x_num, macro_text=macro, micro_text=micro, x_time=x_time, y_tar=targets)
        prediction, aux_loss = result if isinstance(result, tuple) else (result, None)
        return prediction, targets, aux_loss


    def _metric_value_for_selection(self, metrics: dict):
        if metrics is None or not metrics.get("had_batch", False):
            return float("inf")
        return float(metrics.get(self.selection_metric, metrics["mse"]))

    def _save_checkpoint(self, settings: str, epoch: int):
        if not self.save_artifacts:
            return None
        am = self._make_artifact_manager(settings)
        ckpt_path = am.path("best_model.pt")
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "args": deepcopy(self.args),
            "model_name": self.model_name,
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def _load_best_checkpoint(self):
        if self.best_ckpt_path is None or not self.best_ckpt_path.exists():
            return
        ckpt = torch.load(self.best_ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

    def train(self, settings=None):
        settings = settings or "deep_run"
        print("🚀 Start Training...")
        self._ensure_model_initialized()

        optimizer = build_optimizer(
            self.model.parameters(),
            optimizer_name=self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.optimizer_kwargs,
        )

        fit_start = time.perf_counter()
        best_score = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        no_improve = 0

        for epoch in range(int(self.epochs)):
            self.model.train()
            total_loss = 0.0
            total_aux_loss = 0.0
            aux_count = 0
            first_iter_loss = None
            first_iter_aux_loss = None

            for batch in self.train_loader:
                pred, yb, aux_loss = self._forward_batch(batch)
                mse_loss = torch.mean((pred.float() - yb.float()) ** 2)
                loss = mse_loss if aux_loss is None else (mse_loss + aux_loss)

                if first_iter_loss is None:
                    first_iter_loss = float(loss.item())
                if aux_loss is not None and first_iter_aux_loss is None:
                    first_iter_aux_loss = float(aux_loss.detach().item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                if aux_loss is not None:
                    total_aux_loss += float(aux_loss.detach().item())
                    aux_count += 1

            val_metrics = self.evaluate(self.val_loader, quiet=True)
            test_metrics = self.evaluate(self.test_loader, quiet=True)

            train_loss_epoch = total_loss / max(1, len(self.train_loader))
            train_aux_loss_epoch = (total_aux_loss / aux_count) if aux_count > 0 else None
            current_score = self._metric_value_for_selection(val_metrics)

            self.training_log.append({
                "epoch": epoch + 1,
                "train_loss": train_loss_epoch,
                "train_aux_loss": train_aux_loss_epoch,
                "first_iter_loss": first_iter_loss,
                "first_iter_aux_loss": first_iter_aux_loss,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            })

            improved = False
            if val_metrics["had_batch"]:
                if best_score is None or current_score < best_score:
                    best_score = current_score
                    self.best_epoch = epoch + 1
                    self.best_val_metrics = val_metrics
                    self.best_test_metrics = test_metrics
                    self.best_ckpt_path = self._save_checkpoint(settings, epoch + 1)
                    improved = True
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                self.best_epoch = epoch + 1
                self.best_val_metrics = val_metrics
                self.best_test_metrics = test_metrics
                self.best_ckpt_path = self._save_checkpoint(settings, epoch + 1)

            def _fmt_metrics(m, tag):
                if m is None or not m.get("had_batch", False):
                    return f"{tag}=— mse=— mae=— rmse=— mape=—"
                return (
                    f"{tag}={m['loss']:.7f} mse={m['mse']:.7f} "
                    f"mae={m['mae']:.4f} rmse={m['rmse']:.4f} mape={m['mape'] * 100:.5f}%"
                )

            aux_msg = ""
            if train_aux_loss_epoch is not None:
                aux_msg = f" aux={train_aux_loss_epoch:.7f}"

            print(
                f"[E{epoch + 1}/{self.epochs}] "
                f"train_loss={train_loss_epoch:.7f}{aux_msg} "
                f"first_iter_loss={first_iter_loss:.7f} | "
                f"{_fmt_metrics(val_metrics, 'val')} | "
                f"{_fmt_metrics(test_metrics, 'test')} | "
                f"improved={improved}"
            )

            if val_metrics["had_batch"] and no_improve >= self.early_stopping_patience:
                print(f"[EarlyStopping] stop at epoch {epoch + 1}")
                break

        fit_seconds = time.perf_counter() - fit_start

        self._load_best_checkpoint()

        eval_start = time.perf_counter()
        final_val = self.evaluate(self.val_loader, quiet=True)
        final_test = self.evaluate(self.test_loader, quiet=True)
        predictions = self.predict_loader(self.test_loader)
        eval_seconds = time.perf_counter() - eval_start

        self.best_val_metrics = final_val
        self.best_test_metrics = final_test

        print("\n" + "=" * 100)
        print("[Final Best Checkpoint Result]")
        print(f"best_epoch      = {self.best_epoch}")
        print(f"best_ckpt_path  = {self.best_ckpt_path}")
        print(_fmt_metrics(self.best_val_metrics, "best_val"))
        print(_fmt_metrics(self.best_test_metrics, "best_test"))
        print("=" * 100 + "\n")

        self._save_common_artifacts(settings, fit_seconds, eval_seconds, predictions)

    def predict_loader(self, data_loader):
        if data_loader is None:
            return None

        self.model.eval()
        all_preds, all_trues = [], []

        with torch.no_grad():
            for batch in data_loader:
                pred, yb, _ = self._forward_batch(batch)
                all_preds.append(pred.detach().cpu())
                all_trues.append(yb.detach().cpu())

        if not all_preds:
            return None

        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_trues, dim=0)
        preds_denorm, trues_denorm = denorm_preds(preds, trues, self._y_mean, self._y_std)

        return {
            "y_true": trues_denorm.numpy().reshape(-1).tolist(),
            "y_pred": preds_denorm.numpy().reshape(-1).tolist(),
        }

    def evaluate(self, data_loader, quiet: bool = True):
        if data_loader is None:
            return {"had_batch": False, "loss": 0.0, "mse": None, "mae": None, "rmse": None, "mape": None}

        self.model.eval()
        total_loss = 0.0
        all_preds, all_trues = [], []
        had_batch = False

        with torch.no_grad():
            for batch in data_loader:
                had_batch = True
                pred, yb, _ = self._forward_batch(batch)

                loss = torch.mean((pred.float() - yb.float()) ** 2)
                total_loss += float(loss.item())

                all_preds.append(pred.detach().cpu())
                all_trues.append(yb.detach().cpu())

        if not had_batch or not all_preds:
            return {"had_batch": False, "loss": 0.0, "mse": None, "mae": None, "rmse": None, "mape": None}

        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_trues, dim=0)

        preds_denorm, trues_denorm = denorm_preds(preds, trues, self._y_mean, self._y_std)
        return evaluate_from_tensors(
            preds_denorm=preds_denorm,
            trues_denorm=trues_denorm,
            avg_loss=total_loss / max(1, len(data_loader)),
            had_batch=True,
        )
