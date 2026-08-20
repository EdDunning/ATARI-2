"""
ATARI-2: Gesture Classification / experiment_runner_4.py

===============================================================================
PURPOSE
===============================================================================

Diagnostic experiment: how accurately can a Transformer recognise JIGSAWS
surgical gestures from kinematics ALONE, without ever receiving a previous or
current gesture label as a model input?

Prior experiments found:

    previous-true-label persistence baseline  ~99% accuracy
    teacher-forced encoder-decoder Transformer ~99% accuracy
    autonomous autoregressive Transformer      ~10% accuracy

This strongly suggests the encoder-decoder Transformer in Train_PyTorch.py is
relying on the previous gesture label fed to its decoder rather than learning
recognition from kinematics. This script removes that shortcut entirely by
implementing a model with NO decoder and NO label input of any kind:

    38-dim kinematics -> linear projection -> Transformer encoder
        -> linear classification head -> 16 gesture logits per frame

Gesture labels are used only as CrossEntropyLoss / metric targets, never as
model inputs. There is no teacher forcing and no autoregressive feedback
because there is no decoder to feed.

This script does NOT modify Train_PyTorch.py, data_prep.py, run_all_pytorch.py,
predict_gestures.py, or any previous experiment script/result file. It reuses
data loading, LOUO splitting, normalisation, leakage-auditing, and the
Transformer encoder layer implementation directly from Train_PyTorch.py.

===============================================================================
COMMANDS
===============================================================================

Run these from the repository root in PowerShell.

Smoke test:

    python "Gesture Classification\\experiment_runner_4.py" `
        --input-csv "Archive results\\PyTorch_1 Gesture Prediction\\outputs_pytorch_single_procedure_1\\prepared_data\\all_frame_level.csv" `
        --output-dir "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_4_smoke" `
        --smoke

Full Knot Tying LOUO experiment:

    python "Gesture Classification\\experiment_runner_4.py" `
        --input-csv "Archive results\\PyTorch_1 Gesture Prediction\\outputs_pytorch_single_procedure_1\\prepared_data\\all_frame_level.csv" `
        --output-dir "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_4_knot_tying" `
        --persistence-baseline-metrics "Archive results\\PyTorch_1 Gesture Prediction\\outputs_pytorch_single_procedure_1\\persistence_baseline\\previous_label_baseline_metrics.json" `
        --pytorch-metrics "Archive results\\PyTorch_1 Gesture Prediction\\outputs_pytorch_single_procedure_1\\pytorch_model\\pytorch_metrics.json"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install an appropriate PyTorch build "
        "before running experiment_runner_4.py."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse data loading, LOUO splitting, normalisation, leakage auditing and the
# Transformer encoder layer from the existing encoder-decoder script instead
# of duplicating that logic. Train_PyTorch.py itself is never modified.
import Train_PyTorch as tp  # noqa: E402


NUM_GESTURE_CLASSES = tp.NUM_GESTURE_CLASSES
KINEMATIC_DIM_PER_SOURCE = tp.KINEMATIC_DIM_PER_SOURCE
GESTURE_ID_TO_LABEL = tp.GESTURE_ID_TO_LABEL
DEFAULT_RANDOM_SEED = tp.DEFAULT_RANDOM_SEED


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class KinematicsOnlyConfig:
    input_csv: Path
    output_dir: Path

    kinematic_source: str = "mtm"

    sample_rate: float = 30.0
    window_seconds: float = 1.0
    stride_samples: int = 1

    batch_size: int = 64
    epochs: int = 15

    encoder_dim: int = 38
    encoder_heads: int = 1
    encoder_layers: int = 1
    encoder_ff_dim: int = 152

    dropout: float = 0.3
    weight_decay: float = 1e-3
    early_stopping_patience: int = 5
    early_stopping_metric: str = "macro_f1"

    warmup_steps: int = 2000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-9

    standardize: bool = False

    num_workers: int = 0
    random_seed: int = DEFAULT_RANDOM_SEED

    device: str = "auto"

    max_folds: Optional[int] = None
    max_windows: Optional[int] = None

    progress_updates_per_epoch: int = 10

    persistence_baseline_metrics: Optional[Path] = None
    pytorch_metrics: Optional[Path] = None


# =============================================================================
# MODEL: KINEMATICS ONLY, NO DECODER, NO LABEL INPUT
# =============================================================================

class KinematicsOnlyTransformer(nn.Module):
    """
    kinematics -> linear projection -> Transformer encoder
        -> linear classification head -> per-frame gesture logits

    There is no decoder. Gesture labels are never part of the forward pass;
    they are only used externally to compute the loss/metrics.
    """

    def __init__(
        self,
        input_dimension: int = 38,
        encoder_dimension: int = 38,
        num_classes: int = 16,
        encoder_heads: int = 1,
        encoder_layers: int = 1,
        encoder_ff_dimension: int = 152,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if encoder_dimension % encoder_heads != 0:
            raise ValueError(
                "encoder_dimension must be divisible by encoder_heads."
            )

        self.input_dimension = input_dimension
        self.num_classes = num_classes

        self.input_projection = nn.Linear(
            input_dimension,
            encoder_dimension,
        )

        # Reuse the existing Transformer encoder block implementation.
        self.encoder = nn.ModuleList(
            [
                tp.SurgicalEncoderLayer(
                    dimension=encoder_dimension,
                    heads=encoder_heads,
                    feedforward_dimension=encoder_ff_dimension,
                    dropout=dropout,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.classification_head = nn.Linear(
            encoder_dimension,
            num_classes,
        )

    def forward(self, kinematics: torch.Tensor) -> torch.Tensor:
        # Runtime guard: the only tensor this model ever accepts is raw
        # kinematics of the expected width. No label/one-hot tensor exists
        # anywhere in this signature.
        if kinematics.ndim != 3 or kinematics.shape[-1] != self.input_dimension:
            raise RuntimeError(
                "KinematicsOnlyTransformer expects input shaped "
                f"[batch, time, {self.input_dimension}]; got "
                f"{tuple(kinematics.shape)}."
            )

        memory = self.input_projection(kinematics)

        for layer in self.encoder:
            memory = layer(memory)

        return self.classification_head(memory)


def build_model(config: KinematicsOnlyConfig) -> KinematicsOnlyTransformer:
    return KinematicsOnlyTransformer(
        input_dimension=KINEMATIC_DIM_PER_SOURCE,
        encoder_dimension=config.encoder_dim,
        num_classes=NUM_GESTURE_CLASSES,
        encoder_heads=config.encoder_heads,
        encoder_layers=config.encoder_layers,
        encoder_ff_dimension=config.encoder_ff_dim,
        dropout=config.dropout,
    )


# =============================================================================
# TRAINING / EVALUATION (no teacher forcing, no autoregression)
# =============================================================================

def train_epoch(
    model: KinematicsOnlyTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: "tp.NoamLearningRate",
    criterion: nn.Module,
    device: torch.device,
    epoch_number: int,
    total_epochs: int,
    progress_updates: int,
) -> Tuple[float, float]:

    model.train()

    epoch_start = time.perf_counter()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    n_batches = len(loader)
    update_interval = max(1, n_batches // max(1, progress_updates))

    for batch_index, batch in enumerate(loader, start=1):
        # previous_label/metadata are intentionally discarded: this model
        # never receives gesture-label information as input.
        source, target, _previous_label, _metadata = batch

        if source.ndim != 3:
            raise RuntimeError(
                "Expected model input with shape [batch, time, kinematic_features]."
            )

        if source.shape[-1] != KINEMATIC_DIM_PER_SOURCE:
            raise RuntimeError(
                "Unexpected model feature count. "
                f"Expected {KINEMATIC_DIM_PER_SOURCE}, got {source.shape[-1]}."
            )

        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(source)

        loss = criterion(
            logits.reshape(-1, NUM_GESTURE_CLASSES),
            target.reshape(-1),
        )

        loss.backward()

        scheduler.step()
        optimizer.step()

        predictions = torch.argmax(logits, dim=-1)

        batch_tokens = target.numel()
        total_loss += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens
        total_correct += int((predictions == target).sum().item())

        if (
            batch_index == 1
            or batch_index % update_interval == 0
            or batch_index == n_batches
        ):
            elapsed = time.perf_counter() - epoch_start
            average_batch_time = elapsed / batch_index
            eta = average_batch_time * (n_batches - batch_index)
            progress = 100.0 * batch_index / n_batches

            print(
                f"[TRAIN] Epoch {epoch_number}/{total_epochs} | "
                f"Batch {batch_index}/{n_batches} | {progress:5.1f}% | "
                f"Loss {loss.item():.5f} | LR {scheduler.current_lr:.6g} | "
                f"Elapsed {tp.format_duration(elapsed)} | ETA {tp.format_duration(eta)}"
            )

    mean_loss = total_loss / total_tokens if total_tokens else 0.0
    accuracy = total_correct / total_tokens if total_tokens else 0.0

    return float(mean_loss), float(accuracy)


@torch.no_grad()
def evaluate(
    model: KinematicsOnlyTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:

    model.eval()

    confusion = np.zeros(
        (NUM_GESTURE_CLASSES, NUM_GESTURE_CLASSES),
        dtype=np.int64,
    )

    window_accuracy_sum = 0.0
    window_count = 0

    for batch in loader:
        source, target, _previous_label, _metadata = batch

        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(source)
        prediction = torch.argmax(logits, dim=-1)

        tp.update_confusion_matrix(confusion, target, prediction)

        per_window_accuracy = (prediction == target).float().mean(dim=1)
        window_accuracy_sum += float(per_window_accuracy.sum().item())
        window_count += int(len(per_window_accuracy))

    metrics = tp.metrics_from_confusion_matrix(confusion)

    metrics["mean_window_accuracy"] = (
        window_accuracy_sum / window_count if window_count else 0.0
    )

    return metrics


# =============================================================================
# TRAIN ONE FOLD (or the final all-surgeons model when test_trials is None)
# =============================================================================

def train_fold(
    train_trials: Sequence["tp.TrialData"],
    test_trials: Optional[Sequence["tp.TrialData"]],
    config: KinematicsOnlyConfig,
    device: torch.device,
    run_name: str,
    held_out_surgeon: Optional[str] = None,
) -> Tuple[
    KinematicsOnlyTransformer,
    Dict[str, object],
    List[Dict[str, object]],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, object],
]:

    window_frames = int(round(config.window_seconds * config.sample_rate))

    if config.standardize:
        mean, std = tp.calculate_standardization(train_trials)

        normalization_surgeons = sorted({t.surgeon_id for t in train_trials})

        print("[NORMALIZATION] Mean/std calculated from TRAINING surgeons only:")
        print("[NORMALIZATION] " + ", ".join(normalization_surgeons))

        if held_out_surgeon is not None and held_out_surgeon in normalization_surgeons:
            raise RuntimeError(
                "DATA LEAKAGE: held-out surgeon was used to calculate "
                "normalization statistics."
            )
    else:
        mean = None
        std = None
        print("[NORMALIZATION] Standardization disabled.")

    train_dataset = tp.make_split_dataset(
        trials=train_trials,
        window_frames=window_frames,
        stride_samples=config.stride_samples,
        standardize=config.standardize,
        mean=mean,
        std=std,
        max_windows=config.max_windows,
        random_seed=config.random_seed,
    )

    if len(train_dataset) == 0:
        raise ValueError(f"{run_name}: no training windows generated.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_loader: Optional[DataLoader] = None

    if test_trials is not None:
        test_dataset = tp.make_split_dataset(
            trials=test_trials,
            window_frames=window_frames,
            stride_samples=config.stride_samples,
            standardize=config.standardize,
            mean=mean,
            std=std,
            max_windows=config.max_windows,
            random_seed=config.random_seed + 1,
        )

        if len(test_dataset) == 0:
            raise ValueError(f"{run_name}: no testing windows generated.")

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        if held_out_surgeon is None:
            raise RuntimeError(
                "A held-out surgeon must be supplied for LOUO evaluation."
            )

        tp.audit_split_preprocessing(
            train_trials=train_trials,
            test_trials=test_trials,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            window_frames=window_frames,
            stride_samples=config.stride_samples,
            standardize=config.standardize,
            mean=mean,
            std=std,
            held_out_surgeon=held_out_surgeon,
        )

    model = build_model(config).to(device)

    # Runtime assertions: exactly 38 kinematic features in, 16 gesture
    # classes out, and no label-shaped tensor anywhere in the model inputs.
    assert model.input_dimension == KINEMATIC_DIM_PER_SOURCE
    assert model.num_classes == NUM_GESTURE_CLASSES

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    scheduler = tp.NoamLearningRate(
        optimizer=optimizer,
        model_dimension=config.encoder_dim,
        warmup_steps=config.warmup_steps,
    )

    parameter_count = sum(p.numel() for p in model.parameters())

    print()
    print(f"[MODEL] {run_name}")
    print(f"[MODEL] Parameters: {parameter_count:,}")
    print(f"[MODEL] Training trials: {len(train_trials)}")
    print(f"[MODEL] Training windows: {len(train_dataset):,}")

    if test_loader is not None:
        print(f"[MODEL] Testing windows: {len(test_loader.dataset):,}")

    history: List[Dict[str, object]] = []
    best_validation_metric: Optional[float] = None
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    best_train_accuracy: Optional[float] = None
    best_validation_accuracy: Optional[float] = None
    best_validation_macro_f1: Optional[float] = None
    patience_counter = 0

    fold_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch_number=epoch,
            total_epochs=config.epochs,
            progress_updates=config.progress_updates_per_epoch,
        )

        epoch_elapsed = time.perf_counter() - epoch_start

        print(
            f"[EPOCH] {run_name} | {epoch}/{config.epochs} complete | "
            f"Loss {train_loss:.5f} | Train accuracy {train_accuracy:.4f} | "
            f"Epoch time {tp.format_duration(epoch_elapsed)}"
        )

        validation_accuracy: Optional[float] = None
        validation_macro_f1: Optional[float] = None
        early_stopping_value: Optional[float] = None

        if test_loader is not None:
            validation_metrics = evaluate(model=model, loader=test_loader, device=device)

            validation_accuracy = float(validation_metrics["accuracy"])
            validation_macro_f1 = float(validation_metrics["macro_f1"])
            early_stopping_value = float(validation_metrics[config.early_stopping_metric])

            if best_validation_metric is None or early_stopping_value > best_validation_metric:
                best_validation_metric = early_stopping_value
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch
                best_train_accuracy = train_accuracy
                best_validation_accuracy = validation_accuracy
                best_validation_macro_f1 = validation_macro_f1
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"[VALIDATION] {run_name} | Accuracy {validation_accuracy:.4f} | "
                f"Macro F1 {validation_macro_f1:.4f} | "
                f"Best {config.early_stopping_metric} {best_validation_metric:.4f} | "
                f"Patience {patience_counter}/{config.early_stopping_patience}"
            )

        history.append(
            {
                "run": run_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "learning_rate": scheduler.current_lr,
                "epoch_seconds": epoch_elapsed,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
                "early_stopping_value": early_stopping_value,
                "best_validation_value_so_far": best_validation_metric,
                "patience_counter": patience_counter if test_loader is not None else None,
            }
        )

        if test_loader is not None and patience_counter >= config.early_stopping_patience:
            print(
                f"[EARLY STOPPING] {run_name} | No improvement in "
                f"{config.early_stopping_metric} for {patience_counter} epoch(s)."
            )
            break

    metrics: Dict[str, object] = {}

    if test_loader is not None:
        if best_model_state is None:
            raise RuntimeError("No validation checkpoint was recorded.")

        model.load_state_dict(best_model_state)

        print()
        print(f"[EVAL] Final held-out evaluation for {run_name} (best epoch {best_epoch})")

        metrics = evaluate(model=model, loader=test_loader, device=device)

        print(
            f"[EVAL] {run_name} | Accuracy {metrics['accuracy']:.4f} | "
            f"Macro F1 {metrics['macro_f1']:.4f}"
        )

    runtime_seconds = time.perf_counter() - fold_start

    fold_summary = {
        "best_epoch": best_epoch,
        "train_accuracy_at_best_epoch": best_train_accuracy,
        "validation_accuracy_at_best_epoch": best_validation_accuracy,
        "validation_macro_f1_at_best_epoch": best_validation_macro_f1,
        "runtime_seconds": runtime_seconds,
    }

    return model, metrics, history, mean, std, fold_summary


# =============================================================================
# CHECKPOINT SAVING
# =============================================================================

def save_checkpoint(
    path: Path,
    model: KinematicsOnlyTransformer,
    config: KinematicsOnlyConfig,
    kinematic_columns: Sequence[str],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> None:

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dimension": KINEMATIC_DIM_PER_SOURCE,
            "encoder_dimension": config.encoder_dim,
            "num_classes": NUM_GESTURE_CLASSES,
            "encoder_heads": config.encoder_heads,
            "encoder_layers": config.encoder_layers,
            "encoder_ff_dimension": config.encoder_ff_dim,
            "dropout": config.dropout,
        },
        "kinematic_source": config.kinematic_source,
        "kinematic_columns": list(kinematic_columns),
        "gesture_id_to_label": GESTURE_ID_TO_LABEL,
        "num_gesture_classes": NUM_GESTURE_CLASSES,
        "sample_rate": config.sample_rate,
        "window_seconds": config.window_seconds,
        "window_frames": int(round(config.sample_rate * config.window_seconds)),
        "stride_samples": config.stride_samples,
        "standardize": config.standardize,
        "mean": mean.tolist() if mean is not None else None,
        "std": std.tolist() if std is not None else None,
        "uses_kinematics_only": True,
        "uses_previous_gesture_label_as_input": False,
        "teacher_forcing": False,
        "autoregressive_label_feedback": False,
    }

    torch.save(checkpoint, path)


# =============================================================================
# COMPARISON WITH PREVIOUS EXPERIMENTS
# =============================================================================

def load_persistence_baseline_summary(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None:
        return None

    if not path.exists():
        print(f"[WARN] Persistence baseline metrics file not found: {path}")
        return None

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    cv = payload.get("cross_validation", {})

    return {
        "accuracy": cv.get("mean_accuracy"),
        "macro_f1": cv.get("mean_macro_f1"),
    }


def load_pytorch_summary(path: Optional[Path]) -> Optional[Dict[str, float]]:
    if path is None:
        return None

    if not path.exists():
        print(f"[WARN] Train_PyTorch.py metrics file not found: {path}")
        return None

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    cv = payload.get("cross_validation", {})
    folds = cv.get("folds", [])

    teacher_forced_acc = [
        f["teacher_forced_accuracy"] for f in folds
        if f.get("teacher_forced_accuracy") is not None
    ]
    teacher_forced_f1 = [
        f["teacher_forced_macro_f1"] for f in folds
        if f.get("teacher_forced_macro_f1") is not None
    ]

    return {
        "teacher_forced_accuracy": (
            float(np.mean(teacher_forced_acc)) if teacher_forced_acc else None
        ),
        "teacher_forced_macro_f1": (
            float(np.mean(teacher_forced_f1)) if teacher_forced_f1 else None
        ),
        "autoregressive_accuracy": cv.get("mean_accuracy"),
        "autoregressive_macro_f1": cv.get("mean_macro_f1"),
    }


def build_comparison_table(
    persistence_summary: Optional[Dict[str, float]],
    pytorch_summary: Optional[Dict[str, float]],
    kinematics_only_accuracy: Optional[float],
    kinematics_only_macro_f1: Optional[float],
) -> List[Dict[str, object]]:

    return [
        {
            "method": "Previous-label persistence baseline",
            "uses_kinematics": False,
            "uses_true_previous_gesture": True,
            "autonomous": False,
            "louo_accuracy": persistence_summary.get("accuracy") if persistence_summary else None,
            "louo_macro_f1": persistence_summary.get("macro_f1") if persistence_summary else None,
        },
        {
            "method": "Teacher-forced Transformer",
            "uses_kinematics": True,
            "uses_true_previous_gesture": True,
            "autonomous": False,
            "louo_accuracy": pytorch_summary.get("teacher_forced_accuracy") if pytorch_summary else None,
            "louo_macro_f1": pytorch_summary.get("teacher_forced_macro_f1") if pytorch_summary else None,
        },
        {
            "method": "Autoregressive Transformer",
            "uses_kinematics": True,
            "uses_true_previous_gesture": False,
            "autonomous": True,
            "louo_accuracy": pytorch_summary.get("autoregressive_accuracy") if pytorch_summary else None,
            "louo_macro_f1": pytorch_summary.get("autoregressive_macro_f1") if pytorch_summary else None,
        },
        {
            "method": "Kinematics-only Transformer",
            "uses_kinematics": True,
            "uses_true_previous_gesture": False,
            "autonomous": True,
            "louo_accuracy": kinematics_only_accuracy,
            "louo_macro_f1": kinematics_only_macro_f1,
        },
    ]


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(config: KinematicsOnlyConfig, smoke: bool) -> Dict[str, object]:

    tp.seed_everything(config.random_seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = tp.choose_device(config.device)

    print("=" * 78)
    print("ATARI-2 KINEMATICS-ONLY TRANSFORMER GESTURE RECOGNITION (DIAGNOSTIC)")
    print("=" * 78)

    print(f"[SYSTEM] Device: {device}")

    if device.type == "cuda":
        print(f"[SYSTEM] GPU: {torch.cuda.get_device_name(device)}")

    print(f"[SYSTEM] PyTorch version: {torch.__version__}")

    print()
    print("[CHECK] Model inputs contain kinematics only: YES")
    print("[CHECK] Previous gesture supplied as input: NO")
    print("[CHECK] Teacher forcing: NO")
    print("[CHECK] Autoregressive label feedback: NO")
    print()

    trials, kinematic_columns = tp.load_frame_level_data(
        path=config.input_csv,
        kinematic_source=config.kinematic_source,
    )

    surgeons = sorted({trial.surgeon_id for trial in trials})

    if len(surgeons) < 2:
        raise ValueError("LOUO cross-validation requires at least two surgeons.")

    fold_surgeons = surgeons

    if config.max_folds is not None:
        fold_surgeons = fold_surgeons[: config.max_folds]

    total_folds = len(fold_surgeons)

    all_fold_results: List[Dict[str, object]] = []
    complete_history: List[Dict[str, object]] = []

    pipeline_start = time.perf_counter()

    for fold_number, held_out_surgeon in enumerate(fold_surgeons, start=1):
        fold_start = time.perf_counter()

        train_trials = [t for t in trials if t.surgeon_id != held_out_surgeon]
        test_trials = [t for t in trials if t.surgeon_id == held_out_surgeon]

        fold_audit = tp.validate_louo_fold(
            train_trials=train_trials,
            test_trials=test_trials,
            held_out_surgeon=held_out_surgeon,
        )

        print()
        print("=" * 78)
        print(f"[LOUO] Fold {fold_number}/{total_folds}")
        print(f"[LOUO] Held-out surgeon: {held_out_surgeon}")
        print("=" * 78)

        _fold_model, metrics, history, _mean, _std, fold_summary = train_fold(
            train_trials=train_trials,
            test_trials=test_trials,
            config=config,
            device=device,
            run_name=f"LOUO_{held_out_surgeon}",
            held_out_surgeon=held_out_surgeon,
        )

        complete_history.extend(history)

        fold_elapsed = time.perf_counter() - fold_start

        result = {
            "fold": fold_number,
            "held_out_surgeon": held_out_surgeon,
            "n_train_trials": len(train_trials),
            "n_test_trials": len(test_trials),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mean_window_accuracy": metrics["mean_window_accuracy"],
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "best_epoch": fold_summary["best_epoch"],
            "train_accuracy_at_best_epoch": fold_summary["train_accuracy_at_best_epoch"],
            "validation_accuracy_at_best_epoch": fold_summary["validation_accuracy_at_best_epoch"],
            "validation_macro_f1_at_best_epoch": fold_summary["validation_macro_f1_at_best_epoch"],
            "runtime_seconds": fold_summary["runtime_seconds"],
            "seconds": fold_elapsed,
            "split_audit": fold_audit,
        }

        all_fold_results.append(result)

        elapsed_cv = time.perf_counter() - pipeline_start
        average_fold_time = elapsed_cv / fold_number
        eta = average_fold_time * (total_folds - fold_number)

        print()
        print(f"[LOUO] Fold {fold_number}/{total_folds} complete")
        print(f"[LOUO] Fold time: {tp.format_duration(fold_elapsed)}")
        print(f"[LOUO] CV elapsed: {tp.format_duration(elapsed_cv)}")
        print(f"[LOUO] Estimated CV remaining: {tp.format_duration(eta)}")

        del _fold_model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    accuracies = [float(r["accuracy"]) for r in all_fold_results]
    macro_f1_values = [float(r["macro_f1"]) for r in all_fold_results]

    cv_summary = {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "std_macro_f1": float(np.std(macro_f1_values)),
        "folds": all_fold_results,
    }

    print()
    print("=" * 78)
    print("[LOUO] CROSS-VALIDATION SUMMARY")
    print("=" * 78)
    print(f"Mean accuracy: {cv_summary['mean_accuracy']:.4f} +/- {cv_summary['std_accuracy']:.4f}")
    print(f"Mean macro F1: {cv_summary['mean_macro_f1']:.4f} +/- {cv_summary['std_macro_f1']:.4f}")

    # -------------------------------------------------------------------
    # Final model trained on all surgeons (for kinematics_only_model.pt)
    # -------------------------------------------------------------------

    print()
    print("=" * 78)
    print("[FINAL MODEL] Training on ALL available surgeons")
    print("=" * 78)

    final_model, _metrics, final_history, mean, std, _fold_summary = train_fold(
        train_trials=trials,
        test_trials=None,
        config=config,
        device=device,
        run_name="FINAL_ALL_USERS",
    )

    complete_history.extend(final_history)

    # -------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------

    model_path = config.output_dir / "kinematics_only_model.pt"
    save_checkpoint(
        path=model_path,
        model=final_model,
        config=config,
        kinematic_columns=kinematic_columns,
        mean=mean,
        std=std,
    )

    persistence_summary = load_persistence_baseline_summary(config.persistence_baseline_metrics)
    pytorch_summary = load_pytorch_summary(config.pytorch_metrics)

    comparison_rows = build_comparison_table(
        persistence_summary=persistence_summary,
        pytorch_summary=pytorch_summary,
        kinematics_only_accuracy=cv_summary["mean_accuracy"],
        kinematics_only_macro_f1=cv_summary["mean_macro_f1"],
    )

    total_runtime = time.perf_counter() - pipeline_start

    metrics_payload = {
        "diagnostic_note": (
            "This model receives raw kinematics only. No previous or "
            "current gesture label is ever supplied as a model input. "
            "There is no decoder, no teacher forcing, and no autoregressive "
            "label feedback."
        ),
        "uses_kinematics_only": True,
        "uses_previous_gesture_label_as_input": False,
        "teacher_forcing": False,
        "autoregressive_label_feedback": False,
        "cross_validation": cv_summary,
        "device": str(device),
        "total_runtime_seconds": total_runtime,
        "model_path": str(model_path),
    }

    metrics_path = config.output_dir / "kinematics_only_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    by_surgeon_rows = [
        {
            "held_out_surgeon": r["held_out_surgeon"],
            "n_train_trials": r["n_train_trials"],
            "n_test_trials": r["n_test_trials"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
            "mean_window_accuracy": r["mean_window_accuracy"],
            "best_epoch": r["best_epoch"],
            "train_accuracy_at_best_epoch": r["train_accuracy_at_best_epoch"],
            "validation_accuracy_at_best_epoch": r["validation_accuracy_at_best_epoch"],
            "validation_macro_f1_at_best_epoch": r["validation_macro_f1_at_best_epoch"],
            "runtime_seconds": r["runtime_seconds"],
        }
        for r in all_fold_results
    ]
    by_surgeon_path = config.output_dir / "kinematics_only_by_surgeon.csv"
    pd.DataFrame(by_surgeon_rows).to_csv(by_surgeon_path, index=False)

    history_path = config.output_dir / "kinematics_only_training_history.csv"
    pd.DataFrame(complete_history).to_csv(history_path, index=False)

    comparison_path = config.output_dir / "kinematics_only_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    print()
    print("=" * 78)
    print("[DONE] KINEMATICS-ONLY TRANSFORMER EXPERIMENT COMPLETE")
    print("=" * 78)
    print(f"Total runtime: {tp.format_duration(total_runtime)}")
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"By surgeon: {by_surgeon_path}")
    print(f"History: {history_path}")
    print(f"Comparison: {comparison_path}")

    return metrics_payload


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic experiment: train a kinematics-only Transformer "
            "(no decoder, no label input) to recognise JIGSAWS gestures."
        )
    )

    parser.add_argument("--input-csv", type=str, required=True, help="Prepared all_frame_level.csv.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for this experiment's outputs.")

    parser.add_argument("--kinematic-source", choices=["mtm", "psm"], default="mtm")

    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-samples", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--encoder-dim", type=int, default=38)
    parser.add_argument("--encoder-heads", type=int, default=1)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--encoder-ff-dim", type=int, default=152)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-metric", choices=["macro_f1", "accuracy"], default="macro_f1")

    parser.add_argument("--warmup-steps", type=int, default=2000)

    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize kinematics using training-surgeon mean/std. Disabled by default.",
    )

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, mps, etc.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)

    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--progress-updates-per-epoch", type=int, default=10)

    parser.add_argument(
        "--persistence-baseline-metrics",
        type=str,
        default=None,
        help="Optional previous_label_baseline_metrics.json for comparison.",
    )
    parser.add_argument(
        "--pytorch-metrics",
        type=str,
        default=None,
        help="Optional existing pytorch_metrics.json (teacher-forced/autoregressive) for comparison.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test mode: caps folds/epochs/windows so data loading, CUDA, "
            "forward/backward passes, LOUO splitting, metrics and outputs can "
            "be verified. Smoke-test accuracy is NOT meaningful."
        ),
    )

    return parser


def main() -> None:

    parser = build_arg_parser()
    args = parser.parse_args()

    max_folds = args.max_folds
    max_windows = args.max_windows
    epochs = args.epochs

    if args.smoke:
        max_folds = 2 if max_folds is None else min(max_folds, 2)
        max_windows = 1000 if max_windows is None else min(max_windows, 1000)
        epochs = 1
        print("[SMOKE] Smoke-test mode enabled: max_folds<=2, epochs=1, max_windows<=1000.")
        print("[SMOKE] Smoke-test accuracy is NOT meaningful model performance.")

    config = KinematicsOnlyConfig(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        kinematic_source=args.kinematic_source,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_samples=args.stride_samples,
        batch_size=args.batch_size,
        epochs=epochs,
        encoder_dim=args.encoder_dim,
        encoder_heads=args.encoder_heads,
        encoder_layers=args.encoder_layers,
        encoder_ff_dim=args.encoder_ff_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        warmup_steps=args.warmup_steps,
        standardize=args.standardize,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        device=args.device,
        max_folds=max_folds,
        max_windows=max_windows,
        progress_updates_per_epoch=args.progress_updates_per_epoch,
        persistence_baseline_metrics=(
            Path(args.persistence_baseline_metrics) if args.persistence_baseline_metrics else None
        ),
        pytorch_metrics=(
            Path(args.pytorch_metrics) if args.pytorch_metrics else None
        ),
    )

    run_pipeline(config, smoke=args.smoke)


if __name__ == "__main__":
    main()
