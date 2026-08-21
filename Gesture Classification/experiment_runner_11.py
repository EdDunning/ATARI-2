"""
ATARI-2: Gesture Classification / experiment_runner_11.py

===============================================================================
PURPOSE
===============================================================================

Determine whether MTM contains information complementary to the current best
PSM kinematics-only Transformer, and determine the best way to combine MTM and
PSM. Input source / fusion architecture is the only intended independent
variable. Compares exactly four architectures against the same LOUO folds:

    A: PSM-only baseline   (k39-k76, 38 features)
    B: MTM-only reference  (k01-k38, 38 features)
    C: Early fusion        (k01-k76, 76 features -> one encoder)
    D: Late fusion         (independent MTM/PSM encoders -> fusion layer)

===============================================================================
REUSE, NOT DUPLICATION
===============================================================================

A and B are ordinary 38-feature single-source kinematics-only Transformers,
so they reuse experiment_runner_4.py's KinematicsOnlyConfig/train_fold/
save_checkpoint and experiment_runner_8.py's predict_trial_frame_level
completely unmodified -- only the kinematic source differs.

C and D cannot reuse those same functions unmodified for two structural
reasons:
    - experiment_runner_4.train_epoch()/experiment_runner_8.
      predict_trial_frame_level() both assert the model receives exactly the
      module-level KINEMATIC_DIM_PER_SOURCE (38) features; C's model receives
      76, and D's model receives two separate 38-feature tensors.
    - experiment_runner_4.build_model()/save_checkpoint() hardcode a single
      38-feature KinematicsOnlyTransformer; D's dual-encoder model is a
      different architecture entirely.
    - Train_PyTorch.calculate_standardization() hardcodes a 38-feature output
      shape, so it cannot standardise the 76-feature combined input.

For C and D this script therefore uses train_fold_fusion()/train_epoch_fusion()/
evaluate_fusion()/predict_trial_frame_level_fusion()/calculate_standardization_generic(),
which mirror the equivalent experiment_runner_4.py/experiment_runner_8.py/
Train_PyTorch.py functions but are parameterised by feature width and a
`forward_fn` so the same code works for both a single wide encoder (C) and a
dual-branch model (D). Everything else -- LOUO splitting, leakage auditing,
the window dataset, the Noam learning-rate schedule, confusion-matrix/metric
aggregation, and C's actual encoder block (tp.SurgicalEncoderLayer, reused
inside experiment_runner_4.KinematicsOnlyTransformer for C, and directly for
each of D's two independent branches) -- is reused unmodified.

Train_PyTorch.py, experiment_runner_4/5/6/7/8/9/10.py, data_prep.py, and
run_all_pytorch.py are never modified.

===============================================================================
COMMANDS
===============================================================================

Smoke test:

    python "Gesture Classification\\experiment_runner_11.py" `
        --mode smoke `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_11_smoke"

Single-procedure run (Knot Tying):

    python "Gesture Classification\\experiment_runner_11.py" `
        --mode single `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_11_knot_tying"

Full-dataset run (supported, not run automatically):

    python "Gesture Classification\\experiment_runner_11.py" `
        --mode full `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --kinematics-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\Needle_Passing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\transcriptions" `
        --kinematics-dir "JIGSAW\\Suturing\\Suturing\\Suturing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Suturing\\Suturing\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_11_full"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install an appropriate PyTorch build "
        "before running experiment_runner_11.py."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PREP_SCRIPT = PROJECT_ROOT / "Gesture Data Manipulation" / "data_prep.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the kinematics-only model/config/training loop and the frame-level
# prediction decoder instead of duplicating them wherever they stay compatible.
import experiment_runner_4 as e4  # noqa: E402
import experiment_runner_8 as e8  # noqa: E402

VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CUDA_VENV_PYTHON = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"

NUM_GESTURE_CLASSES = e4.NUM_GESTURE_CLASSES
GESTURE_ID_TO_LABEL = e4.GESTURE_ID_TO_LABEL
MTM_FEATURE_COUNT = 38
PSM_FEATURE_COUNT = 38
COMBINED_FEATURE_COUNT = 76

# Fixed "newly selected best configuration". Input source/fusion architecture
# is the only intended independent variable; none of these are CLI-tunable.
SAMPLE_RATE = 30.0
WINDOW_SECONDS = 1.5
WINDOW_FRAMES = 45
STRIDE_SAMPLES = 1
BATCH_SIZE = 64
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "macro_f1"
MAX_EPOCHS = 15
STANDARDIZE = True
RANDOM_SEED = 42
FUSION_HIDDEN_DIM = 64

BASELINE_ARCHITECTURE = "psm_only"
ATTENTION_GESTURES = ("G1", "G12", "G13", "G14", "G15")


@dataclass(frozen=True)
class TestPreset:
    name: str
    epochs: int
    max_trials: Optional[int]
    max_folds: Optional[int]
    max_windows: Optional[int]


PRESETS = {
    "smoke": TestPreset("smoke", 1, 2, 2, 1000),
    "single": TestPreset("single", 15, None, None, None),
    "full": TestPreset("full", 15, None, None, None),
}


def resolve_python_executable() -> str:
    override = os.environ.get("ATARI_PYTHON")
    if override and Path(override).exists():
        return override

    for candidate in (CUDA_VENV_PYTHON, VENV_PYTHON):
        if not candidate.exists():
            continue
        try:
            result = subprocess.run(
                [str(candidate), "-c", "import torch, pandas"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return str(candidate)

    return sys.executable


PYTHON_EXECUTABLE = resolve_python_executable()


def format_duration(seconds: float) -> str:
    return e4.tp.format_duration(seconds)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find {description}: {path}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"Expected {description}: {path}")


def prepare_data(
    kinematics_dirs: List[Path],
    annotations_dirs: List[Path],
    prepared_data_dir: Path,
    sample_rate: float,
    max_trials: Optional[int],
) -> Path:
    command = [PYTHON_EXECUTABLE, "-u", str(DATA_PREP_SCRIPT)]
    for kinematics_dir, annotations_dir in zip(kinematics_dirs, annotations_dirs):
        command.extend([
            "--kinematics-dir", str(kinematics_dir),
            "--annotations-dir", str(annotations_dir),
        ])
    command.extend([
        "--output-dir", str(prepared_data_dir),
        "--sample-rate", str(sample_rate),
    ])
    if max_trials is not None:
        command.extend(["--max-trials", str(max_trials)])

    print()
    print("=" * 79)
    print("[STARTING] data preparation")
    print("Command:", " ".join(f'"{part}"' if " " in part else part for part in command))

    start = time.perf_counter()
    result = subprocess.run(command)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"[ERROR] data preparation failed with return code {result.returncode}.")
        raise SystemExit(result.returncode)

    print(f"[SUCCESS] data preparation ({format_duration(elapsed)})")

    frame_file = prepared_data_dir / "all_frame_level.csv"
    require_file(frame_file, "frame-level dataset")
    return frame_file


# =============================================================================
# COMBINED MTM+PSM (76-FEATURE) TRIAL DATA
# =============================================================================

def build_combined_trials(
    trials_mtm: Sequence["e4.tp.TrialData"],
    trials_psm: Sequence["e4.tp.TrialData"],
) -> List["e4.tp.TrialData"]:
    """MTM (k01-k38) concatenated with PSM (k39-k76) into one 76-wide trial."""

    psm_by_trial = {trial.trial_id: trial for trial in trials_psm}
    combined: List["e4.tp.TrialData"] = []

    for mtm_trial in trials_mtm:
        psm_trial = psm_by_trial.get(mtm_trial.trial_id)
        if psm_trial is None:
            raise RuntimeError(f"Trial {mtm_trial.trial_id} present in MTM load but missing from PSM load.")
        if not np.array_equal(mtm_trial.frame_indices, psm_trial.frame_indices):
            raise RuntimeError(f"Trial {mtm_trial.trial_id}: MTM/PSM frame indices do not match.")
        if not np.array_equal(mtm_trial.labels, psm_trial.labels):
            raise RuntimeError(f"Trial {mtm_trial.trial_id}: MTM/PSM gesture labels do not match.")
        if mtm_trial.surgeon_id != psm_trial.surgeon_id:
            raise RuntimeError(f"Trial {mtm_trial.trial_id}: MTM/PSM surgeon_id mismatch.")

        combined_kinematics = np.concatenate([mtm_trial.kinematics, psm_trial.kinematics], axis=1)

        combined.append(
            e4.tp.TrialData(
                trial_id=mtm_trial.trial_id,
                surgeon_id=mtm_trial.surgeon_id,
                task=mtm_trial.task,
                kinematics=combined_kinematics,
                labels=mtm_trial.labels,
                frame_indices=mtm_trial.frame_indices,
            )
        )

    return combined


def calculate_standardization_generic(
    trials: Sequence["e4.tp.TrialData"],
    num_features: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirrors tp.calculate_standardization, parameterized: tp's version hardcodes 38 features."""

    total_count = 0
    running_sum = np.zeros(num_features, dtype=np.float64)
    running_sq_sum = np.zeros(num_features, dtype=np.float64)

    for trial in trials:
        x = trial.kinematics.astype(np.float64, copy=False)
        running_sum += x.sum(axis=0)
        running_sq_sum += np.square(x).sum(axis=0)
        total_count += len(x)

    mean = running_sum / total_count
    variance = np.maximum(running_sq_sum / total_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


# =============================================================================
# LATE-FUSION MODEL (new architecture; independent MTM/PSM encoders)
# =============================================================================

class LateFusionTransformer(nn.Module):
    """
    MTM 38D -> MTM projection -> MTM encoder -> MTM representation
    PSM 38D -> PSM projection -> PSM encoder -> PSM representation
    concat(MTM repr, PSM repr) -> Linear -> ReLU -> Dropout -> Linear -> 16 logits

    The two encoders (tp.SurgicalEncoderLayer stacks) have independent
    weights -- separate ModuleList instances, never shared.
    """

    def __init__(
        self,
        mtm_dim: int = 38,
        psm_dim: int = 38,
        encoder_dim: int = 38,
        num_classes: int = 16,
        encoder_heads: int = 1,
        encoder_layers: int = 1,
        encoder_ff_dimension: int = 152,
        dropout: float = 0.1,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.mtm_dim = mtm_dim
        self.psm_dim = psm_dim
        self.num_classes = num_classes

        self.mtm_input_projection = nn.Linear(mtm_dim, encoder_dim)
        self.psm_input_projection = nn.Linear(psm_dim, encoder_dim)

        self.mtm_encoder = nn.ModuleList(
            [
                e4.tp.SurgicalEncoderLayer(
                    dimension=encoder_dim,
                    heads=encoder_heads,
                    feedforward_dimension=encoder_ff_dimension,
                    dropout=dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.psm_encoder = nn.ModuleList(
            [
                e4.tp.SurgicalEncoderLayer(
                    dimension=encoder_dim,
                    heads=encoder_heads,
                    feedforward_dimension=encoder_ff_dimension,
                    dropout=dropout,
                )
                for _ in range(encoder_layers)
            ]
        )

        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, mtm_source: torch.Tensor, psm_source: torch.Tensor) -> torch.Tensor:
        if mtm_source.shape[-1] != self.mtm_dim or psm_source.shape[-1] != self.psm_dim:
            raise RuntimeError(
                f"LateFusionTransformer expects MTM/PSM widths "
                f"{self.mtm_dim}/{self.psm_dim}; got "
                f"{mtm_source.shape[-1]}/{psm_source.shape[-1]}."
            )

        mtm_repr = self.mtm_input_projection(mtm_source)
        for layer in self.mtm_encoder:
            mtm_repr = layer(mtm_repr)

        psm_repr = self.psm_input_projection(psm_source)
        for layer in self.psm_encoder:
            psm_repr = layer(psm_repr)

        fused = torch.cat([mtm_repr, psm_repr], dim=-1)
        return self.fusion(fused)


# =============================================================================
# GENERIC (WIDTH/forward_fn-PARAMETERIZED) TRAIN/EVAL/PREDICT FOR C AND D
# =============================================================================

def train_epoch_fusion(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: "e4.tp.NoamLearningRate",
    criterion: nn.Module,
    device: torch.device,
    epoch_number: int,
    total_epochs: int,
    progress_updates: int,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
) -> Tuple[float, float]:

    model.train()
    epoch_start = time.perf_counter()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    n_batches = len(loader)
    update_interval = max(1, n_batches // max(1, progress_updates))

    for batch_index, batch in enumerate(loader, start=1):
        source, target, _previous_label, _metadata = batch

        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = forward_fn(model, source)

        loss = criterion(logits.reshape(-1, NUM_GESTURE_CLASSES), target.reshape(-1))
        loss.backward()

        scheduler.step()
        optimizer.step()

        predictions = torch.argmax(logits, dim=-1)
        batch_tokens = target.numel()
        total_loss += float(loss.item()) * batch_tokens
        total_tokens += batch_tokens
        total_correct += int((predictions == target).sum().item())

        if batch_index == 1 or batch_index % update_interval == 0 or batch_index == n_batches:
            elapsed = time.perf_counter() - epoch_start
            eta = (elapsed / batch_index) * (n_batches - batch_index)
            progress = 100.0 * batch_index / n_batches
            print(
                f"[TRAIN] Epoch {epoch_number}/{total_epochs} | Batch {batch_index}/{n_batches} | "
                f"{progress:5.1f}% | Loss {loss.item():.5f} | LR {scheduler.current_lr:.6g} | "
                f"Elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
            )

    mean_loss = total_loss / total_tokens if total_tokens else 0.0
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return float(mean_loss), float(accuracy)


@torch.no_grad()
def evaluate_fusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
) -> Dict[str, object]:

    model.eval()
    confusion = np.zeros((NUM_GESTURE_CLASSES, NUM_GESTURE_CLASSES), dtype=np.int64)
    window_accuracy_sum = 0.0
    window_count = 0

    for batch in loader:
        source, target, _previous_label, _metadata = batch
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = forward_fn(model, source)
        prediction = torch.argmax(logits, dim=-1)

        e4.tp.update_confusion_matrix(confusion, target, prediction)

        per_window_accuracy = (prediction == target).float().mean(dim=1)
        window_accuracy_sum += float(per_window_accuracy.sum().item())
        window_count += int(len(per_window_accuracy))

    metrics = e4.tp.metrics_from_confusion_matrix(confusion)
    metrics["mean_window_accuracy"] = window_accuracy_sum / window_count if window_count else 0.0
    return metrics


def train_fold_fusion(
    model: nn.Module,
    train_trials: Sequence["e4.tp.TrialData"],
    test_trials: Sequence["e4.tp.TrialData"],
    config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    run_name: str,
    held_out_surgeon: str,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    num_features: int,
    lr_model_dimension: int,
) -> Tuple[nn.Module, Dict[str, object], List[Dict[str, object]], Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    """Mirrors experiment_runner_4.train_fold(); parameterised for width/forward_fn (see module docstring)."""

    window_frames = WINDOW_FRAMES

    if config.standardize:
        mean, std = calculate_standardization_generic(train_trials, num_features)
        normalization_surgeons = sorted({t.surgeon_id for t in train_trials})
        if held_out_surgeon in normalization_surgeons:
            raise RuntimeError("DATA LEAKAGE: held-out surgeon was used to calculate normalization statistics.")
    else:
        mean = None
        std = None

    train_dataset = e4.tp.make_split_dataset(
        trials=train_trials, window_frames=window_frames, stride_samples=config.stride_samples,
        standardize=config.standardize, mean=mean, std=std, max_windows=config.max_windows,
        random_seed=config.random_seed,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"{run_name}: no training windows generated.")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    test_dataset = e4.tp.make_split_dataset(
        trials=test_trials, window_frames=window_frames, stride_samples=config.stride_samples,
        standardize=config.standardize, mean=mean, std=std, max_windows=config.max_windows,
        random_seed=config.random_seed + 1,
    )
    if len(test_dataset) == 0:
        raise ValueError(f"{run_name}: no testing windows generated.")

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    e4.tp.audit_split_preprocessing(
        train_trials=train_trials, test_trials=test_trials,
        train_dataset=train_dataset, test_dataset=test_dataset,
        window_frames=window_frames, stride_samples=config.stride_samples,
        standardize=config.standardize, mean=mean, std=std,
        held_out_surgeon=held_out_surgeon,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0,
        betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    scheduler = e4.tp.NoamLearningRate(optimizer=optimizer, model_dimension=lr_model_dimension, warmup_steps=config.warmup_steps)

    print(f"[MODEL] {run_name} | Training windows: {len(train_dataset):,} | Testing windows: {len(test_dataset):,}")

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

        train_loss, train_accuracy = train_epoch_fusion(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler,
            criterion=criterion, device=device, epoch_number=epoch, total_epochs=config.epochs,
            progress_updates=config.progress_updates_per_epoch, forward_fn=forward_fn,
        )

        epoch_elapsed = time.perf_counter() - epoch_start
        print(
            f"[EPOCH] {run_name} | {epoch}/{config.epochs} complete | Loss {train_loss:.5f} | "
            f"Train accuracy {train_accuracy:.4f} | Epoch time {format_duration(epoch_elapsed)}"
        )

        validation_metrics = evaluate_fusion(model=model, loader=test_loader, device=device, forward_fn=forward_fn)
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
            f"[VALIDATION] {run_name} | Accuracy {validation_accuracy:.4f} | Macro F1 {validation_macro_f1:.4f} | "
            f"Best {config.early_stopping_metric} {best_validation_metric:.4f} | "
            f"Patience {patience_counter}/{config.early_stopping_patience}"
        )

        history.append(
            {
                "run": run_name, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                "learning_rate": scheduler.current_lr, "epoch_seconds": epoch_elapsed,
                "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_macro_f1,
                "early_stopping_value": early_stopping_value,
                "best_validation_value_so_far": best_validation_metric,
                "patience_counter": patience_counter,
            }
        )

        if patience_counter >= config.early_stopping_patience:
            print(f"[EARLY STOPPING] {run_name} | No improvement in {config.early_stopping_metric} for {patience_counter} epoch(s).")
            break

    if best_model_state is None:
        raise RuntimeError("No validation checkpoint was recorded.")

    model.load_state_dict(best_model_state)

    print(f"[EVAL] Final held-out evaluation for {run_name} (best epoch {best_epoch})")
    metrics = evaluate_fusion(model=model, loader=test_loader, device=device, forward_fn=forward_fn)
    print(f"[EVAL] {run_name} | Accuracy {metrics['accuracy']:.4f} | Macro F1 {metrics['macro_f1']:.4f}")

    runtime_seconds = time.perf_counter() - fold_start

    fold_summary = {
        "best_epoch": best_epoch,
        "train_accuracy_at_best_epoch": best_train_accuracy,
        "validation_accuracy_at_best_epoch": best_validation_accuracy,
        "validation_macro_f1_at_best_epoch": best_validation_macro_f1,
        "runtime_seconds": runtime_seconds,
    }

    return model, metrics, history, mean, std, fold_summary


@torch.no_grad()
def predict_trial_frame_level_fusion(
    model: nn.Module,
    trial: "e4.tp.TrialData",
    window_frames: int,
    stride_samples: int,
    standardize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    device: torch.device,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    expected_width: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Mirrors experiment_runner_8.predict_trial_frame_level(); parameterised for width/forward_fn."""

    total_frames = len(trial.labels)
    if total_frames < window_frames:
        return None

    starts = list(range(0, total_frames - window_frames + 1, stride_samples))
    windows = np.stack([trial.kinematics[s:s + window_frames] for s in starts])

    if standardize:
        windows = (windows - mean) / std

    assert windows.shape[-1] == expected_width, (
        f"Expected {expected_width} kinematic features, got {windows.shape[-1]}."
    )

    windows_tensor = torch.from_numpy(windows).float().to(device)
    logits = forward_fn(model, windows_tensor)
    probabilities = torch.softmax(logits, dim=-1)
    confidences, predictions = probabilities.max(dim=-1)

    predictions = predictions.cpu().numpy()
    confidences = confidences.cpu().numpy()

    frame_predictions = np.full(total_frames, -1, dtype=np.int64)
    frame_confidences = np.full(total_frames, np.nan, dtype=np.float64)

    first_start = starts[0]
    for offset in range(window_frames):
        frame_predictions[first_start + offset] = predictions[0, offset]
        frame_confidences[first_start + offset] = confidences[0, offset]

    for window_index in range(1, len(starts)):
        start = starts[window_index]
        last_frame = start + window_frames - 1
        frame_predictions[last_frame] = predictions[window_index, window_frames - 1]
        frame_confidences[last_frame] = confidences[window_index, window_frames - 1]

    if stride_samples == 1 and (frame_predictions < 0).any():
        raise RuntimeError(f"Trial {trial.trial_id}: frame coverage gap found with stride=1.")

    return frame_predictions, frame_confidences


def save_fusion_checkpoint(
    path: Path,
    model: nn.Module,
    architecture_name: str,
    input_description: str,
    config: "e4.KinematicsOnlyConfig",
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> None:
    """e4.save_checkpoint() assumes a single 38-feature model; C/D need their own metadata."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture_name,
        "input_description": input_description,
        "sample_rate": config.sample_rate,
        "window_seconds": config.window_seconds,
        "window_frames": WINDOW_FRAMES,
        "stride_samples": config.stride_samples,
        "standardize": config.standardize,
        "mean": mean.tolist() if mean is not None else None,
        "std": std.tolist() if std is not None else None,
        "gesture_id_to_label": GESTURE_ID_TO_LABEL,
        "num_gesture_classes": NUM_GESTURE_CLASSES,
    }
    torch.save(checkpoint, path)


# =============================================================================
# PER-ARCHITECTURE ORCHESTRATION
# =============================================================================

def print_common_check_lines() -> None:
    print(f"[CHECK] Window: {WINDOW_SECONDS} sec / {WINDOW_FRAMES} frames")
    print(f"[CHECK] Standardisation: {'ON' if STANDARDIZE else 'OFF'}")
    print(f"[CHECK] Dropout: {DROPOUT}")
    print(f"[CHECK] Weight decay: {WEIGHT_DECAY}")
    print("[CHECK] Loss: unweighted CrossEntropyLoss")
    print("[CHECK] Previous gesture supplied as input: NO")
    print("[CHECK] Teacher forcing: NO")
    print("[CHECK] Autoregressive label feedback: NO")


def run_single_source_architecture(
    architecture_name: str,
    label: str,
    input_description: str,
    kinematic_source: str,
    trials: List["e4.tp.TrialData"],
    fold_surgeons: List[str],
    base_config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict[str, object], pd.DataFrame]:

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_models_dir = output_dir / "fold_models"
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    config = e4.KinematicsOnlyConfig(
        input_csv=base_config.input_csv, output_dir=output_dir, kinematic_source=kinematic_source,
        sample_rate=base_config.sample_rate, window_seconds=base_config.window_seconds,
        stride_samples=base_config.stride_samples, batch_size=base_config.batch_size,
        epochs=base_config.epochs, dropout=base_config.dropout, weight_decay=base_config.weight_decay,
        early_stopping_patience=base_config.early_stopping_patience,
        early_stopping_metric=base_config.early_stopping_metric, standardize=base_config.standardize,
        random_seed=base_config.random_seed, device=base_config.device, max_windows=base_config.max_windows,
    )

    print()
    print("=" * 78)
    print(f"[ARCHITECTURE] {label}")
    print("=" * 78)
    print(f"[CHECK] Input: {input_description}")
    print(f"[CHECK] Input dimensions: {e4.KINEMATIC_DIM_PER_SOURCE}")
    print_common_check_lines()

    fold_results: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    trainable_parameters: Optional[int] = None

    experiment_start = time.perf_counter()

    for fold_number, held_out_surgeon in enumerate(fold_surgeons, start=1):
        fold_start = time.perf_counter()

        train_trials = [t for t in trials if t.surgeon_id != held_out_surgeon]
        test_trials = [t for t in trials if t.surgeon_id == held_out_surgeon]

        e4.tp.validate_louo_fold(train_trials=train_trials, test_trials=test_trials, held_out_surgeon=held_out_surgeon)

        print()
        print(f"[FOLD] {label} | Fold {fold_number}/{len(fold_surgeons)} | Held-out: {held_out_surgeon}")

        model, metrics, _history, mean, std, fold_summary = e4.train_fold(
            train_trials=train_trials, test_trials=test_trials, config=config, device=device,
            run_name=f"{architecture_name}_LOUO_{held_out_surgeon}", held_out_surgeon=held_out_surgeon,
        )

        if trainable_parameters is None:
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[MODEL] {label} | Trainable parameters: {trainable_parameters:,}")

        e4.save_checkpoint(
            path=fold_models_dir / f"LOUO_{held_out_surgeon}.pt", model=model, config=config,
            kinematic_columns=e4.tp.get_kinematic_columns(kinematic_source), mean=mean, std=std,
        )

        model.eval()
        for trial in test_trials:
            result = e8.predict_trial_frame_level(
                model=model, trial=trial, window_frames=WINDOW_FRAMES, stride_samples=STRIDE_SAMPLES,
                standardize=config.standardize, mean=mean, std=std, device=device,
            )
            if result is None:
                print(f"[WARN] Trial {trial.trial_id} is shorter than the window length; skipped.")
                continue
            frame_predictions, frame_confidences = result
            for i in range(len(trial.labels)):
                prediction_rows.append(
                    {
                        "architecture": architecture_name, "held_out_surgeon": held_out_surgeon,
                        "trial_id": trial.trial_id, "frame_idx": int(trial.frame_indices[i]),
                        "true_gesture": int(trial.labels[i]), "predicted_gesture": int(frame_predictions[i]),
                        "confidence": float(frame_confidences[i]),
                    }
                )

        fold_elapsed = time.perf_counter() - fold_start
        experiment_elapsed = time.perf_counter() - experiment_start
        eta = (experiment_elapsed / fold_number) * (len(fold_surgeons) - fold_number)
        print(f"[RUNTIME] Fold {fold_number}/{len(fold_surgeons)} time: {format_duration(fold_elapsed)}")
        print(f"[RUNTIME] {label} elapsed: {format_duration(experiment_elapsed)} | ETA: {format_duration(eta)}")

        fold_results.append(
            {
                "held_out_surgeon": held_out_surgeon, "architecture": architecture_name,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "best_epoch": fold_summary["best_epoch"],
                "train_accuracy_at_best_epoch": fold_summary["train_accuracy_at_best_epoch"],
                "validation_accuracy_at_best_epoch": fold_summary["validation_accuracy_at_best_epoch"],
                "validation_macro_f1_at_best_epoch": fold_summary["validation_macro_f1_at_best_epoch"],
                "runtime_seconds": fold_summary["runtime_seconds"],
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_runtime = time.perf_counter() - experiment_start

    method_metrics = build_architecture_metrics(
        architecture_name, label, input_description, trainable_parameters, fold_results, total_runtime,
    )

    with (output_dir / "kinematics_only_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(method_metrics, file, indent=2)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(output_dir / "kinematics_only_predictions.csv", index=False)

    print()
    print(f"[DONE] {label} complete. Total runtime: {format_duration(total_runtime)}")

    return method_metrics, predictions_df


def run_fusion_architecture(
    architecture_name: str,
    label: str,
    input_description: str,
    extra_check_lines: List[str],
    model_builder: Callable[[], nn.Module],
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    num_features: int,
    lr_model_dimension: int,
    combined_trials: List["e4.tp.TrialData"],
    fold_surgeons: List[str],
    base_config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict[str, object], pd.DataFrame]:

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_models_dir = output_dir / "fold_models"
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"[ARCHITECTURE] {label}")
    print("=" * 78)
    for line in extra_check_lines:
        print(line)
    print_common_check_lines()

    fold_results: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    trainable_parameters: Optional[int] = None

    experiment_start = time.perf_counter()

    for fold_number, held_out_surgeon in enumerate(fold_surgeons, start=1):
        fold_start = time.perf_counter()

        train_trials = [t for t in combined_trials if t.surgeon_id != held_out_surgeon]
        test_trials = [t for t in combined_trials if t.surgeon_id == held_out_surgeon]

        e4.tp.validate_louo_fold(train_trials=train_trials, test_trials=test_trials, held_out_surgeon=held_out_surgeon)

        print()
        print(f"[FOLD] {label} | Fold {fold_number}/{len(fold_surgeons)} | Held-out: {held_out_surgeon}")

        model = model_builder().to(device)

        if trainable_parameters is None:
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[MODEL] {label} | Trainable parameters: {trainable_parameters:,}")

        model, metrics, _history, mean, std, fold_summary = train_fold_fusion(
            model=model, train_trials=train_trials, test_trials=test_trials, config=base_config,
            device=device, run_name=f"{architecture_name}_LOUO_{held_out_surgeon}",
            held_out_surgeon=held_out_surgeon, forward_fn=forward_fn, num_features=num_features,
            lr_model_dimension=lr_model_dimension,
        )

        save_fusion_checkpoint(
            path=fold_models_dir / f"LOUO_{held_out_surgeon}.pt", model=model,
            architecture_name=architecture_name, input_description=input_description,
            config=base_config, mean=mean, std=std,
        )

        model.eval()
        for trial in test_trials:
            result = predict_trial_frame_level_fusion(
                model=model, trial=trial, window_frames=WINDOW_FRAMES, stride_samples=STRIDE_SAMPLES,
                standardize=base_config.standardize, mean=mean, std=std, device=device,
                forward_fn=forward_fn, expected_width=num_features,
            )
            if result is None:
                print(f"[WARN] Trial {trial.trial_id} is shorter than the window length; skipped.")
                continue
            frame_predictions, frame_confidences = result
            for i in range(len(trial.labels)):
                prediction_rows.append(
                    {
                        "architecture": architecture_name, "held_out_surgeon": held_out_surgeon,
                        "trial_id": trial.trial_id, "frame_idx": int(trial.frame_indices[i]),
                        "true_gesture": int(trial.labels[i]), "predicted_gesture": int(frame_predictions[i]),
                        "confidence": float(frame_confidences[i]),
                    }
                )

        fold_elapsed = time.perf_counter() - fold_start
        experiment_elapsed = time.perf_counter() - experiment_start
        eta = (experiment_elapsed / fold_number) * (len(fold_surgeons) - fold_number)
        print(f"[RUNTIME] Fold {fold_number}/{len(fold_surgeons)} time: {format_duration(fold_elapsed)}")
        print(f"[RUNTIME] {label} elapsed: {format_duration(experiment_elapsed)} | ETA: {format_duration(eta)}")

        fold_results.append(
            {
                "held_out_surgeon": held_out_surgeon, "architecture": architecture_name,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "best_epoch": fold_summary["best_epoch"],
                "train_accuracy_at_best_epoch": fold_summary["train_accuracy_at_best_epoch"],
                "validation_accuracy_at_best_epoch": fold_summary["validation_accuracy_at_best_epoch"],
                "validation_macro_f1_at_best_epoch": fold_summary["validation_macro_f1_at_best_epoch"],
                "runtime_seconds": fold_summary["runtime_seconds"],
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_runtime = time.perf_counter() - experiment_start

    method_metrics = build_architecture_metrics(
        architecture_name, label, input_description, trainable_parameters, fold_results, total_runtime,
    )

    with (output_dir / "kinematics_only_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(method_metrics, file, indent=2)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(output_dir / "kinematics_only_predictions.csv", index=False)

    print()
    print(f"[DONE] {label} complete. Total runtime: {format_duration(total_runtime)}")

    return method_metrics, predictions_df


def build_architecture_metrics(
    architecture_name: str,
    label: str,
    input_description: str,
    trainable_parameters: Optional[int],
    fold_results: List[Dict[str, object]],
    total_runtime: float,
) -> Dict[str, object]:
    accuracies = [float(r["accuracy"]) for r in fold_results]
    macro_f1_values = [float(r["macro_f1"]) for r in fold_results]

    return {
        "architecture": architecture_name,
        "label": label,
        "input_description": input_description,
        "trainable_parameters": trainable_parameters,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "std_macro_f1": float(np.std(macro_f1_values)),
        "folds": fold_results,
        "total_runtime_seconds": total_runtime,
    }


# =============================================================================
# CROSS-ARCHITECTURE SUMMARY / BY-SURGEON / BY-GESTURE / CONFUSION OUTPUTS
# =============================================================================

def build_summary_rows(all_architecture_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for architecture_metrics in all_architecture_metrics:
        fold_results = architecture_metrics["folds"]
        best_epochs = [f["best_epoch"] for f in fold_results if f.get("best_epoch") is not None]
        train_acc = [f["train_accuracy_at_best_epoch"] for f in fold_results if f.get("train_accuracy_at_best_epoch") is not None]
        val_acc = [f["validation_accuracy_at_best_epoch"] for f in fold_results if f.get("validation_accuracy_at_best_epoch") is not None]

        rows.append(
            {
                "architecture": architecture_metrics["architecture"],
                "input_features": architecture_metrics["input_description"],
                "trainable_parameters": architecture_metrics["trainable_parameters"],
                "mean_louo_accuracy": architecture_metrics["mean_accuracy"],
                "std_louo_accuracy": architecture_metrics["std_accuracy"],
                "mean_louo_macro_f1": architecture_metrics["mean_macro_f1"],
                "std_louo_macro_f1": architecture_metrics["std_macro_f1"],
                "mean_training_accuracy_at_best_epoch": float(np.mean(train_acc)) if train_acc else None,
                "mean_validation_accuracy_at_best_epoch": float(np.mean(val_acc)) if val_acc else None,
                "mean_best_epoch": float(np.mean(best_epochs)) if best_epochs else None,
                "total_runtime_seconds": architecture_metrics["total_runtime_seconds"],
            }
        )

    baseline_row = next(row for row in rows if row["architecture"] == BASELINE_ARCHITECTURE)
    for row in rows:
        row["change_in_accuracy_vs_psm"] = row["mean_louo_accuracy"] - baseline_row["mean_louo_accuracy"]
        row["change_in_macro_f1_vs_psm"] = row["mean_louo_macro_f1"] - baseline_row["mean_louo_macro_f1"]

    rows.sort(key=lambda row: float(row["mean_louo_macro_f1"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def build_by_surgeon_rows(all_architecture_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for architecture_metrics in all_architecture_metrics:
        for fold in architecture_metrics["folds"]:
            rows.append(
                {
                    "architecture": fold["architecture"], "held_out_surgeon": fold["held_out_surgeon"],
                    "accuracy": fold["accuracy"], "macro_f1": fold["macro_f1"],
                    "best_epoch": fold["best_epoch"],
                    "training_accuracy_at_best_epoch": fold["train_accuracy_at_best_epoch"],
                    "validation_accuracy_at_best_epoch": fold["validation_accuracy_at_best_epoch"],
                    "runtime": fold["runtime_seconds"],
                }
            )
    rows.sort(key=lambda row: (row["held_out_surgeon"], row["architecture"]))
    return rows


def build_by_gesture_rows(all_predictions: pd.DataFrame) -> List[Dict[str, object]]:
    gesture_tables: Dict[str, pd.DataFrame] = {}

    for architecture_name, architecture_predictions in all_predictions.groupby("architecture"):
        confusion = e8.build_confusion_matrix(architecture_predictions)
        rows = []
        for gesture_id in range(NUM_GESTURE_CLASSES):
            true_positive = int(confusion[gesture_id, gesture_id])
            false_positive = int(confusion[:, gesture_id].sum() - true_positive)
            false_negative = int(confusion[gesture_id, :].sum() - true_positive)
            support = int(confusion[gesture_id, :].sum())

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            rows.append(
                {
                    "gesture_id": gesture_id, "gesture": GESTURE_ID_TO_LABEL[gesture_id],
                    "support": support, "precision": precision, "recall": recall, "f1": f1,
                }
            )
        gesture_tables[architecture_name] = pd.DataFrame(rows).set_index("gesture_id")

    baseline_table = gesture_tables[BASELINE_ARCHITECTURE]

    output_rows = []
    for architecture_name, table in gesture_tables.items():
        for gesture_id, row in table.iterrows():
            baseline_row = baseline_table.loc[gesture_id]
            output_rows.append(
                {
                    "architecture": architecture_name, "gesture": row["gesture"], "support": row["support"],
                    "precision": row["precision"], "recall": row["recall"], "f1": row["f1"],
                    "change_in_F1_vs_psm": row["f1"] - baseline_row["f1"],
                    "change_in_recall_vs_psm": row["recall"] - baseline_row["recall"],
                }
            )

    return output_rows


def save_confusion_matrices(all_predictions: pd.DataFrame, output_root: Path) -> None:
    labels = [GESTURE_ID_TO_LABEL[g] for g in range(NUM_GESTURE_CLASSES)]

    for architecture_name, architecture_predictions in all_predictions.groupby("architecture"):
        confusion = e8.build_confusion_matrix(architecture_predictions)

        counts_df = pd.DataFrame(confusion, index=labels, columns=labels)
        counts_df.index.name = "true_gesture"
        counts_df.to_csv(output_root / f"kinematics_only_fusion_confusion_counts_{architecture_name}.csv")

        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(
            confusion.astype(np.float64) * 100.0, row_sums,
            out=np.zeros_like(confusion, dtype=np.float64), where=row_sums != 0,
        )
        normalized_df = pd.DataFrame(normalized, index=labels, columns=labels)
        normalized_df.index.name = "true_gesture"
        normalized_df.to_csv(output_root / f"kinematics_only_fusion_confusion_normalized_{architecture_name}.csv")


# =============================================================================
# PIPELINE
# =============================================================================

def run_experiments(args: argparse.Namespace) -> Path:
    preset = PRESETS[args.mode]
    output_root = Path(args.output_root).resolve()
    prepared_data_dir = output_root / "prepared_data"
    experiment_root = output_root / "fusion_experiments"
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    assert WINDOW_FRAMES == round(WINDOW_SECONDS * SAMPLE_RATE)

    e4.tp.seed_everything(RANDOM_SEED)
    device = e4.tp.choose_device(args.device)

    if args.reuse_prepared_data:
        frame_file = prepared_data_dir / "all_frame_level.csv"
        require_file(frame_file, "existing frame-level dataset")
    else:
        frame_file = prepare_data(
            [Path(path).resolve() for path in args.kinematics_dir],
            [Path(path).resolve() for path in args.annotations_dir],
            prepared_data_dir, SAMPLE_RATE, preset.max_trials,
        )

    trials_mtm, _ = e4.tp.load_frame_level_data(path=frame_file, kinematic_source="mtm")
    trials_psm, _ = e4.tp.load_frame_level_data(path=frame_file, kinematic_source="psm")
    combined_trials = build_combined_trials(trials_mtm, trials_psm)

    surgeons = sorted({trial.surgeon_id for trial in trials_mtm})
    fold_surgeons = surgeons if preset.max_folds is None else surgeons[: preset.max_folds]

    max_windows = args.max_windows if args.max_windows is not None else preset.max_windows

    if args.mode == "smoke":
        print()
        print("[SMOKE] Smoke-test mode enabled: max_folds<=2, epochs=1, max_windows<=1000.")
        print("[SMOKE] Verifying: PSM=38 features, MTM=38 features, early fusion=76 features,")
        print("[SMOKE] late fusion=two 38-feature tensors, all four output [batch, time, 16].")
        print("[SMOKE] Smoke-test results are NOT scientifically meaningful.")

    base_config = e4.KinematicsOnlyConfig(
        input_csv=frame_file, output_dir=experiment_root, kinematic_source="mtm",
        sample_rate=SAMPLE_RATE, window_seconds=WINDOW_SECONDS, stride_samples=STRIDE_SAMPLES,
        batch_size=BATCH_SIZE, epochs=preset.epochs, dropout=DROPOUT, weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE, early_stopping_metric=EARLY_STOPPING_METRIC,
        standardize=STANDARDIZE, random_seed=RANDOM_SEED, device=args.device, max_windows=max_windows,
    )

    print(
        f"Running exactly 4 fusion-architecture configurations with seed {RANDOM_SEED} "
        f"against the same prepared dataset and the same {len(fold_surgeons)} LOUO fold(s) "
        f"(data preparation is not rerun between configurations). Input source/fusion "
        f"architecture is the only intended independent variable."
    )

    pipeline_start = time.perf_counter()
    all_architecture_metrics: List[Dict[str, object]] = []
    all_predictions_frames: List[pd.DataFrame] = []

    psm_metrics, psm_predictions = run_single_source_architecture(
        architecture_name="psm_only", label="A - PSM only", input_description="PSM k39-k76 (38 features)",
        kinematic_source="psm", trials=trials_psm, fold_surgeons=fold_surgeons, base_config=base_config,
        device=device, output_dir=experiment_root / "psm_only",
    )
    all_architecture_metrics.append(psm_metrics)
    all_predictions_frames.append(psm_predictions)

    mtm_metrics, mtm_predictions = run_single_source_architecture(
        architecture_name="mtm_only", label="B - MTM only", input_description="MTM k01-k38 (38 features)",
        kinematic_source="mtm", trials=trials_mtm, fold_surgeons=fold_surgeons, base_config=base_config,
        device=device, output_dir=experiment_root / "mtm_only",
    )
    all_architecture_metrics.append(mtm_metrics)
    all_predictions_frames.append(mtm_predictions)

    def build_early_fusion_model() -> nn.Module:
        return e4.KinematicsOnlyTransformer(
            input_dimension=COMBINED_FEATURE_COUNT, encoder_dimension=COMBINED_FEATURE_COUNT,
            num_classes=NUM_GESTURE_CLASSES, encoder_heads=1, encoder_layers=1,
            encoder_ff_dimension=152, dropout=DROPOUT,
        )

    def early_fusion_forward(model: nn.Module, source: torch.Tensor) -> torch.Tensor:
        return model(source)

    early_metrics, early_predictions = run_fusion_architecture(
        architecture_name="early_fusion", label="C - Early fusion (MTM+PSM, one encoder)",
        input_description="MTM+PSM k01-k76 (76 features, early fusion, one encoder)",
        extra_check_lines=["[CHECK] Input: MTM+PSM k01-k76", f"[CHECK] Input dimensions: {COMBINED_FEATURE_COUNT}"],
        model_builder=build_early_fusion_model, forward_fn=early_fusion_forward,
        num_features=COMBINED_FEATURE_COUNT, lr_model_dimension=COMBINED_FEATURE_COUNT,
        combined_trials=combined_trials, fold_surgeons=fold_surgeons, base_config=base_config,
        device=device, output_dir=experiment_root / "early_fusion",
    )
    all_architecture_metrics.append(early_metrics)
    all_predictions_frames.append(early_predictions)

    def build_late_fusion_model() -> nn.Module:
        return LateFusionTransformer(
            mtm_dim=MTM_FEATURE_COUNT, psm_dim=PSM_FEATURE_COUNT, encoder_dim=38,
            num_classes=NUM_GESTURE_CLASSES, encoder_heads=1, encoder_layers=1,
            encoder_ff_dimension=152, dropout=DROPOUT, fusion_hidden_dim=FUSION_HIDDEN_DIM,
        )

    def late_fusion_forward(model: nn.Module, source: torch.Tensor) -> torch.Tensor:
        return model(source[..., :MTM_FEATURE_COUNT], source[..., MTM_FEATURE_COUNT:])

    late_metrics, late_predictions = run_fusion_architecture(
        architecture_name="late_fusion", label="D - Late fusion (independent MTM/PSM encoders)",
        input_description="MTM 38 + PSM 38 (76 features, late fusion, independent encoders)",
        extra_check_lines=[
            f"[CHECK] MTM branch dimensions: {MTM_FEATURE_COUNT}",
            f"[CHECK] PSM branch dimensions: {PSM_FEATURE_COUNT}",
            "[CHECK] Independent encoders: YES",
        ],
        model_builder=build_late_fusion_model, forward_fn=late_fusion_forward,
        num_features=COMBINED_FEATURE_COUNT, lr_model_dimension=38,
        combined_trials=combined_trials, fold_surgeons=fold_surgeons, base_config=base_config,
        device=device, output_dir=experiment_root / "late_fusion",
    )
    all_architecture_metrics.append(late_metrics)
    all_predictions_frames.append(late_predictions)

    all_predictions = pd.concat(all_predictions_frames, ignore_index=True)

    summary_rows = build_summary_rows(all_architecture_metrics)
    by_surgeon_rows = build_by_surgeon_rows(all_architecture_metrics)
    by_gesture_rows = build_by_gesture_rows(all_predictions)

    summary_path = output_root / "kinematics_only_fusion_summary.csv"
    summary_fieldnames = [
        "rank", "architecture", "input_features", "trainable_parameters",
        "mean_louo_accuracy", "std_louo_accuracy", "mean_louo_macro_f1", "std_louo_macro_f1",
        "mean_training_accuracy_at_best_epoch", "mean_validation_accuracy_at_best_epoch",
        "mean_best_epoch", "total_runtime_seconds",
        "change_in_accuracy_vs_psm", "change_in_macro_f1_vs_psm",
    ]
    pd.DataFrame(summary_rows)[summary_fieldnames].to_csv(summary_path, index=False)

    by_surgeon_path = output_root / "kinematics_only_fusion_by_surgeon.csv"
    pd.DataFrame(by_surgeon_rows).to_csv(by_surgeon_path, index=False)

    by_gesture_path = output_root / "kinematics_only_fusion_by_gesture.csv"
    pd.DataFrame(by_gesture_rows).to_csv(by_gesture_path, index=False)

    save_confusion_matrices(all_predictions, output_root)

    predictions_path = output_root / "kinematics_only_fusion_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False)

    total_runtime = time.perf_counter() - pipeline_start

    attention_gesture_rows = [row for row in by_gesture_rows if row["gesture"] in ATTENTION_GESTURES]

    print()
    print("=" * 79)
    print("[DONE] KINEMATICS-ONLY FUSION EXPERIMENT COMPLETE")
    print("=" * 79)
    print(f"Complete experiment runtime: {format_duration(total_runtime)}")
    print(f"Summary written to {summary_path}")
    print(f"Per-surgeon comparison written to {by_surgeon_path}")
    print(f"Per-gesture comparison written to {by_gesture_path}")
    print(f"Confusion matrices written under {output_root}")
    print(f"Held-out predictions written to {predictions_path}")
    print(f"Per-fold checkpoints written under {experiment_root}/<architecture>/fold_models/")
    print()
    print(f"[G1/G12-G15] {len(attention_gesture_rows)} rows reported in {by_gesture_path.name} for review.")

    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled MTM-vs-PSM-vs-fusion kinematics-only Transformer "
            "experiment (experiment_runner_4.py/experiment_runner_8.py helpers, not "
            "Train_PyTorch.py). Input source/fusion architecture is the only "
            "intended independent variable."
        )
    )
    parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
    parser.add_argument("--kinematics-dir", action="append", required=True)
    parser.add_argument("--annotations-dir", action="append", required=True)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs_pytorch_kinematics_only_fusion"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-prepared-data", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    kinematics_dirs = [Path(path) for path in args.kinematics_dir]
    annotations_dirs = [Path(path) for path in args.annotations_dir]
    if len(kinematics_dirs) != len(annotations_dirs):
        raise ValueError("The number of kinematics and annotation directories must match.")
    for path in kinematics_dirs:
        require_directory(path, "kinematic data directory")
    for path in annotations_dirs:
        require_directory(path, "annotation directory")
    if not DATA_PREP_SCRIPT.is_file():
        raise FileNotFoundError(f"Required script was not found: {DATA_PREP_SCRIPT}")
    run_experiments(args)


if __name__ == "__main__":
    main()
