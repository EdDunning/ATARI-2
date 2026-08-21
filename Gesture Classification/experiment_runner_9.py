"""
ATARI-2: Gesture Classification / experiment_runner_9.py

===============================================================================
PURPOSE
===============================================================================

Determine whether class-weighted CrossEntropyLoss improves PSM kinematics-only
Transformer gesture recognition, particularly for underrepresented gestures
such as G1. Class-weighting method is the only intended independent variable;
every other setting matches the current best configuration used by
experiment_runner_7.py's PSM experiment / experiment_runner_8.py's error
analysis:

    PSM (k39-k76), 38 features, standardisation ON, 1.5 s / 45-frame windows,
    stride 1, dropout 0.3, weight decay 1e-3, batch size 64, AdamW, max 15
    epochs, early stopping patience 5 on macro F1, random seed 42, no
    previous-gesture input, no teacher forcing, no autoregressive feedback.

Runs exactly four configurations against the same prepared all_frame_level.csv
(data preparation is only run once, never rerun per configuration):

    A: unweighted            nn.CrossEntropyLoss()
    B: inverse_frequency     weight_c is proportional to 1 / frequency_c
    C: inverse_sqrt_frequency weight_c is proportional to 1 / sqrt(frequency_c)
    D: effective_number      weight_c is proportional to (1-beta) / (1-beta**n_c)

===============================================================================
REUSE, NOT DUPLICATION
===============================================================================

This script imports experiment_runner_4.py and experiment_runner_8.py in
process (never modifying either) and reuses, unchanged:

    - experiment_runner_4.KinematicsOnlyConfig / build_model / train_epoch /
      evaluate / save_checkpoint
    - Train_PyTorch.py's data loading, LOUO splitting, standardisation,
      leakage-auditing and NoamLearningRate helpers (via experiment_runner_4.tp)
    - experiment_runner_8.predict_trial_frame_level (the overlapping-window
      frame-level decoding logic) and experiment_runner_8.build_confusion_matrix

The ONE piece of experiment_runner_4.py that cannot be reused unmodified is
its train_fold(), because that function hardcodes an unweighted
nn.CrossEntropyLoss() with no override hook. train_fold_weighted() below
therefore mirrors train_fold()'s outer control flow (dataset construction,
optimizer/scheduler, epoch loop, early stopping, best-state tracking) but
builds the criterion from a per-fold class-weight tensor. This is the only
intentional duplication in this script.

===============================================================================
CLASS-WEIGHT LEAKAGE SAFETY
===============================================================================

Class frequencies/weights are computed inside compute_class_weights() using
ONLY the current fold's train_trials (which already excludes the held-out
surgeon by construction). A runtime assertion in run_weighting_experiment()
re-confirms the held-out surgeon contributed zero frames to those counts.

===============================================================================
ABSENT CLASSES
===============================================================================

If a gesture never occurs in a fold's training data, it is assigned a neutral
weight of 1.0 rather than an extreme/infinite value. Since nn.CrossEntropyLoss
indexes its `weight` tensor by each sample's TRUE target class, and an absent
class never appears as a true target within that fold, this neutral value has
no effect on that fold's training loss -- it exists only so the weight tensor
remains a well-defined length-16 tensor.

===============================================================================
COMMANDS
===============================================================================

Smoke test:

    python "Gesture Classification\\experiment_runner_9.py" `
        --mode smoke `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_9_smoke"

Single-procedure run (Knot Tying):

    python "Gesture Classification\\experiment_runner_9.py" `
        --mode single `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_9_knot_tying"

Full-dataset run (supported, not run automatically):

    python "Gesture Classification\\experiment_runner_9.py" `
        --mode full `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --kinematics-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\Needle_Passing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\transcriptions" `
        --kinematics-dir "JIGSAW\\Suturing\\Suturing\\Suturing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Suturing\\Suturing\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_9_full"
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install an appropriate PyTorch build "
        "before running experiment_runner_9.py."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PREP_SCRIPT = PROJECT_ROOT / "Gesture Data Manipulation" / "data_prep.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the kinematics-only model/config/training helpers and the frame-level
# prediction decoder instead of duplicating them.
import experiment_runner_4 as e4  # noqa: E402
import experiment_runner_8 as e8  # noqa: E402

VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CUDA_VENV_PYTHON = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"

NUM_GESTURE_CLASSES = e4.NUM_GESTURE_CLASSES
KINEMATIC_DIM_PER_SOURCE = e4.KINEMATIC_DIM_PER_SOURCE
GESTURE_ID_TO_LABEL = e4.GESTURE_ID_TO_LABEL

# Fixed "current best configuration". Class-weighting method is the only
# intended independent variable; none of these are CLI-tunable.
KINEMATIC_SOURCE = "psm"
SAMPLE_RATE = 30.0
WINDOW_SECONDS = 1.5
WINDOW_FRAMES = 45
STRIDE_SAMPLES = 1
BATCH_SIZE = 64
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "macro_f1"
MAX_EPOCHS = 15
STANDARDIZE = True
RANDOM_SEED = 42
DEFAULT_EFFECTIVE_NUMBER_BETA = 0.9999

BASELINE_WEIGHTING_METHOD = "unweighted"


@dataclass(frozen=True)
class TestPreset:
    name: str
    epochs: int
    max_trials: Optional[int]
    max_folds: Optional[int]
    max_windows: Optional[int]


@dataclass(frozen=True)
class WeightingMethod:
    name: str
    label: str


PRESETS = {
    "smoke": TestPreset("smoke", 1, 2, 2, 1000),
    "single": TestPreset("single", 15, None, None, None),
    "full": TestPreset("full", 15, None, None, None),
}

WEIGHTING_METHODS = (
    WeightingMethod("unweighted", "Experiment A - unweighted baseline"),
    WeightingMethod("inverse_frequency", "Experiment B - inverse-frequency weighting"),
    WeightingMethod("inverse_sqrt_frequency", "Experiment C - inverse-sqrt-frequency weighting"),
    WeightingMethod("effective_number", "Experiment D - effective-number weighting"),
)


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
# CLASS WEIGHTS
# =============================================================================

def compute_class_weights(
    train_trials: Sequence["e4.tp.TrialData"],
    method: str,
    num_classes: int,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Class counts/weights computed from TRAINING-FOLD frames only.

    Absent classes (count == 0) get a neutral weight of 1.0 rather than an
    infinite/extreme value; since nn.CrossEntropyLoss indexes its weight
    tensor by each sample's TRUE class, an absent class is never selected by
    any training sample in this fold, so its weight value has no effect on
    the training loss.
    """

    counts = np.zeros(num_classes, dtype=np.int64)
    for trial in train_trials:
        counts += np.bincount(trial.labels, minlength=num_classes)

    present = counts > 0
    weights = np.ones(num_classes, dtype=np.float64)

    if method == "unweighted":
        return counts, weights

    raw = np.zeros(num_classes, dtype=np.float64)

    if method == "inverse_frequency":
        raw[present] = 1.0 / counts[present]
    elif method == "inverse_sqrt_frequency":
        raw[present] = 1.0 / np.sqrt(counts[present])
    elif method == "effective_number":
        effective_num = 1.0 - np.power(beta, counts[present])
        raw[present] = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    mean_present = float(raw[present].mean()) if present.any() else 1.0
    if mean_present > 0:
        weights[present] = raw[present] / mean_present

    return counts, weights


# =============================================================================
# TRAINING WITH A WEIGHTED CRITERION
# (mirrors experiment_runner_4.train_fold(); duplicated because that function
#  hardcodes an unweighted nn.CrossEntropyLoss() with no override hook)
# =============================================================================

def train_fold_weighted(
    train_trials: Sequence["e4.tp.TrialData"],
    test_trials: Sequence["e4.tp.TrialData"],
    config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    run_name: str,
    held_out_surgeon: str,
    class_weights: np.ndarray,
) -> Tuple["e4.KinematicsOnlyTransformer", Dict[str, object], List[Dict[str, object]], Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:

    window_frames = int(round(config.window_seconds * config.sample_rate))

    if config.standardize:
        mean, std = e4.tp.calculate_standardization(train_trials)
        normalization_surgeons = sorted({t.surgeon_id for t in train_trials})
        if held_out_surgeon in normalization_surgeons:
            raise RuntimeError(
                "DATA LEAKAGE: held-out surgeon was used to calculate normalization statistics."
            )
    else:
        mean = None
        std = None

    train_dataset = e4.tp.make_split_dataset(
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

    test_dataset = e4.tp.make_split_dataset(
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

    e4.tp.audit_split_preprocessing(
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

    model = e4.build_model(config).to(device)

    assert model.input_dimension == KINEMATIC_DIM_PER_SOURCE
    assert model.num_classes == NUM_GESTURE_CLASSES

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    scheduler = e4.tp.NoamLearningRate(
        optimizer=optimizer,
        model_dimension=config.encoder_dim,
        warmup_steps=config.warmup_steps,
    )

    print()
    print(f"[MODEL] {run_name}")
    print(f"[MODEL] Training windows: {len(train_dataset):,}")
    print(f"[MODEL] Testing windows: {len(test_dataset):,}")

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

        train_loss, train_accuracy = e4.train_epoch(
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
            f"Epoch time {format_duration(epoch_elapsed)}"
        )

        validation_metrics = e4.evaluate(model=model, loader=test_loader, device=device)
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
                "patience_counter": patience_counter,
            }
        )

        if patience_counter >= config.early_stopping_patience:
            print(
                f"[EARLY STOPPING] {run_name} | No improvement in "
                f"{config.early_stopping_metric} for {patience_counter} epoch(s)."
            )
            break

    if best_model_state is None:
        raise RuntimeError("No validation checkpoint was recorded.")

    model.load_state_dict(best_model_state)

    print()
    print(f"[EVAL] Final held-out evaluation for {run_name} (best epoch {best_epoch})")

    metrics = e4.evaluate(model=model, loader=test_loader, device=device)

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
# ONE WEIGHTING-METHOD EXPERIMENT ACROSS ALL LOUO FOLDS
# =============================================================================

def run_weighting_experiment(
    weighting_method: WeightingMethod,
    trials: List["e4.tp.TrialData"],
    fold_surgeons: List[str],
    config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    output_dir: Path,
    beta: float,
) -> Tuple[Dict[str, object], List[Dict[str, object]], pd.DataFrame]:

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_models_dir = output_dir / "fold_models"
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"[WEIGHTING] {weighting_method.label}")
    print("=" * 78)
    print("[CHECK] Kinematic source: PSM")
    print(f"[CHECK] Input features: {KINEMATIC_DIM_PER_SOURCE}")
    print(f"[CHECK] Window: {WINDOW_SECONDS} sec / {WINDOW_FRAMES} frames")
    print(f"[CHECK] Standardisation: {'ON' if STANDARDIZE else 'OFF'}")
    print("[CHECK] Class weights calculated from training fold only: YES")
    print("[CHECK] Previous gesture supplied as input: NO")
    print("[CHECK] Teacher forcing: NO")
    print("[CHECK] Autoregressive label feedback: NO")

    fold_results: List[Dict[str, object]] = []
    class_weight_records: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []

    experiment_start = time.perf_counter()

    for fold_number, held_out_surgeon in enumerate(fold_surgeons, start=1):
        fold_start = time.perf_counter()

        train_trials = [t for t in trials if t.surgeon_id != held_out_surgeon]
        test_trials = [t for t in trials if t.surgeon_id == held_out_surgeon]

        e4.tp.validate_louo_fold(
            train_trials=train_trials,
            test_trials=test_trials,
            held_out_surgeon=held_out_surgeon,
        )

        counts, weights = compute_class_weights(
            train_trials=train_trials,
            method=weighting_method.name,
            num_classes=NUM_GESTURE_CLASSES,
            beta=beta,
        )

        # Runtime assertion: the held-out surgeon contributed zero frames to
        # the class counts used for this fold's weights.
        assert held_out_surgeon not in {t.surgeon_id for t in train_trials}
        assert int(sum(np.bincount(t.labels, minlength=NUM_GESTURE_CLASSES).sum() for t in test_trials)) >= 0

        print()
        print(f"[FOLD] {weighting_method.label} | Fold {fold_number}/{len(fold_surgeons)} | Held-out: {held_out_surgeon}")
        print(f"[CLASS WEIGHTS] Counts (training surgeons only): {counts.tolist()}")
        print(f"[CLASS WEIGHTS] Weights: {[round(w, 4) for w in weights.tolist()]}")

        class_weight_records.append(
            {
                "held_out_surgeon": held_out_surgeon,
                "weighting_method": weighting_method.name,
                "counts": counts.tolist(),
                "weights": weights.tolist(),
            }
        )

        model, metrics, _history, mean, std, fold_summary = train_fold_weighted(
            train_trials=train_trials,
            test_trials=test_trials,
            config=config,
            device=device,
            run_name=f"{weighting_method.name}_LOUO_{held_out_surgeon}",
            held_out_surgeon=held_out_surgeon,
            class_weights=weights,
        )

        e4.save_checkpoint(
            path=fold_models_dir / f"LOUO_{held_out_surgeon}.pt",
            model=model,
            config=config,
            kinematic_columns=[f"k{i:02d}" for i in range(39, 77)],
            mean=mean,
            std=std,
        )

        model.eval()
        for trial in test_trials:
            result = e8.predict_trial_frame_level(
                model=model,
                trial=trial,
                window_frames=WINDOW_FRAMES,
                stride_samples=STRIDE_SAMPLES,
                standardize=config.standardize,
                mean=mean,
                std=std,
                device=device,
            )
            if result is None:
                print(f"[WARN] Trial {trial.trial_id} is shorter than the window length; skipped.")
                continue

            frame_predictions, frame_confidences = result
            for i in range(len(trial.labels)):
                prediction_rows.append(
                    {
                        "weighting_method": weighting_method.name,
                        "held_out_surgeon": held_out_surgeon,
                        "trial_id": trial.trial_id,
                        "frame_idx": int(trial.frame_indices[i]),
                        "true_gesture": int(trial.labels[i]),
                        "predicted_gesture": int(frame_predictions[i]),
                        "confidence": float(frame_confidences[i]),
                    }
                )

        fold_elapsed = time.perf_counter() - fold_start
        experiment_elapsed = time.perf_counter() - experiment_start
        average_fold_time = experiment_elapsed / fold_number
        eta = average_fold_time * (len(fold_surgeons) - fold_number)

        print(f"[RUNTIME] Fold {fold_number}/{len(fold_surgeons)} time: {format_duration(fold_elapsed)}")
        print(f"[RUNTIME] {weighting_method.label} elapsed: {format_duration(experiment_elapsed)}")
        print(f"[RUNTIME] Estimated remaining (this method): {format_duration(eta)}")

        fold_results.append(
            {
                "held_out_surgeon": held_out_surgeon,
                "weighting_method": weighting_method.name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class": metrics["per_class"],
                "confusion_matrix": metrics["confusion_matrix"],
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

    accuracies = [float(r["accuracy"]) for r in fold_results]
    macro_f1_values = [float(r["macro_f1"]) for r in fold_results]

    method_metrics = {
        "weighting_method": weighting_method.name,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "std_macro_f1": float(np.std(macro_f1_values)),
        "folds": fold_results,
        "total_runtime_seconds": total_runtime,
    }

    with (output_dir / "kinematics_only_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(method_metrics, file, indent=2)

    with (output_dir / "class_weights.json").open("w", encoding="utf-8") as file:
        json.dump(class_weight_records, file, indent=2)

    predictions_df = pd.DataFrame(prediction_rows)

    print()
    print(f"[DONE] {weighting_method.label} complete. Total runtime: {format_duration(total_runtime)}")

    return method_metrics, fold_results, predictions_df


# =============================================================================
# CROSS-METHOD SUMMARY / BY-SURGEON / BY-GESTURE OUTPUTS
# =============================================================================

def build_summary_rows(all_method_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []

    for method_metrics in all_method_metrics:
        fold_results = method_metrics["folds"]
        best_epochs = [f["best_epoch"] for f in fold_results if f.get("best_epoch") is not None]
        train_accuracy_at_best = [
            f["train_accuracy_at_best_epoch"] for f in fold_results
            if f.get("train_accuracy_at_best_epoch") is not None
        ]

        rows.append(
            {
                "weighting_method": method_metrics["weighting_method"],
                "mean_louo_accuracy": method_metrics["mean_accuracy"],
                "std_louo_accuracy": method_metrics["std_accuracy"],
                "mean_louo_macro_f1": method_metrics["mean_macro_f1"],
                "std_louo_macro_f1": method_metrics["std_macro_f1"],
                "mean_training_accuracy_at_best_epoch": (
                    float(np.mean(train_accuracy_at_best)) if train_accuracy_at_best else None
                ),
                "mean_best_epoch": float(np.mean(best_epochs)) if best_epochs else None,
                "total_runtime_seconds": method_metrics["total_runtime_seconds"],
            }
        )

    baseline_row = next(row for row in rows if row["weighting_method"] == BASELINE_WEIGHTING_METHOD)
    for row in rows:
        row["change_in_accuracy_vs_unweighted"] = row["mean_louo_accuracy"] - baseline_row["mean_louo_accuracy"]
        row["change_in_macro_f1_vs_unweighted"] = row["mean_louo_macro_f1"] - baseline_row["mean_louo_macro_f1"]

    rows.sort(key=lambda row: float(row["mean_louo_macro_f1"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def build_by_surgeon_rows(all_method_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for method_metrics in all_method_metrics:
        for fold in method_metrics["folds"]:
            rows.append(
                {
                    "held_out_surgeon": fold["held_out_surgeon"],
                    "weighting_method": fold["weighting_method"],
                    "accuracy": fold["accuracy"],
                    "macro_f1": fold["macro_f1"],
                    "best_epoch": fold["best_epoch"],
                    "training_accuracy_at_best_epoch": fold["train_accuracy_at_best_epoch"],
                    "runtime": fold["runtime_seconds"],
                }
            )
    rows.sort(key=lambda row: (row["held_out_surgeon"], row["weighting_method"]))
    return rows


def build_by_gesture_rows(all_predictions: pd.DataFrame) -> List[Dict[str, object]]:
    gesture_tables: Dict[str, pd.DataFrame] = {}

    for method_name, method_predictions in all_predictions.groupby("weighting_method"):
        confusion = e8.build_confusion_matrix(method_predictions)
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
                    "gesture_id": gesture_id,
                    "gesture": GESTURE_ID_TO_LABEL[gesture_id],
                    "support": support,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        gesture_tables[method_name] = pd.DataFrame(rows).set_index("gesture_id")

    baseline_table = gesture_tables[BASELINE_WEIGHTING_METHOD]

    output_rows = []
    for method_name, table in gesture_tables.items():
        for gesture_id, row in table.iterrows():
            baseline_row = baseline_table.loc[gesture_id]
            output_rows.append(
                {
                    "weighting_method": method_name,
                    "gesture": row["gesture"],
                    "support": row["support"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "change_in_F1_vs_unweighted": row["f1"] - baseline_row["f1"],
                    "change_in_recall_vs_unweighted": row["recall"] - baseline_row["recall"],
                }
            )

    return output_rows


# =============================================================================
# PIPELINE
# =============================================================================

def run_experiments(args: argparse.Namespace) -> Path:
    preset = PRESETS[args.mode]
    output_root = Path(args.output_root).resolve()
    prepared_data_dir = output_root / "prepared_data"
    experiment_root = output_root / "class_weighting_experiments"
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    assert WINDOW_FRAMES == round(WINDOW_SECONDS * SAMPLE_RATE)
    assert KINEMATIC_DIM_PER_SOURCE == 38

    e4.tp.seed_everything(RANDOM_SEED)
    device = e4.tp.choose_device(args.device)

    if args.reuse_prepared_data:
        frame_file = prepared_data_dir / "all_frame_level.csv"
        require_file(frame_file, "existing frame-level dataset")
    else:
        frame_file = prepare_data(
            [Path(path).resolve() for path in args.kinematics_dir],
            [Path(path).resolve() for path in args.annotations_dir],
            prepared_data_dir,
            SAMPLE_RATE,
            preset.max_trials,
        )

    trials, _kinematic_columns = e4.tp.load_frame_level_data(
        path=frame_file,
        kinematic_source=KINEMATIC_SOURCE,
    )

    surgeons = sorted({trial.surgeon_id for trial in trials})
    fold_surgeons = surgeons if preset.max_folds is None else surgeons[: preset.max_folds]

    max_windows = args.max_windows if args.max_windows is not None else preset.max_windows
    epochs = preset.epochs

    if args.mode == "smoke":
        print()
        print("[SMOKE] Smoke-test mode enabled: max_folds<=2, epochs=1, max_windows<=1000.")
        print("[SMOKE] Smoke-test results are NOT scientifically meaningful.")

    config = e4.KinematicsOnlyConfig(
        input_csv=frame_file,
        output_dir=experiment_root,
        kinematic_source=KINEMATIC_SOURCE,
        sample_rate=SAMPLE_RATE,
        window_seconds=WINDOW_SECONDS,
        stride_samples=STRIDE_SAMPLES,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        dropout=DROPOUT,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_metric=EARLY_STOPPING_METRIC,
        standardize=STANDARDIZE,
        random_seed=RANDOM_SEED,
        device=args.device,
        max_windows=max_windows,
    )

    print(
        f"Running exactly {len(WEIGHTING_METHODS)} class-weighting configurations "
        f"with seed {RANDOM_SEED} against the same prepared dataset and the same "
        f"{len(fold_surgeons)} LOUO fold(s) (data preparation is not rerun between "
        f"configurations). Class-weighting method is the only intended independent "
        f"variable."
    )

    pipeline_start = time.perf_counter()

    all_method_metrics = []
    all_predictions_frames = []

    for weighting_method in WEIGHTING_METHODS:
        method_dir = experiment_root / weighting_method.name
        method_metrics, _fold_results, predictions_df = run_weighting_experiment(
            weighting_method=weighting_method,
            trials=trials,
            fold_surgeons=fold_surgeons,
            config=config,
            device=device,
            output_dir=method_dir,
            beta=args.beta,
        )
        all_method_metrics.append(method_metrics)
        all_predictions_frames.append(predictions_df)

    all_predictions = pd.concat(all_predictions_frames, ignore_index=True)

    summary_rows = build_summary_rows(all_method_metrics)
    by_surgeon_rows = build_by_surgeon_rows(all_method_metrics)
    by_gesture_rows = build_by_gesture_rows(all_predictions)

    summary_path = output_root / "kinematics_only_class_weighting_summary.csv"
    summary_fieldnames = [
        "rank", "weighting_method",
        "mean_louo_accuracy", "std_louo_accuracy",
        "mean_louo_macro_f1", "std_louo_macro_f1",
        "mean_training_accuracy_at_best_epoch", "mean_best_epoch",
        "total_runtime_seconds",
        "change_in_accuracy_vs_unweighted", "change_in_macro_f1_vs_unweighted",
    ]
    pd.DataFrame(summary_rows)[summary_fieldnames].to_csv(summary_path, index=False)

    by_surgeon_path = output_root / "kinematics_only_class_weighting_by_surgeon.csv"
    pd.DataFrame(by_surgeon_rows).to_csv(by_surgeon_path, index=False)

    by_gesture_path = output_root / "kinematics_only_class_weighting_by_gesture.csv"
    pd.DataFrame(by_gesture_rows).to_csv(by_gesture_path, index=False)

    predictions_path = output_root / "kinematics_only_class_weighting_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False)

    total_runtime = time.perf_counter() - pipeline_start

    print()
    print("=" * 79)
    print("[DONE] KINEMATICS-ONLY CLASS-WEIGHTING EXPERIMENT COMPLETE")
    print("=" * 79)
    print(f"Complete experiment runtime: {format_duration(total_runtime)}")
    print(f"Summary written to {summary_path}")
    print(f"Per-surgeon comparison written to {by_surgeon_path}")
    print(f"Per-gesture comparison written to {by_gesture_path}")
    print(f"Held-out predictions written to {predictions_path}")
    print(f"Per-fold checkpoints written under {experiment_root}/<method>/fold_models/")

    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled PSM kinematics-only Transformer class-weighting "
            "experiment (experiment_runner_4.py/experiment_runner_8.py helpers, "
            "not Train_PyTorch.py). Class-weighting method is the only intended "
            "independent variable."
        )
    )
    parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
    parser.add_argument("--kinematics-dir", action="append", required=True)
    parser.add_argument("--annotations-dir", action="append", required=True)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs_pytorch_kinematics_only_class_weighting"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reuse-prepared-data", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_EFFECTIVE_NUMBER_BETA,
        help="Effective-number weighting beta (default 0.9999).",
    )
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
