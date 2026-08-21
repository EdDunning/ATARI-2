"""
ATARI-2: Gesture Classification / experiment_runner_10.py

===============================================================================
PURPOSE
===============================================================================

Retune dropout and weight decay for the CURRENT best kinematics-only PSM
Transformer (not the old encoder-decoder Train_PyTorch.py model -- that
model's regularisation experiment, experiment_runner_1.py/experiment_runner_2.py,
is no longer representative because the architecture and inference method have
changed substantially). Dropout and weight decay are the only intended
independent variables; every other setting matches the current best PSM
configuration used by experiment_runner_7/8/9.py:

    PSM (k39-k76), 38 features, standardisation ON, 1.5 s / 45-frame windows,
    stride 1, unweighted CrossEntropyLoss, batch size 64, AdamW, max 15
    epochs, early stopping patience 5 on macro F1, random seed 42, no
    previous-gesture input, no teacher forcing, no autoregressive feedback.

Runs exactly eight configurations against the same prepared all_frame_level.csv
(data preparation is only run once, never rerun per configuration):

    A: dropout=0.0, weight_decay=0
    B: dropout=0.1, weight_decay=1e-4
    C: dropout=0.2, weight_decay=1e-4
    D: dropout=0.3, weight_decay=1e-3   (current baseline)
    E: dropout=0.4, weight_decay=1e-3
    F: dropout=0.5, weight_decay=1e-3
    G: dropout=0.2, weight_decay=1e-3
    H: dropout=0.3, weight_decay=1e-4

===============================================================================
REUSE, NOT DUPLICATION
===============================================================================

Because this experiment uses ordinary unweighted CrossEntropyLoss (unlike
experiment_runner_9.py's class-weighting experiment), experiment_runner_4.py's
train_fold() can be reused completely unmodified for every configuration --
only its config.dropout / config.weight_decay values change. This script also
reuses experiment_runner_8.predict_trial_frame_level() for frame-level held-out
inference and experiment_runner_4.save_checkpoint() for per-fold checkpoints.
Nothing in Train_PyTorch.py, experiment_runner_4/5/6/7/8/9.py, or data_prep.py
is modified.

===============================================================================
COMMANDS
===============================================================================

Smoke test:

    python "Gesture Classification\\experiment_runner_10.py" `
        --mode smoke `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_10_smoke"

Single-procedure run (Knot Tying):

    python "Gesture Classification\\experiment_runner_10.py" `
        --mode single `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_10_knot_tying"

Full-dataset run (supported, not run automatically):

    python "Gesture Classification\\experiment_runner_10.py" `
        --mode full `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --kinematics-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\Needle_Passing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\transcriptions" `
        --kinematics-dir "JIGSAW\\Suturing\\Suturing\\Suturing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Suturing\\Suturing\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_10_full"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install an appropriate PyTorch build "
        "before running experiment_runner_10.py."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PREP_SCRIPT = PROJECT_ROOT / "Gesture Data Manipulation" / "data_prep.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the kinematics-only model/config/training loop and the frame-level
# prediction decoder instead of duplicating them.
import experiment_runner_4 as e4  # noqa: E402
import experiment_runner_8 as e8  # noqa: E402

VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CUDA_VENV_PYTHON = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"

NUM_GESTURE_CLASSES = e4.NUM_GESTURE_CLASSES
KINEMATIC_DIM_PER_SOURCE = e4.KINEMATIC_DIM_PER_SOURCE

# Fixed "current best configuration". Dropout/weight decay are the only
# intended independent variables; none of these are CLI-tunable.
KINEMATIC_SOURCE = "psm"
SAMPLE_RATE = 30.0
WINDOW_SECONDS = 1.5
WINDOW_FRAMES = 45
STRIDE_SAMPLES = 1
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "macro_f1"
MAX_EPOCHS = 15
STANDARDIZE = True
RANDOM_SEED = 42

BASELINE_CONFIG_NAME = "D"


@dataclass(frozen=True)
class TestPreset:
    name: str
    epochs: int
    max_trials: Optional[int]
    max_folds: Optional[int]
    max_windows: Optional[int]


@dataclass(frozen=True)
class RegularizationConfig:
    name: str
    label: str
    dropout: float
    weight_decay: float


PRESETS = {
    "smoke": TestPreset("smoke", 1, 2, 2, 1000),
    "single": TestPreset("single", 15, None, None, None),
    "full": TestPreset("full", 15, None, None, None),
}

REGULARIZATION_CONFIGS = (
    RegularizationConfig("A", "Config A - dropout=0.0, wd=0", 0.0, 0.0),
    RegularizationConfig("B", "Config B - dropout=0.1, wd=1e-4", 0.1, 1e-4),
    RegularizationConfig("C", "Config C - dropout=0.2, wd=1e-4", 0.2, 1e-4),
    RegularizationConfig("D", "Config D - dropout=0.3, wd=1e-3 (current baseline)", 0.3, 1e-3),
    RegularizationConfig("E", "Config E - dropout=0.4, wd=1e-3", 0.4, 1e-3),
    RegularizationConfig("F", "Config F - dropout=0.5, wd=1e-3", 0.5, 1e-3),
    RegularizationConfig("G", "Config G - dropout=0.2, wd=1e-3", 0.2, 1e-3),
    RegularizationConfig("H", "Config H - dropout=0.3, wd=1e-4", 0.3, 1e-4),
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
# ONE REGULARIZATION CONFIGURATION ACROSS ALL LOUO FOLDS
# =============================================================================

def run_regularization_experiment(
    reg_config: RegularizationConfig,
    trials: List["e4.tp.TrialData"],
    fold_surgeons: List[str],
    base_config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_models_dir = output_dir / "fold_models"
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    config = e4.KinematicsOnlyConfig(
        input_csv=base_config.input_csv,
        output_dir=output_dir,
        kinematic_source=base_config.kinematic_source,
        sample_rate=base_config.sample_rate,
        window_seconds=base_config.window_seconds,
        stride_samples=base_config.stride_samples,
        batch_size=base_config.batch_size,
        epochs=base_config.epochs,
        dropout=reg_config.dropout,
        weight_decay=reg_config.weight_decay,
        early_stopping_patience=base_config.early_stopping_patience,
        early_stopping_metric=base_config.early_stopping_metric,
        standardize=base_config.standardize,
        random_seed=base_config.random_seed,
        device=base_config.device,
        max_windows=base_config.max_windows,
    )

    print()
    print("=" * 78)
    print(f"[CONFIG] {reg_config.label}")
    print("=" * 78)
    print(f"[CONFIG] Dropout: {reg_config.dropout}")
    print(f"[CONFIG] Weight decay: {reg_config.weight_decay}")
    print("[CONFIG] Kinematic source: PSM")
    print(f"[CONFIG] Window: {WINDOW_SECONDS} sec / {WINDOW_FRAMES} frames")
    print(f"[CONFIG] Standardisation: {'ON' if STANDARDIZE else 'OFF'}")
    print("[CONFIG] Loss: unweighted CrossEntropyLoss")
    print(f"[CONFIG] Random seed: {RANDOM_SEED}")
    print("[CHECK] Previous gesture supplied as input: NO")
    print("[CHECK] Teacher forcing: NO")
    print("[CHECK] Autoregressive label feedback: NO")

    fold_results: List[Dict[str, object]] = []
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

        print()
        print(f"[FOLD] {reg_config.label} | Fold {fold_number}/{len(fold_surgeons)} | Held-out: {held_out_surgeon}")

        model, metrics, _history, mean, std, fold_summary = e4.train_fold(
            train_trials=train_trials,
            test_trials=test_trials,
            config=config,
            device=device,
            run_name=f"{reg_config.name}_LOUO_{held_out_surgeon}",
            held_out_surgeon=held_out_surgeon,
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
                        "configuration": reg_config.name,
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
        print(f"[RUNTIME] {reg_config.label} elapsed: {format_duration(experiment_elapsed)}")
        print(f"[RUNTIME] Estimated remaining (this configuration): {format_duration(eta)}")

        fold_results.append(
            {
                "held_out_surgeon": held_out_surgeon,
                "configuration": reg_config.name,
                "dropout": reg_config.dropout,
                "weight_decay": reg_config.weight_decay,
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
        "configuration": reg_config.name,
        "dropout": reg_config.dropout,
        "weight_decay": reg_config.weight_decay,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "std_macro_f1": float(np.std(macro_f1_values)),
        "folds": fold_results,
        "total_runtime_seconds": total_runtime,
    }

    with (output_dir / "kinematics_only_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(method_metrics, file, indent=2)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(output_dir / "kinematics_only_predictions.csv", index=False)

    print()
    print(f"[DONE] {reg_config.label} complete. Total runtime: {format_duration(total_runtime)}")

    return method_metrics


# =============================================================================
# CROSS-CONFIGURATION SUMMARY / BY-SURGEON OUTPUTS
# =============================================================================

def build_summary_rows(all_config_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []

    for config_metrics in all_config_metrics:
        fold_results = config_metrics["folds"]
        best_epochs = [f["best_epoch"] for f in fold_results if f.get("best_epoch") is not None]
        train_accuracy_at_best = [
            f["train_accuracy_at_best_epoch"] for f in fold_results
            if f.get("train_accuracy_at_best_epoch") is not None
        ]
        validation_accuracy_at_best = [
            f["validation_accuracy_at_best_epoch"] for f in fold_results
            if f.get("validation_accuracy_at_best_epoch") is not None
        ]

        rows.append(
            {
                "configuration": config_metrics["configuration"],
                "dropout": config_metrics["dropout"],
                "weight_decay": config_metrics["weight_decay"],
                "mean_louo_accuracy": config_metrics["mean_accuracy"],
                "std_louo_accuracy": config_metrics["std_accuracy"],
                "mean_louo_macro_f1": config_metrics["mean_macro_f1"],
                "std_louo_macro_f1": config_metrics["std_macro_f1"],
                "mean_training_accuracy_at_best_epoch": (
                    float(np.mean(train_accuracy_at_best)) if train_accuracy_at_best else None
                ),
                "mean_validation_accuracy_at_best_epoch": (
                    float(np.mean(validation_accuracy_at_best)) if validation_accuracy_at_best else None
                ),
                "mean_best_epoch": float(np.mean(best_epochs)) if best_epochs else None,
                "total_runtime_seconds": config_metrics["total_runtime_seconds"],
            }
        )

    baseline_row = next(row for row in rows if row["configuration"] == BASELINE_CONFIG_NAME)
    for row in rows:
        row["change_in_accuracy_vs_current_baseline"] = row["mean_louo_accuracy"] - baseline_row["mean_louo_accuracy"]
        row["change_in_macro_f1_vs_current_baseline"] = row["mean_louo_macro_f1"] - baseline_row["mean_louo_macro_f1"]

    rows.sort(key=lambda row: float(row["mean_louo_macro_f1"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def build_by_surgeon_rows(all_config_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for config_metrics in all_config_metrics:
        for fold in config_metrics["folds"]:
            rows.append(
                {
                    "configuration": fold["configuration"],
                    "dropout": fold["dropout"],
                    "weight_decay": fold["weight_decay"],
                    "held_out_surgeon": fold["held_out_surgeon"],
                    "accuracy": fold["accuracy"],
                    "macro_f1": fold["macro_f1"],
                    "best_epoch": fold["best_epoch"],
                    "training_accuracy_at_best_epoch": fold["train_accuracy_at_best_epoch"],
                    "runtime": fold["runtime_seconds"],
                }
            )
    rows.sort(key=lambda row: (row["held_out_surgeon"], row["configuration"]))
    return rows


# =============================================================================
# PIPELINE
# =============================================================================

def run_experiments(args: argparse.Namespace) -> Path:
    preset = PRESETS[args.mode]
    output_root = Path(args.output_root).resolve()
    prepared_data_dir = output_root / "prepared_data"
    experiment_root = output_root / "regularization_experiments"
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

    if args.mode == "smoke":
        print()
        print("[SMOKE] Smoke-test mode enabled: max_folds<=2, epochs=1, max_windows<=1000.")
        print("[SMOKE] Smoke-test results are NOT scientifically meaningful.")

    base_config = e4.KinematicsOnlyConfig(
        input_csv=frame_file,
        output_dir=experiment_root,
        kinematic_source=KINEMATIC_SOURCE,
        sample_rate=SAMPLE_RATE,
        window_seconds=WINDOW_SECONDS,
        stride_samples=STRIDE_SAMPLES,
        batch_size=BATCH_SIZE,
        epochs=preset.epochs,
        dropout=0.3,
        weight_decay=1e-3,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_metric=EARLY_STOPPING_METRIC,
        standardize=STANDARDIZE,
        random_seed=RANDOM_SEED,
        device=args.device,
        max_windows=max_windows,
    )

    print(
        f"Running exactly {len(REGULARIZATION_CONFIGS)} dropout/weight-decay "
        f"configurations with seed {RANDOM_SEED} against the same prepared "
        f"dataset and the same {len(fold_surgeons)} LOUO fold(s) (data "
        f"preparation is not rerun between configurations). Dropout and "
        f"weight decay are the only intended independent variables."
    )

    pipeline_start = time.perf_counter()

    all_config_metrics = []

    for reg_config in REGULARIZATION_CONFIGS:
        config_dir = experiment_root / reg_config.name
        config_metrics = run_regularization_experiment(
            reg_config=reg_config,
            trials=trials,
            fold_surgeons=fold_surgeons,
            base_config=base_config,
            device=device,
            output_dir=config_dir,
        )
        all_config_metrics.append(config_metrics)

    summary_rows = build_summary_rows(all_config_metrics)
    by_surgeon_rows = build_by_surgeon_rows(all_config_metrics)

    summary_path = output_root / "kinematics_only_regularization_summary.csv"
    summary_fieldnames = [
        "rank", "dropout", "weight_decay",
        "mean_louo_accuracy", "std_louo_accuracy",
        "mean_louo_macro_f1", "std_louo_macro_f1",
        "mean_training_accuracy_at_best_epoch",
        "mean_validation_accuracy_at_best_epoch",
        "mean_best_epoch", "total_runtime_seconds",
        "change_in_accuracy_vs_current_baseline",
        "change_in_macro_f1_vs_current_baseline",
    ]
    pd.DataFrame(summary_rows)[summary_fieldnames].to_csv(summary_path, index=False)

    by_surgeon_path = output_root / "kinematics_only_regularization_by_surgeon.csv"
    pd.DataFrame(by_surgeon_rows).to_csv(by_surgeon_path, index=False)

    total_runtime = time.perf_counter() - pipeline_start

    print()
    print("=" * 79)
    print("[DONE] KINEMATICS-ONLY REGULARIZATION EXPERIMENT COMPLETE")
    print("=" * 79)
    print(f"Complete experiment runtime: {format_duration(total_runtime)}")
    print(f"Summary written to {summary_path}")
    print(f"Per-surgeon comparison written to {by_surgeon_path}")
    print(f"Per-configuration held-out predictions written under {experiment_root}/<config>/kinematics_only_predictions.csv")
    print(f"Per-fold checkpoints written under {experiment_root}/<config>/fold_models/")

    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled PSM kinematics-only Transformer dropout/"
            "weight-decay retuning experiment (experiment_runner_4.py/"
            "experiment_runner_8.py helpers, not Train_PyTorch.py). Dropout "
            "and weight decay are the only intended independent variables."
        )
    )
    parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
    parser.add_argument("--kinematics-dir", action="append", required=True)
    parser.add_argument("--annotations-dir", action="append", required=True)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs_pytorch_kinematics_only_regularization"),
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
