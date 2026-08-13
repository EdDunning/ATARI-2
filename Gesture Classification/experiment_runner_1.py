"""
ATARI-2 regularization experiment runner

This script runs four Train_PyTorch.py experiments that differ only in dropout
and AdamW weight decay:

    1. dropout=0.1, weight_decay=0
    2. dropout=0.2, weight_decay=1e-4
    3. dropout=0.3, weight_decay=1e-4
    4. dropout=0.3, weight_decay=1e-3

All experiments use the same prepared dataset, random seed, LOUO fold limit,
window settings, batch size, early-stopping settings, and maximum epoch count.
Each experiment writes to its own directory under:

    <output-root>/regularization_experiments/

The final ranked summary is written to:

    <output-root>/regularization_experiment_summary.csv

Run a smoke experiment grid with:

    python "Gesture Classification/experiment_runner_1.py" `
        --mode smoke `
        --kinematics-dir "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures" `
        --annotations-dir "JIGSAW/Knot_Tying/Knot_Tying/transcriptions"

single procedure run with:

    python "Gesture Classification/experiment_runner_1.py" `
        --mode single `
        --kinematics-dir "...Knot_Tying...\AllGestures" `
        --annotations-dir "...Knot_Tying...\transcriptions" `
        --kinematics-dir "...Suturing...\AllGestures" `
        --annotations-dir "...Suturing...\transcriptions" `
        --kinematics-dir "...Needle_Passing...\AllGestures" `
        --annotations-dir "...Needle_Passing...\transcriptions"

Whole dataset run with:

python ".\Gesture Classification\experiment_runner_1.py" `
    --mode full `
    --kinematics-dir ".\JIGSAW\Knot_Tying\Knot_Tying\Knot_Tying kinematics\AllGestures" `
    --annotations-dir ".\JIGSAW\Knot_Tying\Knot_Tying\transcriptions" `
    --kinematics-dir ".\JIGSAW\Suturing\Suturing\Suturing kinematics\AllGestures" `
    --annotations-dir ".\JIGSAW\Suturing\Suturing\transcriptions" `
    --kinematics-dir ".\JIGSAW\Needle_Passing\Needle_Passing\Needle_Passing kinematics\AllGestures" `
    --annotations-dir ".\JIGSAW\Needle_Passing\Needle_Passing\transcriptions" `
    --output-root ".\outputs_pytorch_experiments" `
    --device auto

For a full single-procedure or multi-procedure run, use `--mode single` or
`--mode full` and provide the matching directory pairs. Use `--reuse-prepared-data`
to reuse an existing all_frame_level.csv under the selected output root.
Prediction is not run by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PREP_SCRIPT = PROJECT_ROOT / "Gesture Data Manipulation" / "data_prep.py"
TRAIN_SCRIPT = SCRIPT_DIR / "Train_PyTorch.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CUDA_VENV_PYTHON = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"
EXPERIMENT_RANDOM_SEED = 42


@dataclass(frozen=True)
class TestPreset:
    name: str
    epochs: int
    max_trials: Optional[int]
    max_folds: Optional[int]
    max_windows: Optional[int]
    description: str


@dataclass(frozen=True)
class RegularizationExperiment:
    name: str
    dropout: float
    weight_decay: float


PRESETS = {
    "smoke": TestPreset(
        name="smoke",
        epochs=1,
        max_trials=2,
        max_folds=2,
        max_windows=1000,
        description="Small integration test.",
    ),
    "single": TestPreset(
        name="single",
        epochs=15,
        max_trials=None,
        max_folds=None,
        max_windows=None,
        description="Full LOUO experiment for one procedure.",
    ),
    "full": TestPreset(
        name="full",
        epochs=15,
        max_trials=None,
        max_folds=None,
        max_windows=None,
        description="Full multi-procedure LOUO experiment.",
    ),
}

REGULARIZATION_EXPERIMENTS = (
    RegularizationExperiment("dropout_0.1_weight_decay_0", 0.1, 0.0),
    RegularizationExperiment("dropout_0.2_weight_decay_1e-4", 0.2, 1e-4),
    RegularizationExperiment("dropout_0.3_weight_decay_1e-4", 0.3, 1e-4),
    RegularizationExperiment("dropout_0.3_weight_decay_1e-3", 0.3, 1e-3),
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
        except Exception:
            continue
        if result.returncode == 0:
            return str(candidate)

    return sys.executable


PYTHON_EXECUTABLE = resolve_python_executable()


def separator() -> None:
    print()
    print("=" * 79)
    print()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    return f"{int(round(max(0, seconds)))} sec"


def require_script(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required script was not found: {path}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"Expected {description}: {path}")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find {description}: {path}")


def run_command(command: List[str], stage_name: str) -> float:
    separator()
    print(f"[STARTING] {stage_name}")
    print(f"Started: {timestamp()}")
    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    start = time.perf_counter()
    result = subprocess.run(command)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"[ERROR] {stage_name} failed with return code {result.returncode}.")
        raise SystemExit(result.returncode)

    print(f"[SUCCESS] {stage_name} ({format_duration(elapsed)})")
    return elapsed


def prepare_data(
    kinematics_dirs: List[Path],
    annotations_dirs: List[Path],
    prepared_data_dir: Path,
    sample_rate: float,
    max_trials: Optional[int],
) -> Tuple[Path, float]:
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

    elapsed = run_command(command, "data preparation")
    frame_file = prepared_data_dir / "all_frame_level.csv"
    require_file(frame_file, "frame-level dataset")
    return frame_file, elapsed


def run_training(
    frame_file: Path,
    model_dir: Path,
    preset: TestPreset,
    kinematic_source: str,
    sample_rate: float,
    window_seconds: float,
    stride_samples: int,
    batch_size: int,
    experiment: RegularizationExperiment,
    early_stopping_patience: int,
    early_stopping_metric: str,
    device: str,
    standardize: bool,
    save_fold_models: bool,
) -> float:
    command = [
        PYTHON_EXECUTABLE, "-u", str(TRAIN_SCRIPT),
        "--input-csv", str(frame_file),
        "--output-dir", str(model_dir),
        "--kinematic-source", kinematic_source,
        "--sample-rate", str(sample_rate),
        "--window-seconds", str(window_seconds),
        "--stride-samples", str(stride_samples),
        "--batch-size", str(batch_size),
        "--dropout", str(experiment.dropout),
        "--weight-decay", str(experiment.weight_decay),
        "--early-stopping-patience", str(early_stopping_patience),
        "--early-stopping-metric", early_stopping_metric,
        "--random-seed", str(EXPERIMENT_RANDOM_SEED),
        "--epochs", str(preset.epochs),
        "--device", device,
    ]
    if preset.max_folds is not None:
        command.extend(["--max-folds", str(preset.max_folds)])
    if preset.max_windows is not None:
        command.extend(["--max-windows", str(preset.max_windows)])
    if standardize:
        command.append("--standardize")
    if save_fold_models:
        command.append("--save-fold-models")

    elapsed = run_command(command, f"training {experiment.name}")
    require_file(model_dir / "pytorch_metrics.json", "training metrics")
    require_file(model_dir / "pytorch_training_history.csv", "training history")
    return elapsed


def read_summary(
    model_dir: Path,
    experiment: RegularizationExperiment,
    early_stopping_metric: str,
) -> dict[str, object]:
    with (model_dir / "pytorch_metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    with (model_dir / "pytorch_training_history.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        history = list(csv.DictReader(file))

    cross_validation = metrics["cross_validation"]
    best_epochs = []
    for fold in cross_validation.get("folds", []):
        fold_name = f"LOUO_{fold['held_out_surgeon']}"
        rows = [
            row for row in history
            if row.get("run") == fold_name
            and row.get("early_stopping_value") not in (None, "")
        ]
        if rows:
            best_row = max(rows, key=lambda row: float(row["early_stopping_value"]))
            best_epochs.append(int(best_row["epoch"]))

    final_rows = [row for row in history if row.get("run") == "FINAL_ALL_USERS"]
    final_accuracy = None
    if final_rows:
        final_accuracy = float(final_rows[-1]["teacher_forced_train_accuracy"])

    return {
        "experiment": experiment.name,
        "dropout": experiment.dropout,
        "weight_decay": experiment.weight_decay,
        "early_stopping_metric": early_stopping_metric,
        "mean_louo_accuracy": cross_validation["mean_accuracy"],
        "mean_louo_macro_f1": cross_validation["mean_macro_f1"],
        "best_validation_epoch": (
            sum(best_epochs) / len(best_epochs) if best_epochs else None
        ),
        "total_runtime_seconds": metrics["total_runtime_seconds"],
        "final_teacher_forced_training_accuracy": final_accuracy,
    }


def run_experiments(args: argparse.Namespace) -> Path:
    preset = PRESETS[args.mode]
    output_root = Path(args.output_root).resolve()
    prepared_data_dir = output_root / "prepared_data"
    experiment_root = output_root / "regularization_experiments"
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    if args.reuse_prepared_data:
        frame_file = prepared_data_dir / "all_frame_level.csv"
        require_file(frame_file, "existing frame-level dataset")
    else:
        frame_file, _ = prepare_data(
            [Path(path).resolve() for path in args.kinematics_dir],
            [Path(path).resolve() for path in args.annotations_dir],
            prepared_data_dir,
            args.sample_rate,
            preset.max_trials,
        )

    print(f"Running {len(REGULARIZATION_EXPERIMENTS)} experiments with seed {EXPERIMENT_RANDOM_SEED}.")
    rows = []
    for experiment in REGULARIZATION_EXPERIMENTS:
        model_dir = experiment_root / experiment.name
        run_training(
            frame_file=frame_file,
            model_dir=model_dir,
            preset=preset,
            kinematic_source=args.kinematic_source,
            sample_rate=args.sample_rate,
            window_seconds=args.window_seconds,
            stride_samples=args.stride_samples,
            batch_size=args.batch_size,
            experiment=experiment,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_metric=args.early_stopping_metric,
            device=args.device,
            standardize=args.standardize,
            save_fold_models=args.save_fold_models,
        )
        rows.append(read_summary(model_dir, experiment, args.early_stopping_metric))

    rows.sort(key=lambda row: float(row["mean_louo_macro_f1"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    summary_path = output_root / "regularization_experiment_summary.csv"
    fieldnames = [
        "rank", "experiment", "dropout", "weight_decay",
        "early_stopping_metric", "mean_louo_accuracy",
        "mean_louo_macro_f1", "best_validation_epoch",
        "total_runtime_seconds", "final_teacher_forced_training_accuracy",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to {summary_path}")
    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ATARI-2 PyTorch regularization experiment grid.")
    parser.add_argument("--mode", choices=["smoke", "single", "full"], required=True)
    parser.add_argument("--kinematics-dir", action="append", required=True)
    parser.add_argument("--annotations-dir", action="append", required=True)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs_pytorch_experiments"))
    parser.add_argument("--kinematic-source", choices=["mtm", "psm"], default="mtm")
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--reuse-prepared-data", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    kinematics_dirs = [Path(path) for path in args.kinematics_dir]
    annotations_dirs = [Path(path) for path in args.annotations_dir]
    if len(kinematics_dirs) != len(annotations_dirs):
        raise ValueError("The number of kinematics and annotation directories must match.")
    if args.mode == "single" and len(kinematics_dirs) != 1:
        raise ValueError("--mode single expects exactly one directory pair.")
    for path in kinematics_dirs:
        require_directory(path, "kinematic data directory")
    for path in annotations_dirs:
        require_directory(path, "annotation directory")
    require_script(DATA_PREP_SCRIPT)
    require_script(TRAIN_SCRIPT)
    run_experiments(args)


if __name__ == "__main__":
    main()
