"""
ATARI-2: Gesture Classification / run_all_pytorch.py

===============================================================================
PURPOSE
===============================================================================

This script controls the complete ATARI-2 PyTorch gesture-recognition pipeline.

It is the PyTorch equivalent of the existing run_all.py script.

The pipeline is:

    JIGSAWS raw kinematics + gesture annotations
                        |
                        v
                  data_prep.py
                        |
                        v
              all_frame_level.csv
                        |
                        v
               Train_PyTorch.py
                        |
                        v
             pytorch_gesture_model.pt
                        |
                        v
              predict_gestures.py
                        |
                        v
            predicted gesture segments


===============================================================================
WHY A SEPARATE PYTORCH RUNNER IS USED
===============================================================================

The XGBoost and Transformer models use the data differently.

XGBoost:
    engineered window features
        -> all_window_features.csv
        -> Train_xgboost.py

PyTorch Transformer:
    raw frame-by-frame kinematic sequences
        -> all_frame_level.csv
        -> Train_PyTorch.py

Keeping run_all_pytorch.py separate while developing the Transformer allows the
existing XGBoost pipeline to remain unchanged and provides a simple way to test
the new model.

Once both pipelines are stable, they could later be combined into one runner
with a command such as:

    --model xgboost

or:

    --model pytorch


===============================================================================
THREE TEST MODES
===============================================================================

This script provides three predefined modes.

-------------------------------------------------------------------------------
1. SMOKE TEST
-------------------------------------------------------------------------------

    --mode smoke

Purpose:
    Check that the complete software and hardware pipeline actually works.

It deliberately uses a very small workload:

    maximum source trials per dataset = 2
    LOUO folds                       = 2
    epochs                           = 1
    windows per train/test dataset   = 1000

This test is NOT intended to measure model accuracy.

It checks:

    - data_prep.py runs
    - frame-level CSV is generated
    - PyTorch imports correctly
    - DataLoader works
    - Transformer forward pass works
    - backpropagation works
    - optimizer works
    - LOUO splitting works
    - checkpoint saving works
    - prediction works
    - CPU/GPU device detection works
    - progress and ETA reporting works

This should always be run first.


-------------------------------------------------------------------------------
2. SINGLE-PROCEDURE FULL TEST
-------------------------------------------------------------------------------

    --mode single

Purpose:
    Train and evaluate the full Transformer using one procedure only.

Example:

    Knot Tying

This uses:

    all available Knot Tying trials
    all available Knot Tying surgeons
    full LOUO evaluation
    15 epochs
    paper-style 30-frame windows
    1-frame sequence stride

This gives a meaningful test of:

    - training runtime
    - model convergence
    - LOUO performance
    - CPU/GPU practicality

before attempting the much larger combined dataset.


-------------------------------------------------------------------------------
3. FULL MULTI-TASK LOUO TEST
-------------------------------------------------------------------------------

    --mode full

Purpose:
    Train one gesture-recognition Transformer using:

        Knot Tying
        Suturing
        Needle Passing

The trials are combined before training.

LOUO then holds out an entire surgeon.

For example:

    training:
        all trials from B,C,D,E,F,G,H

    testing:
        every available trial from surgeon I
        across all supplied procedures

This is the most computationally expensive experiment and is the one most
relevant to testing whether the gesture model generalises to an unseen surgeon.


===============================================================================
PYTORCH DATA PIPELINE
===============================================================================

Unlike XGBoost, Train_PyTorch.py reads:

    all_frame_level.csv

and NOT:

    all_window_features.csv

The Transformer internally creates its own overlapping 30-frame sequences.

This is important because the Transformer needs the original temporal structure
of the 38-dimensional MTM or PSM kinematics.

The default is:

    --kinematic-source mtm

because the paper reports its best gesture-recognition performance using the
MTM kinematics.


===============================================================================
OUTPUT STRUCTURE
===============================================================================

By default:

    ATARI-2/
        outputs_pytorch/

            prepared_data/
                all_frame_level.csv
                ...

            pytorch_model/
                pytorch_gesture_model.pt
                pytorch_config.json
                pytorch_metrics.json
                pytorch_training_history.csv
                pytorch_kinematic_columns.json

            predictions/
                <trial>_predicted_segments.txt
                <trial>_predicted_segments.csv
                ...


===============================================================================
IMPORTANT DEPENDENCY
===============================================================================

predict_gestures.py must eventually support:

    --model-type pytorch

and must understand the checkpoint:

    pytorch_gesture_model.pt

Until that modification has been made, use:

    --skip-prediction

to test preprocessing and PyTorch training without running Stage 3.

This is useful because Train_PyTorch.py can be tested before the prediction
script has been modified.


===============================================================================
EXAMPLE: SMOKE TEST
===============================================================================

python "Gesture Classification/run_all_pytorch.py" `
    --mode smoke `
    --kinematics-dir "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures" `
    --annotations-dir "JIGSAW/Knot_Tying/Knot_Tying/transcriptions" `
    --predict-file "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures/Knot_Tying_B001.txt" `


===============================================================================
EXAMPLE: FULL KNOT TYING
===============================================================================

python "Gesture Classification/run_all_pytorch.py" `
    --mode single `
    --kinematics-dir "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures" `
    --annotations-dir "JIGSAW/Knot_Tying/Knot_Tying/transcriptions" `
    --predict-file "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures/Knot_Tying_B001.txt" `



===============================================================================
EXAMPLE: ALL THREE PROCEDURES
===============================================================================

python "Gesture Classification/run_all_pytorch.py" `
    --mode full `
    --kinematics-dir "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures" `
    --annotations-dir "JIGSAW/Knot_Tying/Knot_Tying/transcriptions" `
    --kinematics-dir "JIGSAW/Suturing/Suturing/Suturing kinematics/AllGestures" `
    --annotations-dir "JIGSAW/Suturing/Suturing/transcriptions" `
    --kinematics-dir "JIGSAW/Needle_Passing/Needle_Passing/Needle_Passing kinematics/AllGestures" `
    --annotations-dir "JIGSAW/Needle_Passing/Needle_Passing/transcriptions" `
    --predict-file "JIGSAW/Suturing/Suturing/Suturing kinematics/AllGestures/Suturing_B001.txt" `


"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PREP_SCRIPT = PROJECT_ROOT / "Gesture Data Manipulation" / "data_prep.py"
TRAIN_SCRIPT = SCRIPT_DIR / "Train_PyTorch.py"
PREDICT_SCRIPT = SCRIPT_DIR / "predict_gestures.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CUDA_VENV_PYTHON = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class TestPreset:
    name: str
    epochs: int
    max_trials: Optional[int]
    max_folds: Optional[int]
    max_windows: Optional[int]
    description: str


PRESETS = {
    "smoke": TestPreset("smoke", 1, 2, 2, 1000, "Small integration test."),
    "single": TestPreset("single", 15, None, None, None, "One-procedure LOUO."),
    "full": TestPreset("full", 15, None, None, None, "Multi-procedure LOUO."),
}


def resolve_python_executable() -> str:
    override = os.environ.get("ATARI_PYTHON")
    if override and Path(override).exists():
        return override
    for candidate in (CUDA_VENV_PYTHON, VENV_PYTHON):
        if candidate.exists():
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


def require_script(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required script was not found: {path}")


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"Expected {description}: {path}")


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find {description}: {path}")


def run_command(command: List[str], stage_name: str) -> None:
    print(f"[STARTING] {stage_name}")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_data_preparation(
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
    run_command(command, "STAGE 1: PyTorch data preparation")
    frame_file = prepared_data_dir / "all_frame_level.csv"
    require_file(frame_file, "combined frame-level dataset")
    return frame_file


def run_pytorch_training(
    frame_file: Path,
    model_dir: Path,
    preset: TestPreset,
    kinematic_source: str,
    sample_rate: float,
    window_seconds: float,
    stride_samples: int,
    batch_size: int,
    dropout: float,
    weight_decay: float,
    early_stopping_patience: int,
    early_stopping_metric: str,
    device: str,
    standardize: bool,
    save_fold_models: bool,
) -> Path:
    command = [
        PYTHON_EXECUTABLE, "-u", str(TRAIN_SCRIPT),
        "--input-csv", str(frame_file),
        "--output-dir", str(model_dir),
        "--kinematic-source", kinematic_source,
        "--sample-rate", str(sample_rate),
        "--window-seconds", str(window_seconds),
        "--stride-samples", str(stride_samples),
        "--batch-size", str(batch_size),
        "--dropout", str(dropout),
        "--weight-decay", str(weight_decay),
        "--early-stopping-patience", str(early_stopping_patience),
        "--early-stopping-metric", early_stopping_metric,
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
    run_command(command, "STAGE 2: PyTorch Transformer training")
    model_file = model_dir / "pytorch_gesture_model.pt"
    require_file(model_file, "trained PyTorch gesture model")
    return model_file


def run_pytorch_prediction(
    predict_file: Path,
    model_dir: Path,
    prediction_dir: Path,
    sample_rate: float,
    window_seconds: float,
) -> None:
    command = [
        PYTHON_EXECUTABLE, "-u", str(PREDICT_SCRIPT),
        "--kinematics", str(predict_file),
        "--model-dir", str(model_dir),
        "--model-type", "pytorch",
        "--output-dir", str(prediction_dir),
        "--sample-rate", str(sample_rate),
        "--window-seconds", str(window_seconds),
    ]
    run_command(command, "STAGE 3: PyTorch gesture prediction")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ATARI-2 PyTorch pipeline.")
    parser.add_argument("--mode", choices=["smoke", "single", "full"], required=True)
    parser.add_argument("--kinematics-dir", action="append", required=True)
    parser.add_argument("--annotations-dir", action="append", required=True)
    parser.add_argument("--predict-file")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs_pytorch"))
    parser.add_argument("--kinematic-source", choices=["mtm", "psm"], default="mtm")
    parser.add_argument("--sample-rate", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-metric", choices=["macro_f1", "accuracy"], default="macro_f1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--reuse-prepared-data", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    preset = PRESETS[args.mode]
    kinematics_dirs = [Path(path).resolve() for path in args.kinematics_dir]
    annotations_dirs = [Path(path).resolve() for path in args.annotations_dir]
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

    output_root = Path(args.output_root).resolve()
    prepared_data_dir = output_root / "prepared_data"
    model_dir = output_root / "pytorch_model"
    prediction_dir = output_root / "predictions"
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_prepared_data:
        frame_file = prepared_data_dir / "all_frame_level.csv"
        require_file(frame_file, "existing frame-level dataset")
    else:
        frame_file = run_data_preparation(
            kinematics_dirs, annotations_dirs, prepared_data_dir,
            args.sample_rate, preset.max_trials,
        )

    model_file = run_pytorch_training(
        frame_file=frame_file,
        model_dir=model_dir,
        preset=preset,
        kinematic_source=args.kinematic_source,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_samples=args.stride_samples,
        batch_size=args.batch_size,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        device=args.device,
        standardize=args.standardize,
        save_fold_models=args.save_fold_models,
    )

    if args.skip_prediction:
        print("[SKIP] Stage 3 prediction skipped.")
        return
    if args.predict_file is None:
        raise ValueError("--predict-file is required unless --skip-prediction is supplied.")
    predict_file = Path(args.predict_file).resolve()
    require_file(predict_file, "prediction kinematic file")
    require_script(PREDICT_SCRIPT)
    run_pytorch_prediction(
        predict_file, model_dir, prediction_dir,
        args.sample_rate, args.window_seconds,
    )
    print(f"Final PyTorch model: {model_file}")


if __name__ == "__main__":
    main()
