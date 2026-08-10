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
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PREP_SCRIPT = (
    PROJECT_ROOT
    / "Gesture Data Manipulation"
    / "data_prep.py"
)

TRAIN_SCRIPT = (
    SCRIPT_DIR
    / "Train_PyTorch.py"
)

PREDICT_SCRIPT = (
    SCRIPT_DIR
    / "predict_gestures.py"
)

VENV_PYTHON = (
    PROJECT_ROOT
    / ".venv"
    / "Scripts"
    / "python.exe"
)


def resolve_python_executable() -> str:
    """
    Prefer the workspace .venv interpreter whenever it exists and can import
    the PyTorch runtime stack. Otherwise fall back to the interpreter that is
    currently running this launcher.
    """

    if VENV_PYTHON.exists():
        probe = [
            str(VENV_PYTHON),
            "-c",
            "import torch, torchvision, torchaudio, pandas; print('ok')",
        ]

        try:
            result = subprocess.run(
                probe,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return sys.executable

        if result.returncode == 0:
            return str(VENV_PYTHON)

    return sys.executable


PYTHON_EXECUTABLE = resolve_python_executable()


# =============================================================================
# TEST PRESETS
# =============================================================================

@dataclass(frozen=True)
class TestPreset:
    name: str

    epochs: int

    max_trials: Optional[int]
    max_folds: Optional[int]
    max_windows: Optional[int]

    description: str


PRESETS = {

    "smoke": TestPreset(
        name="smoke",
        epochs=1,
        max_trials=2,
        max_folds=2,
        max_windows=1000,
        description=(
            "Small integration and hardware test. "
            "Not intended for accuracy measurement."
        ),
    ),

    "single": TestPreset(
        name="single",
        epochs=15,
        max_trials=None,
        max_folds=None,
        max_windows=None,
        description=(
            "Full LOUO Transformer experiment using "
            "one supplied surgical procedure."
        ),
    ),

    "full": TestPreset(
        name="full",
        epochs=15,
        max_trials=None,
        max_folds=None,
        max_windows=None,
        description=(
            "Full multi-task LOUO Transformer experiment "
            "using all supplied surgical procedures."
        ),
    ),
}


# =============================================================================
# FORMATTING
# =============================================================================

def separator() -> None:
    print()
    print("=" * 79)
    print()


def format_duration(seconds: float) -> str:

    seconds = max(
        0,
        int(round(seconds)),
    )

    hours, remainder = divmod(
        seconds,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    parts: List[str] = []

    if hours:
        parts.append(
            f"{hours} hr"
        )

    if minutes or hours:
        parts.append(
            f"{minutes} min"
        )

    parts.append(
        f"{seconds} sec"
    )

    return " ".join(parts)


def timestamp() -> str:
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =============================================================================
# VALIDATION
# =============================================================================

def require_script(
    path: Path,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            "\nRequired script was not found:\n"
            f"{path}\n"
        )


def require_directory(
    path: Path,
    description: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find {description}:\n"
            f"{path}\n"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"\nExpected a directory for "
            f"{description}:\n"
            f"{path}\n"
        )


def require_file(
    path: Path,
    description: str,
) -> None:

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find {description}:\n"
            f"{path}\n"
        )


# =============================================================================
# SUBPROCESS EXECUTION
# =============================================================================

def run_command(
    command: List[str],
    stage_name: str,
) -> float:
    """
    Run one pipeline stage and return its elapsed runtime.

    Child Python processes are launched using -u so progress output appears
    immediately in the terminal.
    """

    separator()

    print(
        f"[STARTING] {stage_name}"
    )

    print(
        f"Started: {timestamp()}"
    )

    separator()

    print("Command:")

    print(
        " ".join(
            (
                f'"{part}"'
                if " " in part
                else part
            )
            for part in command
        )
    )

    print()

    start = time.perf_counter()

    result = subprocess.run(
        command
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if result.returncode != 0:

        separator()

        print(
            f"[ERROR] {stage_name} failed."
        )

        print(
            f"Return code: "
            f"{result.returncode}"
        )

        print(
            f"Elapsed: "
            f"{format_duration(elapsed)}"
        )

        print()

        print(
            "The pipeline has been stopped."
        )

        raise SystemExit(
            result.returncode
        )

    print()

    print(
        f"[SUCCESS] {stage_name}"
    )

    print(
        f"Finished: {timestamp()}"
    )

    print(
        f"Elapsed: "
        f"{format_duration(elapsed)}"
    )

    return elapsed


# =============================================================================
# STAGE 1 - DATA PREPARATION
# =============================================================================

def run_data_preparation(
    kinematics_dirs: List[Path],
    annotations_dirs: List[Path],
    prepared_data_dir: Path,
    sample_rate: float,
    max_trials: Optional[int],
) -> Tuple[Path, float]:

    command: List[str] = [
        PYTHON_EXECUTABLE,
        "-u",
        str(DATA_PREP_SCRIPT),
    ]

    for kin_dir, ann_dir in zip(
        kinematics_dirs,
        annotations_dirs,
    ):

        command.extend(
            [
                "--kinematics-dir",
                str(kin_dir),

                "--annotations-dir",
                str(ann_dir),
            ]
        )

    command.extend(
        [
            "--output-dir",
            str(prepared_data_dir),

            "--sample-rate",
            str(sample_rate),
        ]
    )

    if max_trials is not None:

        command.extend(
            [
                "--max-trials",
                str(max_trials),
            ]
        )

    elapsed = run_command(
        command,
        "STAGE 1: PyTorch data preparation",
    )

    frame_file = (
        prepared_data_dir
        / "all_frame_level.csv"
    )

    require_file(
        frame_file,
        "combined frame-level dataset",
    )

    print()
    print(
        "[FOUND] PyTorch frame-level dataset:"
    )
    print(frame_file)

    return (
        frame_file,
        elapsed,
    )


# =============================================================================
# STAGE 2 - PYTORCH TRAINING
# =============================================================================

def run_pytorch_training(
    frame_file: Path,
    model_dir: Path,
    preset: TestPreset,
    kinematic_source: str,
    sample_rate: float,
    window_seconds: float,
    stride_samples: int,
    batch_size: int,
    device: str,
    standardize: bool,
    save_fold_models: bool,
) -> Tuple[Path, float]:

    command: List[str] = [
        PYTHON_EXECUTABLE,
        "-u",
        str(TRAIN_SCRIPT),

        "--input-csv",
        str(frame_file),

        "--output-dir",
        str(model_dir),

        "--kinematic-source",
        kinematic_source,

        "--sample-rate",
        str(sample_rate),

        "--window-seconds",
        str(window_seconds),

        "--stride-samples",
        str(stride_samples),

        "--batch-size",
        str(batch_size),

        "--epochs",
        str(preset.epochs),

        "--device",
        device,
    ]

    if preset.max_folds is not None:

        command.extend(
            [
                "--max-folds",
                str(preset.max_folds),
            ]
        )

    if preset.max_windows is not None:

        command.extend(
            [
                "--max-windows",
                str(preset.max_windows),
            ]
        )

    if standardize:
        command.append(
            "--standardize"
        )

    if save_fold_models:
        command.append(
            "--save-fold-models"
        )

    elapsed = run_command(
        command,
        "STAGE 2: PyTorch Transformer training",
    )

    model_file = (
        model_dir
        / "pytorch_gesture_model.pt"
    )

    require_file(
        model_file,
        "trained PyTorch gesture model",
    )

    print()
    print(
        "[FOUND] PyTorch model:"
    )
    print(model_file)

    return (
        model_file,
        elapsed,
    )


# =============================================================================
# STAGE 3 - PYTORCH GESTURE PREDICTION
# =============================================================================

def run_pytorch_prediction(
    predict_file: Path,
    model_dir: Path,
    prediction_dir: Path,
    sample_rate: float,
    window_seconds: float,
) -> Tuple[Path, float]:

    """
    This assumes predict_gestures.py has been modified to support:

        --model-type pytorch

    and knows how to load:

        pytorch_gesture_model.pt
    """

    command = [
        PYTHON_EXECUTABLE,
        "-u",
        str(PREDICT_SCRIPT),

        "--kinematics",
        str(predict_file),

        "--model-dir",
        str(model_dir),

        "--model-type",
        "pytorch",

        "--output-dir",
        str(prediction_dir),

        "--sample-rate",
        str(sample_rate),

        "--window-seconds",
        str(window_seconds),
    ]

    elapsed = run_command(
        command,
        "STAGE 3: PyTorch gesture prediction and segmentation",
    )

    trial_id = (
        predict_file.stem
    )

    segment_file = (
        prediction_dir
        / (
            f"{trial_id}"
            "_predicted_segments.txt"
        )
    )

    require_file(
        segment_file,
        "predicted gesture transcription",
    )

    print()
    print(
        "[FOUND] Predicted gesture transcription:"
    )
    print(segment_file)

    return (
        segment_file,
        elapsed,
    )


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run ATARI-2 PyTorch Transformer gesture-recognition tests."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "single",
            "full",
        ],
        required=True,
        help=(
            "smoke = minimal pipeline test; "
            "single = full LOUO on one procedure; "
            "full = full LOUO on all supplied procedures."
        ),
    )

    parser.add_argument(
        "--kinematics-dir",
        action="append",
        required=True,
        help=(
            "Directory containing raw JIGSAWS kinematic files. "
            "Supply multiple times for multi-task training."
        ),
    )

    parser.add_argument(
        "--annotations-dir",
        action="append",
        required=True,
        help=(
            "Directory containing matching gesture transcription files. "
            "Supply multiple times for multi-task training."
        ),
    )

    parser.add_argument(
        "--predict-file",
        type=str,
        required=False,
        help=(
            "Raw kinematic file on which the final model should "
            "generate gesture predictions."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(
            PROJECT_ROOT
            / "outputs_pytorch"
        ),
    )

    parser.add_argument(
        "--kinematic-source",
        choices=[
            "mtm",
            "psm",
        ],
        default="mtm",
        help=(
            "38-dimensional robot stream used by the Transformer. "
            "Default is MTM."
        ),
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--stride-samples",
        type=int,
        default=1,
        help=(
            "Transformer sliding-window stride in raw frames. "
            "Paper-style default is 1."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "auto, cpu, cuda, cuda:0, mps, etc."
        ),
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
    )

    parser.add_argument(
        "--save-fold-models",
        action="store_true",
    )

    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help=(
            "Run preprocessing and training only. "
            "Useful until predict_gestures.py supports PyTorch."
        ),
    )

    parser.add_argument(
        "--reuse-prepared-data",
        action="store_true",
        help=(
            "Skip data_prep.py and reuse an existing "
            "all_frame_level.csv under the selected output root."
        ),
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = build_arg_parser()
    args = parser.parse_args()

    preset = PRESETS[
        args.mode
    ]

    # -------------------------------------------------------------------------
    # Convert paths
    # -------------------------------------------------------------------------

    kinematics_dirs = [
        Path(path).resolve()
        for path in args.kinematics_dir
    ]

    annotations_dirs = [
        Path(path).resolve()
        for path in args.annotations_dir
    ]

    if (
        len(kinematics_dirs)
        != len(annotations_dirs)
    ):
        raise ValueError(
            "The number of --kinematics-dir values must "
            "equal the number of --annotations-dir values."
        )

    predict_file: Optional[Path]

    if args.predict_file is not None:
        predict_file = Path(
            args.predict_file
        ).resolve()
    else:
        predict_file = None

    output_root = Path(
        args.output_root
    ).resolve()

    prepared_data_dir = (
        output_root
        / "prepared_data"
    )

    model_dir = (
        output_root
        / "pytorch_model"
    )

    prediction_dir = (
        output_root
        / "predictions"
    )

    # -------------------------------------------------------------------------
    # Validate scripts
    # -------------------------------------------------------------------------

    require_script(
        DATA_PREP_SCRIPT
    )

    require_script(
        TRAIN_SCRIPT
    )

    if not args.skip_prediction:
        require_script(
            PREDICT_SCRIPT
        )

    # -------------------------------------------------------------------------
    # Validate input folders
    # -------------------------------------------------------------------------

    for path in kinematics_dirs:
        require_directory(
            path,
            "kinematic data directory",
        )

    for path in annotations_dirs:
        require_directory(
            path,
            "gesture annotation directory",
        )

    if not args.skip_prediction:

        if predict_file is None:
            raise ValueError(
                "--predict-file is required unless "
                "--skip-prediction is supplied."
            )

        require_file(
            predict_file,
            "prediction kinematic file",
        )

    # -------------------------------------------------------------------------
    # Validate mode semantics
    # -------------------------------------------------------------------------

    if (
        args.mode == "single"
        and len(kinematics_dirs) != 1
    ):
        raise ValueError(
            "--mode single expects exactly one "
            "kinematics/annotation directory pair."
        )

    # -------------------------------------------------------------------------
    # Create outputs
    # -------------------------------------------------------------------------

    prepared_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Print configuration
    # -------------------------------------------------------------------------

    separator()

    print(
        "ATARI-2 PYTORCH GESTURE RECOGNITION PIPELINE"
    )

    separator()

    print(
        f"Mode: {preset.name}"
    )

    print(
        f"Description: {preset.description}"
    )

    print()

    print(
        f"Python: {PYTHON_EXECUTABLE}"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print()

    print(
        f"Kinematic source: "
        f"{args.kinematic_source.upper()}"
    )

    print(
        f"Device request: {args.device}"
    )

    print()

    print(
        "Training kinematic directories:"
    )

    for index, path in enumerate(
        kinematics_dirs,
        start=1,
    ):
        print(
            f"  {index}. {path}"
        )

    print()

    print(
        "Annotation directories:"
    )

    for index, path in enumerate(
        annotations_dirs,
        start=1,
    ):
        print(
            f"  {index}. {path}"
        )

    if predict_file is not None:
        print()
        print(
            f"Prediction file: "
            f"{predict_file}"
        )

    print()

    print(
        "Test settings:"
    )

    print(
        f"  Epochs: {preset.epochs}"
    )

    print(
        f"  Max trials per dataset: "
        f"{preset.max_trials}"
    )

    print(
        f"  Max LOUO folds: "
        f"{preset.max_folds}"
    )

    print(
        f"  Max windows: "
        f"{preset.max_windows}"
    )

    print(
        f"  Batch size: "
        f"{args.batch_size}"
    )

    print(
        f"  Window: "
        f"{args.window_seconds} sec"
    )

    print(
        f"  Sequence stride: "
        f"{args.stride_samples} frame(s)"
    )

    print()

    print(
        f"Output root: "
        f"{output_root}"
    )

    # -------------------------------------------------------------------------
    # Pipeline timer
    # -------------------------------------------------------------------------

    pipeline_start = (
        time.perf_counter()
    )

    stage1_elapsed = 0.0
    stage2_elapsed = 0.0
    stage3_elapsed = 0.0

    # -------------------------------------------------------------------------
    # STAGE 1
    # -------------------------------------------------------------------------

    frame_file = (
        prepared_data_dir
        / "all_frame_level.csv"
    )

    if args.reuse_prepared_data:

        print()
        print(
            "[SKIP] Reusing existing prepared frame-level data."
        )

        require_file(
            frame_file,
            "existing all_frame_level.csv",
        )

    else:

        (
            frame_file,
            stage1_elapsed,
        ) = run_data_preparation(
            kinematics_dirs=kinematics_dirs,
            annotations_dirs=annotations_dirs,
            prepared_data_dir=prepared_data_dir,
            sample_rate=args.sample_rate,
            max_trials=preset.max_trials,
        )

    # -------------------------------------------------------------------------
    # STAGE 2
    # -------------------------------------------------------------------------

    (
        model_file,
        stage2_elapsed,
    ) = run_pytorch_training(
        frame_file=frame_file,
        model_dir=model_dir,
        preset=preset,
        kinematic_source=args.kinematic_source,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_samples=args.stride_samples,
        batch_size=args.batch_size,
        device=args.device,
        standardize=args.standardize,
        save_fold_models=args.save_fold_models,
    )

    # -------------------------------------------------------------------------
    # STAGE 3
    # -------------------------------------------------------------------------

    segment_file: Optional[Path] = None

    if args.skip_prediction:

        print()
        print(
            "[SKIP] Stage 3 prediction skipped."
        )

    else:

        assert predict_file is not None

        (
            segment_file,
            stage3_elapsed,
        ) = run_pytorch_prediction(
            predict_file=predict_file,
            model_dir=model_dir,
            prediction_dir=prediction_dir,
            sample_rate=args.sample_rate,
            window_seconds=args.window_seconds,
        )

    # -------------------------------------------------------------------------
    # COMPLETE
    # -------------------------------------------------------------------------

    total_elapsed = (
        time.perf_counter()
        - pipeline_start
    )

    separator()

    print(
        "[COMPLETE] ATARI-2 PyTorch pipeline finished."
    )

    separator()

    print(
        "Runtime summary:"
    )

    print(
        "Stage 1 - Data preparation:       "
        f"{format_duration(stage1_elapsed)}"
    )

    print(
        "Stage 2 - PyTorch training:       "
        f"{format_duration(stage2_elapsed)}"
    )

    if args.skip_prediction:

        print(
            "Stage 3 - Gesture prediction:   SKIPPED"
        )

    else:

        print(
            "Stage 3 - Gesture prediction:   "
            f"{format_duration(stage3_elapsed)}"
        )

    print(
        "Total pipeline runtime:           "
        f"{format_duration(total_elapsed)}"
    )

    print()

    print(
        "Frame-level training data:"
    )
    print(
        frame_file
    )

    print()

    print(
        "Final PyTorch model:"
    )
    print(
        model_file
    )

    if segment_file is not None:

        print()

        print(
            "Predicted gesture transcription:"
        )

        print(
            segment_file
        )

    separator()


if __name__ == "__main__":
    main()