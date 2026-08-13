r"""
ATARI-2: run_all.py

Purpose
=======

This script runs the complete ATARI-2 gesture-recognition pipeline from one
command.

Instead of manually running:

    1. data_prep.py
    2. Train_xgboost.py
    3. predict_gestures.py

this script runs them automatically in the correct order and passes the output
of each stage into the next stage.

The pipeline is:

RAW JIGSAWS DATA
       |
       v
data_prep.py
       |
       | produces
       v
all_window_features.csv
       |
       v
Train_xgboost.py
       |
       | produces
       v
trained XGBoost model
       |
       v
predict_gestures.py
       |
       | produces
       v
predicted gesture segments
       |
       v
start_frame   end_frame   gesture


What each stage does
====================

STAGE 1 - DATA PREPARATION

`data_prep.py` reads:
    - raw JIGSAWS kinematic files
    - matching JIGSAWS gesture annotation files

It aligns the gesture annotations with the kinematic data and creates sliding
windows.

For each window, it calculates features such as:
    - mean
    - standard deviation
    - minimum
    - maximum
    - range
    - slope
    - signal energy
    - changes in the signal
    - jerk
    - frequency-domain features
    - coordination features between robot manipulators

It then creates:

    all_window_features.csv

This is the training dataset used by XGBoost.


STAGE 2 - MODEL TRAINING

`Train_xgboost.py` reads:

    all_window_features.csv

It trains an XGBoost classifier to predict surgical gestures from the engineered
kinematic features.

The trained model and supporting files are saved into:

    outputs/xgboost_model/

including:

    xgboost_gesture_model.json
    label_encoder.pkl
    feature_columns.json
    metrics.json
    cv_predictions.csv
    feature_importance.csv


STAGE 3 - GESTURE PREDICTION

`predict_gestures.py` loads the trained XGBoost model.

It then takes a raw kinematic file that was NOT manually annotated for the
prediction step.

It:
    - divides the recording into sliding windows
    - calculates exactly the same features used during training
    - predicts a gesture for each window
    - smooths the predictions
    - joins neighbouring predictions of the same gesture
    - creates start and end frame boundaries

For example, the final output may look like:

    80 219 G1
    220 370 G5
    371 590 G8
    591 660 G2

This has the same basic structure as a JIGSAWS gesture transcription file.


Why this script is useful
=========================

The scripts in this project are deliberately separate because each one has a
different responsibility.

However, during normal use it is inconvenient to type three long commands every
time.

This script acts as the pipeline controller.

It also checks that each stage actually produced the files required by the next
stage.

If something goes wrong, it stops immediately and reports which stage failed.


IMPORTANT: Training and prediction data
=======================================

During development, you can predict a recording that also exists in your
training dataset simply to check that the complete software pipeline works.

However, this is NOT a valid test of model generalisation.

For proper evaluation, the surgeon/trial being predicted should be excluded
from training.

That will be handled later with a proper leave-one-user-out or held-out test
pipeline.


Expected folder structure for the JIGSAWS data
==============================================

You should have one folder containing kinematic files:

    kinematics/
        Suturing_B001.txt
        Suturing_B002.txt
        Suturing_B003.txt
        ...

and another containing matching gesture annotation files:

    annotations/
        Suturing_B001.txt
        Suturing_B002.txt
        Suturing_B003.txt
        ...

The filenames must match.

For example:

    kinematics/Suturing_B001.txt

must correspond to:

    annotations/Suturing_B001.txt


Example command
===============

From the ATARI-2 root folder:

two-trial smoke test:
python "Gesture Classification\run_all.py" `
  --kinematics-dir "JIGSAW\Knot_Tying\Knot_Tying\Knot_Tying kinematics\AllGestures" `
  --annotations-dir "JIGSAW\Knot_Tying\Knot_Tying\transcriptions" `
  --predict-file "JIGSAW\Knot_Tying\Knot_Tying\Knot_Tying kinematics\AllGestures\Knot_Tying_B001.txt" `
  --max-trials 2 `
  --n-estimators 20 `
  --max-depth 3 `
  --stride-seconds 1.0

For training on only the knot tying task, and predicting on one knot tying recording:

python "Gesture Classification\run_all.py" `
  --kinematics-dir "JIGSAW\Knot_Tying\Knot_Tying\Knot_Tying kinematics\AllGestures" `
  --annotations-dir "JIGSAW\Knot_Tying\Knot_Tying\transcriptions" `
  --predict-file "JIGSAW\Knot_Tying\Knot_Tying\Knot_Tying kinematics\AllGestures\Knot_Tying_B001.txt" `
  --n-estimators 300 `
  --max-depth 6 `
  --stride-seconds 0.5

For training on all of the dataset, use: 

python "Gesture Classification\run_all.py" `
  --kinematics-dir "JIGSAW\Knot_Tying\Knot_Tying\Knot_Tying kinematics\AllGestures" `
  --annotations-dir "JIGSAW\Knot_Tying\Knot_Tying\transcriptions" `
  --kinematics-dir "JIGSAW\Suturing\Suturing\Suturing kinematics\AllGestures" `
  --annotations-dir "JIGSAW\Suturing\Suturing\transcriptions" `
  --kinematics-dir "JIGSAW\Needle_Passing\Needle_Passing\Needle_Passing kinematics\AllGestures" `
  --annotations-dir "JIGSAW\Needle_Passing\Needle_Passing\transcriptions" `
  --predict-file "JIGSAW\Suturing\Suturing\Suturing kinematics\AllGestures\Suturing_B001.txt" `
  --n-estimators 300 `
  --max-depth 6 `
  --stride-seconds 0.5

On Windows PowerShell, you can enter it on one line:

python run_all.py --kinematics-dir "C:\\path\\to\\kinematics" --annotations-dir "C:\\path\\to\\annotations" --predict-file "C:\\path\\to\\kinematics\\Suturing_B001.txt"


Outputs
=======

By default, everything is placed under:

    ATARI-2/outputs/

The structure will be:

    outputs/
        prepared_data/
            all_frame_level.csv
            all_window_features.csv
            ...

        xgboost_model/
            xgboost_gesture_model.json
            label_encoder.pkl
            feature_columns.json
            metrics.json
            cv_predictions.csv
            feature_importance.csv

        predictions/
            Suturing_B001_window_predictions.csv
            Suturing_B001_predicted_segments.csv
            Suturing_B001_predicted_segments.txt
            Suturing_B001_prediction_summary.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# run_all.py lives in the Gesture Classification folder, but the project root
# is one level above it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PREP_SCRIPT = (
    PROJECT_ROOT
    / "Gesture Data Manipulation"
    / "data_prep.py"
)

TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "Gesture Classification"
    / "Train_xgboost.py"
)

PREDICT_SCRIPT = (
    PROJECT_ROOT
    / "Gesture Classification"
    / "predict_gestures.py"
)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def print_separator() -> None:
    print("\n" + "=" * 75 + "\n")


def format_duration(seconds: float) -> str:
    """
    Return a human-readable duration string.
    """
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} second{'' if seconds == 1 else 's'}"

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {sec} sec"

    hours, minutes = divmod(minutes, 60)
    return f"{hours} hr {minutes} min {sec} sec"


def run_command(command: list[str], stage_name: str) -> float:
    """
    Run one stage of the pipeline.

    If the command fails, stop the entire pipeline immediately.
    """

    print_separator()
    print(f"[STARTING] {stage_name}")
    print_separator()

    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    print()

    start_time = time.time()
    result = subprocess.run(command)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print_separator()
        print(f"[ERROR] {stage_name} failed.")
        print(f"Return code: {result.returncode}")
        print()
        print(
            "The pipeline has been stopped because the next stage depends "
            "on this stage completing successfully."
        )

        raise SystemExit(result.returncode)

    print()
    print(f"[SUCCESS] {stage_name} completed.")
    print(f"Duration: {format_duration(elapsed)}")

    return elapsed


def require_file(path: Path, description: str) -> None:
    """
    Stop execution if an expected output file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nExpected {description} was not created:\n"
            f"{path}\n"
        )


def require_directory(path: Path, description: str) -> None:
    """
    Stop execution if a required input directory does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nCould not find {description}:\n"
            f"{path}\n"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"\nExpected a directory for {description}:\n"
            f"{path}\n"
        )


def require_script(path: Path) -> None:
    """
    Make sure one of the ATARI-2 scripts exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired ATARI-2 script not found:\n"
            f"{path}\n"
        )


# ---------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------

def run_data_preparation(
    kinematics_dirs: list[Path],
    annotations_dirs: list[Path],
    prepared_data_dir: Path,
    window_seconds: float,
    stride_seconds: float,
    sample_rate: float,
    max_trials: int | None = None,
) -> tuple[Path, float]:
    """
    Run data_prep.py and return the all_window_features.csv path and elapsed time.
    """

    command = [
        sys.executable,
        "-u",
        str(DATA_PREP_SCRIPT),
    ]

    for kinematics_dir, annotations_dir in zip(kinematics_dirs, annotations_dirs):
        command.extend([
            "--kinematics-dir",
            str(kinematics_dir),
            "--annotations-dir",
            str(annotations_dir),
        ])

    command.extend([
        "--output-dir",
        str(prepared_data_dir),

        "--sample-rate",
        str(sample_rate),

        "--window-seconds",
        str(window_seconds),

        "--stride-seconds",
        str(stride_seconds),
    ])

    if max_trials is not None:
        command.extend(["--max-trials", str(max_trials)])

    elapsed = run_command(
        command,
        stage_name="STAGE 1: Gesture data preparation",
    )

    feature_file = prepared_data_dir / "all_window_features.csv"

    require_file(
        feature_file,
        "combined window feature dataset",
    )

    print()
    print("[FOUND] Training feature file:")
    print(feature_file)

    return feature_file, elapsed


# ---------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------

def run_xgboost_training(
    feature_file: Path,
    model_dir: Path,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> tuple[Path, float]:
    """
    Train the XGBoost model.

    Returns the model directory and elapsed time.
    """

    command = [
        sys.executable,
        "-u",
        str(TRAIN_SCRIPT),

        "--input-csv",
        str(feature_file),

        "--output-dir",
        str(model_dir),

        "--n-estimators",
        str(n_estimators),

        "--learning-rate",
        str(learning_rate),

        "--max-depth",
        str(max_depth),
    ]

    elapsed = run_command(
        command,
        stage_name="STAGE 2: XGBoost gesture model training",
    )

    model_file = model_dir / "xgboost_gesture_model.json"
    encoder_file = model_dir / "label_encoder.pkl"
    features_file = model_dir / "feature_columns.json"

    require_file(
        model_file,
        "trained XGBoost model",
    )

    require_file(
        encoder_file,
        "gesture label encoder",
    )

    require_file(
        features_file,
        "feature column definition",
    )

    print()
    print("[FOUND] Trained model:")
    print(model_file)

    return model_dir, elapsed


# ---------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------

def run_gesture_prediction(
    predict_file: Path,
    model_dir: Path,
    prediction_dir: Path,
    window_seconds: float,
    stride_seconds: float,
    sample_rate: float,
    smoothing_window: int,
) -> tuple[Path, float]:
    """
    Run gesture prediction on one raw kinematic recording.

    Returns the predicted segment text file and elapsed time.
    """

    command = [
        sys.executable,
        "-u",
        str(PREDICT_SCRIPT),

        "--kinematics",
        str(predict_file),

        "--model-dir",
        str(model_dir),

        "--output-dir",
        str(prediction_dir),

        "--sample-rate",
        str(sample_rate),

        "--window-seconds",
        str(window_seconds),

        "--stride-seconds",
        str(stride_seconds),

        "--smoothing-window",
        str(smoothing_window),
    ]

    elapsed = run_command(
        command,
        stage_name="STAGE 3: Gesture prediction and segmentation",
    )

    trial_id = predict_file.stem

    segment_file = (
        prediction_dir
        / f"{trial_id}_predicted_segments.txt"
    )

    require_file(
        segment_file,
        "predicted gesture transcription",
    )

    print()
    print("[FOUND] Predicted gesture file:")
    print(segment_file)

    return segment_file, elapsed


# ---------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete ATARI-2 XGBoost gesture-recognition pipeline."
        )
    )

    # Required dataset locations

    parser.add_argument(
        "--kinematics-dir",
        type=str,
        action="append",
        required=True,
        help=(
            "Directory containing the JIGSAWS raw kinematic text files. "
            "Can be supplied multiple times to include multiple tasks."
        ),
    )

    parser.add_argument(
        "--annotations-dir",
        type=str,
        action="append",
        required=True,
        help=(
            "Directory containing the matching JIGSAWS gesture annotation files. "
            "Can be supplied multiple times to include multiple tasks."
        ),
    )

    parser.add_argument(
        "--predict-file",
        type=str,
        required=True,
        help=(
            "Raw kinematic file on which the trained model should predict gestures."
        ),
    )

    # Output

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help=(
            "Root directory for all generated files. "
            "Defaults to ATARI-2/outputs."
        ),
    )

    # Preprocessing parameters

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=30.0,
        help=(
            "Kinematic sampling rate in Hz. "
            "JIGSAWS uses 30 Hz."
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help=(
            "Sliding window duration in seconds."
        ),
    )

    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=0.5,
        help=(
            "Distance between the start of neighbouring windows."
        ),
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help=(
            "If set, pass the limit through to data_prep.py so at most this many "
            "matched trials are processed per dataset pair."
        ),
    )

    # XGBoost parameters

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help=(
            "Number of XGBoost trees."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help=(
            "XGBoost learning rate."
        ),
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help=(
            "Maximum XGBoost tree depth."
        ),
    )

    # Prediction parameters

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help=(
            "Number of neighbouring predictions used during "
            "majority-vote temporal smoothing."
        ),
    )

    return parser


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main() -> None:

    parser = build_arg_parser()
    args = parser.parse_args()
    pipeline_start_time = time.time()

    # -------------------------------------------------------------
    # Convert paths
    # -------------------------------------------------------------

    kinematics_dirs = [Path(p).resolve() for p in args.kinematics_dir]
    annotations_dirs = [Path(p).resolve() for p in args.annotations_dir]
    predict_file = Path(args.predict_file).resolve()
    output_root = Path(args.output_root).resolve()

    prepared_data_dir = output_root / "prepared_data"
    model_dir = output_root / "xgboost_model"
    prediction_dir = output_root / "predictions"

    # -------------------------------------------------------------
    # Initial checks
    # -------------------------------------------------------------

    print_separator()

    print("ATARI-2 GESTURE RECOGNITION PIPELINE")

    print_separator()

    print("Python:")
    print(sys.executable)

    print()
    print("ATARI-2 root:")
    print(PROJECT_ROOT)

    print()
    print("Kinematic dataset directories:")
    for dir_path in kinematics_dirs:
        print(dir_path)

    print()
    print("Gesture annotation directories:")
    for dir_path in annotations_dirs:
        print(dir_path)

    print()
    print("Prediction file:")
    print(predict_file)

    if args.max_trials is not None:
        print()
        print(f"Max trials per dataset pair: {args.max_trials}")

    print()
    print("Output directory:")
    print(output_root)

    # Make sure scripts exist.

    require_script(DATA_PREP_SCRIPT)
    require_script(TRAIN_SCRIPT)
    require_script(PREDICT_SCRIPT)

    if len(kinematics_dirs) != len(annotations_dirs):
        raise ValueError(
            "--kinematics-dir and --annotations-dir must be supplied the same number of times. "
            f"Found {len(kinematics_dirs)} kinematics and {len(annotations_dirs)} annotations directories."
        )

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

    require_file(
        predict_file,
        "kinematic prediction file",
    )

    # Create output folders.

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

    # -------------------------------------------------------------
    # STAGE 1
    # -------------------------------------------------------------

    feature_file, stage1_elapsed = run_data_preparation(
        kinematics_dirs=kinematics_dirs,
        annotations_dirs=annotations_dirs,
        prepared_data_dir=prepared_data_dir,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        sample_rate=args.sample_rate,
        max_trials=args.max_trials,
    )

    # -------------------------------------------------------------
    # STAGE 2
    # -------------------------------------------------------------

    trained_model_dir, stage2_elapsed = run_xgboost_training(
        feature_file=feature_file,
        model_dir=model_dir,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )

    # -------------------------------------------------------------
    # STAGE 3
    # -------------------------------------------------------------

    segment_file, stage3_elapsed = run_gesture_prediction(
        predict_file=predict_file,
        model_dir=trained_model_dir,
        prediction_dir=prediction_dir,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        sample_rate=args.sample_rate,
        smoothing_window=args.smoothing_window,
    )

    # -------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------

    print_separator()

    print("[COMPLETE] ATARI-2 gesture-recognition pipeline finished.")

    print()
    print("Prepared training data:")
    print(feature_file)

    print()
    print("Trained XGBoost model:")
    print(model_dir / "xgboost_gesture_model.json")

    print()
    print("Final predicted gesture transcription:")
    print(segment_file)

    total_elapsed = time.time() - pipeline_start_time
    print()
    print("Stage runtime summary:")
    print(f"Stage 1 - Data preparation:        {format_duration(stage1_elapsed)}")
    print(f"Stage 2 - XGBoost training:        {format_duration(stage2_elapsed)}")
    print(f"Stage 3 - Gesture prediction:       {format_duration(stage3_elapsed)}")
    print(f"Total pipeline runtime:            {format_duration(total_elapsed)}")

    print_separator()


if __name__ == "__main__":
    main()