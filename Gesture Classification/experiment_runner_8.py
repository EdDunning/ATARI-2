"""
ATARI-2: Gesture Classification / experiment_runner_8.py

===============================================================================
PURPOSE
===============================================================================

Diagnostic error analysis for the current best kinematics-only Transformer
configuration (Experiment 7, Experiment B - PSM):

    kinematics-only Transformer, PSM (k39-k76), 38 input features,
    standardisation ON, 1.5 s / 45-frame windows, stride 1, dropout 0.3,
    weight decay 1e-3, batch size 64, AdamW, max 15 epochs, early stopping
    patience 5 on macro F1, random seed 42, no previous-gesture input, no
    teacher forcing, no autoregressive label feedback.

===============================================================================
CAN EXPERIMENT 7's EXISTING OUTPUTS BE REUSED?
===============================================================================

No. experiment_runner_7.py only orchestrates experiment_runner_4.py, and
experiment_runner_4.py's run_pipeline():

    - deletes each LOUO fold's trained model immediately after that fold's
      aggregated evaluation ("del _fold_model"); no per-fold checkpoint is
      ever written to disk.
    - only writes AGGREGATED per-fold metrics to kinematics_only_metrics.json
      / kinematics_only_by_surgeon.csv (accuracy, macro F1, per-class
      precision/recall/F1/support, an aggregated confusion matrix). There is
      no per-frame trial_id / frame_idx / predicted_gesture / confidence
      record anywhere in Experiment 7's outputs.
    - the one saved checkpoint (kinematics_only_model.pt) is a FINAL model
      trained on ALL surgeons together, not a LOUO fold model, so it cannot
      be used to produce leakage-free held-out predictions for any surgeon.

This is a structural property of experiment_runner_4.py itself (which this
script must not modify), so no existing Experiment 7 run -- regardless of
whether one has actually been executed in this workspace -- can contain the
frame-level predictions this analysis requires. Retraining the PSM LOUO
experiment is therefore necessary. check_experiment7_reusable() below performs
this check explicitly and prints its reasoning before training begins.

===============================================================================
WHAT THIS SCRIPT DOES
===============================================================================

Reproduces the PSM LOUO experiment using the exact best configuration above,
reusing experiment_runner_4.py's KinematicsOnlyConfig/build_model/train_fold
(and the Train_PyTorch.py data-loading/LOUO/standardisation/audit helpers that
experiment_runner_4.py itself already reuses) rather than duplicating that
implementation. For every LOUO fold it then runs a frame-level inference pass
over the held-out surgeon's trials to recover one prediction + confidence per
frame (overlapping windows: the first window in a trial supplies predictions
for all of its frames, every subsequent window supplies only its final
frame's prediction -- with stride=1 this gives exactly one prediction per
frame with no gaps and no duplicated frame predictions).

This script does NOT modify Train_PyTorch.py, experiment_runner_4.py,
experiment_runner_5.py, experiment_runner_6.py, experiment_runner_7.py,
data_prep.py, or any previous result file.

===============================================================================
OUTPUTS
===============================================================================

    psm_per_gesture_metrics.csv
    psm_confusion_matrix_counts.csv
    psm_confusion_matrix_normalized.csv
    psm_top_confusions.csv
    psm_gesture_by_surgeon.csv
    psm_class_distribution.csv
    psm_transition_error_analysis.csv
    psm_louo_predictions.csv
    psm_error_analysis_summary.json

===============================================================================
COMMANDS
===============================================================================

Smoke test:

    python "Gesture Classification\\experiment_runner_8.py" `
        --input-csv "Archive results\\PyTorch_1 Gesture Prediction\\outputs_pytorch_single_procedure_1\\prepared_data\\all_frame_level.csv" `
        --output-dir "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_8_smoke" `
        --smoke

Full Knot Tying PSM error analysis:

    python "Gesture Classification\\experiment_runner_8.py" `
        --input-csv "Archive results\\PyTorch_1 Gesture Prediction\\outputs_pytorch_single_procedure_1\\prepared_data\\all_frame_level.csv" `
        --output-dir "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_8_knot_tying"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install an appropriate PyTorch build "
        "before running experiment_runner_8.py."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the kinematics-only model/config/training loop instead of duplicating
# them. experiment_runner_4.py itself reuses Train_PyTorch.py's data loading,
# LOUO splitting, normalisation and leakage-auditing helpers.
import experiment_runner_4 as e4  # noqa: E402


NUM_GESTURE_CLASSES = e4.NUM_GESTURE_CLASSES
KINEMATIC_DIM_PER_SOURCE = e4.KINEMATIC_DIM_PER_SOURCE
GESTURE_ID_TO_LABEL = e4.GESTURE_ID_TO_LABEL

# Fixed "current best configuration". This is a diagnostic error-analysis
# experiment: none of these are exposed as CLI-tunable hyperparameters.
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

TRANSITION_THRESHOLDS = (1, 5, 15)


def format_duration(seconds: float) -> str:
    return e4.tp.format_duration(seconds)


# =============================================================================
# STEP 0: CAN EXPERIMENT 7's OUTPUTS BE REUSED?
# =============================================================================

def check_experiment7_reusable(experiment7_output_dir: Optional[str]) -> bool:
    """Explain, and confirm, that Experiment 7 outputs cannot be reused."""

    print()
    print("=" * 78)
    print("[CHECK] Can existing Experiment 7 (PSM) outputs be reused?")
    print("=" * 78)

    if experiment7_output_dir is not None:
        directory = Path(experiment7_output_dir)
        metrics_path = directory / "kinematics_only_metrics.json"
        by_surgeon_path = directory / "kinematics_only_by_surgeon.csv"

        print(f"[CHECK] Inspecting supplied directory: {directory}")

        if metrics_path.exists() and by_surgeon_path.exists():
            with metrics_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            fold_keys: set = set()
            for fold in payload.get("cross_validation", {}).get("folds", []):
                fold_keys.update(fold.keys())

            print(f"[CHECK] Found kinematics_only_metrics.json with fold fields: {sorted(fold_keys)}")
        else:
            print("[CHECK] kinematics_only_metrics.json / kinematics_only_by_surgeon.csv not found.")
    else:
        print("[CHECK] No --experiment7-output-dir supplied.")

    print(
        "[CHECK] experiment_runner_4.py (used by experiment_runner_7.py) only ever "
        "writes AGGREGATED per-fold metrics -- accuracy, macro F1, per-class "
        "precision/recall/F1/support, and an aggregated confusion matrix. It "
        "contains no trial_id / frame_idx / predicted_gesture / confidence records."
    )
    print(
        "[CHECK] Its LOUO fold models are deleted immediately after evaluation "
        "('del _fold_model' in run_pipeline); only a FINAL model trained on ALL "
        "surgeons together is ever saved to kinematics_only_model.pt, so it cannot "
        "produce leakage-free held-out predictions for any single surgeon."
    )
    print(
        "[RETRAIN] Missing output: a per-frame LOUO prediction/confidence table "
        "and a per-fold held-out model. Neither exists in Experiment 7's output "
        "format, so this script must retrain the PSM LOUO experiment itself to "
        "obtain them."
    )

    return False


# =============================================================================
# STEP 1: PER-FRAME LOUO INFERENCE
# =============================================================================

@torch.no_grad()
def predict_trial_frame_level(
    model: "e4.KinematicsOnlyTransformer",
    trial: "e4.tp.TrialData",
    window_frames: int,
    stride_samples: int,
    standardize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    device: torch.device,
) -> Optional[tuple]:
    """
    One prediction + confidence per frame, built from overlapping windows.

    The first window in the trial supplies predictions for every one of its
    frames; every later window supplies only its own final frame's
    prediction. With stride=1 this covers every frame in the trial exactly
    once, with no gaps and no duplicated frame predictions.
    """

    total_frames = len(trial.labels)

    if total_frames < window_frames:
        return None

    starts = list(range(0, total_frames - window_frames + 1, stride_samples))

    windows = np.stack([trial.kinematics[s:s + window_frames] for s in starts])

    if standardize:
        windows = (windows - mean) / std

    # Runtime assertion: exactly 38 kinematic features, never gesture labels.
    assert windows.shape[-1] == KINEMATIC_DIM_PER_SOURCE, (
        f"Expected {KINEMATIC_DIM_PER_SOURCE} kinematic features, got {windows.shape[-1]}."
    )

    windows_tensor = torch.from_numpy(windows).float().to(device)

    logits = model(windows_tensor)
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
        raise RuntimeError(
            f"Trial {trial.trial_id}: frame coverage gap found with stride=1."
        )

    return frame_predictions, frame_confidences


def compute_transition_distances(labels: np.ndarray) -> np.ndarray:
    """Distance (in frames) from every frame to the nearest true-gesture transition."""

    n_frames = len(labels)
    transition_positions = [i for i in range(1, n_frames) if labels[i] != labels[i - 1]]

    if not transition_positions:
        return np.full(n_frames, 10**9, dtype=np.int64)

    positions = np.array(transition_positions)
    frame_indices = np.arange(n_frames)

    return np.min(np.abs(frame_indices[:, None] - positions[None, :]), axis=1)


def run_louo_and_collect_predictions(
    trials: List["e4.tp.TrialData"],
    config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    max_folds: Optional[int],
) -> pd.DataFrame:

    surgeons = sorted({trial.surgeon_id for trial in trials})
    fold_surgeons = surgeons if max_folds is None else surgeons[:max_folds]

    all_rows: List[Dict[str, object]] = []

    pipeline_start = time.perf_counter()

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
        print("=" * 78)
        print(f"[LOUO] Fold {fold_number}/{len(fold_surgeons)} | Held-out surgeon: {held_out_surgeon}")
        print("=" * 78)

        model, _metrics, _history, mean, std, fold_summary = e4.train_fold(
            train_trials=train_trials,
            test_trials=test_trials,
            config=config,
            device=device,
            run_name=f"LOUO_{held_out_surgeon}",
            held_out_surgeon=held_out_surgeon,
        )

        assert model.input_dimension == KINEMATIC_DIM_PER_SOURCE
        assert model.num_classes == NUM_GESTURE_CLASSES

        model.eval()

        for trial in test_trials:
            result = predict_trial_frame_level(
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
            distances = compute_transition_distances(trial.labels)

            for i in range(len(trial.labels)):
                true_gesture = int(trial.labels[i])
                predicted_gesture = int(frame_predictions[i])

                all_rows.append(
                    {
                        "held_out_surgeon": held_out_surgeon,
                        "trial_id": trial.trial_id,
                        "task": trial.task,
                        "frame_idx": int(trial.frame_indices[i]),
                        "true_gesture": true_gesture,
                        "true_gesture_label": GESTURE_ID_TO_LABEL[true_gesture],
                        "predicted_gesture": predicted_gesture,
                        "predicted_gesture_label": GESTURE_ID_TO_LABEL[predicted_gesture],
                        "confidence": float(frame_confidences[i]),
                        "distance_to_nearest_true_transition": int(distances[i]),
                    }
                )

        fold_elapsed = time.perf_counter() - fold_start
        elapsed_total = time.perf_counter() - pipeline_start
        average_fold_time = elapsed_total / fold_number
        eta = average_fold_time * (len(fold_surgeons) - fold_number)

        print()
        print(f"[RUNTIME] Fold {fold_number}/{len(fold_surgeons)} ({held_out_surgeon}) time: {format_duration(fold_elapsed)}")
        print(f"[RUNTIME] Best epoch: {fold_summary['best_epoch']}")
        print(f"[RUNTIME] Experiment elapsed: {format_duration(elapsed_total)}")
        print(f"[RUNTIME] Estimated remaining: {format_duration(eta)}")

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_runtime = time.perf_counter() - pipeline_start
    print()
    print(f"[RUNTIME] Total LOUO retraining + inference runtime: {format_duration(total_runtime)}")

    return pd.DataFrame(all_rows)


# =============================================================================
# STEP 2: ERROR ANALYSIS
# =============================================================================

def build_confusion_matrix(predictions: pd.DataFrame) -> np.ndarray:
    confusion = np.zeros((NUM_GESTURE_CLASSES, NUM_GESTURE_CLASSES), dtype=np.int64)

    indices = (
        predictions["true_gesture"].to_numpy() * NUM_GESTURE_CLASSES
        + predictions["predicted_gesture"].to_numpy()
    )
    counts = np.bincount(indices, minlength=NUM_GESTURE_CLASSES * NUM_GESTURE_CLASSES)
    confusion += counts.reshape(NUM_GESTURE_CLASSES, NUM_GESTURE_CLASSES)

    return confusion


def gesture_segments(predictions: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    """Per-gesture segment counts/lengths computed from the true-label sequence."""

    segments: Dict[int, Dict[str, object]] = {
        gesture_id: {"count": 0, "lengths": []} for gesture_id in range(NUM_GESTURE_CLASSES)
    }

    for _trial_id, trial_rows in predictions.groupby("trial_id"):
        trial_rows = trial_rows.sort_values("frame_idx")
        labels = trial_rows["true_gesture"].to_numpy()

        run_start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[run_start]:
                gesture_id = int(labels[run_start])
                segments[gesture_id]["count"] += 1
                segments[gesture_id]["lengths"].append(i - run_start)
                run_start = i

    return segments


def compute_per_gesture_metrics(
    predictions: pd.DataFrame,
    confusion: np.ndarray,
    segments: Dict[int, Dict[str, object]],
) -> pd.DataFrame:

    total_frames = len(predictions)
    rows = []

    for gesture_id in range(NUM_GESTURE_CLASSES):
        true_positive = int(confusion[gesture_id, gesture_id])
        false_positive = int(confusion[:, gesture_id].sum() - true_positive)
        false_negative = int(confusion[gesture_id, :].sum() - true_positive)
        n_frames = int(confusion[gesture_id, :].sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "gesture_id": gesture_id,
                "gesture_label": GESTURE_ID_TO_LABEL[gesture_id],
                "n_frames": n_frames,
                "n_segments": segments[gesture_id]["count"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positive,
                "false_positives": false_positive,
                "false_negatives": false_negative,
                "pct_of_total_dataset": 100.0 * n_frames / total_frames if total_frames else 0.0,
            }
        )

    return pd.DataFrame(rows)


def compute_top_confusions(confusion: np.ndarray, top_n: int = 3) -> pd.DataFrame:
    rows = []

    for true_gesture_id in range(NUM_GESTURE_CLASSES):
        row_counts = confusion[true_gesture_id, :].copy()
        row_total_errors = int(row_counts.sum() - row_counts[true_gesture_id])
        row_counts[true_gesture_id] = 0

        top_indices = np.argsort(row_counts)[::-1][:top_n]

        for predicted_gesture_id in top_indices:
            error_count = int(row_counts[predicted_gesture_id])
            if error_count <= 0:
                continue

            rows.append(
                {
                    "true_gesture": GESTURE_ID_TO_LABEL[true_gesture_id],
                    "confused_gesture": GESTURE_ID_TO_LABEL[int(predicted_gesture_id)],
                    "n_errors": error_count,
                    "pct_of_gesture_errors": (
                        100.0 * error_count / row_total_errors if row_total_errors else 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def compute_gesture_by_surgeon(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for surgeon, surgeon_rows in predictions.groupby("held_out_surgeon"):
        confusion = build_confusion_matrix(surgeon_rows)

        for gesture_id in range(NUM_GESTURE_CLASSES):
            true_positive = int(confusion[gesture_id, gesture_id])
            false_positive = int(confusion[:, gesture_id].sum() - true_positive)
            false_negative = int(confusion[gesture_id, :].sum() - true_positive)
            support = int(confusion[gesture_id, :].sum())

            if support == 0:
                continue

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            rows.append(
                {
                    "held_out_surgeon": surgeon,
                    "gesture_id": gesture_id,
                    "gesture_label": GESTURE_ID_TO_LABEL[gesture_id],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )

    return pd.DataFrame(rows)


def compute_class_distribution(
    per_gesture_metrics: pd.DataFrame,
    segments: Dict[int, Dict[str, object]],
) -> pd.DataFrame:

    rows = []

    for _, row in per_gesture_metrics.iterrows():
        gesture_id = int(row["gesture_id"])
        lengths = segments[gesture_id]["lengths"]

        rows.append(
            {
                "gesture_id": gesture_id,
                "gesture_label": row["gesture_label"],
                "total_frames": row["n_frames"],
                "n_segments": row["n_segments"],
                "mean_segment_duration_frames": float(np.mean(lengths)) if lengths else 0.0,
                "median_segment_duration_frames": float(np.median(lengths)) if lengths else 0.0,
                "pct_of_total_dataset": row["pct_of_total_dataset"],
            }
        )

    return pd.DataFrame(rows)


def compute_transition_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    correct = predictions["true_gesture"] == predictions["predicted_gesture"]
    total_errors = int((~correct).sum())

    rows = []

    for threshold in TRANSITION_THRESHOLDS:
        near = predictions["distance_to_nearest_true_transition"] <= threshold
        away = ~near

        n_near = int(near.sum())
        n_away = int(away.sum())

        near_accuracy = float(correct[near].mean()) if n_near else None
        away_accuracy = float(correct[away].mean()) if n_away else None

        near_errors = int((~correct & near).sum())

        rows.append(
            {
                "transition_window_frames": threshold,
                "n_near_frames": n_near,
                "n_away_frames": n_away,
                "near_accuracy": near_accuracy,
                "near_error_rate": (1.0 - near_accuracy) if near_accuracy is not None else None,
                "away_accuracy": away_accuracy,
                "away_error_rate": (1.0 - away_accuracy) if away_accuracy is not None else None,
                "pct_of_all_errors_near_transition": (
                    100.0 * near_errors / total_errors if total_errors else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# STEP 3: SAVE OUTPUTS + TERMINAL SUMMARY
# =============================================================================

def save_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    confusion: np.ndarray,
    per_gesture_metrics: pd.DataFrame,
    top_confusions: pd.DataFrame,
    gesture_by_surgeon: pd.DataFrame,
    class_distribution: pd.DataFrame,
    transition_analysis: pd.DataFrame,
    total_runtime_seconds: float,
) -> Dict[str, object]:

    labels = [GESTURE_ID_TO_LABEL[g] for g in range(NUM_GESTURE_CLASSES)]

    counts_df = pd.DataFrame(confusion, index=labels, columns=labels)
    counts_df.index.name = "true_gesture"
    counts_df.to_csv(output_dir / "psm_confusion_matrix_counts.csv")

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion.astype(np.float64) * 100.0,
        row_sums,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sums != 0,
    )
    normalized_df = pd.DataFrame(normalized, index=labels, columns=labels)
    normalized_df.index.name = "true_gesture"
    normalized_df.to_csv(output_dir / "psm_confusion_matrix_normalized.csv")

    per_gesture_metrics.to_csv(output_dir / "psm_per_gesture_metrics.csv", index=False)
    top_confusions.to_csv(output_dir / "psm_top_confusions.csv", index=False)
    gesture_by_surgeon.to_csv(output_dir / "psm_gesture_by_surgeon.csv", index=False)
    class_distribution.to_csv(output_dir / "psm_class_distribution.csv", index=False)
    transition_analysis.to_csv(output_dir / "psm_transition_error_analysis.csv", index=False)
    predictions.to_csv(output_dir / "psm_louo_predictions.csv", index=False)

    overall_accuracy = float((predictions["true_gesture"] == predictions["predicted_gesture"]).mean())

    supported_gestures = per_gesture_metrics[per_gesture_metrics["n_frames"] > 0]
    overall_macro_f1 = float(supported_gestures["f1"].mean()) if len(supported_gestures) else 0.0

    best_gesture_row = supported_gestures.loc[supported_gestures["f1"].idxmax()]
    worst_gesture_row = supported_gestures.loc[supported_gestures["f1"].idxmin()]

    most_underrepresented_row = supported_gestures.loc[supported_gestures["n_frames"].idxmin()]

    most_common_confusion = None
    off_diagonal = confusion.copy()
    np.fill_diagonal(off_diagonal, 0)
    if off_diagonal.sum() > 0:
        flat_index = int(np.argmax(off_diagonal))
        true_id, predicted_id = np.unravel_index(flat_index, off_diagonal.shape)
        most_common_confusion = {
            "true_gesture": GESTURE_ID_TO_LABEL[int(true_id)],
            "confused_gesture": GESTURE_ID_TO_LABEL[int(predicted_id)],
            "n_errors": int(off_diagonal[true_id, predicted_id]),
        }

    per_surgeon_accuracy = (
        predictions.groupby("held_out_surgeon")
        .apply(lambda rows: float((rows["true_gesture"] == rows["predicted_gesture"]).mean()))
        .to_dict()
    )
    best_surgeon = max(per_surgeon_accuracy, key=per_surgeon_accuracy.get)
    worst_surgeon = min(per_surgeon_accuracy, key=per_surgeon_accuracy.get)

    transition_15 = transition_analysis[transition_analysis["transition_window_frames"] == 15].iloc[0]

    summary = {
        "diagnostic_note": (
            "Frame-level LOUO predictions for the PSM kinematics-only Transformer "
            "(current best configuration), retrained here because "
            "experiment_runner_4.py/experiment_runner_7.py never save per-frame "
            "predictions, confidence, or per-fold LOUO checkpoints."
        ),
        "kinematic_source": KINEMATIC_SOURCE,
        "window_seconds": WINDOW_SECONDS,
        "window_frames": WINDOW_FRAMES,
        "standardize": STANDARDIZE,
        "overall_louo_accuracy": overall_accuracy,
        "overall_louo_macro_f1": overall_macro_f1,
        "best_performing_gesture": best_gesture_row["gesture_label"],
        "best_performing_gesture_f1": float(best_gesture_row["f1"]),
        "worst_performing_gesture": worst_gesture_row["gesture_label"],
        "worst_performing_gesture_f1": float(worst_gesture_row["f1"]),
        "most_common_confusion": most_common_confusion,
        "most_underrepresented_gesture": most_underrepresented_row["gesture_label"],
        "most_underrepresented_gesture_pct_of_dataset": float(most_underrepresented_row["pct_of_total_dataset"]),
        "pct_of_errors_near_transitions": {
            f"within_{threshold}_frames": float(
                transition_analysis.loc[
                    transition_analysis["transition_window_frames"] == threshold,
                    "pct_of_all_errors_near_transition",
                ].iloc[0]
            )
            for threshold in TRANSITION_THRESHOLDS
        },
        "surgeon_with_highest_accuracy": best_surgeon,
        "surgeon_with_highest_accuracy_value": per_surgeon_accuracy[best_surgeon],
        "surgeon_with_lowest_accuracy": worst_surgeon,
        "surgeon_with_lowest_accuracy_value": per_surgeon_accuracy[worst_surgeon],
        "total_runtime_seconds": total_runtime_seconds,
    }

    with (output_dir / "psm_error_analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print()
    print("=" * 78)
    print("[SUMMARY] PSM kinematics-only Transformer error analysis")
    print("=" * 78)
    print(f"Overall LOUO accuracy: {overall_accuracy:.4f}")
    print(f"Overall LOUO macro F1: {overall_macro_f1:.4f}")
    print(f"Best-performing gesture: {best_gesture_row['gesture_label']} (F1={best_gesture_row['f1']:.4f})")
    print(f"Worst-performing gesture: {worst_gesture_row['gesture_label']} (F1={worst_gesture_row['f1']:.4f})")
    if most_common_confusion:
        print(
            f"Most common confusion: {most_common_confusion['true_gesture']} -> "
            f"{most_common_confusion['confused_gesture']} ({most_common_confusion['n_errors']} errors)"
        )
    print(
        f"Most underrepresented gesture: {most_underrepresented_row['gesture_label']} "
        f"({most_underrepresented_row['pct_of_total_dataset']:.2f}% of dataset)"
    )
    print(f"Errors within 15 frames of a transition: {transition_15['pct_of_all_errors_near_transition']:.2f}%")
    print(f"Surgeon with highest accuracy: {best_surgeon} ({per_surgeon_accuracy[best_surgeon]:.4f})")
    print(f"Surgeon with lowest accuracy: {worst_surgeon} ({per_surgeon_accuracy[worst_surgeon]:.4f})")

    return summary


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    check_experiment7_reusable(args.experiment7_output_dir)

    assert WINDOW_FRAMES == round(WINDOW_SECONDS * SAMPLE_RATE)
    assert KINEMATIC_DIM_PER_SOURCE == 38

    e4.tp.seed_everything(RANDOM_SEED)
    device = e4.tp.choose_device(args.device)

    print()
    print("=" * 78)
    print("ATARI-2 PSM KINEMATICS-ONLY TRANSFORMER ERROR ANALYSIS (EXPERIMENT 8)")
    print("=" * 78)
    print(f"[SYSTEM] Device: {device}")
    print(f"[SYSTEM] PyTorch version: {torch.__version__}")
    print()
    print("[EXPERIMENT] Kinematic source: PSM")
    print("[CHECK] Input features: k39-k76")
    print(f"[CHECK] Number of model input features: {KINEMATIC_DIM_PER_SOURCE}")
    print(f"[EXPERIMENT] Window: {WINDOW_SECONDS} sec / {WINDOW_FRAMES} frames")
    print(f"[EXPERIMENT] Standardisation: {'ON' if STANDARDIZE else 'OFF'}")
    print("[CHECK] Previous gesture supplied as input: NO")
    print("[CHECK] Teacher forcing: NO")
    print("[CHECK] Autoregressive label feedback: NO")

    max_folds = args.max_folds
    max_windows = args.max_windows
    epochs = MAX_EPOCHS

    if args.smoke:
        max_folds = 2 if max_folds is None else min(max_folds, 2)
        max_windows = 1000 if max_windows is None else min(max_windows, 1000)
        epochs = 1
        print()
        print("[SMOKE] Smoke-test mode enabled: max_folds<=2, epochs=1, max_windows<=1000.")
        print("[SMOKE] Smoke-test results are NOT scientifically meaningful.")

    trials, _kinematic_columns = e4.tp.load_frame_level_data(
        path=Path(args.input_csv),
        kinematic_source=KINEMATIC_SOURCE,
    )

    config = e4.KinematicsOnlyConfig(
        input_csv=Path(args.input_csv),
        output_dir=output_dir,
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

    pipeline_start = time.perf_counter()

    predictions = run_louo_and_collect_predictions(
        trials=trials,
        config=config,
        device=device,
        max_folds=max_folds,
    )

    if predictions.empty:
        raise RuntimeError("No held-out predictions were produced; cannot run error analysis.")

    confusion = build_confusion_matrix(predictions)
    segments = gesture_segments(predictions)
    per_gesture_metrics = compute_per_gesture_metrics(predictions, confusion, segments)
    top_confusions = compute_top_confusions(confusion)
    gesture_by_surgeon = compute_gesture_by_surgeon(predictions)
    class_distribution = compute_class_distribution(per_gesture_metrics, segments)
    transition_analysis = compute_transition_error_analysis(predictions)

    total_runtime = time.perf_counter() - pipeline_start

    save_outputs(
        output_dir=output_dir,
        predictions=predictions,
        confusion=confusion,
        per_gesture_metrics=per_gesture_metrics,
        top_confusions=top_confusions,
        gesture_by_surgeon=gesture_by_surgeon,
        class_distribution=class_distribution,
        transition_analysis=transition_analysis,
        total_runtime_seconds=total_runtime,
    )

    print()
    print(f"[DONE] Total experiment_runner_8.py runtime: {format_duration(total_runtime)}")
    print(f"[DONE] Outputs written to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic error analysis for the current best PSM kinematics-only "
            "Transformer LOUO configuration."
        )
    )

    parser.add_argument("--input-csv", type=str, required=True, help="Prepared all_frame_level.csv.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for error-analysis outputs.")
    parser.add_argument(
        "--experiment7-output-dir",
        type=str,
        default=None,
        help="Optional existing Experiment 7 PSM output directory to inspect before retraining.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test mode: caps folds/epochs/windows to verify the pipeline "
            "runs end to end. Smoke-test results are NOT scientifically meaningful."
        ),
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
