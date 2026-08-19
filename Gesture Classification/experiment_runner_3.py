"""Evaluate a previous-label persistence baseline on JIGSAWS frame labels.

This is a diagnostic baseline, not a deployable recognition model.  If simply
copying the previous *true* gesture achieves performance close to the
teacher-forced Transformer, then that teacher-forced metric is heavily
influenced by temporal gesture persistence.  It is consequently not a good
measure of deployable gesture recognition, because true labels are unavailable
at inference time.
to run this: 

python "Gesture Classification\experiment_runner_3.py" `
  --input-csv "Archive results\PyTorch_1 Gesture Prediction\outputs_pytorch_single_procedure_1\prepared_data\all_frame_level.csv" `
  --pytorch-metrics "Archive results\PyTorch_1 Gesture Prediction\outputs_pytorch_single_procedure_1\pytorch_model\pytorch_metrics.json" `
  --output-dir "Archive results\PyTorch_1 Gesture Prediction\outputs_pytorch_single_procedure_1\persistence_baseline"

Notice the output directory is different from the PyTorch model output directory, so that the baseline metrics do not overwrite the PyTorch metrics.  
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NUM_GESTURE_CLASSES = 16
BACKGROUND_ID = 0
GESTURE_ID_TO_LABEL = {
	0: "BACKGROUND",
	**{gesture_id: f"G{gesture_id}" for gesture_id in range(1, 16)},
}


def infer_surgeon_id(trial_id: str) -> str:
	"""Match Train_PyTorch.py's fallback surgeon-ID convention."""
	suffix = trial_id.rsplit("_", 1)[-1]
	if not suffix:
		raise ValueError(f"Cannot infer surgeon from trial ID: {trial_id}")
	return suffix[0]


def load_frame_labels(input_csv: Path) -> pd.DataFrame:
	"""Load only frame metadata and labels; kinematic data are never read."""
	if not input_csv.exists():
		raise FileNotFoundError(f"Prepared frame-level CSV does not exist: {input_csv}")

	frame_data = pd.read_csv(input_csv)
	required_columns = {"trial_id", "frame_idx", "gesture_id"}
	missing_columns = sorted(required_columns.difference(frame_data.columns))
	if missing_columns:
		raise ValueError(
			"Prepared frame-level CSV is missing required columns: "
			+ ", ".join(missing_columns)
		)

	if "surgeon_id" not in frame_data.columns:
		frame_data["surgeon_id"] = frame_data["trial_id"].astype(str).map(
			infer_surgeon_id
		)

	for column in ("trial_id", "surgeon_id"):
		if frame_data[column].isna().any():
			raise ValueError(f"Prepared frame-level CSV contains missing {column} values.")
		frame_data[column] = frame_data[column].astype(str).str.strip()
		if (frame_data[column] == "").any():
			raise ValueError(f"Prepared frame-level CSV contains blank {column} values.")

	frame_data["frame_idx"] = pd.to_numeric(
		frame_data["frame_idx"], errors="raise"
	).astype(int)
	frame_data["gesture_id"] = pd.to_numeric(
		frame_data["gesture_id"], errors="raise"
	).astype(int)

	invalid_gesture_ids = sorted(
		set(frame_data["gesture_id"].unique()).difference(range(NUM_GESTURE_CLASSES))
	)
	if invalid_gesture_ids:
		raise ValueError(
			"Gesture IDs outside the expected 0-15 range were found: "
			f"{invalid_gesture_ids}"
		)

	return frame_data


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
	"""Calculate Train_PyTorch-compatible frame-level classification metrics."""
	indices = target * NUM_GESTURE_CLASSES + prediction
	confusion = np.bincount(
		indices, minlength=NUM_GESTURE_CLASSES * NUM_GESTURE_CLASSES
	).reshape(NUM_GESTURE_CLASSES, NUM_GESTURE_CLASSES)

	per_gesture: dict[str, dict[str, float | int]] = {}
	supported_f1_values: list[float] = []
	for gesture_id in range(NUM_GESTURE_CLASSES):
		true_positive = float(confusion[gesture_id, gesture_id])
		false_positive = float(confusion[:, gesture_id].sum() - true_positive)
		false_negative = float(confusion[gesture_id, :].sum() - true_positive)
		support = int(confusion[gesture_id, :].sum())
		precision = (
			true_positive / (true_positive + false_positive)
			if true_positive + false_positive
			else 0.0
		)
		recall = (
			true_positive / (true_positive + false_negative)
			if true_positive + false_negative
			else 0.0
		)
		f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
		if support:
			supported_f1_values.append(f1)
		per_gesture[GESTURE_ID_TO_LABEL[gesture_id]] = {
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"support": support,
		}

	return {
		"accuracy": float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0,
		"macro_f1": float(np.mean(supported_f1_values)) if supported_f1_values else 0.0,
		"per_gesture": per_gesture,
		"confusion_matrix": confusion.tolist(),
	}


def evaluate_baseline(frame_data: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
	"""Evaluate each LOUO held-out surgeon without fitting or using features."""
	fold_results: list[dict[str, Any]] = []
	for held_out_surgeon in sorted(frame_data["surgeon_id"].unique()):
		held_out_data = frame_data.loc[
			frame_data["surgeon_id"] == held_out_surgeon
		].copy()
		predictions: list[np.ndarray] = []
		targets: list[np.ndarray] = []

		for trial_id, trial_data in held_out_data.groupby("trial_id", sort=True):
			trial_data = trial_data.sort_values("frame_idx")
			frame_indices = trial_data["frame_idx"].to_numpy()
			if len(np.unique(frame_indices)) != len(frame_indices):
				raise ValueError(f"Trial {trial_id} contains duplicate frame_idx values.")
			if len(frame_indices) > 1 and not np.all(np.diff(frame_indices) == 1):
				raise ValueError(
					f"Trial {trial_id} has non-contiguous frame_idx values; "
					"do not create predictions across missing frames."
				)

			labels = trial_data["gesture_id"].to_numpy(dtype=np.int64)
			trial_predictions = np.empty_like(labels)
			trial_predictions[0] = BACKGROUND_ID
			if len(labels) > 1:
				trial_predictions[1:] = labels[:-1]
			targets.append(labels)
			predictions.append(trial_predictions)

		target = np.concatenate(targets)
		prediction = np.concatenate(predictions)
		metrics = calculate_metrics(target, prediction)
		fold_results.append(
			{
				"held_out_surgeon": str(held_out_surgeon),
				"n_test_trials": int(held_out_data["trial_id"].nunique()),
				"n_test_frames": int(len(held_out_data)),
				"metrics": metrics,
			}
		)

	accuracies = [fold["metrics"]["accuracy"] for fold in fold_results]
	macro_f1_values = [fold["metrics"]["macro_f1"] for fold in fold_results]
	summary = {
		"mean_accuracy": float(np.mean(accuracies)),
		"std_accuracy": float(np.std(accuracies)),
		"mean_macro_f1": float(np.mean(macro_f1_values)),
		"std_macro_f1": float(np.std(macro_f1_values)),
	}
	return fold_results, summary


def load_transformer_folds(metrics_path: Path | None) -> dict[str, dict[str, float]]:
	"""Read fold metrics from Train_PyTorch.py when an output file is supplied."""
	if metrics_path is None:
		return {}
	if not metrics_path.exists():
		print(f"[WARN] PyTorch metrics file not found; comparison columns will be blank: {metrics_path}")
		return {}

	with metrics_path.open("r", encoding="utf-8") as metrics_file:
		payload = json.load(metrics_file)
	folds = payload.get("cross_validation", {}).get("folds", [])
	transformer_folds: dict[str, dict[str, float]] = {}
	for fold in folds:
		surgeon = str(fold.get("held_out_surgeon", ""))
		if not surgeon:
			continue
		transformer_folds[surgeon] = {
			"teacher_forced_accuracy": fold.get("teacher_forced_accuracy"),
			"teacher_forced_macro_f1": fold.get("teacher_forced_macro_f1"),
			"autoregressive_accuracy": fold.get("autoregressive_accuracy"),
			"autoregressive_macro_f1": fold.get("autoregressive_macro_f1"),
		}
	return transformer_folds


def write_outputs(
	output_dir: Path,
	input_csv: Path,
	pytorch_metrics: Path | None,
	fold_results: list[dict[str, Any]],
	summary: dict[str, Any],
	transformer_folds: dict[str, dict[str, float]],
) -> None:
	"""Write the requested JSON and surgeon-level CSV artifacts."""
	output_dir.mkdir(parents=True, exist_ok=True)
	metrics_payload = {
		"diagnostic_note": (
			"This baseline copies the true previous frame label within each trial. "
			"Close teacher-forced Transformer performance indicates temporal gesture "
			"persistence, not deployable kinematic gesture recognition."
		),
		"input_csv": str(input_csv),
		"pytorch_metrics_json": str(pytorch_metrics) if pytorch_metrics else None,
		"baseline": "previous true label; BACKGROUND at each trial's first frame",
		"uses_kinematic_features": False,
		"uses_future_labels": False,
		"trains_model": False,
		"cross_validation": {"folds": fold_results, **summary},
	}
	with (output_dir / "previous_label_baseline_metrics.json").open("w", encoding="utf-8") as output_file:
		json.dump(metrics_payload, output_file, indent=2)

	surgeon_rows = [
		{
			"held_out_surgeon": fold["held_out_surgeon"],
			"n_test_trials": fold["n_test_trials"],
			"n_test_frames": fold["n_test_frames"],
			"previous_label_accuracy": fold["metrics"]["accuracy"],
			"previous_label_macro_f1": fold["metrics"]["macro_f1"],
		}
		for fold in fold_results
	]
	pd.DataFrame(surgeon_rows).to_csv(
		output_dir / "previous_label_baseline_by_surgeon.csv", index=False
	)

	comparison_rows: list[dict[str, Any]] = []
	for row in surgeon_rows:
		transformer_row = transformer_folds.get(row["held_out_surgeon"], {})
		comparison_rows.append(
			{
				"held_out_surgeon": row["held_out_surgeon"],
				"previous_label_accuracy": row["previous_label_accuracy"],
				"previous_label_macro_f1": row["previous_label_macro_f1"],
				"transformer_teacher_forced_accuracy": transformer_row.get("teacher_forced_accuracy"),
				"transformer_teacher_forced_macro_f1": transformer_row.get("teacher_forced_macro_f1"),
				"transformer_autoregressive_accuracy": transformer_row.get("autoregressive_accuracy"),
				"transformer_autoregressive_macro_f1": transformer_row.get("autoregressive_macro_f1"),
			}
		)
	comparison_rows.append(
		{
			"held_out_surgeon": "MEAN",
			"previous_label_accuracy": summary["mean_accuracy"],
			"previous_label_macro_f1": summary["mean_macro_f1"],
			"transformer_teacher_forced_accuracy": np.mean(
				[row["transformer_teacher_forced_accuracy"] for row in comparison_rows if row["transformer_teacher_forced_accuracy"] is not None]
			) if any(row["transformer_teacher_forced_accuracy"] is not None for row in comparison_rows) else None,
			"transformer_teacher_forced_macro_f1": np.mean(
				[row["transformer_teacher_forced_macro_f1"] for row in comparison_rows if row["transformer_teacher_forced_macro_f1"] is not None]
			) if any(row["transformer_teacher_forced_macro_f1"] is not None for row in comparison_rows) else None,
			"transformer_autoregressive_accuracy": np.mean(
				[row["transformer_autoregressive_accuracy"] for row in comparison_rows if row["transformer_autoregressive_accuracy"] is not None]
			) if any(row["transformer_autoregressive_accuracy"] is not None for row in comparison_rows) else None,
			"transformer_autoregressive_macro_f1": np.mean(
				[row["transformer_autoregressive_macro_f1"] for row in comparison_rows if row["transformer_autoregressive_macro_f1"] is not None]
			) if any(row["transformer_autoregressive_macro_f1"] is not None for row in comparison_rows) else None,
		}
	)
	pd.DataFrame(comparison_rows).to_csv(
		output_dir / "previous_label_baseline_comparison.csv", index=False
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate the previous true-label persistence diagnostic baseline."
	)
	parser.add_argument(
		"--input-csv", required=True, type=Path,
		help="Prepared all_frame_level.csv produced by data_prep.py.",
	)
	parser.add_argument(
		"--pytorch-metrics", type=Path, default=None,
		help="Optional existing pytorch_metrics.json for Transformer comparison.",
	)
	parser.add_argument(
		"--output-dir", type=Path, default=None,
		help="Directory for baseline outputs; defaults to the input CSV directory.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output_dir or args.input_csv.parent
	frame_data = load_frame_labels(args.input_csv)
	fold_results, summary = evaluate_baseline(frame_data)
	transformer_folds = load_transformer_folds(args.pytorch_metrics)
	write_outputs(
		output_dir, args.input_csv, args.pytorch_metrics, fold_results, summary,
		transformer_folds,
	)

	print("[DONE] Previous-label persistence diagnostic baseline")
	print(f"[DATA] Trials: {frame_data['trial_id'].nunique()} | Surgeons: {len(fold_results)}")
	print(f"[LOUO] Accuracy: {summary['mean_accuracy']:.4f} +/- {summary['std_accuracy']:.4f}")
	print(f"[LOUO] Macro F1: {summary['mean_macro_f1']:.4f} +/- {summary['std_macro_f1']:.4f}")
	print(f"[OUTPUT] {output_dir}")


if __name__ == "__main__":
	main()
