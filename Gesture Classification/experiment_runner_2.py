"""
ATARI-2 controlled kinematic standardization experiment runner.

Runs exactly two PyTorch gesture-recognition configurations against the same
prepared all_frame_level.csv:

	A: dropout=0.3, weight_decay=1e-3, standardize=False
	B: dropout=0.3, weight_decay=1e-3, standardize=True

Each configuration writes to its own directory under the output root. The
final summary is ranked by mean LOUO autoregressive macro F1.

Smoke test:

	py -3 "Gesture Classification/experiment_runner_2.py" `
		--mode smoke `
		--kinematics-dir "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures" `
		--annotations-dir "JIGSAW/Knot_Tying/Knot_Tying/transcriptions" `
		--output-root "Archive results/PyTorch_2 Experements/outputs_pytorch_experiment_2_smoke"

Single-procedure run (Knot Tying):

	py -3 "Gesture Classification/experiment_runner_2.py" `
		--mode single `
		--kinematics-dir "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures" `
		--annotations-dir "JIGSAW/Knot_Tying/Knot_Tying/transcriptions" `
		--output-root "Archive results/PyTorch_2 Experements/outputs_pytorch_experiment_2_knot_tying"

Full-dataset run:

	py -3 "Gesture Classification/experiment_runner_2.py" `
		--mode full `
		--kinematics-dir "JIGSAW/Knot_Tying/Knot_Tying/Knot_Tying kinematics/AllGestures" `
		--annotations-dir "JIGSAW/Knot_Tying/Knot_Tying/transcriptions" `
		--kinematics-dir "JIGSAW/Needle_Passing/Needle_Passing/Needle_Passing kinematics/AllGestures" `
		--annotations-dir "JIGSAW/Needle_Passing/Needle_Passing/transcriptions" `
		--kinematics-dir "JIGSAW/Suturing/Suturing/Suturing kinematics/AllGestures" `
		--annotations-dir "JIGSAW/Suturing/Suturing/transcriptions" `
		--output-root "Archive results/PyTorch_2 Experements/outputs_pytorch_experiment_2_full"
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
from typing import List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PREP_SCRIPT = PROJECT_ROOT / "Gesture Data Manipulation" / "data_prep.py"
TRAIN_SCRIPT = SCRIPT_DIR / "Train_PyTorch.py"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CUDA_VENV_PYTHON = PROJECT_ROOT / ".venv312" / "Scripts" / "python.exe"

EXPERIMENT_RANDOM_SEED = 42
KINEMATIC_SOURCE = "mtm"
SAMPLE_RATE = 30.0
WINDOW_SECONDS = 1.0
STRIDE_SAMPLES = 1
BATCH_SIZE = 64
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "macro_f1"


@dataclass(frozen=True)
class TestPreset:
	name: str
	epochs: int
	max_trials: Optional[int]
	max_folds: Optional[int]
	max_windows: Optional[int]


@dataclass(frozen=True)
class StandardizationExperiment:
	name: str
	label: str
	standardize: bool


PRESETS = {
	"smoke": TestPreset("smoke", 1, 2, 2, 1000),
	"single": TestPreset("single", 15, None, None, None),
	"full": TestPreset("full", 15, None, None, None),
}

STANDARDIZATION_EXPERIMENTS = (
	StandardizationExperiment(
		name="control_unstandardized",
		label="Experiment A - control",
		standardize=False,
	),
	StandardizationExperiment(
		name="standardized",
		label="Experiment B - standardized",
		standardize=True,
	),
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


def timestamp() -> str:
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
	return f"{int(round(max(0.0, seconds)))} sec"


def require_script(path: Path) -> None:
	if not path.is_file():
		raise FileNotFoundError(f"Required script was not found: {path}")


def require_directory(path: Path, description: str) -> None:
	if not path.is_dir():
		raise NotADirectoryError(f"Expected {description}: {path}")


def require_file(path: Path, description: str) -> None:
	if not path.is_file():
		raise FileNotFoundError(f"Could not find {description}: {path}")


def run_command(command: List[str], stage_name: str) -> float:
	print()
	print("=" * 79)
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

	run_command(command, "data preparation")
	frame_file = prepared_data_dir / "all_frame_level.csv"
	require_file(frame_file, "frame-level dataset")
	return frame_file


def run_training(
	frame_file: Path,
	model_dir: Path,
	preset: TestPreset,
	experiment: StandardizationExperiment,
	device: str,
	save_fold_models: bool,
) -> float:
	command = [
		PYTHON_EXECUTABLE, "-u", str(TRAIN_SCRIPT),
		"--input-csv", str(frame_file),
		"--output-dir", str(model_dir),
		"--kinematic-source", KINEMATIC_SOURCE,
		"--sample-rate", str(SAMPLE_RATE),
		"--window-seconds", str(WINDOW_SECONDS),
		"--stride-samples", str(STRIDE_SAMPLES),
		"--batch-size", str(BATCH_SIZE),
		"--dropout", str(DROPOUT),
		"--weight-decay", str(WEIGHT_DECAY),
		"--early-stopping-patience", str(EARLY_STOPPING_PATIENCE),
		"--early-stopping-metric", EARLY_STOPPING_METRIC,
		"--random-seed", str(EXPERIMENT_RANDOM_SEED),
		"--epochs", str(preset.epochs),
		"--device", device,
	]
	if preset.max_folds is not None:
		command.extend(["--max-folds", str(preset.max_folds)])
	if preset.max_windows is not None:
		command.extend(["--max-windows", str(preset.max_windows)])
	if experiment.standardize:
		command.append("--standardize")
	if save_fold_models:
		command.append("--save-fold-models")

	elapsed = run_command(command, f"training {experiment.label}")
	require_file(model_dir / "pytorch_metrics.json", "training metrics")
	require_file(model_dir / "pytorch_training_history.csv", "training history")
	return elapsed


def read_summary(
	model_dir: Path,
	experiment: StandardizationExperiment,
) -> dict[str, object]:
	with (model_dir / "pytorch_metrics.json").open(encoding="utf-8") as file:
		metrics = json.load(file)
	with (model_dir / "pytorch_training_history.csv").open(
		encoding="utf-8", newline=""
	) as file:
		history = list(csv.DictReader(file))

	cross_validation = metrics["cross_validation"]
	folds = cross_validation.get("folds", [])
	best_epochs = []
	teacher_forced_accuracy = []
	autoregressive_accuracy = []
	teacher_forced_macro_f1 = []
	autoregressive_macro_f1 = []

	for fold in folds:
		teacher_forced_accuracy.append(float(fold["teacher_forced_accuracy"]))
		autoregressive_accuracy.append(float(fold["autoregressive_accuracy"]))
		teacher_forced_macro_f1.append(float(fold["teacher_forced_macro_f1"]))
		autoregressive_macro_f1.append(float(fold["autoregressive_macro_f1"]))

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
	final_training_accuracy = None
	if final_rows:
		final_training_accuracy = float(
			final_rows[-1]["teacher_forced_train_accuracy"]
		)

	mean_teacher_forced_accuracy = sum(teacher_forced_accuracy) / len(teacher_forced_accuracy)
	mean_autoregressive_accuracy = sum(autoregressive_accuracy) / len(autoregressive_accuracy)
	mean_teacher_forced_macro_f1 = sum(teacher_forced_macro_f1) / len(teacher_forced_macro_f1)
	mean_autoregressive_macro_f1 = sum(autoregressive_macro_f1) / len(autoregressive_macro_f1)

	return {
		"experiment": experiment.label,
		"standardize": experiment.standardize,
		"dropout": DROPOUT,
		"weight_decay": WEIGHT_DECAY,
		"mean_louo_autoregressive_accuracy": mean_autoregressive_accuracy,
		"mean_louo_autoregressive_macro_f1": mean_autoregressive_macro_f1,
		"mean_louo_teacher_forced_accuracy": mean_teacher_forced_accuracy,
		"mean_louo_teacher_forced_macro_f1": mean_teacher_forced_macro_f1,
		"louo_teacher_forced_minus_autoregressive_accuracy": (
			mean_teacher_forced_accuracy - mean_autoregressive_accuracy
		),
		"louo_teacher_forced_minus_autoregressive_macro_f1": (
			mean_teacher_forced_macro_f1 - mean_autoregressive_macro_f1
		),
		"final_teacher_forced_training_accuracy": final_training_accuracy,
		"mean_best_epoch": (
			sum(best_epochs) / len(best_epochs) if best_epochs else None
		),
		"total_runtime_seconds": metrics["total_runtime_seconds"],
	}


def run_experiments(args: argparse.Namespace) -> Path:
	preset = PRESETS[args.mode]
	output_root = Path(args.output_root).resolve()
	prepared_data_dir = output_root / "prepared_data"
	experiment_root = output_root / "standardization_experiments"
	output_root.mkdir(parents=True, exist_ok=True)
	experiment_root.mkdir(parents=True, exist_ok=True)

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

	print(
		f"Running exactly {len(STANDARDIZATION_EXPERIMENTS)} experiments "
		f"with seed {EXPERIMENT_RANDOM_SEED}."
	)
	rows = []
	for experiment in STANDARDIZATION_EXPERIMENTS:
		model_dir = experiment_root / experiment.name
		run_training(
			frame_file=frame_file,
			model_dir=model_dir,
			preset=preset,
			experiment=experiment,
			device=args.device,
			save_fold_models=args.save_fold_models,
		)
		rows.append(read_summary(model_dir, experiment))

	rows.sort(
		key=lambda row: float(
			row["mean_louo_autoregressive_macro_f1"]
		),
		reverse=True,
	)
	for rank, row in enumerate(rows, start=1):
		row["rank"] = rank

	summary_path = output_root / "standardization_experiment_summary.csv"
	fieldnames = [
		"rank",
		"experiment",
		"standardize",
		"dropout",
		"weight_decay",
		"mean_louo_autoregressive_accuracy",
		"mean_louo_autoregressive_macro_f1",
		"mean_louo_teacher_forced_accuracy",
		"mean_louo_teacher_forced_macro_f1",
		"louo_teacher_forced_minus_autoregressive_accuracy",
		"louo_teacher_forced_minus_autoregressive_macro_f1",
		"final_teacher_forced_training_accuracy",
		"mean_best_epoch",
		"total_runtime_seconds",
	]
	with summary_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Summary written to {summary_path}")
	return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Run the controlled ATARI-2 kinematic standardization experiment."
	)
	parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
	parser.add_argument("--kinematics-dir", action="append", required=True)
	parser.add_argument("--annotations-dir", action="append", required=True)
	parser.add_argument(
		"--output-root",
		default=str(PROJECT_ROOT / "outputs_pytorch_standardization"),
	)
	parser.add_argument("--device", default="auto")
	parser.add_argument("--save-fold-models", action="store_true")
	parser.add_argument("--reuse-prepared-data", action="store_true")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	kinematics_dirs = [Path(path) for path in args.kinematics_dir]
	annotations_dirs = [Path(path) for path in args.annotations_dir]
	if len(kinematics_dirs) != len(annotations_dirs):
		raise ValueError(
			"The number of kinematics and annotation directories must match."
		)
	for path in kinematics_dirs:
		require_directory(path, "kinematic data directory")
	for path in annotations_dirs:
		require_directory(path, "annotation directory")
	require_script(DATA_PREP_SCRIPT)
	require_script(TRAIN_SCRIPT)
	run_experiments(args)


if __name__ == "__main__":
	main()
