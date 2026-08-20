"""
ATARI-2 controlled kinematics-only Transformer standardization experiment runner.

Structural template: experiment_runner_2.py (which ran the same A/B
standardization comparison against the old encoder-decoder Train_PyTorch.py
model). This script runs the identical A/B comparison instead against the
diagnostic kinematics-only Transformer implemented in experiment_runner_4.py:

    38-dim kinematics -> linear projection -> Transformer encoder
        -> linear classification head -> 16 gesture logits per frame

No decoder, no previous/current gesture label input, no teacher forcing, no
autoregressive label feedback. This script reuses experiment_runner_4.py as a
subprocess CLI rather than duplicating its model/training implementation, and
never modifies Train_PyTorch.py, experiment_runner_4.py, or data_prep.py.

Runs exactly two configurations against the same prepared all_frame_level.csv
(data preparation is only run once, never rerun between A and B):

	A: standardize=False (control)
	B: standardize=True  (standardized)

Every other setting is identical between A and B: MTM kinematics, 38 input
features, window=1.0s/30 frames, stride=1, batch size=64, dropout=0.3,
weight decay=1e-3, AdamW, max epochs=15, early stopping patience=5 on macro
F1, random seed=42, no positional encoding.

Each configuration writes to its own directory under the output root. The
final summary is ranked by mean LOUO macro F1.

Smoke test:

	python "Gesture Classification\\experiment_runner_5.py" `
		--mode smoke `
		--kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
		--output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_5_smoke"

Single-procedure run (Knot Tying):

	python "Gesture Classification\\experiment_runner_5.py" `
		--mode single `
		--kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
		--output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_5_knot_tying"

Full-dataset run:

	python "Gesture Classification\\experiment_runner_5.py" `
		--mode full `
		--kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
		--kinematics-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\Needle_Passing kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\transcriptions" `
		--kinematics-dir "JIGSAW\\Suturing\\Suturing\\Suturing kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Suturing\\Suturing\\transcriptions" `
		--output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_5_full"
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
KINEMATICS_ONLY_SCRIPT = SCRIPT_DIR / "experiment_runner_4.py"
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
) -> float:
	print()
	print(f"[CONFIG] {experiment.label}")
	print(f"[CONFIG] Standardization enabled: {experiment.standardize}")
	print(f"[CONFIG] Kinematic source: {KINEMATIC_SOURCE.upper()}")
	print(f"[CONFIG] Window length: {WINDOW_SECONDS} s ({int(WINDOW_SECONDS * SAMPLE_RATE)} frames)")
	print(f"[CONFIG] Dropout: {DROPOUT}")
	print(f"[CONFIG] Weight decay: {WEIGHT_DECAY}")
	print(f"[CONFIG] Random seed: {EXPERIMENT_RANDOM_SEED}")
	print("[CHECK] Model inputs contain kinematics only: YES")
	print("[CHECK] Previous gesture supplied as input: NO")
	print("[CHECK] Teacher forcing: NO")
	print("[CHECK] Autoregressive label feedback: NO")

	command = [
		PYTHON_EXECUTABLE, "-u", str(KINEMATICS_ONLY_SCRIPT),
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

	elapsed = run_command(command, f"training {experiment.label}")
	require_file(model_dir / "kinematics_only_metrics.json", "kinematics-only metrics")
	require_file(model_dir / "kinematics_only_by_surgeon.csv", "kinematics-only per-surgeon results")
	require_file(model_dir / "kinematics_only_training_history.csv", "kinematics-only training history")
	return elapsed


def read_summary(
	model_dir: Path,
	experiment: StandardizationExperiment,
) -> dict[str, object]:
	with (model_dir / "kinematics_only_metrics.json").open(encoding="utf-8") as file:
		metrics = json.load(file)
	with (model_dir / "kinematics_only_by_surgeon.csv").open(
		encoding="utf-8", newline=""
	) as file:
		by_surgeon = list(csv.DictReader(file))

	cross_validation = metrics["cross_validation"]

	best_epochs = [
		int(row["best_epoch"]) for row in by_surgeon
		if row.get("best_epoch") not in (None, "")
	]
	train_accuracy_at_best = [
		float(row["train_accuracy_at_best_epoch"]) for row in by_surgeon
		if row.get("train_accuracy_at_best_epoch") not in (None, "")
	]

	return {
		"experiment": experiment.label,
		"standardize": experiment.standardize,
		"kinematic_source": KINEMATIC_SOURCE,
		"window_seconds": WINDOW_SECONDS,
		"dropout": DROPOUT,
		"weight_decay": WEIGHT_DECAY,
		"random_seed": EXPERIMENT_RANDOM_SEED,
		"mean_louo_accuracy": cross_validation["mean_accuracy"],
		"std_louo_accuracy": cross_validation["std_accuracy"],
		"mean_louo_macro_f1": cross_validation["mean_macro_f1"],
		"std_louo_macro_f1": cross_validation["std_macro_f1"],
		"mean_best_epoch": (
			sum(best_epochs) / len(best_epochs) if best_epochs else None
		),
		"mean_train_accuracy_at_best_epoch": (
			sum(train_accuracy_at_best) / len(train_accuracy_at_best)
			if train_accuracy_at_best else None
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
		f"Running exactly {len(STANDARDIZATION_EXPERIMENTS)} kinematics-only "
		f"Transformer experiments with seed {EXPERIMENT_RANDOM_SEED} against "
		f"the same prepared dataset (data preparation is not rerun between "
		f"experiments)."
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
		)
		rows.append(read_summary(model_dir, experiment))

	control_row = next(row for row in rows if not row["standardize"])
	for row in rows:
		row["accuracy_change_vs_control"] = (
			row["mean_louo_accuracy"] - control_row["mean_louo_accuracy"]
		)
		row["macro_f1_change_vs_control"] = (
			row["mean_louo_macro_f1"] - control_row["mean_louo_macro_f1"]
		)

	rows.sort(key=lambda row: float(row["mean_louo_macro_f1"]), reverse=True)
	for rank, row in enumerate(rows, start=1):
		row["rank"] = rank

	summary_path = output_root / "kinematics_only_standardization_summary.csv"
	fieldnames = [
		"rank",
		"experiment",
		"standardize",
		"kinematic_source",
		"window_seconds",
		"dropout",
		"weight_decay",
		"random_seed",
		"mean_louo_accuracy",
		"std_louo_accuracy",
		"mean_louo_macro_f1",
		"std_louo_macro_f1",
		"mean_best_epoch",
		"mean_train_accuracy_at_best_epoch",
		"total_runtime_seconds",
		"accuracy_change_vs_control",
		"macro_f1_change_vs_control",
	]
	with summary_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print()
	print(f"Summary written to {summary_path}")
	return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Run the controlled kinematics-only Transformer standardization "
			"experiment (experiment_runner_4.py, not Train_PyTorch.py)."
		)
	)
	parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
	parser.add_argument("--kinematics-dir", action="append", required=True)
	parser.add_argument("--annotations-dir", action="append", required=True)
	parser.add_argument(
		"--output-root",
		default=str(PROJECT_ROOT / "outputs_pytorch_kinematics_only_standardization"),
	)
	parser.add_argument("--device", default="auto")
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
	require_script(KINEMATICS_ONLY_SCRIPT)
	run_experiments(args)


if __name__ == "__main__":
	main()
