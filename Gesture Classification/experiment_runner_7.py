"""
ATARI-2 kinematics-only Transformer MTM-vs-PSM experiment runner.

Structural reference: experiment_runner_6.py (which ran a controlled
window-length sweep against the kinematics-only Transformer). This script
keeps that same subprocess-orchestration approach but instead varies which
38-dimensional kinematic source is used while holding every other setting
fixed at the current provisional best configuration.

Model reference: experiment_runner_4.py, which implements the diagnostic
kinematics-only Transformer:

    38-dim kinematics -> linear projection -> Transformer encoder
        -> linear classification head -> 16 gesture logits per frame

No decoder, no previous/current gesture label input, no teacher forcing, no
autoregressive label feedback. This script calls experiment_runner_4.py as a
subprocess CLI rather than duplicating its model/training implementation, and
never modifies Train_PyTorch.py, experiment_runner_4.py, experiment_runner_5.py,
experiment_runner_6.py, data_prep.py, or run_all_pytorch.py.

Runs exactly two configurations against the same prepared all_frame_level.csv
(which already contains all 76 kinematic variables; data preparation is only
run once, never rerun between configurations):

    A: kinematic_source="mtm" (k01-k38)
    B: kinematic_source="psm" (k39-k76)

Both configurations receive exactly 38 features. Every other setting is
identical: window=1.5s/45 frames, stride=1, standardize=True, dropout=0.3,
weight_decay=1e-3, batch_size=64, max epochs=15, early stopping patience=5 on
macro F1, random seed=42. MTM vs PSM is the only intended independent
variable.

Each configuration writes to its own directory under the output root. The
final summary is ranked by mean LOUO macro F1, using MTM as the baseline for
the "change vs MTM" comparison columns.

Smoke test:

	python "Gesture Classification\\experiment_runner_7.py" `
		--mode smoke `
		--kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
		--output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_7_smoke"

Single-procedure run (Knot Tying):

	python "Gesture Classification\\experiment_runner_7.py" `
		--mode single `
		--kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
		--output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_7_knot_tying"

Full-dataset run (supported, not run automatically):

	python "Gesture Classification\\experiment_runner_7.py" `
		--mode full `
		--kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
		--kinematics-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\Needle_Passing kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\transcriptions" `
		--kinematics-dir "JIGSAW\\Suturing\\Suturing\\Suturing kinematics\\AllGestures" `
		--annotations-dir "JIGSAW\\Suturing\\Suturing\\transcriptions" `
		--output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_7_full"
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
KINEMATIC_FEATURE_COUNT = 38
SAMPLE_RATE = 30.0
WINDOW_SECONDS = 1.5
WINDOW_FRAMES = 45
STRIDE_SAMPLES = 1
BATCH_SIZE = 64
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "macro_f1"
STANDARDIZE = True

BASELINE_KINEMATIC_SOURCE = "mtm"


@dataclass(frozen=True)
class TestPreset:
	name: str
	epochs: int
	max_trials: Optional[int]
	max_folds: Optional[int]
	max_windows: Optional[int]


@dataclass(frozen=True)
class KinematicSourceExperiment:
	name: str
	label: str
	kinematic_source: str
	feature_range: str


PRESETS = {
	"smoke": TestPreset("smoke", 1, 2, 2, 1000),
	"single": TestPreset("single", 15, None, None, None),
	"full": TestPreset("full", 15, None, None, None),
}

KINEMATIC_SOURCE_EXPERIMENTS = (
	KinematicSourceExperiment("mtm", "Experiment A - MTM", "mtm", "k01-k38"),
	KinematicSourceExperiment("psm", "Experiment B - PSM", "psm", "k39-k76"),
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
	seconds = max(0, int(round(seconds)))
	hours, remainder = divmod(seconds, 3600)
	minutes, secs = divmod(remainder, 60)
	parts = []
	if hours:
		parts.append(f"{hours} hr")
	if minutes or hours:
		parts.append(f"{minutes} min")
	parts.append(f"{secs} sec")
	return " ".join(parts)


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
	experiment: KinematicSourceExperiment,
	device: str,
) -> float:
	# Runtime assertion: both sources must select exactly 38 kinematic features.
	assert KINEMATIC_FEATURE_COUNT == 38, "Kinematic source must select exactly 38 features."

	print()
	print(f"[EXPERIMENT] Kinematic source: {experiment.kinematic_source.upper()}")
	print(f"[CHECK] Input features: {experiment.feature_range}")
	print(f"[CHECK] Number of model input features: {KINEMATIC_FEATURE_COUNT}")
	print(f"[EXPERIMENT] Window: {WINDOW_SECONDS} sec / {WINDOW_FRAMES} frames")
	print(f"[EXPERIMENT] Standardisation: {'ON' if STANDARDIZE else 'OFF'}")
	print("[CHECK] Previous gesture supplied as input: NO")
	print("[CHECK] Teacher forcing: NO")
	print("[CHECK] Autoregressive label feedback: NO")

	command = [
		PYTHON_EXECUTABLE, "-u", str(KINEMATICS_ONLY_SCRIPT),
		"--input-csv", str(frame_file),
		"--output-dir", str(model_dir),
		"--kinematic-source", experiment.kinematic_source,
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
		"--standardize",
	]
	if preset.max_folds is not None:
		command.extend(["--max-folds", str(preset.max_folds)])
	if preset.max_windows is not None:
		command.extend(["--max-windows", str(preset.max_windows)])

	elapsed = run_command(command, f"training {experiment.label}")
	require_file(model_dir / "kinematics_only_metrics.json", "kinematics-only metrics")
	require_file(model_dir / "kinematics_only_by_surgeon.csv", "kinematics-only per-surgeon results")
	require_file(model_dir / "kinematics_only_training_history.csv", "kinematics-only training history")
	return elapsed


def read_summary(
	model_dir: Path,
	experiment: KinematicSourceExperiment,
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
		"kinematic_source": experiment.kinematic_source,
		"number_of_features": KINEMATIC_FEATURE_COUNT,
		"window_seconds": WINDOW_SECONDS,
		"window_frames": WINDOW_FRAMES,
		"standardized": STANDARDIZE,
		"mean_louo_accuracy": cross_validation["mean_accuracy"],
		"std_louo_accuracy": cross_validation["std_accuracy"],
		"mean_louo_macro_f1": cross_validation["mean_macro_f1"],
		"std_louo_macro_f1": cross_validation["std_macro_f1"],
		"mean_training_accuracy_at_best_epoch": (
			sum(train_accuracy_at_best) / len(train_accuracy_at_best)
			if train_accuracy_at_best else None
		),
		"mean_best_epoch": (
			sum(best_epochs) / len(best_epochs) if best_epochs else None
		),
		"total_runtime_seconds": metrics["total_runtime_seconds"],
	}


def read_by_surgeon_rows(
	model_dir: Path,
	experiment: KinematicSourceExperiment,
) -> List[dict[str, object]]:
	with (model_dir / "kinematics_only_by_surgeon.csv").open(
		encoding="utf-8", newline=""
	) as file:
		by_surgeon = list(csv.DictReader(file))

	rows = []
	for row in by_surgeon:
		rows.append(
			{
				"held_out_surgeon": row["held_out_surgeon"],
				"kinematic_source": experiment.kinematic_source,
				"accuracy": float(row["accuracy"]),
				"macro_f1": float(row["macro_f1"]),
				"best_epoch": row["best_epoch"],
				"training_accuracy_at_best_epoch": row["train_accuracy_at_best_epoch"],
				"runtime": row["runtime_seconds"],
			}
		)
	return rows


def add_psm_minus_mtm_columns(
	by_surgeon_rows: List[dict[str, object]],
) -> None:
	"""Attach PSM-minus-MTM accuracy/macro-F1 deltas to every row for that surgeon."""

	mtm_by_surgeon = {
		row["held_out_surgeon"]: row
		for row in by_surgeon_rows
		if row["kinematic_source"] == "mtm"
	}
	psm_by_surgeon = {
		row["held_out_surgeon"]: row
		for row in by_surgeon_rows
		if row["kinematic_source"] == "psm"
	}

	for row in by_surgeon_rows:
		surgeon = row["held_out_surgeon"]
		mtm_row = mtm_by_surgeon.get(surgeon)
		psm_row = psm_by_surgeon.get(surgeon)

		if mtm_row is not None and psm_row is not None:
			row["psm_minus_mtm_accuracy"] = psm_row["accuracy"] - mtm_row["accuracy"]
			row["psm_minus_mtm_macro_f1"] = psm_row["macro_f1"] - mtm_row["macro_f1"]
		else:
			row["psm_minus_mtm_accuracy"] = None
			row["psm_minus_mtm_macro_f1"] = None


def run_experiments(args: argparse.Namespace) -> Path:
	preset = PRESETS[args.mode]
	output_root = Path(args.output_root).resolve()
	prepared_data_dir = output_root / "prepared_data"
	experiment_root = output_root / "mtm_vs_psm_experiments"
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
		f"Running exactly {len(KINEMATIC_SOURCE_EXPERIMENTS)} kinematics-only "
		f"Transformer MTM-vs-PSM experiments with seed {EXPERIMENT_RANDOM_SEED} "
		f"against the same prepared dataset (data preparation is not rerun "
		f"between configurations). MTM vs PSM is the only intended independent "
		f"variable; window stays fixed at {WINDOW_SECONDS} s / {WINDOW_FRAMES} frames."
	)

	pipeline_start = time.perf_counter()

	summary_rows = []
	by_surgeon_rows: List[dict[str, object]] = []
	runtime_by_source: dict[str, float] = {}

	for index, experiment in enumerate(KINEMATIC_SOURCE_EXPERIMENTS, start=1):
		model_dir = experiment_root / experiment.name
		elapsed = run_training(
			frame_file=frame_file,
			model_dir=model_dir,
			preset=preset,
			experiment=experiment,
			device=args.device,
		)
		runtime_by_source[experiment.kinematic_source] = elapsed
		summary_rows.append(read_summary(model_dir, experiment))
		by_surgeon_rows.extend(read_by_surgeon_rows(model_dir, experiment))

		elapsed_total = time.perf_counter() - pipeline_start
		average_per_config = elapsed_total / index
		remaining_configs = len(KINEMATIC_SOURCE_EXPERIMENTS) - index
		eta = average_per_config * remaining_configs

		print()
		print(f"[RUNTIME] {experiment.label} configuration time: {format_duration(elapsed)}")
		print(f"[RUNTIME] Experiment elapsed: {format_duration(elapsed_total)}")
		print(f"[RUNTIME] Estimated remaining: {format_duration(eta)}")

	baseline_row = next(
		row for row in summary_rows
		if row["kinematic_source"] == BASELINE_KINEMATIC_SOURCE
	)
	for row in summary_rows:
		row["change_in_accuracy_vs_mtm"] = (
			row["mean_louo_accuracy"] - baseline_row["mean_louo_accuracy"]
		)
		row["change_in_macro_f1_vs_mtm"] = (
			row["mean_louo_macro_f1"] - baseline_row["mean_louo_macro_f1"]
		)

	summary_rows.sort(key=lambda row: float(row["mean_louo_macro_f1"]), reverse=True)
	for rank, row in enumerate(summary_rows, start=1):
		row["rank"] = rank

	summary_path = output_root / "kinematics_only_mtm_vs_psm_summary.csv"
	summary_fieldnames = [
		"rank",
		"kinematic_source",
		"number_of_features",
		"window_seconds",
		"window_frames",
		"standardized",
		"mean_louo_accuracy",
		"std_louo_accuracy",
		"mean_louo_macro_f1",
		"std_louo_macro_f1",
		"mean_training_accuracy_at_best_epoch",
		"mean_best_epoch",
		"total_runtime_seconds",
		"change_in_accuracy_vs_mtm",
		"change_in_macro_f1_vs_mtm",
	]
	with summary_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=summary_fieldnames, extrasaction="ignore")
		writer.writeheader()
		writer.writerows(summary_rows)

	add_psm_minus_mtm_columns(by_surgeon_rows)

	by_surgeon_path = output_root / "kinematics_only_mtm_vs_psm_by_surgeon.csv"
	by_surgeon_rows.sort(key=lambda row: (row["held_out_surgeon"], row["kinematic_source"]))
	by_surgeon_fieldnames = [
		"held_out_surgeon",
		"kinematic_source",
		"accuracy",
		"macro_f1",
		"best_epoch",
		"training_accuracy_at_best_epoch",
		"runtime",
		"psm_minus_mtm_accuracy",
		"psm_minus_mtm_macro_f1",
	]
	with by_surgeon_path.open("w", encoding="utf-8", newline="") as file:
		writer = csv.DictWriter(file, fieldnames=by_surgeon_fieldnames)
		writer.writeheader()
		writer.writerows(by_surgeon_rows)

	total_runtime = time.perf_counter() - pipeline_start

	print()
	print("=" * 79)
	print("[DONE] KINEMATICS-ONLY MTM-VS-PSM EXPERIMENT COMPLETE")
	print("=" * 79)
	print(f"Total MTM runtime: {format_duration(runtime_by_source.get('mtm', 0.0))}")
	print(f"Total PSM runtime: {format_duration(runtime_by_source.get('psm', 0.0))}")
	print(f"Complete experiment runtime: {format_duration(total_runtime)}")
	print(f"Summary written to {summary_path}")
	print(f"Per-surgeon comparison written to {by_surgeon_path}")

	return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Run the controlled kinematics-only Transformer MTM-vs-PSM "
			"experiment (experiment_runner_4.py, not Train_PyTorch.py). "
			"MTM vs PSM is the only intended independent variable."
		)
	)
	parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
	parser.add_argument("--kinematics-dir", action="append", required=True)
	parser.add_argument("--annotations-dir", action="append", required=True)
	parser.add_argument(
		"--output-root",
		default=str(PROJECT_ROOT / "outputs_pytorch_kinematics_only_mtm_vs_psm"),
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
