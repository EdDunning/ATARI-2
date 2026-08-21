"""
ATARI-2: Gesture Classification / experiment_runner_12.py

===============================================================================
PURPOSE
===============================================================================

Final model-optimisation experiment: determine whether explicitly providing
temporal position information improves the current PSM-only kinematics-only
Transformer. Positional-encoding strategy is the only intended independent
variable; every other setting matches the current best PSM configuration:

    PSM (k39-k76), 38 features, standardisation ON, 1.5 s / 45-frame windows,
    stride 1, dropout 0.3, weight decay 1e-4, unweighted CrossEntropyLoss,
    batch size 64, AdamW, max 15 epochs, early stopping patience 5 on macro
    F1, random seed 42, no previous-gesture input, no teacher forcing, no
    autoregressive feedback.

Compares exactly three configurations against identical LOUO folds:

    A: none        (current baseline: no positional information at all)
    B: sinusoidal  (fixed sin/cos positional encoding, zero trainable params)
    C: learned     (one trainable embedding per of the 45 positions)

===============================================================================
REUSE, NOT DUPLICATION
===============================================================================

experiment_runner_4.py's KinematicsOnlyTransformer is NEVER modified.
Configuration A instantiates it completely unchanged via experiment_runner_4.
build_model(). Configurations B/C are new classes defined here
(SinusoidalPositionalKinematicsTransformer / LearnedPositionalKinematicsTransformer)
that reuse the same building blocks KinematicsOnlyTransformer itself uses --
an nn.Linear input projection, Train_PyTorch.SurgicalEncoderLayer, and an
nn.Linear classification head -- with a positional-encoding step inserted
between the projection and the encoder, exactly as specified.

Because experiment_runner_4.train_fold() hardcodes model construction via its
own build_model(config) with no override hook, it cannot be reused directly
for B/C. train_fold_with_model() below mirrors train_fold()'s outer loop but
accepts an already-built model; it is used uniformly for all three strategies
for consistency. All three strategies keep 38 PSM features and a plain
model(source)->logits forward signature, so experiment_runner_4.train_epoch(),
experiment_runner_4.evaluate(), experiment_runner_8.predict_trial_frame_level(),
and every Train_PyTorch.py data/LOUO/standardisation/audit helper (via
experiment_runner_4.tp) are reused completely unmodified for all three
configurations. Train_PyTorch.py, experiment_runner_4/5/6/7/8/9/10/11.py,
data_prep.py, and run_all_pytorch.py are never modified.

===============================================================================
COMMANDS
===============================================================================

Smoke test:

    python "Gesture Classification\\experiment_runner_12.py" `
        --mode smoke `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_12_smoke"

Single-procedure run (Knot Tying, primary experiment):

    python "Gesture Classification\\experiment_runner_12.py" `
        --mode single `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_12_knot_tying"

Full-dataset run (supported, not run automatically):

    python "Gesture Classification\\experiment_runner_12.py" `
        --mode full `
        --kinematics-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\Knot_Tying kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Knot_Tying\\Knot_Tying\\transcriptions" `
        --kinematics-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\Needle_Passing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Needle_Passing\\Needle_Passing\\transcriptions" `
        --kinematics-dir "JIGSAW\\Suturing\\Suturing\\Suturing kinematics\\AllGestures" `
        --annotations-dir "JIGSAW\\Suturing\\Suturing\\transcriptions" `
        --output-root "Archive results\\PyTorch_2 Experements\\outputs_pytorch_experiment_12_full"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install an appropriate PyTorch build "
        "before running experiment_runner_12.py."
    ) from exc
from torch.utils.data import DataLoader

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
GESTURE_ID_TO_LABEL = e4.GESTURE_ID_TO_LABEL
KINEMATIC_DIM_PER_SOURCE = e4.KINEMATIC_DIM_PER_SOURCE

# Fixed "current best configuration". Positional-encoding strategy is the
# only intended independent variable; none of these are CLI-tunable.
KINEMATIC_SOURCE = "psm"
SAMPLE_RATE = 30.0
WINDOW_SECONDS = 1.5
WINDOW_FRAMES = 45
STRIDE_SAMPLES = 1
BATCH_SIZE = 64
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_METRIC = "macro_f1"
MAX_EPOCHS = 15
STANDARDIZE = True
RANDOM_SEED = 42

BASELINE_STRATEGY = "none"
ATTENTION_GESTURES = ("G1", "G12", "G13", "G14", "G15")


@dataclass(frozen=True)
class TestPreset:
    name: str
    epochs: int
    max_trials: Optional[int]
    max_folds: Optional[int]
    max_windows: Optional[int]


PRESETS = {
    "smoke": TestPreset("smoke", 1, 2, 2, 1000),
    "single": TestPreset("single", 15, None, None, None),
    "full": TestPreset("full", 15, None, None, None),
}


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
# REPRODUCIBILITY
# =============================================================================

def enable_determinism(seed: int) -> Dict[str, object]:
    """Seed everything; enable deterministic PyTorch behaviour where practical."""

    e4.tp.seed_everything(seed)

    settings: Dict[str, object] = {
        "seed": seed,
        "cudnn_deterministic": False,
        "cudnn_benchmark_disabled": False,
        "torch_deterministic_algorithms": False,
        "deterministic_algorithms_warning": None,
    }

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        settings["cudnn_deterministic"] = True
        settings["cudnn_benchmark_disabled"] = True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Could not configure cuDNN deterministic settings: {exc}")

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        settings["torch_deterministic_algorithms"] = True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] Could not enable strict deterministic algorithms; continuing with seeded reproducibility only: {exc}")
        settings["deterministic_algorithms_warning"] = str(exc)

    return settings


# =============================================================================
# POSITIONAL-ENCODING MODULES (new; do not modify KinematicsOnlyTransformer)
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sin/cos Transformer positional encoding; zero trainable parameters."""

    def __init__(self, dimension: int, max_len: int) -> None:
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension)
        )

        pe = torch.zeros(max_len, dimension, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        # persistent=False: this is a fixed, reconstructible constant, not a trainable parameter.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class LearnedPositionalEmbedding(nn.Module):
    """One trainable embedding vector per position, added to the projected representation."""

    def __init__(self, dimension: int, max_len: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(max_len, dimension))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.embedding[:seq_len].unsqueeze(0)


class SinusoidalPositionalKinematicsTransformer(nn.Module):
    """KinematicsOnlyTransformer + fixed sinusoidal positional encoding after the input projection."""

    def __init__(
        self,
        input_dimension: int = 38,
        encoder_dimension: int = 38,
        num_classes: int = 16,
        encoder_heads: int = 1,
        encoder_layers: int = 1,
        encoder_ff_dimension: int = 152,
        dropout: float = 0.1,
        max_len: int = 45,
    ) -> None:
        super().__init__()

        if encoder_dimension % encoder_heads != 0:
            raise ValueError("encoder_dimension must be divisible by encoder_heads.")

        self.input_dimension = input_dimension
        self.num_classes = num_classes

        self.input_projection = nn.Linear(input_dimension, encoder_dimension)
        self.positional_encoding = SinusoidalPositionalEncoding(encoder_dimension, max_len)

        self.encoder = nn.ModuleList(
            [
                e4.tp.SurgicalEncoderLayer(
                    dimension=encoder_dimension, heads=encoder_heads,
                    feedforward_dimension=encoder_ff_dimension, dropout=dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.classification_head = nn.Linear(encoder_dimension, num_classes)

    def forward(self, kinematics: torch.Tensor) -> torch.Tensor:
        if kinematics.ndim != 3 or kinematics.shape[-1] != self.input_dimension:
            raise RuntimeError(
                f"Expected input shaped [batch, time, {self.input_dimension}]; got {tuple(kinematics.shape)}."
            )

        memory = self.input_projection(kinematics)
        memory = self.positional_encoding(memory)

        for layer in self.encoder:
            memory = layer(memory)

        return self.classification_head(memory)


class LearnedPositionalKinematicsTransformer(nn.Module):
    """KinematicsOnlyTransformer + trainable positional embedding after the input projection."""

    def __init__(
        self,
        input_dimension: int = 38,
        encoder_dimension: int = 38,
        num_classes: int = 16,
        encoder_heads: int = 1,
        encoder_layers: int = 1,
        encoder_ff_dimension: int = 152,
        dropout: float = 0.1,
        max_len: int = 45,
    ) -> None:
        super().__init__()

        if encoder_dimension % encoder_heads != 0:
            raise ValueError("encoder_dimension must be divisible by encoder_heads.")

        self.input_dimension = input_dimension
        self.num_classes = num_classes

        self.input_projection = nn.Linear(input_dimension, encoder_dimension)
        self.positional_encoding = LearnedPositionalEmbedding(encoder_dimension, max_len)

        self.encoder = nn.ModuleList(
            [
                e4.tp.SurgicalEncoderLayer(
                    dimension=encoder_dimension, heads=encoder_heads,
                    feedforward_dimension=encoder_ff_dimension, dropout=dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.classification_head = nn.Linear(encoder_dimension, num_classes)

    def forward(self, kinematics: torch.Tensor) -> torch.Tensor:
        if kinematics.ndim != 3 or kinematics.shape[-1] != self.input_dimension:
            raise RuntimeError(
                f"Expected input shaped [batch, time, {self.input_dimension}]; got {tuple(kinematics.shape)}."
            )

        memory = self.input_projection(kinematics)
        memory = self.positional_encoding(memory)

        for layer in self.encoder:
            memory = layer(memory)

        return self.classification_head(memory)


def build_model_for_strategy(strategy: str, config: "e4.KinematicsOnlyConfig") -> nn.Module:
    if strategy == "none":
        return e4.build_model(config)

    if strategy == "sinusoidal":
        return SinusoidalPositionalKinematicsTransformer(
            input_dimension=KINEMATIC_DIM_PER_SOURCE, encoder_dimension=config.encoder_dim,
            num_classes=NUM_GESTURE_CLASSES, encoder_heads=config.encoder_heads,
            encoder_layers=config.encoder_layers, encoder_ff_dimension=config.encoder_ff_dim,
            dropout=config.dropout, max_len=WINDOW_FRAMES,
        )

    if strategy == "learned":
        return LearnedPositionalKinematicsTransformer(
            input_dimension=KINEMATIC_DIM_PER_SOURCE, encoder_dimension=config.encoder_dim,
            num_classes=NUM_GESTURE_CLASSES, encoder_heads=config.encoder_heads,
            encoder_layers=config.encoder_layers, encoder_ff_dimension=config.encoder_ff_dim,
            dropout=config.dropout, max_len=WINDOW_FRAMES,
        )

    raise ValueError(f"Unknown positional-encoding strategy: {strategy}")


# =============================================================================
# TRAINING (mirrors experiment_runner_4.train_fold(); duplicated because that
# function hardcodes model construction via its own build_model() with no
# override hook -- everything else it calls (train_epoch/evaluate/tp.*) is
# reused completely unmodified since all three strategies keep 38 features)
# =============================================================================

def train_fold_with_model(
    model: nn.Module,
    train_trials: List["e4.tp.TrialData"],
    test_trials: List["e4.tp.TrialData"],
    config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    run_name: str,
    held_out_surgeon: str,
) -> Tuple[nn.Module, Dict[str, object], List[Dict[str, object]], Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    window_frames = WINDOW_FRAMES

    if config.standardize:
        mean, std = e4.tp.calculate_standardization(train_trials)
        normalization_surgeons = sorted({t.surgeon_id for t in train_trials})
        if held_out_surgeon in normalization_surgeons:
            raise RuntimeError("DATA LEAKAGE: held-out surgeon was used to calculate normalization statistics.")
    else:
        mean = None
        std = None

    train_dataset = e4.tp.make_split_dataset(
        trials=train_trials, window_frames=window_frames, stride_samples=config.stride_samples,
        standardize=config.standardize, mean=mean, std=std, max_windows=config.max_windows,
        random_seed=config.random_seed,
    )
    if len(train_dataset) == 0:
        raise ValueError(f"{run_name}: no training windows generated.")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    test_dataset = e4.tp.make_split_dataset(
        trials=test_trials, window_frames=window_frames, stride_samples=config.stride_samples,
        standardize=config.standardize, mean=mean, std=std, max_windows=config.max_windows,
        random_seed=config.random_seed + 1,
    )
    if len(test_dataset) == 0:
        raise ValueError(f"{run_name}: no testing windows generated.")

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"), drop_last=False,
    )

    e4.tp.audit_split_preprocessing(
        train_trials=train_trials, test_trials=test_trials,
        train_dataset=train_dataset, test_dataset=test_dataset,
        window_frames=window_frames, stride_samples=config.stride_samples,
        standardize=config.standardize, mean=mean, std=std,
        held_out_surgeon=held_out_surgeon,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0,
        betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    scheduler = e4.tp.NoamLearningRate(optimizer=optimizer, model_dimension=config.encoder_dim, warmup_steps=config.warmup_steps)

    print(f"[MODEL] {run_name} | Training windows: {len(train_dataset):,} | Testing windows: {len(test_dataset):,}")

    history: List[Dict[str, object]] = []
    best_validation_metric: Optional[float] = None
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: Optional[int] = None
    best_train_accuracy: Optional[float] = None
    best_validation_accuracy: Optional[float] = None
    best_validation_macro_f1: Optional[float] = None
    patience_counter = 0

    fold_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = e4.train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler,
            criterion=criterion, device=device, epoch_number=epoch, total_epochs=config.epochs,
            progress_updates=config.progress_updates_per_epoch,
        )

        epoch_elapsed = time.perf_counter() - epoch_start
        print(
            f"[EPOCH] {run_name} | {epoch}/{config.epochs} complete | Loss {train_loss:.5f} | "
            f"Train accuracy {train_accuracy:.4f} | Epoch time {format_duration(epoch_elapsed)}"
        )

        validation_metrics = e4.evaluate(model=model, loader=test_loader, device=device)
        validation_accuracy = float(validation_metrics["accuracy"])
        validation_macro_f1 = float(validation_metrics["macro_f1"])
        early_stopping_value = float(validation_metrics[config.early_stopping_metric])

        if best_validation_metric is None or early_stopping_value > best_validation_metric:
            best_validation_metric = early_stopping_value
            best_model_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_train_accuracy = train_accuracy
            best_validation_accuracy = validation_accuracy
            best_validation_macro_f1 = validation_macro_f1
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[VALIDATION] {run_name} | Accuracy {validation_accuracy:.4f} | Macro F1 {validation_macro_f1:.4f} | "
            f"Best {config.early_stopping_metric} {best_validation_metric:.4f} | "
            f"Patience {patience_counter}/{config.early_stopping_patience}"
        )

        history.append(
            {
                "run": run_name, "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                "learning_rate": scheduler.current_lr, "epoch_seconds": epoch_elapsed,
                "validation_accuracy": validation_accuracy, "validation_macro_f1": validation_macro_f1,
                "early_stopping_value": early_stopping_value,
                "best_validation_value_so_far": best_validation_metric,
                "patience_counter": patience_counter,
            }
        )

        if patience_counter >= config.early_stopping_patience:
            print(f"[EARLY STOPPING] {run_name} | No improvement in {config.early_stopping_metric} for {patience_counter} epoch(s).")
            break

    if best_model_state is None:
        raise RuntimeError("No validation checkpoint was recorded.")

    model.load_state_dict(best_model_state)

    print(f"[EVAL] Final held-out evaluation for {run_name} (best epoch {best_epoch})")
    metrics = e4.evaluate(model=model, loader=test_loader, device=device)
    print(f"[EVAL] {run_name} | Accuracy {metrics['accuracy']:.4f} | Macro F1 {metrics['macro_f1']:.4f}")

    runtime_seconds = time.perf_counter() - fold_start

    fold_summary = {
        "best_epoch": best_epoch,
        "train_accuracy_at_best_epoch": best_train_accuracy,
        "validation_accuracy_at_best_epoch": best_validation_accuracy,
        "validation_macro_f1_at_best_epoch": best_validation_macro_f1,
        "runtime_seconds": runtime_seconds,
    }

    return model, metrics, history, mean, std, fold_summary


def save_positional_checkpoint(
    path: Path,
    model: nn.Module,
    strategy: str,
    config: "e4.KinematicsOnlyConfig",
    kinematic_columns: List[str],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> None:
    """e4.save_checkpoint() has no field for positional-encoding strategy; record it explicitly."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "positional_encoding": strategy,
        "kinematic_source": config.kinematic_source,
        "kinematic_columns": list(kinematic_columns),
        "gesture_id_to_label": GESTURE_ID_TO_LABEL,
        "num_gesture_classes": NUM_GESTURE_CLASSES,
        "sample_rate": config.sample_rate,
        "window_seconds": config.window_seconds,
        "window_frames": WINDOW_FRAMES,
        "stride_samples": config.stride_samples,
        "standardize": config.standardize,
        "mean": mean.tolist() if mean is not None else None,
        "std": std.tolist() if std is not None else None,
        "encoder_dim": config.encoder_dim,
        "encoder_heads": config.encoder_heads,
        "encoder_layers": config.encoder_layers,
        "encoder_ff_dim": config.encoder_ff_dim,
        "dropout": config.dropout,
    }
    torch.save(checkpoint, path)


# =============================================================================
# PER-STRATEGY ORCHESTRATION
# =============================================================================

def print_check_lines(strategy_label: str) -> None:
    print(f"[EXPERIMENT] Positional encoding: {strategy_label}")
    print("[CHECK] Kinematic source: PSM")
    print("[CHECK] Input features: k39-k76")
    print(f"[CHECK] Number of kinematic features: {KINEMATIC_DIM_PER_SOURCE}")
    print(f"[CHECK] Window: {WINDOW_SECONDS} sec / {WINDOW_FRAMES} frames")
    print(f"[CHECK] Standardisation: {'ON' if STANDARDIZE else 'OFF'}")
    print(f"[CHECK] Dropout: {DROPOUT}")
    print(f"[CHECK] Weight decay: {WEIGHT_DECAY}")
    print("[CHECK] Loss: unweighted CrossEntropyLoss")
    print("[CHECK] Previous gesture supplied as input: NO")
    print("[CHECK] Teacher forcing: NO")
    print("[CHECK] Autoregressive label feedback: NO")
    print(f"[CHECK] Random seed: {RANDOM_SEED}")


def run_positional_strategy(
    strategy: str,
    label: str,
    trials: List["e4.tp.TrialData"],
    fold_surgeons: List[str],
    config: "e4.KinematicsOnlyConfig",
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_models_dir = output_dir / "fold_models"
    fold_models_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"[STRATEGY] {label}")
    print("=" * 78)
    print_check_lines(strategy.upper())

    fold_results: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    trainable_parameters: Optional[int] = None

    experiment_start = time.perf_counter()

    for fold_number, held_out_surgeon in enumerate(fold_surgeons, start=1):
        fold_start = time.perf_counter()

        train_trials = [t for t in trials if t.surgeon_id != held_out_surgeon]
        test_trials = [t for t in trials if t.surgeon_id == held_out_surgeon]

        e4.tp.validate_louo_fold(train_trials=train_trials, test_trials=test_trials, held_out_surgeon=held_out_surgeon)

        print()
        print(f"[FOLD] {label} | Fold {fold_number}/{len(fold_surgeons)} | Held-out: {held_out_surgeon}")

        model = build_model_for_strategy(strategy, config).to(device)

        if trainable_parameters is None:
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[MODEL] {label} | Trainable parameters: {trainable_parameters:,}")

        model, metrics, _history, mean, std, fold_summary = train_fold_with_model(
            model=model, train_trials=train_trials, test_trials=test_trials, config=config,
            device=device, run_name=f"{strategy}_LOUO_{held_out_surgeon}", held_out_surgeon=held_out_surgeon,
        )

        save_positional_checkpoint(
            path=fold_models_dir / f"LOUO_{held_out_surgeon}.pt", model=model, strategy=strategy,
            config=config, kinematic_columns=e4.tp.get_kinematic_columns(KINEMATIC_SOURCE), mean=mean, std=std,
        )

        model.eval()
        for trial in test_trials:
            result = e8.predict_trial_frame_level(
                model=model, trial=trial, window_frames=WINDOW_FRAMES, stride_samples=STRIDE_SAMPLES,
                standardize=config.standardize, mean=mean, std=std, device=device,
            )
            if result is None:
                print(f"[WARN] Trial {trial.trial_id} is shorter than the window length; skipped.")
                continue
            frame_predictions, frame_confidences = result
            for i in range(len(trial.labels)):
                prediction_rows.append(
                    {
                        "positional_encoding": strategy, "held_out_surgeon": held_out_surgeon,
                        "trial_id": trial.trial_id, "frame_idx": int(trial.frame_indices[i]),
                        "true_gesture": int(trial.labels[i]), "predicted_gesture": int(frame_predictions[i]),
                        "confidence": float(frame_confidences[i]),
                    }
                )

        fold_elapsed = time.perf_counter() - fold_start
        experiment_elapsed = time.perf_counter() - experiment_start
        eta = (experiment_elapsed / fold_number) * (len(fold_surgeons) - fold_number)
        print(f"[RUNTIME] Fold {fold_number}/{len(fold_surgeons)} time: {format_duration(fold_elapsed)}")
        print(f"[RUNTIME] {label} elapsed: {format_duration(experiment_elapsed)} | ETA: {format_duration(eta)}")

        fold_results.append(
            {
                "held_out_surgeon": held_out_surgeon, "positional_encoding": strategy,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
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

    strategy_metrics = {
        "positional_encoding": strategy,
        "label": label,
        "trainable_parameters": trainable_parameters,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_macro_f1": float(np.mean(macro_f1_values)),
        "std_macro_f1": float(np.std(macro_f1_values)),
        "folds": fold_results,
        "total_runtime_seconds": total_runtime,
    }

    with (output_dir / "kinematics_only_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(strategy_metrics, file, indent=2)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(output_dir / "kinematics_only_predictions.csv", index=False)

    print()
    print(f"[DONE] {label} complete. Total runtime: {format_duration(total_runtime)}")

    return strategy_metrics, predictions_df


# =============================================================================
# CROSS-STRATEGY SUMMARY / BY-SURGEON / BY-GESTURE / CONFUSION OUTPUTS
# =============================================================================

def build_summary_rows(all_strategy_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for strategy_metrics in all_strategy_metrics:
        fold_results = strategy_metrics["folds"]
        best_epochs = [f["best_epoch"] for f in fold_results if f.get("best_epoch") is not None]
        train_acc = [f["train_accuracy_at_best_epoch"] for f in fold_results if f.get("train_accuracy_at_best_epoch") is not None]
        val_acc = [f["validation_accuracy_at_best_epoch"] for f in fold_results if f.get("validation_accuracy_at_best_epoch") is not None]

        rows.append(
            {
                "positional_encoding": strategy_metrics["positional_encoding"],
                "trainable_parameters": strategy_metrics["trainable_parameters"],
                "mean_louo_accuracy": strategy_metrics["mean_accuracy"],
                "std_louo_accuracy": strategy_metrics["std_accuracy"],
                "mean_louo_macro_f1": strategy_metrics["mean_macro_f1"],
                "std_louo_macro_f1": strategy_metrics["std_macro_f1"],
                "mean_training_accuracy_at_best_epoch": float(np.mean(train_acc)) if train_acc else None,
                "mean_validation_accuracy_at_best_epoch": float(np.mean(val_acc)) if val_acc else None,
                "mean_best_epoch": float(np.mean(best_epochs)) if best_epochs else None,
                "total_runtime_seconds": strategy_metrics["total_runtime_seconds"],
            }
        )

    baseline_row = next(row for row in rows if row["positional_encoding"] == BASELINE_STRATEGY)
    for row in rows:
        row["change_in_accuracy_vs_none"] = row["mean_louo_accuracy"] - baseline_row["mean_louo_accuracy"]
        row["change_in_macro_f1_vs_none"] = row["mean_louo_macro_f1"] - baseline_row["mean_louo_macro_f1"]

    rows.sort(key=lambda row: float(row["mean_louo_macro_f1"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def build_by_surgeon_rows(all_strategy_metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for strategy_metrics in all_strategy_metrics:
        for fold in strategy_metrics["folds"]:
            rows.append(
                {
                    "positional_encoding": fold["positional_encoding"], "held_out_surgeon": fold["held_out_surgeon"],
                    "accuracy": fold["accuracy"], "macro_f1": fold["macro_f1"], "best_epoch": fold["best_epoch"],
                    "training_accuracy_at_best_epoch": fold["train_accuracy_at_best_epoch"],
                    "validation_accuracy_at_best_epoch": fold["validation_accuracy_at_best_epoch"],
                    "runtime": fold["runtime_seconds"],
                }
            )
    rows.sort(key=lambda row: (row["held_out_surgeon"], row["positional_encoding"]))
    return rows


def build_by_gesture_rows(all_predictions: pd.DataFrame) -> List[Dict[str, object]]:
    gesture_tables: Dict[str, pd.DataFrame] = {}

    for strategy_name, strategy_predictions in all_predictions.groupby("positional_encoding"):
        confusion = e8.build_confusion_matrix(strategy_predictions)
        rows = []
        for gesture_id in range(NUM_GESTURE_CLASSES):
            true_positive = int(confusion[gesture_id, gesture_id])
            false_positive = int(confusion[:, gesture_id].sum() - true_positive)
            false_negative = int(confusion[gesture_id, :].sum() - true_positive)
            support = int(confusion[gesture_id, :].sum())

            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            rows.append(
                {
                    "gesture_id": gesture_id, "gesture": GESTURE_ID_TO_LABEL[gesture_id],
                    "support": support, "precision": precision, "recall": recall, "f1": f1,
                }
            )
        gesture_tables[strategy_name] = pd.DataFrame(rows).set_index("gesture_id")

    baseline_table = gesture_tables[BASELINE_STRATEGY]

    output_rows = []
    for strategy_name, table in gesture_tables.items():
        for gesture_id, row in table.iterrows():
            baseline_row = baseline_table.loc[gesture_id]
            output_rows.append(
                {
                    "positional_encoding": strategy_name, "gesture": row["gesture"], "support": row["support"],
                    "precision": row["precision"], "recall": row["recall"], "f1": row["f1"],
                    "change_in_F1_vs_none": row["f1"] - baseline_row["f1"],
                    "change_in_recall_vs_none": row["recall"] - baseline_row["recall"],
                }
            )

    return output_rows


def save_confusion_matrices(all_predictions: pd.DataFrame, output_root: Path) -> None:
    labels = [GESTURE_ID_TO_LABEL[g] for g in range(NUM_GESTURE_CLASSES)]

    for strategy_name, strategy_predictions in all_predictions.groupby("positional_encoding"):
        confusion = e8.build_confusion_matrix(strategy_predictions)

        counts_df = pd.DataFrame(confusion, index=labels, columns=labels)
        counts_df.index.name = "true_gesture"
        counts_df.to_csv(output_root / f"kinematics_only_positional_encoding_confusion_counts_{strategy_name}.csv")

        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(
            confusion.astype(np.float64) * 100.0, row_sums,
            out=np.zeros_like(confusion, dtype=np.float64), where=row_sums != 0,
        )
        normalized_df = pd.DataFrame(normalized, index=labels, columns=labels)
        normalized_df.index.name = "true_gesture"
        normalized_df.to_csv(output_root / f"kinematics_only_positional_encoding_confusion_normalized_{strategy_name}.csv")


# =============================================================================
# PIPELINE
# =============================================================================

def run_experiments(args: argparse.Namespace) -> Path:
    preset = PRESETS[args.mode]
    output_root = Path(args.output_root).resolve()
    prepared_data_dir = output_root / "prepared_data"
    experiment_root = output_root / "positional_encoding_experiments"
    output_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    assert WINDOW_FRAMES == round(WINDOW_SECONDS * SAMPLE_RATE)
    assert KINEMATIC_DIM_PER_SOURCE == 38

    determinism_settings = enable_determinism(RANDOM_SEED)
    device = e4.tp.choose_device(args.device)

    if args.reuse_prepared_data:
        frame_file = prepared_data_dir / "all_frame_level.csv"
        require_file(frame_file, "existing frame-level dataset")
    else:
        frame_file = prepare_data(
            [Path(path).resolve() for path in args.kinematics_dir],
            [Path(path).resolve() for path in args.annotations_dir],
            prepared_data_dir, SAMPLE_RATE, preset.max_trials,
        )

    trials, _kinematic_columns = e4.tp.load_frame_level_data(path=frame_file, kinematic_source=KINEMATIC_SOURCE)

    surgeons = sorted({trial.surgeon_id for trial in trials})
    fold_surgeons = surgeons if preset.max_folds is None else surgeons[: preset.max_folds]

    max_windows = args.max_windows if args.max_windows is not None else preset.max_windows

    if args.mode == "smoke":
        print()
        print("[SMOKE] Smoke-test mode enabled: max_folds<=2, epochs=1, max_windows<=1000.")
        print("[SMOKE] Verifying: 38 PSM features, sequence length 45, output [batch,45,16],")
        print("[SMOKE] sinusoidal encoding has zero trainable parameters, learned encoding does.")
        print("[SMOKE] Smoke-test results are NOT scientifically meaningful.")

    config = e4.KinematicsOnlyConfig(
        input_csv=frame_file, output_dir=experiment_root, kinematic_source=KINEMATIC_SOURCE,
        sample_rate=SAMPLE_RATE, window_seconds=WINDOW_SECONDS, stride_samples=STRIDE_SAMPLES,
        batch_size=BATCH_SIZE, epochs=preset.epochs, dropout=DROPOUT, weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE, early_stopping_metric=EARLY_STOPPING_METRIC,
        standardize=STANDARDIZE, random_seed=RANDOM_SEED, device=args.device, max_windows=max_windows,
    )

    print(
        f"Running exactly 3 positional-encoding configurations with seed {RANDOM_SEED} against "
        f"the same prepared dataset and the same {len(fold_surgeons)} LOUO fold(s) (data "
        f"preparation is not rerun between configurations). Positional-encoding strategy is "
        f"the only intended independent variable."
    )

    pipeline_start = time.perf_counter()

    strategies = (
        ("none", "A - No positional encoding (current baseline)"),
        ("sinusoidal", "B - Fixed sinusoidal positional encoding"),
        ("learned", "C - Learned positional embedding"),
    )

    all_strategy_metrics: List[Dict[str, object]] = []
    all_predictions_frames: List[pd.DataFrame] = []

    for strategy, label in strategies:
        strategy_metrics, predictions_df = run_positional_strategy(
            strategy=strategy, label=label, trials=trials, fold_surgeons=fold_surgeons,
            config=config, device=device, output_dir=experiment_root / strategy,
        )
        all_strategy_metrics.append(strategy_metrics)
        all_predictions_frames.append(predictions_df)

    all_predictions = pd.concat(all_predictions_frames, ignore_index=True)

    baseline_parameters = next(m["trainable_parameters"] for m in all_strategy_metrics if m["positional_encoding"] == "none")
    for strategy_metrics in all_strategy_metrics:
        delta = strategy_metrics["trainable_parameters"] - baseline_parameters
        print(
            f"[MODEL CAPACITY] {strategy_metrics['label']}: {strategy_metrics['trainable_parameters']:,} "
            f"trainable parameters ({'+' if delta >= 0 else ''}{delta:,} vs no positional encoding)"
        )

    summary_rows = build_summary_rows(all_strategy_metrics)
    by_surgeon_rows = build_by_surgeon_rows(all_strategy_metrics)
    by_gesture_rows = build_by_gesture_rows(all_predictions)

    summary_path = output_root / "kinematics_only_positional_encoding_summary.csv"
    summary_fieldnames = [
        "rank", "positional_encoding", "trainable_parameters",
        "mean_louo_accuracy", "std_louo_accuracy", "mean_louo_macro_f1", "std_louo_macro_f1",
        "mean_training_accuracy_at_best_epoch", "mean_validation_accuracy_at_best_epoch",
        "mean_best_epoch", "total_runtime_seconds",
        "change_in_accuracy_vs_none", "change_in_macro_f1_vs_none",
    ]
    pd.DataFrame(summary_rows)[summary_fieldnames].to_csv(summary_path, index=False)

    by_surgeon_path = output_root / "kinematics_only_positional_encoding_by_surgeon.csv"
    pd.DataFrame(by_surgeon_rows).to_csv(by_surgeon_path, index=False)

    by_gesture_path = output_root / "kinematics_only_positional_encoding_by_gesture.csv"
    pd.DataFrame(by_gesture_rows).to_csv(by_gesture_path, index=False)

    save_confusion_matrices(all_predictions, output_root)

    predictions_path = output_root / "kinematics_only_positional_encoding_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False)

    total_runtime = time.perf_counter() - pipeline_start

    metadata = {
        "reproducibility": determinism_settings,
        "device": str(device),
        "total_runtime_seconds": total_runtime,
    }
    with (output_root / "kinematics_only_positional_encoding_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    attention_gesture_rows = [row for row in by_gesture_rows if row["gesture"] in ATTENTION_GESTURES]

    print()
    print("=" * 79)
    print("[DONE] KINEMATICS-ONLY POSITIONAL-ENCODING EXPERIMENT COMPLETE")
    print("=" * 79)
    print(f"Complete experiment runtime: {format_duration(total_runtime)}")
    print(f"Summary written to {summary_path}")
    print(f"Per-surgeon comparison written to {by_surgeon_path}")
    print(f"Per-gesture comparison written to {by_gesture_path}")
    print(f"Confusion matrices written under {output_root}")
    print(f"Held-out predictions written to {predictions_path}")
    print(f"Per-fold checkpoints written under {experiment_root}/<strategy>/fold_models/")
    print()
    print(f"[G1/G12-G15] {len(attention_gesture_rows)} rows reported in {by_gesture_path.name} for review.")

    return summary_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled PSM kinematics-only Transformer positional-encoding "
            "experiment (experiment_runner_4.py/experiment_runner_8.py helpers, not "
            "Train_PyTorch.py). Positional-encoding strategy is the only intended "
            "independent variable."
        )
    )
    parser.add_argument("--mode", choices=sorted(PRESETS), required=True)
    parser.add_argument("--kinematics-dir", action="append", required=True)
    parser.add_argument("--annotations-dir", action="append", required=True)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs_pytorch_kinematics_only_positional_encoding"),
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
