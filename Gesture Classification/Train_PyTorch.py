"""
ATARI-2: Gesture Classification / Train_PyTorch.py

===============================================================================
PURPOSE
===============================================================================

This script trains a PyTorch Transformer model to recognise surgical gestures
from JIGSAWS robot kinematic data.

It is designed as a gesture-recognition-only implementation based on:

    Chen, K., Bandara, D. S. V., Arata, J.
    "A real-time approach for surgical activity recognition and prediction
    based on transformer models in robot-assisted surgery."
    International Journal of Computer Assisted Radiology and Surgery, 2025.
    DOI: 10.1007/s11548-024-03306-9

and the closely related earlier Transformer work cited by that paper:

    Shi, C., Zheng, Y., Fey, A. M.
    "Recognition and Prediction of Surgical Gestures and Trajectories
    Using Transformer Models in Robot-Assisted Surgery."
    IROS 2022.

This file intentionally implements ONLY GESTURE RECOGNITION.

It does NOT implement:
    - future gesture prediction
    - future trajectory prediction

===============================================================================
CORE IDEA
===============================================================================

The network receives one second of current surgical robot kinematics and
predicts the gesture occurring at every frame within that same second.

JIGSAWS is sampled at 30 Hz.

Therefore, by default:

    input window:
        30 kinematic frames = 1 second

    output:
        30 gesture predictions

For example:

    Kinematics:

        frame 100
        frame 101
        ...
        frame 129

             |
             v

        Transformer

             |
             v

    Gesture predictions:

        G2
        G2
        G2
        ...
        G3

The model therefore performs sequence-to-sequence gesture recognition rather
than assigning one gesture to an entire one-second window.

This is different from the XGBoost pipeline, where engineered statistics are
calculated for each window and the entire window receives a single label.

===============================================================================
INPUT DATA
===============================================================================

This script expects:

    all_frame_level.csv

created by ATARI-2's data_prep.py.

The CSV should contain at least:

    trial_id
    surgeon_id
    frame_idx
    gesture_id
    k01
    k02
    ...
    k76

The current ATARI-2 data preparation pipeline already produces these raw
frame-level kinematic values.

The Transformer therefore does NOT use the engineered XGBoost features such as:

    mean
    standard deviation
    range
    jerk
    spectral entropy

Instead, it learns temporal representations directly from the sequence of raw
kinematic measurements.

===============================================================================
KINEMATIC INPUT
===============================================================================

Each JIGSAWS frame contains 76 kinematic variables:

    k01-k38:
        left + right Master Tool Manipulators (MTMs)

    k39-k76:
        left + right Patient-Side Manipulators (PSMs)

The paper evaluates the two sets independently.

This script defaults to:

    --kinematic-source mtm

which gives 38 input variables per frame.

PSM data can instead be selected with:

    --kinematic-source psm

Do NOT combine all 76 values if the intention is to reproduce the method
described in the paper. The published experiment treats the two 38-dimensional
sources independently.

This separately analyses each hand of the surgeon.

===============================================================================
MODEL ARCHITECTURE
===============================================================================

The gesture recognition network implemented here contains:

    Raw kinematics
          |
          v
    Fully connected input layer
          |
          v
    Transformer encoder
          |
          | memory
          v
    Gesture-recognition Transformer decoder (DEC_GR)
          |
          v
    Fully connected output layer
          |
          v
    16 gesture logits per frame

The model deliberately has:

    NO POSITIONAL ENCODING

in accordance with the 2025 paper.

The encoder contains:

    multi-head self-attention
    residual connection
    normalisation
    feed-forward network
    residual connection
    normalisation

The gesture-recognition decoder contains:

    masked multi-head self-attention
    residual + normalisation

    encoder-decoder attention
    residual + normalisation

    feed-forward network
    residual + normalisation

The decoder uses a causal/look-ahead mask.

This prevents a prediction at time t from seeing future gesture labels.

===============================================================================
GESTURE CLASSES
===============================================================================

The network uses 16 classes:

    0   BACKGROUND
    1   G1
    2   G2
    ...
    15  G15

Not every task uses every gesture.

For example, Knot Tying contains a different subset of the gesture vocabulary
from Suturing.

A combined model can nevertheless learn the shared 16-class vocabulary.

===============================================================================
TEACHER FORCING
===============================================================================

During TRAINING the decoder uses teacher forcing.

For a target sequence:

    G1 G1 G2 G2 G3

the decoder input is shifted by one frame:

    previous_label G1 G1 G2 G2

The true previous gesture is therefore available to the decoder during
training.

During INFERENCE the true sequence is unavailable.

The decoder instead predicts gestures autoregressively:

    predict gesture 1
         |
         v
    feed prediction back into decoder
         |
         v
    predict gesture 2
         |
         v
        ...

This reflects the recognition procedure described in the Transformer work.

===============================================================================
TRAINING
===============================================================================

Default paper-style settings:

    sample rate              = 30 Hz
    window                   = 1 second / 30 frames
    stride                   = 1 frame
    batch size               = 64
    epochs                   = 15
    Adam beta1               = 0.9
    Adam beta2               = 0.98
    Adam epsilon             = 1e-9
    warm-up steps            = 2000

A Transformer-style learning-rate schedule is used:

    learning rate =
        d_model^(-0.5)
        *
        min(
            step^(-0.5),
            step * warmup_steps^(-1.5)
        )

===============================================================================
CROSS-VALIDATION
===============================================================================

The default evaluation method is:

    Leave-One-User-Out (LOUO)

For example:

    Fold 1:
        train = surgeons C,D,E,F,G,H,I
        test  = surgeon B

    Fold 2:
        train = surgeons B,D,E,F,G,H,I
        test  = surgeon C

    ...

This is much more rigorous than randomly mixing windows because the model has
to recognise gestures performed by a surgeon it has never seen during training.

After LOUO evaluation is complete, one final deployment model is trained on ALL
available surgeons.

That final model is saved for predict_gestures.py.

===============================================================================
OUTPUT FILES
===============================================================================

The output directory contains:

    pytorch_gesture_model.pt
        Final trained PyTorch model.

    pytorch_config.json
        Model and preprocessing settings.

    pytorch_metrics.json
        LOUO cross-validation results.

    pytorch_training_history.csv
        Training loss and timing information.

    pytorch_kinematic_columns.json
        The 38 kinematic columns expected by inference.

The .pt checkpoint also stores:

    model weights
    model architecture
    gesture mapping
    selected kinematic source
    window size
    normalisation settings

This will allow predict_gestures.py to recreate exactly the same model at
inference time.

===============================================================================
COMPUTATIONAL COST
===============================================================================

This model is considerably more computationally expensive than XGBoost.

With full JIGSAWS data:

    - windows overlap every single frame
    - each window contains 30 frames
    - the decoder processes complete sequences
    - LOUO trains a separate model for every surgeon
    - each fold runs for multiple epochs

GPU training is strongly preferred. DGX comes in to be very important.

The script automatically uses:

    CUDA GPU -> if available
    Apple MPS -> if available
    CPU       -> otherwise

Progress, elapsed time and ETA are printed during training.

===============================================================================
EXAMPLE
===============================================================================

After data_prep.py has generated:

    outputs/prepared_data/all_frame_level.csv

run:

python "Gesture Classification/Train_PyTorch.py" \
    --input-csv "outputs/prepared_data/all_frame_level.csv" \
    --output-dir "outputs/pytorch_model"

For a quick smoke test:

python "Gesture Classification/Train_PyTorch.py" \
    --input-csv "outputs/prepared_data/all_frame_level.csv" \
    --output-dir "outputs/pytorch_model" \
    --epochs 1 \
    --max-folds 2 \
    --max-windows 1000

===============================================================================
IMPORTANT REPRODUCIBILITY NOTE
===============================================================================

The published 2025 article describes the major network architecture but does
not publicly expose every implementation constant or the authors' source code.

Where exact constants are not stated in the accessible article, this script
uses the standard Transformer settings and the training procedure described in
the closely related 2022 Transformer work.

Those parameters are exposed as command-line options so they can later be
adjusted during replication experiments.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Install an appropriate PyTorch build "
        "before running Train_PyTorch.py."
    ) from exc


# =============================================================================
# CONSTANTS
# =============================================================================

NUM_GESTURE_CLASSES = 16
KINEMATIC_DIM_PER_SOURCE = 38

GESTURE_ID_TO_LABEL: Dict[int, str] = {
    0: "BACKGROUND",
    1: "G1",
    2: "G2",
    3: "G3",
    4: "G4",
    5: "G5",
    6: "G6",
    7: "G7",
    8: "G8",
    9: "G9",
    10: "G10",
    11: "G11",
    12: "G12",
    13: "G13",
    14: "G14",
    15: "G15",
}

DEFAULT_RANDOM_SEED = 42


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class TrainConfig:
    input_csv: Path
    output_dir: Path

    kinematic_source: str = "mtm"

    sample_rate: float = 30.0
    window_seconds: float = 1.0
    stride_samples: int = 1

    batch_size: int = 64
    epochs: int = 15

    encoder_dim: int = 38
    decoder_dim: int = 16

    encoder_heads: int = 1
    decoder_heads: int = 1

    encoder_layers: int = 1
    decoder_layers: int = 1

    encoder_ff_dim: int = 152
    decoder_ff_dim: int = 64

    dropout: float = 0.1
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    early_stopping_metric: str = "macro_f1"

    warmup_steps: int = 2000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-9

    standardize: bool = False

    num_workers: int = 0
    random_seed: int = DEFAULT_RANDOM_SEED

    device: str = "auto"

    max_folds: Optional[int] = None
    max_windows: Optional[int] = None

    skip_cv: bool = False
    save_fold_models: bool = False

    progress_updates_per_epoch: int = 10


# =============================================================================
# BASIC HELPERS
# =============================================================================

def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))

    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts: List[str] = []

    if hours:
        parts.append(f"{hours} hr")

    if minutes or hours:
        parts.append(f"{minutes} min")

    parts.append(f"{secs} sec")

    return " ".join(parts)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()

    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def infer_surgeon_id(trial_id: str) -> str:
    """
    Fallback surgeon inference if surgeon_id is missing from the CSV.

    Examples:
        Suturing_B001       -> B
        Knot_Tying_C003     -> C
        Needle_Passing_H002 -> H
    """

    suffix = trial_id.rsplit("_", 1)[-1]

    if not suffix:
        raise ValueError(f"Cannot infer surgeon from trial ID: {trial_id}")

    return suffix[0]


def get_kinematic_columns(source: str) -> List[str]:
    """
    Select the 38 dimensions corresponding to either MTMs or PSMs.
    """

    source = source.lower()

    if source == "mtm":
        return [f"k{i:02d}" for i in range(1, 39)]

    if source == "psm":
        return [f"k{i:02d}" for i in range(39, 77)]

    raise ValueError(
        f"Unknown kinematic source '{source}'. "
        "Expected 'mtm' or 'psm'."
    )


# =============================================================================
# DATA REPRESENTATION
# =============================================================================

@dataclass
class TrialData:
    trial_id: str
    surgeon_id: str
    task: str

    kinematics: np.ndarray
    labels: np.ndarray
    frame_indices: np.ndarray


def load_frame_level_data(
    path: Path,
    kinematic_source: str,
) -> Tuple[List[TrialData], List[str]]:
    """
    Load all_frame_level.csv and convert it into per-trial arrays.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    print(f"[DATA] Loading {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input frame-level CSV is empty.")

    required = {
        "trial_id",
        "frame_idx",
        "gesture_id",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "Frame-level CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    kinematic_columns = get_kinematic_columns(kinematic_source)

    missing_kin = [
        col for col in kinematic_columns
        if col not in df.columns
    ]

    if missing_kin:
        raise ValueError(
            "Frame-level CSV does not contain the expected raw kinematic "
            f"columns. Missing examples: {missing_kin[:5]}"
        )

    if "surgeon_id" not in df.columns:
        print(
            "[WARN] surgeon_id column not found. "
            "Inferring surgeon IDs from trial_id."
        )
        df["surgeon_id"] = df["trial_id"].astype(str).map(
            infer_surgeon_id
        )

    # Treat identifiers as categorical metadata, not model features.  Cleaning
    # them before constructing folds prevents accidental folds such as "B" and
    # " B " for the same surgeon.
    for column in ("trial_id", "surgeon_id"):
        if df[column].isna().any():
            raise ValueError(
                f"Frame-level CSV contains missing {column} values."
            )
        df[column] = df[column].astype(str).str.strip()
        if (df[column] == "").any():
            raise ValueError(
                f"Frame-level CSV contains blank {column} values."
            )

    if "task" not in df.columns:
        df["task"] = df["trial_id"].astype(str).map(
            lambda value: value.rsplit("_", 1)[0]
        )

    # Ensure valid gesture IDs.
    gesture_values = pd.to_numeric(
        df["gesture_id"],
        errors="raise",
    ).astype(int)

    invalid = sorted(
        set(gesture_values.unique())
        - set(range(NUM_GESTURE_CLASSES))
    )

    if invalid:
        raise ValueError(
            "Gesture IDs outside the expected 0-15 range were found: "
            f"{invalid}"
        )

    df["gesture_id"] = gesture_values

    # Ensure kinematics are numeric and finite.
    df[kinematic_columns] = df[kinematic_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if df[kinematic_columns].isna().any().any():
        raise ValueError(
            "NaN/non-numeric values were found in raw kinematic columns."
        )

    trials: List[TrialData] = []

    for trial_id, trial_df in df.groupby("trial_id", sort=True):
        trial_df = trial_df.sort_values("frame_idx")

        surgeon_values = trial_df["surgeon_id"].unique()
        if len(surgeon_values) != 1:
            raise ValueError(
                f"Trial {trial_id} has multiple surgeon_id values: "
                f"{surgeon_values.tolist()}"
            )

        task_values = trial_df["task"].astype(str).str.strip().unique()
        if len(task_values) != 1 or not task_values[0]:
            raise ValueError(
                f"Trial {trial_id} must have exactly one non-blank task."
            )

        kin = trial_df[kinematic_columns].to_numpy(
            dtype=np.float32
        )

        labels = trial_df["gesture_id"].to_numpy(
            dtype=np.int64
        )

        frame_indices = trial_df["frame_idx"].to_numpy(
            dtype=np.int64
        )

        if len(np.unique(frame_indices)) != len(frame_indices):
            raise ValueError(
                f"Trial {trial_id} contains duplicate frame_idx values."
            )

        if len(frame_indices) > 1 and not np.all(
            np.diff(frame_indices) == 1
        ):
            raise ValueError(
                f"Trial {trial_id} has non-contiguous frame_idx values; "
                "do not create sequence windows across missing frames."
            )

        if not np.isfinite(kin).all():
            raise ValueError(
                f"Non-finite kinematic values in trial {trial_id}"
            )

        trials.append(
            TrialData(
                trial_id=str(trial_id),
                surgeon_id=str(
                    trial_df["surgeon_id"].iloc[0]
                ),
                task=str(
                    trial_df["task"].iloc[0]
                ),
                kinematics=kin,
                labels=labels,
                frame_indices=frame_indices,
            )
        )

    if not trials:
        raise ValueError("No trials were loaded.")

    print(f"[DATA] Trials loaded: {len(trials)}")
    print(
        "[DATA] Surgeons: "
        + ", ".join(sorted({trial.surgeon_id for trial in trials}))
    )
    print(
        "[DATA] Tasks: "
        + ", ".join(sorted({trial.task for trial in trials}))
    )
    print(
        f"[DATA] Kinematic source: {kinematic_source.upper()} "
        f"({len(kinematic_columns)} features)"
    )

    return trials, kinematic_columns


# =============================================================================
# NORMALISATION
# =============================================================================

def calculate_standardization(
    trials: Sequence[TrialData],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean/std from TRAINING TRIALS ONLY.

    This prevents information from held-out surgeons entering the training
    preprocessing.
    """

    if not trials:
        raise ValueError(
            "Cannot calculate standardization from zero trials."
        )

    total_count = 0
    running_sum = np.zeros(
        KINEMATIC_DIM_PER_SOURCE,
        dtype=np.float64,
    )
    running_sq_sum = np.zeros(
        KINEMATIC_DIM_PER_SOURCE,
        dtype=np.float64,
    )

    for trial in trials:
        x = trial.kinematics.astype(
            np.float64,
            copy=False,
        )

        running_sum += x.sum(axis=0)
        running_sq_sum += np.square(x).sum(axis=0)
        total_count += len(x)

    mean = running_sum / total_count

    variance = (
        running_sq_sum / total_count
        - np.square(mean)
    )

    variance = np.maximum(variance, 1e-12)

    std = np.sqrt(variance)

    std[std < 1e-8] = 1.0

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


# =============================================================================
# WINDOW DATASET
# =============================================================================

class JIGSAWSWindowDataset(Dataset):
    """
    Creates overlapping 1-second sequence windows without copying the complete
    dataset into a second large array.

    Each item returns:

        source:
            [T, 38] raw kinematics

        target:
            [T] gesture IDs

        previous_label:
            gesture immediately before the window

        metadata:
            information required for later inspection
    """

    def __init__(
        self,
        trials: Sequence[TrialData],
        window_frames: int,
        stride_samples: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        max_windows: Optional[int] = None,
        random_seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:

        self.trials = list(trials)
        self.window_frames = window_frames
        self.stride_samples = stride_samples

        self.mean = mean
        self.std = std

        self.indices: List[Tuple[int, int]] = []

        for trial_index, trial in enumerate(self.trials):
            n_frames = len(trial.labels)

            if n_frames < window_frames:
                continue

            for start in range(
                0,
                n_frames - window_frames + 1,
                stride_samples,
            ):
                self.indices.append(
                    (trial_index, start)
                )

        if max_windows is not None:
            if max_windows <= 0:
                raise ValueError(
                    "max_windows must be greater than zero."
                )

            if len(self.indices) > max_windows:
                rng = random.Random(random_seed)
                self.indices = rng.sample(
                    self.indices,
                    max_windows,
                )

                self.indices.sort()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, object],
    ]:

        trial_index, start = self.indices[index]

        trial = self.trials[trial_index]

        end = start + self.window_frames

        source = trial.kinematics[
            start:end
        ].copy()

        target = trial.labels[
            start:end
        ].copy()

        if self.mean is not None and self.std is not None:
            source = (
                source - self.mean
            ) / self.std

        if start > 0:
            previous_label = int(
                trial.labels[start - 1]
            )
        else:
            previous_label = 0

        metadata = {
            "trial_id": trial.trial_id,
            "surgeon_id": trial.surgeon_id,
            "task": trial.task,
            "start_frame": int(
                trial.frame_indices[start]
            ),
            "end_frame": int(
                trial.frame_indices[end - 1]
            ),
        }

        return (
            torch.from_numpy(source).float(),
            torch.from_numpy(target).long(),
            torch.tensor(previous_label).long(),
            metadata,
        )


# =============================================================================
# BATCH-NORMALISATION HELPER
# =============================================================================

class SequenceBatchNorm(nn.Module):
    """
    Apply BatchNorm1d to a [batch, time, channels] tensor.

    BatchNorm1d expects [batch, channels, time], so dimensions are temporarily
    transposed.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.norm = nn.BatchNorm1d(
            channels
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        return x


# =============================================================================
# TRANSFORMER ENCODER
# =============================================================================

class SurgicalEncoderLayer(nn.Module):
    """
    Transformer encoder block:

        self attention
            -> residual
            -> normalisation
            -> feed-forward
            -> residual
            -> normalisation

    No positional encoding is added.
    """

    def __init__(
        self,
        dimension: int,
        heads: int,
        feedforward_dimension: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = SequenceBatchNorm(
            dimension
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                dimension,
                feedforward_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_dimension,
                dimension,
            ),
        )

        self.norm2 = SequenceBatchNorm(
            dimension
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        attention_output, _ = self.self_attention(
            x,
            x,
            x,
            need_weights=False,
        )

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        feed_forward_output = self.feed_forward(
            x
        )

        x = self.norm2(
            x + self.dropout(feed_forward_output)
        )

        return x


# =============================================================================
# TRANSFORMER DECODER
# =============================================================================

class GestureDecoderLayer(nn.Module):
    """
    Gesture-recognition decoder block.

    Contains:

        masked self-attention
        encoder-decoder cross-attention
        feed-forward network

    No positional encoding is used.
    """

    def __init__(
        self,
        decoder_dimension: int,
        encoder_dimension: int,
        heads: int,
        feedforward_dimension: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=decoder_dimension,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=decoder_dimension,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
            kdim=encoder_dimension,
            vdim=encoder_dimension,
        )

        self.norm1 = SequenceBatchNorm(
            decoder_dimension
        )
        self.norm2 = SequenceBatchNorm(
            decoder_dimension
        )
        self.norm3 = SequenceBatchNorm(
            decoder_dimension
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                decoder_dimension,
                feedforward_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_dimension,
                decoder_dimension,
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:

        self_attention_output, _ = self.self_attention(
            target,
            target,
            target,
            attn_mask=causal_mask,
            need_weights=False,
        )

        target = self.norm1(
            target
            + self.dropout(self_attention_output)
        )

        cross_attention_output, _ = self.cross_attention(
            target,
            memory,
            memory,
            need_weights=False,
        )

        target = self.norm2(
            target
            + self.dropout(cross_attention_output)
        )

        feed_forward_output = self.feed_forward(
            target
        )

        target = self.norm3(
            target
            + self.dropout(feed_forward_output)
        )

        return target


# =============================================================================
# COMPLETE GESTURE-RECOGNITION MODEL
# =============================================================================

class GestureRecognitionTransformer(nn.Module):
    """
    Gesture recognition portion of the modified Transformer.

    This corresponds conceptually to:

        ENC -> DEC_GR

    from the paper.

    Gesture prediction and trajectory prediction decoders are deliberately
    omitted.
    """

    def __init__(
        self,
        input_dimension: int = 38,
        encoder_dimension: int = 38,
        decoder_dimension: int = 16,
        num_classes: int = 16,
        encoder_heads: int = 1,
        decoder_heads: int = 1,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        encoder_ff_dimension: int = 152,
        decoder_ff_dimension: int = 64,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        if encoder_dimension % encoder_heads != 0:
            raise ValueError(
                "encoder_dimension must be divisible "
                "by encoder_heads."
            )

        if decoder_dimension % decoder_heads != 0:
            raise ValueError(
                "decoder_dimension must be divisible "
                "by decoder_heads."
            )

        self.input_dimension = input_dimension
        self.encoder_dimension = encoder_dimension
        self.decoder_dimension = decoder_dimension
        self.num_classes = num_classes

        # Fully connected encoder input mapping.
        self.encoder_input = nn.Linear(
            input_dimension,
            encoder_dimension,
        )

        self.encoder = nn.ModuleList(
            [
                SurgicalEncoderLayer(
                    dimension=encoder_dimension,
                    heads=encoder_heads,
                    feedforward_dimension=encoder_ff_dimension,
                    dropout=dropout,
                )
                for _ in range(encoder_layers)
            ]
        )

        # One-hot gesture vector -> decoder representation.
        self.decoder_input = nn.Linear(
            num_classes,
            decoder_dimension,
        )

        self.decoder = nn.ModuleList(
            [
                GestureDecoderLayer(
                    decoder_dimension=decoder_dimension,
                    encoder_dimension=encoder_dimension,
                    heads=decoder_heads,
                    feedforward_dimension=decoder_ff_dimension,
                    dropout=dropout,
                )
                for _ in range(decoder_layers)
            ]
        )

        self.output_layer = nn.Linear(
            decoder_dimension,
            num_classes,
        )

    @staticmethod
    def make_causal_mask(
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        True values above the diagonal are blocked.

        A decoder token can therefore attend only to itself and earlier tokens.
        """

        return torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def encode(
        self,
        source: torch.Tensor,
    ) -> torch.Tensor:

        memory = self.encoder_input(
            source
        )

        for layer in self.encoder:
            memory = layer(memory)

        return memory

    def decode(
        self,
        decoder_input: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:

        target = self.decoder_input(
            decoder_input
        )

        causal_mask = self.make_causal_mask(
            target.size(1),
            target.device,
        )

        for layer in self.decoder:
            target = layer(
                target=target,
                memory=memory,
                causal_mask=causal_mask,
            )

        return self.output_layer(
            target
        )

    def forward(
        self,
        source: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:

        memory = self.encode(
            source
        )

        return self.decode(
            decoder_input,
            memory,
        )


# =============================================================================
# DECODER INPUTS
# =============================================================================

def make_teacher_forcing_input(
    target: torch.Tensor,
    previous_label: torch.Tensor,
    num_classes: int = NUM_GESTURE_CLASSES,
) -> torch.Tensor:
    """
    Create shifted-right one-hot gesture sequence.

    target shape:
        [B, T]

    returned shape:
        [B, T, 16]
    """

    target_one_hot = F.one_hot(
        target,
        num_classes=num_classes,
    ).float()

    previous_one_hot = F.one_hot(
        previous_label,
        num_classes=num_classes,
    ).float()

    decoder_input = torch.zeros_like(
        target_one_hot
    )

    decoder_input[:, 0, :] = previous_one_hot

    if target.size(1) > 1:
        decoder_input[:, 1:, :] = (
            target_one_hot[:, :-1, :]
        )

    return decoder_input


@torch.no_grad()
def autoregressive_recognition(
    model: GestureRecognitionTransformer,
    source: torch.Tensor,
    num_classes: int = NUM_GESTURE_CLASSES,
    start_mode: str = "background",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Autoregressive gesture recognition.

    The first decoder vector defaults to the one-hot BACKGROUND token.  This
    is deterministic and matches the decoder input used for a window starting
    at the first frame of a trial.  A random initial token makes a LOUO score
    vary between otherwise identical evaluations.

    Returns:
        predicted IDs: [B, T]
        confidences:   [B, T]
    """

    model.eval()

    batch_size = source.size(0)
    sequence_length = source.size(1)
    device = source.device

    memory = model.encode(
        source
    )

    if start_mode == "background":
        decoder_sequence = F.one_hot(
            torch.zeros(batch_size, dtype=torch.long, device=device),
            num_classes=num_classes,
        ).float().unsqueeze(1)

    elif start_mode == "random":
        decoder_sequence = torch.rand(
            batch_size,
            1,
            num_classes,
            device=device,
        )

    elif start_mode == "zeros":
        decoder_sequence = torch.zeros(
            batch_size,
            1,
            num_classes,
            device=device,
        )

    else:
        raise ValueError(
            "start_mode must be 'background', 'random', or 'zeros'."
        )

    predicted_ids: List[torch.Tensor] = []
    predicted_confidences: List[torch.Tensor] = []

    for _ in range(sequence_length):
        logits = model.decode(
            decoder_sequence,
            memory,
        )

        last_logits = logits[:, -1, :]

        probabilities = torch.softmax(
            last_logits,
            dim=-1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=-1,
        )

        predicted_ids.append(
            prediction
        )
        predicted_confidences.append(
            confidence
        )

        prediction_one_hot = F.one_hot(
            prediction,
            num_classes=num_classes,
        ).float()

        decoder_sequence = torch.cat(
            [
                decoder_sequence,
                prediction_one_hot.unsqueeze(1),
            ],
            dim=1,
        )

    return (
        torch.stack(
            predicted_ids,
            dim=1,
        ),
        torch.stack(
            predicted_confidences,
            dim=1,
        ),
    )


# =============================================================================
# LEARNING-RATE SCHEDULE
# =============================================================================

class NoamLearningRate:
    """
    Transformer learning-rate schedule.

    lr =
        model_dimension^-0.5 *
        min(step^-0.5, step * warmup^-1.5)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_dimension: int,
        warmup_steps: int,
    ) -> None:

        self.optimizer = optimizer
        self.model_dimension = float(
            model_dimension
        )
        self.warmup_steps = float(
            warmup_steps
        )

        self.step_number = 0
        self.current_lr = 0.0

    def step(self) -> float:
        self.step_number += 1

        step = float(
            self.step_number
        )

        lr = (
            self.model_dimension ** -0.5
            * min(
                step ** -0.5,
                step
                * self.warmup_steps ** -1.5,
            )
        )

        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = lr

        self.current_lr = lr

        return lr


# =============================================================================
# MODEL CREATION
# =============================================================================

def build_model(
    config: TrainConfig,
) -> GestureRecognitionTransformer:

    return GestureRecognitionTransformer(
        input_dimension=KINEMATIC_DIM_PER_SOURCE,
        encoder_dimension=config.encoder_dim,
        decoder_dimension=config.decoder_dim,
        num_classes=NUM_GESTURE_CLASSES,
        encoder_heads=config.encoder_heads,
        decoder_heads=config.decoder_heads,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        encoder_ff_dimension=config.encoder_ff_dim,
        decoder_ff_dimension=config.decoder_ff_dim,
        dropout=config.dropout,
    )


# =============================================================================
# CONFUSION MATRIX / METRICS
# =============================================================================

def update_confusion_matrix(
    confusion: np.ndarray,
    target: torch.Tensor,
    prediction: torch.Tensor,
) -> None:

    true_np = target.detach().cpu().numpy().reshape(-1)
    pred_np = prediction.detach().cpu().numpy().reshape(-1)

    indices = (
        true_np * NUM_GESTURE_CLASSES
        + pred_np
    )

    counts = np.bincount(
        indices,
        minlength=(
            NUM_GESTURE_CLASSES
            * NUM_GESTURE_CLASSES
        ),
    )

    confusion += counts.reshape(
        NUM_GESTURE_CLASSES,
        NUM_GESTURE_CLASSES,
    )


def metrics_from_confusion_matrix(
    confusion: np.ndarray,
) -> Dict[str, object]:

    total = confusion.sum()
    correct = np.trace(confusion)

    accuracy = (
        float(correct / total)
        if total
        else 0.0
    )

    per_class: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []

    for class_id in range(NUM_GESTURE_CLASSES):
        tp = float(
            confusion[class_id, class_id]
        )

        fp = float(
            confusion[:, class_id].sum()
            - tp
        )

        fn = float(
            confusion[class_id, :].sum()
            - tp
        )

        support = int(
            confusion[class_id, :].sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        if support > 0:
            f1_values.append(
                f1
            )

        per_class[
            GESTURE_ID_TO_LABEL[class_id]
        ] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = (
        float(np.mean(f1_values))
        if f1_values
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: GestureRecognitionTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamLearningRate,
    criterion: nn.Module,
    device: torch.device,
    epoch_number: int,
    total_epochs: int,
    progress_updates: int,
) -> Tuple[float, float]:

    model.train()

    epoch_start = time.perf_counter()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    n_batches = len(loader)

    update_interval = max(
        1,
        n_batches // max(1, progress_updates),
    )

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        source, target, previous_label, metadata = batch

        if source.ndim != 3:
            raise RuntimeError(
                "Expected model input with shape "
                "[batch, time, kinematic_features]."
            )

        if source.shape[-1] != KINEMATIC_DIM_PER_SOURCE:
            raise RuntimeError(
                "Unexpected model feature count. "
                f"Expected {KINEMATIC_DIM_PER_SOURCE}, "
                f"got {source.shape[-1]}. "
                "Metadata may have entered the model input."
            )

        source = source.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        previous_label = previous_label.to(
            device,
            non_blocking=True,
        )

        decoder_input = make_teacher_forcing_input(
            target=target,
            previous_label=previous_label,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            source,
            decoder_input,
        )

        loss = criterion(
            logits.reshape(
                -1,
                NUM_GESTURE_CLASSES,
            ),
            target.reshape(-1),
        )

        loss.backward()

        scheduler.step()

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=-1,
        )

        batch_tokens = target.numel()

        total_loss += (
            float(loss.item())
            * batch_tokens
        )

        total_tokens += batch_tokens

        total_correct += int(
            (predictions == target)
            .sum()
            .item()
        )

        if (
            batch_index == 1
            or batch_index % update_interval == 0
            or batch_index == n_batches
        ):
            elapsed = (
                time.perf_counter()
                - epoch_start
            )

            average_batch_time = (
                elapsed / batch_index
            )

            remaining_batches = (
                n_batches - batch_index
            )

            eta = (
                average_batch_time
                * remaining_batches
            )

            progress = (
                100.0
                * batch_index
                / n_batches
            )

            print(
                f"[TRAIN] "
                f"Epoch {epoch_number}/{total_epochs} | "
                f"Batch {batch_index}/{n_batches} | "
                f"{progress:5.1f}% | "
                f"Loss {loss.item():.5f} | "
                f"LR {scheduler.current_lr:.6g} | "
                f"Elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(eta)}"
            )

    mean_loss = (
        total_loss / total_tokens
        if total_tokens
        else 0.0
    )

    accuracy = (
        total_correct / total_tokens
        if total_tokens
        else 0.0
    )

    return (
        float(mean_loss),
        float(accuracy),
    )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model: GestureRecognitionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:

    model.eval()

    confusion = np.zeros(
        (
            NUM_GESTURE_CLASSES,
            NUM_GESTURE_CLASSES,
        ),
        dtype=np.int64,
    )

    window_accuracy_sum = 0.0
    window_count = 0

    start_time = time.perf_counter()

    total_batches = len(loader)

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):
        source, target, _, _ = batch

        source = source.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        prediction, _ = autoregressive_recognition(
            model=model,
            source=source,
            start_mode="background",
        )

        update_confusion_matrix(
            confusion,
            target,
            prediction,
        )

        per_window_accuracy = (
            (prediction == target)
            .float()
            .mean(dim=1)
        )

        window_accuracy_sum += float(
            per_window_accuracy.sum().item()
        )

        window_count += int(
            len(per_window_accuracy)
        )

        if (
            batch_number == 1
            or batch_number == total_batches
            or batch_number
            % max(1, total_batches // 10)
            == 0
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            avg_batch = (
                elapsed / batch_number
            )

            remaining = (
                avg_batch
                * (total_batches - batch_number)
            )

            print(
                f"[EVAL] "
                f"Batch {batch_number}/{total_batches} | "
                f"Elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(remaining)}"
            )

    metrics = metrics_from_confusion_matrix(
        confusion
    )

    metrics["mean_window_accuracy"] = (
        window_accuracy_sum / window_count
        if window_count
        else 0.0
    )

    return metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: GestureRecognitionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:

    model.eval()

    confusion = np.zeros(
        (
            NUM_GESTURE_CLASSES,
            NUM_GESTURE_CLASSES,
        ),
        dtype=np.int64,
    )

    window_accuracy_sum = 0.0
    window_count = 0

    for batch in loader:
        source, target, previous_label, _ = batch

        source = source.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        previous_label = previous_label.to(
            device,
            non_blocking=True,
        )

        decoder_input = make_teacher_forcing_input(
            target=target,
            previous_label=previous_label,
        )

        logits = model(
            source,
            decoder_input,
        )

        prediction = torch.argmax(
            logits,
            dim=-1,
        )

        update_confusion_matrix(
            confusion,
            target,
            prediction,
        )

        per_window_accuracy = (
            (prediction == target)
            .float()
            .mean(dim=1)
        )

        window_accuracy_sum += float(
            per_window_accuracy.sum().item()
        )

        window_count += int(
            len(per_window_accuracy)
        )

    metrics = metrics_from_confusion_matrix(
        confusion
    )

    metrics["mean_window_accuracy"] = (
        window_accuracy_sum / window_count
        if window_count
        else 0.0
    )

    return metrics


# =============================================================================
# DATASET CREATION FOR A SPLIT
# =============================================================================

def make_split_dataset(
    trials: Sequence[TrialData],
    window_frames: int,
    stride_samples: int,
    standardize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    max_windows: Optional[int],
    random_seed: int,
) -> JIGSAWSWindowDataset:

    return JIGSAWSWindowDataset(
        trials=trials,
        window_frames=window_frames,
        stride_samples=stride_samples,
        mean=mean if standardize else None,
        std=std if standardize else None,
        max_windows=max_windows,
        random_seed=random_seed,
    )


# =============================================================================
# TRAIN ONE MODEL
# =============================================================================

def train_model(
    train_trials: Sequence[TrialData],
    test_trials: Optional[Sequence[TrialData]],
    config: TrainConfig,
    device: torch.device,
    run_name: str,
    held_out_surgeon: Optional[str] = None,
) -> Tuple[
    GestureRecognitionTransformer,
    Dict[str, object],
    List[Dict[str, object]],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:

    window_frames = int(
        round(
            config.window_seconds
            * config.sample_rate
        )
    )

    if config.standardize:
        mean, std = calculate_standardization(
            train_trials
        )

        normalization_surgeons = sorted(
            {
                trial.surgeon_id
                for trial in train_trials
            }
        )

        print(
            "[NORMALIZATION] Mean/std calculated "
            "from TRAINING surgeons only:"
        )
        print(
            "[NORMALIZATION] "
            + ", ".join(
                normalization_surgeons
            )
        )

        if (
            held_out_surgeon is not None
            and held_out_surgeon
            in normalization_surgeons
        ):
            raise RuntimeError(
                "DATA LEAKAGE: held-out surgeon was used "
                "to calculate normalization statistics."
            )

    else:
        mean = None
        std = None

        print(
            "[NORMALIZATION] Standardization disabled."
        )

    train_dataset = make_split_dataset(
        trials=train_trials,
        window_frames=window_frames,
        stride_samples=config.stride_samples,
        standardize=config.standardize,
        mean=mean,
        std=std,
        max_windows=config.max_windows,
        random_seed=config.random_seed,
    )

    if len(train_dataset) == 0:
        raise ValueError(
            f"{run_name}: no training windows generated."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_loader: Optional[DataLoader] = None

    if test_trials is not None:
        test_dataset = make_split_dataset(
            trials=test_trials,
            window_frames=window_frames,
            stride_samples=config.stride_samples,
            standardize=config.standardize,
            mean=mean,
            std=std,
            max_windows=config.max_windows,
            random_seed=config.random_seed + 1,
        )

        if len(test_dataset) == 0:
            raise ValueError(
                f"{run_name}: no testing windows generated."
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        if held_out_surgeon is None:
            raise RuntimeError(
                "A held-out surgeon must be supplied "
                "for LOUO evaluation."
            )

        audit_split_preprocessing(
            train_trials=train_trials,
            test_trials=test_trials,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            window_frames=window_frames,
            stride_samples=config.stride_samples,
            standardize=config.standardize,
            mean=mean,
            std=std,
            held_out_surgeon=held_out_surgeon,
            ) 

    model = build_model(
        config
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0,
        betas=(
            config.adam_beta1,
            config.adam_beta2,
        ),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    scheduler = NoamLearningRate(
        optimizer=optimizer,
        model_dimension=config.decoder_dim,
        warmup_steps=config.warmup_steps,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"[MODEL] {run_name}"
    )
    print(
        f"[MODEL] Parameters: "
        f"{parameter_count:,}"
    )
    print(
        f"[MODEL] Training trials: "
        f"{len(train_trials)}"
    )
    print(
        f"[MODEL] Training windows: "
        f"{len(train_dataset):,}"
    )

    if test_loader is not None:
        print(
            f"[MODEL] Testing windows: "
            f"{len(test_loader.dataset):,}"
        )

    history: List[Dict[str, object]] = []
    best_validation_metric: Optional[float] = None
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter = 0

    model_start = time.perf_counter()

    for epoch in range(
        1,
        config.epochs + 1,
    ):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epoch_number=epoch,
            total_epochs=config.epochs,
            progress_updates=(
                config.progress_updates_per_epoch
            ),
        )

        epoch_elapsed = (
            time.perf_counter()
            - epoch_start
        )

        total_elapsed = (
            time.perf_counter()
            - model_start
        )

        average_epoch_time = (
            total_elapsed / epoch
        )

        remaining_epochs = (
            config.epochs - epoch
        )

        estimated_remaining = (
            average_epoch_time
            * remaining_epochs
        )

        print(
            f"[EPOCH] {run_name} | "
            f"{epoch}/{config.epochs} complete | "
            f"Loss {train_loss:.5f} | "
            f"Teacher-forced accuracy "
            f"{train_accuracy:.4f} | "
            f"Epoch time "
            f"{format_duration(epoch_elapsed)} | "
            f"Training ETA "
            f"{format_duration(estimated_remaining)}"
        )

        validation_teacher_forced_accuracy: Optional[float] = None
        validation_teacher_forced_macro_f1: Optional[float] = None
        validation_teacher_forced_confusion_matrix: Optional[str] = None
        validation_autoregressive_accuracy: Optional[float] = None
        validation_autoregressive_macro_f1: Optional[float] = None
        validation_autoregressive_confusion_matrix: Optional[str] = None
        early_stopping_metric: Optional[str] = None
        early_stopping_value: Optional[float] = None
        best_validation_value_so_far: Optional[float] = None

        if test_loader is not None:
            validation_teacher_forced_metrics = evaluate_teacher_forced(
                model=model,
                loader=test_loader,
                device=device,
            )
            validation_autoregressive_metrics = evaluate(
                model=model,
                loader=test_loader,
                device=device,
            )
            validation_teacher_forced_accuracy = float(
                validation_teacher_forced_metrics["accuracy"]
            )
            validation_teacher_forced_macro_f1 = float(
                validation_teacher_forced_metrics["macro_f1"]
            )
            validation_teacher_forced_confusion_matrix = json.dumps(
                validation_teacher_forced_metrics[
                    "confusion_matrix"
                ]
            )
            validation_autoregressive_accuracy = float(
                validation_autoregressive_metrics["accuracy"]
            )
            validation_autoregressive_macro_f1 = float(
                validation_autoregressive_metrics["macro_f1"]
            )
            validation_autoregressive_confusion_matrix = json.dumps(
                validation_autoregressive_metrics[
                    "confusion_matrix"
                ]
            )
            early_stopping_metric = (
                config.early_stopping_metric
            )
            early_stopping_value = float(
                validation_autoregressive_metrics[
                    config.early_stopping_metric
                ]
            )

            if (
                best_validation_metric is None
                or early_stopping_value
                > best_validation_metric
            ):
                best_validation_metric = early_stopping_value
                best_model_state = deepcopy(
                    model.state_dict()
                )
                patience_counter = 0
            else:
                patience_counter += 1

            best_validation_value_so_far = (
                best_validation_metric
            )

            print(
                f"[VALIDATION] {run_name} | "
                f"Teacher-forced accuracy "
                f"{validation_teacher_forced_accuracy:.4f} | "
                f"Teacher-forced macro F1 "
                f"{validation_teacher_forced_macro_f1:.4f} | "
                f"Autoregressive accuracy "
                f"{validation_autoregressive_accuracy:.4f} | "
                f"Autoregressive macro F1 "
                f"{validation_autoregressive_macro_f1:.4f} | "
                f"Best {config.early_stopping_metric} "
                f"{best_validation_metric:.4f} | "
                f"Patience "
                f"{patience_counter}/"
                f"{config.early_stopping_patience}"
            )

        history.append(
            {
                "run": run_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "teacher_forced_train_accuracy": (
                    train_accuracy
                ),
                "learning_rate": (
                    scheduler.current_lr
                ),
                "epoch_seconds": (
                    epoch_elapsed
                ),
                "validation_teacher_forced_accuracy": (
                    validation_teacher_forced_accuracy
                ),
                "validation_teacher_forced_macro_f1": (
                    validation_teacher_forced_macro_f1
                ),
                "validation_teacher_forced_confusion_matrix": (
                    validation_teacher_forced_confusion_matrix
                ),
                "validation_autoregressive_accuracy": (
                    validation_autoregressive_accuracy
                ),
                "validation_autoregressive_macro_f1": (
                    validation_autoregressive_macro_f1
                ),
                "validation_autoregressive_confusion_matrix": (
                    validation_autoregressive_confusion_matrix
                ),
                "early_stopping_metric": (
                    early_stopping_metric
                ),
                "early_stopping_value": (
                    early_stopping_value
                ),
                "best_validation_value_so_far": (
                    best_validation_value_so_far
                ),
                "patience_counter": (
                    patience_counter
                    if test_loader is not None
                    else None
                ),
            }
        )

        if test_loader is not None and (
            patience_counter
            >= config.early_stopping_patience
        ):
            print(
                f"[EARLY STOPPING] {run_name} | "
                f"No improvement in "
                f"{config.early_stopping_metric} "
                f"for {patience_counter} epoch(s)."
            )
            break

    metrics: Dict[str, object] = {}

    if test_loader is not None:
        if best_model_state is None:
            raise RuntimeError(
                "No validation checkpoint was recorded."
            )

        model.load_state_dict(best_model_state)

        print()
        print(
            f"[EVAL] Starting teacher-forced and autoregressive "
            f"evaluation for {run_name}"
        )

        teacher_forced_metrics = evaluate_teacher_forced(
            model=model,
            loader=test_loader,
            device=device,
        )
        autoregressive_metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
        )

        metrics = {
            "teacher_forced": teacher_forced_metrics,
            "autoregressive": autoregressive_metrics,
        }

        print(
            f"[EVAL] {run_name} | "
            f"Teacher-forced accuracy "
            f"{teacher_forced_metrics['accuracy']:.4f} | "
            f"Teacher-forced macro F1 "
            f"{teacher_forced_metrics['macro_f1']:.4f} | "
            f"Autoregressive accuracy "
            f"{autoregressive_metrics['accuracy']:.4f} | "
            f"Autoregressive macro F1 "
            f"{autoregressive_metrics['macro_f1']:.4f}"
        )

    return (
        model,
        metrics,
        history,
        mean,
        std,
    )


# =============================================================================
# CHECKPOINT SAVING
# =============================================================================

def model_config_dictionary(
    config: TrainConfig,
) -> Dict[str, object]:

    return {
        "input_dimension": (
            KINEMATIC_DIM_PER_SOURCE
        ),
        "encoder_dimension": (
            config.encoder_dim
        ),
        "decoder_dimension": (
            config.decoder_dim
        ),
        "num_classes": (
            NUM_GESTURE_CLASSES
        ),
        "encoder_heads": (
            config.encoder_heads
        ),
        "decoder_heads": (
            config.decoder_heads
        ),
        "encoder_layers": (
            config.encoder_layers
        ),
        "decoder_layers": (
            config.decoder_layers
        ),
        "encoder_ff_dimension": (
            config.encoder_ff_dim
        ),
        "decoder_ff_dimension": (
            config.decoder_ff_dim
        ),
        "dropout": (
            config.dropout
        ),
    }


def save_checkpoint(
    path: Path,
    model: GestureRecognitionTransformer,
    config: TrainConfig,
    kinematic_columns: Sequence[str],
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
) -> None:

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "model_config": (
            model_config_dictionary(config)
        ),
        "kinematic_source": (
            config.kinematic_source
        ),
        "kinematic_columns": (
            list(kinematic_columns)
        ),
        "gesture_id_to_label": (
            GESTURE_ID_TO_LABEL
        ),
        "num_gesture_classes": (
            NUM_GESTURE_CLASSES
        ),
        "sample_rate": (
            config.sample_rate
        ),
        "window_seconds": (
            config.window_seconds
        ),
        "window_frames": int(
            round(
                config.sample_rate
                * config.window_seconds
            )
        ),
        "stride_samples": (
            config.stride_samples
        ),
        "standardize": (
            config.standardize
        ),
        "mean": (
            mean.tolist()
            if mean is not None
            else None
        ),
        "std": (
            std.tolist()
            if std is not None
            else None
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# COMPLETE TRAINING PIPELINE
# =============================================================================
def audit_split_preprocessing(
    train_trials: Sequence[TrialData],
    test_trials: Sequence[TrialData],
    train_dataset: JIGSAWSWindowDataset,
    test_dataset: JIGSAWSWindowDataset,
    window_frames: int,
    stride_samples: int,
    standardize: bool,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    held_out_surgeon: str,
) -> Dict[str, object]:
    """
    Audit a LOUO fold for preprocessing leakage or inconsistencies.

    This is deliberately strict. If an invalid train/test configuration is
    detected, training stops rather than silently producing misleading
    cross-validation results.
    """

    train_surgeons = {
        trial.surgeon_id
        for trial in train_trials
    }

    test_surgeons = {
        trial.surgeon_id
        for trial in test_trials
    }

    train_trial_ids = {
        trial.trial_id
        for trial in train_trials
    }

    test_trial_ids = {
        trial.trial_id
        for trial in test_trials
    }

    # ------------------------------------------------------------------
    # Surgeon leakage
    # ------------------------------------------------------------------

    surgeon_overlap = (
        train_surgeons
        .intersection(test_surgeons)
    )

    if surgeon_overlap:
        raise RuntimeError(
            "DATA LEAKAGE: surgeon IDs occur in both "
            f"training and testing: {sorted(surgeon_overlap)}"
        )

    if test_surgeons != {
        held_out_surgeon
    }:
        raise RuntimeError(
            "LOUO test set contains surgeons other than "
            f"the held-out surgeon {held_out_surgeon}: "
            f"{sorted(test_surgeons)}"
        )

    # ------------------------------------------------------------------
    # Trial leakage
    # ------------------------------------------------------------------

    trial_overlap = (
        train_trial_ids
        .intersection(test_trial_ids)
    )

    if trial_overlap:
        raise RuntimeError(
            "DATA LEAKAGE: trial IDs occur in both "
            f"training and testing: {sorted(trial_overlap)}"
        )

    # ------------------------------------------------------------------
    # Window configuration
    # ------------------------------------------------------------------

    if (
        train_dataset.window_frames
        != test_dataset.window_frames
    ):
        raise RuntimeError(
            "Training and testing datasets use different "
            "window lengths."
        )

    if (
        train_dataset.stride_samples
        != test_dataset.stride_samples
    ):
        raise RuntimeError(
            "Training and testing datasets use different "
            "window strides."
        )

    if (
        train_dataset.window_frames
        != window_frames
    ):
        raise RuntimeError(
            "Training dataset window length does not match "
            "the configured value."
        )

    if (
        train_dataset.stride_samples
        != stride_samples
    ):
        raise RuntimeError(
            "Training dataset stride does not match "
            "the configured value."
        )

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    if standardize:
        if mean is None or std is None:
            raise RuntimeError(
                "Standardization was requested but mean/std "
                "were not calculated."
            )

        if train_dataset.mean is None:
            raise RuntimeError(
                "Training dataset is missing normalization "
                "statistics."
            )

        if test_dataset.mean is None:
            raise RuntimeError(
                "Testing dataset is missing normalization "
                "statistics."
            )

        if not np.array_equal(
            train_dataset.mean,
            test_dataset.mean,
        ):
            raise RuntimeError(
                "Training and testing datasets received "
                "different normalization means."
            )

        if not np.array_equal(
            train_dataset.std,
            test_dataset.std,
        ):
            raise RuntimeError(
                "Training and testing datasets received "
                "different normalization standard deviations."
            )

    else:
        if (
            train_dataset.mean is not None
            or test_dataset.mean is not None
        ):
            raise RuntimeError(
                "Normalization statistics are being applied "
                "although standardization is disabled."
            )

    # ------------------------------------------------------------------
    # Window ownership
    # ------------------------------------------------------------------

    for trial_index, start in train_dataset.indices:
        trial = train_dataset.trials[
            trial_index
        ]

        if trial.surgeon_id == held_out_surgeon:
            raise RuntimeError(
                "DATA LEAKAGE: a training window belongs "
                f"to held-out surgeon {held_out_surgeon}."
            )

        if (
            start < 0
            or start + window_frames
            > len(trial.labels)
        ):
            raise RuntimeError(
                f"Invalid training window found in "
                f"{trial.trial_id}."
            )

    for trial_index, start in test_dataset.indices:
        trial = test_dataset.trials[
            trial_index
        ]

        if trial.surgeon_id != held_out_surgeon:
            raise RuntimeError(
                "LOUO test window belongs to surgeon "
                f"{trial.surgeon_id}, expected "
                f"{held_out_surgeon}."
            )

        if (
            start < 0
            or start + window_frames
            > len(trial.labels)
        ):
            raise RuntimeError(
                f"Invalid testing window found in "
                f"{trial.trial_id}."
            )

    audit = {
        "held_out_surgeon": (
            held_out_surgeon
        ),
        "train_surgeons": sorted(
            train_surgeons
        ),
        "test_surgeons": sorted(
            test_surgeons
        ),
        "train_trials": len(
            train_trial_ids
        ),
        "test_trials": len(
            test_trial_ids
        ),
        "train_windows": len(
            train_dataset
        ),
        "test_windows": len(
            test_dataset
        ),
        "window_frames": (
            window_frames
        ),
        "stride_samples": (
            stride_samples
        ),
        "standardized": (
            standardize
        ),
        "surgeon_overlap": [],
        "trial_overlap": [],
        "metadata_used_as_model_features": False,
    }

    print()
    print(
        "[AUDIT] LOUO preprocessing check PASSED"
    )
    print(
        f"[AUDIT] Held-out surgeon: "
        f"{held_out_surgeon}"
    )
    print(
        f"[AUDIT] Training surgeons: "
        f"{', '.join(sorted(train_surgeons))}"
    )
    print(
        f"[AUDIT] Testing surgeons: "
        f"{', '.join(sorted(test_surgeons))}"
    )
    print(
        f"[AUDIT] Training trials: "
        f"{len(train_trial_ids)}"
    )
    print(
        f"[AUDIT] Testing trials: "
        f"{len(test_trial_ids)}"
    )
    print(
        f"[AUDIT] Training windows: "
        f"{len(train_dataset):,}"
    )
    print(
        f"[AUDIT] Testing windows: "
        f"{len(test_dataset):,}"
    )
    print(
        f"[AUDIT] Window length: "
        f"{window_frames} frames"
    )
    print(
        f"[AUDIT] Stride: "
        f"{stride_samples} frame(s)"
    )
    print(
        f"[AUDIT] Standardization: "
        f"{standardize}"
    )
    print(
        "[AUDIT] Surgeon overlap: NONE"
    )
    print(
        "[AUDIT] Trial overlap: NONE"
    )
    print(
        "[AUDIT] Metadata passed to model: NO"
    )

    return audit


def validate_louo_fold(
    train_trials: Sequence[TrialData],
    test_trials: Sequence[TrialData],
    held_out_surgeon: str,
) -> Dict[str, object]:
    """Fail closed if a purported LOUO fold contains any metadata leakage."""

    if not train_trials:
        raise ValueError(
            f"LOUO fold for surgeon {held_out_surgeon} has no training trials."
        )

    if not test_trials:
        raise ValueError(
            f"LOUO fold for surgeon {held_out_surgeon} has no testing trials."
        )

    train_surgeons = {trial.surgeon_id for trial in train_trials}
    test_surgeons = {trial.surgeon_id for trial in test_trials}
    train_trial_ids = {trial.trial_id for trial in train_trials}
    test_trial_ids = {trial.trial_id for trial in test_trials}

    if held_out_surgeon in train_surgeons:
        raise RuntimeError(
            f"LOUO leakage: held-out surgeon {held_out_surgeon} is in training."
        )

    if test_surgeons != {held_out_surgeon}:
        raise RuntimeError(
            "LOUO split error: test trials do not belong exclusively to "
            f"held-out surgeon {held_out_surgeon}."
        )

    overlap = train_trial_ids.intersection(test_trial_ids)
    if overlap:
        raise RuntimeError(
            "LOUO leakage: trial IDs appear in both training and testing: "
            f"{sorted(overlap)}"
        )

    return {
        "held_out_surgeon": held_out_surgeon,
        "train_surgeons": sorted(train_surgeons),
        "test_surgeons": sorted(test_surgeons),
        "train_trial_ids": sorted(train_trial_ids),
        "test_trial_ids": sorted(test_trial_ids),
        "trial_metadata_used_as_model_input": False,
    }

def train_pipeline(
    config: TrainConfig,
) -> Dict[str, object]:

    seed_everything(
        config.random_seed
    )

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device(
        config.device
    )

    print("=" * 78)
    print(
        "ATARI-2 PYTORCH TRANSFORMER "
        "GESTURE RECOGNITION"
    )
    print("=" * 78)

    print(
        f"[SYSTEM] Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"[SYSTEM] GPU: "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"[SYSTEM] PyTorch version: "
        f"{torch.__version__}"
    )

    trials, kinematic_columns = (
        load_frame_level_data(
            path=config.input_csv,
            kinematic_source=(
                config.kinematic_source
            ),
        )
    )

    surgeons = sorted(
        {
            trial.surgeon_id
            for trial in trials
        }
    )

    if len(surgeons) < 2 and not config.skip_cv:
        raise ValueError(
            "LOUO cross-validation requires at least "
            "two surgeons."
        )

    all_fold_results: List[Dict[str, object]] = []
    complete_history: List[Dict[str, object]] = []

    pipeline_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # LEAVE-ONE-USER-OUT CROSS-VALIDATION
    # -------------------------------------------------------------------------

    if not config.skip_cv:
        fold_surgeons = surgeons

        if config.max_folds is not None:
            fold_surgeons = fold_surgeons[
                : config.max_folds
            ]

        total_folds = len(
            fold_surgeons
        )

        for fold_number, held_out_surgeon in enumerate(
            fold_surgeons,
            start=1,
        ):
            fold_start = time.perf_counter()

            train_trials = [
                trial
                for trial in trials
                if trial.surgeon_id
                != held_out_surgeon
            ]

            test_trials = [
                trial
                for trial in trials
                if trial.surgeon_id
                == held_out_surgeon
            ]

            fold_audit = validate_louo_fold(
                train_trials=train_trials,
                test_trials=test_trials,
                held_out_surgeon=held_out_surgeon,
            )

            print()
            print("=" * 78)
            print(
                f"[LOUO] Fold "
                f"{fold_number}/{total_folds}"
            )
            print(
                f"[LOUO] Held-out surgeon: "
                f"{held_out_surgeon}"
            )
            print("=" * 78)

            fold_model, metrics, history, mean, std = (
                train_model(
                    train_trials=train_trials,
                    test_trials=test_trials,
                    config=config,
                    device=device,
                    run_name=(
                        f"LOUO_{held_out_surgeon}"
                    ),
                    held_out_surgeon=(
                        held_out_surgeon
                    ),
                )
            )

            complete_history.extend(
                history
            )

            fold_elapsed = (
                time.perf_counter()
                - fold_start
            )

            result = {
                "fold": fold_number,
                "held_out_surgeon": (
                    held_out_surgeon
                ),
                "n_train_trials": (
                    len(train_trials)
                ),
                "n_test_trials": (
                    len(test_trials)
                ),
                "teacher_forced_accuracy": (
                    metrics[
                        "teacher_forced"
                    ]["accuracy"]
                ),
                "teacher_forced_macro_f1": (
                    metrics[
                        "teacher_forced"
                    ]["macro_f1"]
                ),
                "autoregressive_accuracy": (
                    metrics[
                        "autoregressive"
                    ]["accuracy"]
                ),
                "autoregressive_macro_f1": (
                    metrics[
                        "autoregressive"
                    ]["macro_f1"]
                ),
                "accuracy": (
                    metrics[
                        "autoregressive"
                    ]["accuracy"]
                ),
                "mean_window_accuracy": (
                    metrics[
                        "autoregressive"
                    ][
                        "mean_window_accuracy"
                    ]
                ),
                "macro_f1": (
                    metrics[
                        "autoregressive"
                    ]["macro_f1"]
                ),
                "seconds": fold_elapsed,
                "split_audit": fold_audit,
                "metrics": metrics,
            }

            all_fold_results.append(
                result
            )

            if config.save_fold_models:
                fold_path = (
                    config.output_dir
                    / (
                        "pytorch_gesture_model_"
                        f"LOUO_{held_out_surgeon}.pt"
                    )
                )

                save_checkpoint(
                    path=fold_path,
                    model=fold_model,
                    config=config,
                    kinematic_columns=(
                        kinematic_columns
                    ),
                    mean=mean,
                    std=std,
                )

            elapsed_cv = (
                time.perf_counter()
                - pipeline_start
            )

            average_fold_time = (
                elapsed_cv / fold_number
            )

            remaining_folds = (
                total_folds - fold_number
            )

            eta = (
                average_fold_time
                * remaining_folds
            )

            print()
            print(
                f"[LOUO] Fold "
                f"{fold_number}/{total_folds} "
                f"complete"
            )
            print(
                f"[LOUO] Fold time: "
                f"{format_duration(fold_elapsed)}"
            )
            print(
                f"[LOUO] CV elapsed: "
                f"{format_duration(elapsed_cv)}"
            )
            print(
                f"[LOUO] Estimated CV remaining: "
                f"{format_duration(eta)}"
            )

            # Free fold GPU memory before next model.
            del fold_model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # SUMMARISE CROSS-VALIDATION
    # -------------------------------------------------------------------------

    cv_summary: Dict[str, object] = {}

    if all_fold_results:
        accuracies = [
            float(result["accuracy"])
            for result in all_fold_results
        ]

        mean_window_accuracies = [
            float(
                result[
                    "mean_window_accuracy"
                ]
            )
            for result in all_fold_results
        ]

        macro_f1_values = [
            float(result["macro_f1"])
            for result in all_fold_results
        ]

        cv_summary = {
            "mean_accuracy": (
                float(np.mean(accuracies))
            ),
            "std_accuracy": (
                float(np.std(accuracies))
            ),
            "mean_window_accuracy": (
                float(
                    np.mean(
                        mean_window_accuracies
                    )
                )
            ),
            "mean_macro_f1": (
                float(
                    np.mean(
                        macro_f1_values
                    )
                )
            ),
            "folds": (
                all_fold_results
            ),
        }

        print()
        print("=" * 78)
        print(
            "[LOUO] CROSS-VALIDATION SUMMARY"
        )
        print("=" * 78)
        print(
            f"Mean accuracy: "
            f"{cv_summary['mean_accuracy']:.4f}"
        )
        print(
            f"Accuracy SD: "
            f"{cv_summary['std_accuracy']:.4f}"
        )
        print(
            f"Mean window accuracy: "
            f"{cv_summary['mean_window_accuracy']:.4f}"
        )
        print(
            f"Mean macro F1: "
            f"{cv_summary['mean_macro_f1']:.4f}"
        )

    # -------------------------------------------------------------------------
    # FINAL DEPLOYMENT MODEL
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "[FINAL MODEL] Training on ALL available surgeons"
    )
    print("=" * 78)

    final_model, _, final_history, mean, std = (
        train_model(
            train_trials=trials,
            test_trials=None,
            config=config,
            device=device,
            run_name="FINAL_ALL_USERS",
        )
    )

    complete_history.extend(
        final_history
    )

    # -------------------------------------------------------------------------
    # SAVE ARTIFACTS
    # -------------------------------------------------------------------------

    model_path = (
        config.output_dir
        / "pytorch_gesture_model.pt"
    )

    save_checkpoint(
        path=model_path,
        model=final_model,
        config=config,
        kinematic_columns=(
            kinematic_columns
        ),
        mean=mean,
        std=std,
    )

    config_path = (
        config.output_dir
        / "pytorch_config.json"
    )

    metrics_path = (
        config.output_dir
        / "pytorch_metrics.json"
    )

    history_path = (
        config.output_dir
        / "pytorch_training_history.csv"
    )

    columns_path = (
        config.output_dir
        / "pytorch_kinematic_columns.json"
    )

    config_payload = {
        key: (
            str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value
        in asdict(config).items()
    }

    config_payload[
        "model_config"
    ] = model_config_dictionary(
        config
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config_payload,
            file,
            indent=2,
        )

    total_runtime = (
        time.perf_counter()
        - pipeline_start
    )

    metrics_payload = {
        "cross_validation": (
            cv_summary
        ),
        "device": str(device),
        "total_runtime_seconds": (
            total_runtime
        ),
        "model_path": (
            str(model_path)
        ),
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics_payload,
            file,
            indent=2,
        )

    pd.DataFrame(
        complete_history
    ).to_csv(
        history_path,
        index=False,
    )

    with columns_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            list(kinematic_columns),
            file,
            indent=2,
        )

    print()
    print("=" * 78)
    print(
        "[DONE] PYTORCH TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        f"Total runtime: "
        f"{format_duration(total_runtime)}"
    )
    print(
        f"Model: {model_path}"
    )
    print(
        f"Metrics: {metrics_path}"
    )
    print(
        f"History: {history_path}"
    )

    return metrics_payload


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Train a paper-style PyTorch Transformer "
            "for JIGSAWS surgical gesture recognition."
        )
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help=(
            "Path to all_frame_level.csv "
            "generated by data_prep.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help=(
            "Directory in which PyTorch model "
            "artifacts will be saved."
        ),
    )

    parser.add_argument(
        "--kinematic-source",
        choices=["mtm", "psm"],
        default="mtm",
        help=(
            "Use the 38 MTM or 38 PSM "
            "kinematic variables. Default: mtm."
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
            "Sliding-window stride measured in "
            "kinematic samples. Paper-style default: 1."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=38,
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--encoder-heads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--encoder-ff-dim",
        type=int,
        default=152,
    )

    parser.add_argument(
        "--decoder-ff-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--early-stopping-metric",
        choices=["macro_f1", "accuracy"],
        default="macro_f1",
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--standardize",
        action="store_true",
        help=(
            "Standardize kinematics using training-set "
            "mean/std. Disabled by default."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers. 0 is safest on Windows."
        ),
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
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit on number of "
            "LOUO folds."
        ),
    )

    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help=(
            "Optional maximum windows per train/test "
            "dataset. Intended for smoke tests."
        ),
    )

    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help=(
            "Skip LOUO and train only the final "
            "deployment model."
        ),
    )

    parser.add_argument(
        "--save-fold-models",
        action="store_true",
        help=(
            "Save the model from every LOUO fold."
        ),
    )

    parser.add_argument(
        "--progress-updates-per-epoch",
        type=int,
        default=10,
    )

    return parser


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = build_arg_parser()
    args = parser.parse_args()

    config = TrainConfig(
        input_csv=Path(
            args.input_csv
        ),
        output_dir=Path(
            args.output_dir
        ),

        kinematic_source=(
            args.kinematic_source
        ),

        sample_rate=(
            args.sample_rate
        ),
        window_seconds=(
            args.window_seconds
        ),
        stride_samples=(
            args.stride_samples
        ),

        batch_size=(
            args.batch_size
        ),
        epochs=(
            args.epochs
        ),

        encoder_dim=(
            args.encoder_dim
        ),
        decoder_dim=(
            args.decoder_dim
        ),

        encoder_heads=(
            args.encoder_heads
        ),
        decoder_heads=(
            args.decoder_heads
        ),

        encoder_layers=(
            args.encoder_layers
        ),
        decoder_layers=(
            args.decoder_layers
        ),

        encoder_ff_dim=(
            args.encoder_ff_dim
        ),
        decoder_ff_dim=(
            args.decoder_ff_dim
        ),

        dropout=(
            args.dropout
        ),

        weight_decay=(
            args.weight_decay
        ),

        early_stopping_patience=(
            args.early_stopping_patience
        ),

        early_stopping_metric=(
            args.early_stopping_metric
        ),

        warmup_steps=(
            args.warmup_steps
        ),

        standardize=(
            args.standardize
        ),

        num_workers=(
            args.num_workers
        ),

        random_seed=(
            args.random_seed
        ),

        device=(
            args.device
        ),

        max_folds=(
            args.max_folds
        ),

        max_windows=(
            args.max_windows
        ),

        skip_cv=(
            args.skip_cv
        ),

        save_fold_models=(
            args.save_fold_models
        ),

        progress_updates_per_epoch=(
            args.progress_updates_per_epoch
        ),
    )

    train_pipeline(
        config
    )


if __name__ == "__main__":
    main()
