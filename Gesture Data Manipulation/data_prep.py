"""
ATARI-2: Gesture Data Manipulation / data_prep.py

What this file does
===================

This script turns raw JIGSAWS kinematic recordings plus gesture annotation files
into two reusable training datasets:

1) Frame-level data
   - One row per kinematic frame.
   - Includes the original 76 kinematic variables.
   - Includes the gesture label for that frame.
   - This is useful for:
       - debugging alignment problems
       - training sequence models later
       - inspecting where gesture boundaries occur

2) Window-level feature data
   - One row per sliding time window.
   - The raw kinematic window is converted into engineered numeric features.
   - The window is assigned a single gesture label using the dominant label
     inside the window.
   - This is the dataset you will usually feed into classical ML models such as
     XGBoost or Random Forest.

Why this file exists
====================

Do not repeat preprocessing logic inside every model script.

If feature extraction, label alignment, and window generation are duplicated in
multiple places, the project becomes inconsistent very quickly:
- one script will label windows differently from another
- one script will handle boundaries differently
- debugging becomes painful
- model comparisons stop being fair

This file is meant to be the single source of truth for:
- loading kinematic data
- loading gesture annotations
- aligning frames to gesture labels
- generating sliding windows
- extracting engineered features
- saving clean training files

Expected input format
=====================

Kinematic file:
- Plain text file with 76 whitespace-separated numeric columns per frame
- No header
- One row per frame

Gesture annotation file:
- Plain text file with rows like:
      80 219 G1
      220 370 G5
- Each row should contain:
      start_frame  end_frame  gesture_label
- The script assumes the annotation frames refer to the same temporal sequence
  as the kinematic file.

Important indexing note
=======================

Some JIGSAWS annotation files are effectively 1-based frame indexed, while some
workflows treat them as 0-based when converting to arrays.

This script exposes `frame_index_base` so you can control that explicitly:
- use 1 if the annotation file starts counting from frame 1
- use 0 if the annotation file already matches Python array indexing

If gesture boundaries look shifted by one frame, this is the first thing to fix.

Outputs
=======

For each trial, the script saves:
- <trial_id>_frame_level.csv
- <trial_id>_window_features.csv

The frame-level file contains:
- trial metadata
- frame index
- gesture label
- gesture id
- all 76 raw kinematic columns

The window-level file contains:
- trial metadata
- window start/end/center
- dominant gesture label for the window
- label purity
- engineered features from the kinematic window

Feature ideas included
======================

This script computes a fairly broad set of features:
- mean, std, min, max, range, median
- slope / trend
- first difference statistics
- jerk-related statistics
- signal energy
- frequency-domain features
- pairwise block-difference summary features across the four 19-variable
  kinematic blocks

That makes the output suitable for classical supervised models such as:
- XGBoost
- Random Forest
- logistic regression
- SVM
- shallow neural nets

Later PyTorch sequence models can still use the frame-level output.

Usage example
=============

Single trial:
    python data_prep.py ^
        --kinematics path/to/Suturing_B001.txt ^
        --annotations path/to/Suturing_B001.txt ^
        --output-dir output/data ^
        --window-seconds 1.0 ^
        --stride-seconds 0.5

Batch mode:
    python data_prep.py ^
        --kinematics-dir path/to/kinematics ^
        --annotations-dir path/to/annotations ^
        --output-dir output/data

The script will match kinematics and annotations by file stem.

Notes
=====

- The JIGSAWS paper states that kinematic data are sampled at 30 Hz and that
  the kinematics and gesture annotations are synchronised frame-for-frame.
- The paper also defines the gesture vocabulary G1-G15 and recommends subject-
  independent evaluation schemes such as leave-one-user-out.
- This script does not train any model. It only prepares data for later scripts.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

KINEMATIC_DIM = 76
BLOCK_SIZE = 19
NUM_BLOCKS = 4

GESTURE_LABEL_MAP = {
    "G1": 1,
    "G2": 2,
    "G3": 3,
    "G4": 4,
    "G5": 5,
    "G6": 6,
    "G7": 7,
    "G8": 8,
    "G9": 9,
    "G10": 10,
    "G11": 11,
    "G12": 12,
    "G13": 13,
    "G14": 14,
    "G15": 15,
    "BACKGROUND": 0,
    "BG": 0,
    "O": 0,
}


@dataclass(frozen=True)
class PrepConfig:
    sample_rate: float = 30.0
    window_seconds: float = 1.0
    stride_seconds: float = 0.5
    frame_index_base: int = 1
    inclusive_end: bool = True
    min_window_purity: float = 0.50
    include_frequency_features: bool = True


def format_duration(seconds: float) -> str:
    """
    Convert seconds into a human-readable string.
    """
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} second{'' if seconds == 1 else 's'}"

    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {sec} sec"

    hours, minutes = divmod(minutes, 60)
    return f"{hours} hr {minutes} min {sec} sec"


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_kinematics(path: Path) -> np.ndarray:
    """
    Load a raw kinematic text file into a NumPy array.

    Expected shape:
        (n_frames, 76)

    The JIGSAWS kinematic files are whitespace-delimited and usually do not
    contain a header row.
    """
    try:
        data = np.loadtxt(path, dtype=float)
    except Exception:
        # Fallback for odd whitespace / formatting issues.
        data = pd.read_csv(path, header=None, delim_whitespace=True).to_numpy(dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != KINEMATIC_DIM:
        raise ValueError(
            f"{path.name}: expected {KINEMATIC_DIM} kinematic columns, found {data.shape[1]}"
        )

    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: kinematic file contains NaN or infinite values")

    return data


def load_annotations(path: Path) -> pd.DataFrame:
    """
    Load gesture annotation intervals.

    Supported row format:
        start_frame end_frame gesture_label

    Example:
        80 219 G1
        220 370 G5
    """
    rows: List[Tuple[int, int, str]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue

            try:
                start = int(parts[0])
                end = int(parts[1])
                label = parts[2].strip()
            except ValueError:
                continue

            rows.append((start, end, label))

    if not rows:
        raise ValueError(f"{path.name}: no valid annotation rows found")

    ann = pd.DataFrame(rows, columns=["start_frame", "end_frame", "gesture_label"])
    ann["gesture_id"] = ann["gesture_label"].map(GESTURE_LABEL_MAP).fillna(-1).astype(int)

    return ann


# ---------------------------------------------------------------------
# Label alignment
# ---------------------------------------------------------------------
def validate_annotation_alignment(
    n_frames: int,
    annotations: pd.DataFrame,
    frame_index_base: int,
    inclusive_end: bool,
    trial_id: str,
) -> None:
    """
    Validate that gesture annotations can be aligned safely to the raw
    kinematic frame sequence.

    Checks:
        - all gesture labels are recognised
        - annotation intervals are valid
        - converted intervals overlap the recorded kinematic sequence
        - annotations do not overlap each other after conversion
    """

    if annotations.empty:
        raise ValueError(
            f"{trial_id}: annotation table is empty."
        )

    unknown = annotations.loc[
        annotations["gesture_id"] < 0,
        "gesture_label",
    ].unique()

    if len(unknown) > 0:
        raise ValueError(
            f"{trial_id}: unknown gesture labels found: "
            f"{sorted(map(str, unknown))}"
        )

    occupied = np.zeros(
        n_frames,
        dtype=bool,
    )

    previous_start = None

    for row_number, row in annotations.iterrows():
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])

        if end_frame < start_frame:
            raise ValueError(
                f"{trial_id}: invalid annotation interval "
                f"{start_frame}-{end_frame}."
            )

        if (
            previous_start is not None
            and start_frame < previous_start
        ):
            raise ValueError(
                f"{trial_id}: annotation intervals are not "
                "in chronological order."
            )

        previous_start = start_frame

        start_idx, end_idx = _normalise_annotation_bounds(
            start_frame=start_frame,
            end_frame=end_frame,
            frame_index_base=frame_index_base,
            inclusive_end=inclusive_end,
        )

        if end_idx < 0 or start_idx >= n_frames:
            raise ValueError(
                f"{trial_id}: annotation "
                f"{start_frame}-{end_frame} lies completely "
                "outside the kinematic recording."
            )

        clipped_start = max(
            0,
            start_idx,
        )

        clipped_end = min(
            n_frames - 1,
            end_idx,
        )

        if occupied[
            clipped_start : clipped_end + 1
        ].any():
            raise ValueError(
                f"{trial_id}: overlapping gesture annotations "
                f"detected around frames "
                f"{start_frame}-{end_frame}."
            )

        occupied[
            clipped_start : clipped_end + 1
        ] = True

    annotated_frames = int(
        occupied.sum()
    )

    background_frames = (
        n_frames - annotated_frames
    )

    print(
        f"[ALIGNMENT] {trial_id}: "
        f"{len(annotations)} annotation intervals | "
        f"{annotated_frames}/{n_frames} frames annotated | "
        f"{background_frames} background/unannotated frames"
    )


def _normalise_annotation_bounds(
    start_frame: int,
    end_frame: int,
    frame_index_base: int,
    inclusive_end: bool,
) -> Tuple[int, int]:
    """
    Convert annotation frame numbers into Python array indices.
    """
    start_idx = start_frame - frame_index_base
    end_idx = end_frame - frame_index_base

    if not inclusive_end:
        end_idx -= 1

    return start_idx, end_idx


def build_frame_labels(
    n_frames: int,
    annotations: pd.DataFrame,
    frame_index_base: int = 1,
    inclusive_end: bool = True,
) -> pd.DataFrame:
    """
    Create one label per frame from the interval annotations.
    """
    labels = np.full(n_frames, "BACKGROUND", dtype=object)
    gesture_ids = np.zeros(n_frames, dtype=int)

    for _, row in annotations.iterrows():
        start_idx, end_idx = _normalise_annotation_bounds(
            int(row["start_frame"]),
            int(row["end_frame"]),
            frame_index_base=frame_index_base,
            inclusive_end=inclusive_end,
        )

        start_idx = max(0, start_idx)
        end_idx = min(n_frames - 1, end_idx)

        if end_idx < start_idx:
            continue

        labels[start_idx : end_idx + 1] = row["gesture_label"]
        gesture_ids[start_idx : end_idx + 1] = int(row["gesture_id"])

    return pd.DataFrame(
        {
            "frame_idx": np.arange(n_frames, dtype=int),
            "gesture_label": labels,
            "gesture_id": gesture_ids,
        }
    )


# ---------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------

def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


def _safe_range(x: np.ndarray) -> float:
    return float(np.max(x) - np.min(x)) if x.size else 0.0


def _linear_slope(x: np.ndarray) -> float:
    """
    Least-squares slope of x against time.
    """
    if x.size < 2:
        return 0.0

    t = np.arange(x.size, dtype=float)
    t_mean = float(t.mean())
    x_mean = float(x.mean())
    denom = float(np.sum((t - t_mean) ** 2))
    if denom == 0.0:
        return 0.0

    numer = float(np.sum((t - t_mean) * (x - x_mean)))
    return numer / denom


def _spectral_features(x: np.ndarray, sample_rate: float) -> Dict[str, float]:
    """
    Simple frequency-domain descriptors.
    """
    feats = {
        "dominant_freq": 0.0,
        "spectral_entropy": 0.0,
    }

    if x.size < 4:
        return feats

    x0 = x - float(np.mean(x))
    spectrum = np.fft.rfft(x0)
    power = np.abs(spectrum) ** 2

    if power.size <= 1:
        return feats

    power[0] = 0.0  # remove DC component

    total_power = float(power.sum())
    if total_power <= 0.0:
        return feats

    dominant_idx = int(np.argmax(power))
    feats["dominant_freq"] = float(dominant_idx * sample_rate / x.size)

    p = power / total_power
    p = p[p > 0]
    if p.size > 1:
        entropy = -float(np.sum(p * np.log2(p)))
        feats["spectral_entropy"] = entropy / float(np.log2(power.size))

    return feats


def _series_features(
    x: np.ndarray,
    prefix: str,
    sample_rate: float,
    include_frequency_features: bool,
) -> Dict[str, float]:
    """
    Compute per-channel statistics for a 1D time series.
    """
    x = np.asarray(x, dtype=float)

    feats: Dict[str, float] = {
        f"{prefix}mean": float(np.mean(x)) if x.size else 0.0,
        f"{prefix}std": _safe_std(x),
        f"{prefix}min": float(np.min(x)) if x.size else 0.0,
        f"{prefix}max": float(np.max(x)) if x.size else 0.0,
        f"{prefix}range": _safe_range(x),
        f"{prefix}median": float(np.median(x)) if x.size else 0.0,
        f"{prefix}slope": _linear_slope(x),
        f"{prefix}energy": float(np.mean(x**2)) if x.size else 0.0,
    }

    if x.size >= 2:
        d1 = np.diff(x)
        feats[f"{prefix}diff_mean_abs"] = float(np.mean(np.abs(d1)))
        feats[f"{prefix}diff_std"] = _safe_std(d1)
        feats[f"{prefix}diff_max_abs"] = float(np.max(np.abs(d1)))
        feats[f"{prefix}first_val"] = float(x[0])
        feats[f"{prefix}last_val"] = float(x[-1])
    else:
        feats[f"{prefix}diff_mean_abs"] = 0.0
        feats[f"{prefix}diff_std"] = 0.0
        feats[f"{prefix}diff_max_abs"] = 0.0
        feats[f"{prefix}first_val"] = float(x[0]) if x.size else 0.0
        feats[f"{prefix}last_val"] = float(x[-1]) if x.size else 0.0

    if x.size >= 4:
        d3 = np.diff(x, n=3)
        feats[f"{prefix}jerk_rms"] = float(np.sqrt(np.mean(d3**2))) if d3.size else 0.0
        feats[f"{prefix}jerk_mean_abs"] = float(np.mean(np.abs(d3))) if d3.size else 0.0
    else:
        feats[f"{prefix}jerk_rms"] = 0.0
        feats[f"{prefix}jerk_mean_abs"] = 0.0

    if include_frequency_features:
        feats.update(
            {
                f"{prefix}dominant_freq": 0.0,
                f"{prefix}spectral_entropy": 0.0,
            }
        )
        spectral = _spectral_features(x, sample_rate=sample_rate)
        feats[f"{prefix}dominant_freq"] = spectral["dominant_freq"]
        feats[f"{prefix}spectral_entropy"] = spectral["spectral_entropy"]

    return feats


def _block_pair_features(blocks: Sequence[np.ndarray]) -> Dict[str, float]:
    """
    Compare the four 19-variable manipulator blocks against each other.

    This adds coordination-style features, which are often useful for gesture
    recognition because gestures depend not just on a single channel but on how
    the arms move relative to each other.
    """
    if len(blocks) != 4:
        raise ValueError("Expected exactly four 19-variable blocks")

    pairs = {
        "mtm_left_minus_mtm_right": (0, 1),
        "psm_left_minus_psm_right": (2, 3),
        "mtm_left_minus_psm_left": (0, 2),
        "mtm_right_minus_psm_right": (1, 3),
    }

    feats: Dict[str, float] = {}

    for name, (a, b) in pairs.items():
        a_mean = np.mean(blocks[a], axis=0)
        b_mean = np.mean(blocks[b], axis=0)
        diff = a_mean - b_mean

        feats[f"{name}_mean"] = float(np.mean(diff))
        feats[f"{name}_std"] = float(np.std(diff))
        feats[f"{name}_min"] = float(np.min(diff))
        feats[f"{name}_max"] = float(np.max(diff))
        feats[f"{name}_energy"] = float(np.mean(diff**2))

    return feats


def extract_window_features(
    window: np.ndarray,
    sample_rate: float,
    include_frequency_features: bool = True,
) -> Dict[str, float]:
    """
    Compute engineered features from a single window of shape (T, 76).
    """
    if window.ndim != 2 or window.shape[1] != KINEMATIC_DIM:
        raise ValueError(f"Expected window shape (T, {KINEMATIC_DIM}), got {window.shape}")

    feats: Dict[str, float] = {
        "window_length": float(window.shape[0]),
        "window_duration_sec": float(window.shape[0] / sample_rate),
    }

    # Per-channel features.
    for c in range(KINEMATIC_DIM):
        col = window[:, c]
        prefix = f"k{c+1:02d}_"
        feats.update(
            _series_features(
                col,
                prefix=prefix,
                sample_rate=sample_rate,
                include_frequency_features=include_frequency_features,
            )
        )

    # Block-level coordination features.
    blocks = [window[:, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE] for i in range(NUM_BLOCKS)]
    feats.update(_block_pair_features(blocks))

    return feats


# ---------------------------------------------------------------------
# Window labelling
# ---------------------------------------------------------------------

def dominant_label(labels: np.ndarray) -> Tuple[str, int, float]:
    """
    Choose the dominant label in a window.

    Returns:
        (label_string, label_id, purity)

    If there is a tie, the center frame label is used as the tie-breaker.
    """
    if labels.size == 0:
        return "BACKGROUND", 0, 0.0

    unique, counts = np.unique(labels, return_counts=True)
    max_count = int(counts.max())
    winners = unique[counts == max_count]

    if winners.size == 1:
        label = str(winners[0])
    else:
        label = str(labels[len(labels) // 2])

    purity = max_count / float(labels.size)
    label_id = int(GESTURE_LABEL_MAP.get(label, -1))
    return label, label_id, purity


# ---------------------------------------------------------------------
# Trial processing
# ---------------------------------------------------------------------

def extract_surgeon_id(trial_id: str) -> str:
    """
    Infer the surgeon ID from a JIGSAWS trial id.

    Example trial ids:
        Suturing_B001 -> B
        Knot_Tying_C003 -> C
        Needle_Passing_H002 -> H
    """
    parts = trial_id.rsplit("_", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Cannot infer surgeon id from trial id: {trial_id}")

    return parts[1][0]


def prepare_trial(
    kinematics_path: Path,
    annotations_path: Path,
    output_dir: Path,
    config: PrepConfig,
    save_per_trial_files: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process one trial and return both the frame-level and window-level tables.
    """
    trial_id = kinematics_path.stem
    task_name = trial_id.rsplit("_", 1)[0] if "_" in trial_id else trial_id
    surgeon_id = extract_surgeon_id(trial_id)

    kinematics = load_kinematics(kinematics_path)
    annotations = load_annotations(annotations_path)

    if kinematics.shape[0] == 0:
        raise ValueError(
            f"{kinematics_path.name}: no kinematic frames found"
        )

    validate_annotation_alignment(
        n_frames=kinematics.shape[0],
        annotations=annotations,
        frame_index_base=config.frame_index_base,
        inclusive_end=config.inclusive_end,
        trial_id=trial_id,
            )

    frame_labels = build_frame_labels(
        n_frames=kinematics.shape[0],
        annotations=annotations,
        frame_index_base=config.frame_index_base,
        inclusive_end=config.inclusive_end,
    )

    raw_cols = [f"k{i:02d}" for i in range(1, KINEMATIC_DIM + 1)]
    frame_df = pd.DataFrame(kinematics, columns=raw_cols)
    frame_df.insert(0, "frame_idx", np.arange(kinematics.shape[0], dtype=int))
    frame_df.insert(0, "surgeon_id", surgeon_id)
    frame_df.insert(0, "task", task_name)
    frame_df.insert(0, "trial_id", trial_id)
    frame_df = pd.concat([frame_df, frame_labels[["gesture_label", "gesture_id"]]], axis=1)

    window_size = max(1, int(round(config.window_seconds * config.sample_rate)))
    stride = max(1, int(round(config.stride_seconds * config.sample_rate)))

    window_rows: List[Dict[str, float]] = []
    n_frames = kinematics.shape[0]

    if n_frames >= window_size:
        total_windows = (n_frames - window_size) // stride + 1
    else:
        total_windows = 0

    if total_windows > 0:
        report_every = max(1, total_windows // 10)
        next_report = report_every
        print(f"[INFO] {trial_id}: extracting {total_windows} windows")

    window_idx = 0
    for start in range(0, n_frames - window_size + 1, stride):
        window_idx += 1
        if total_windows > 0 and (window_idx == next_report or window_idx == total_windows):
            percent_complete = window_idx / total_windows * 100
            print(
                f"[INFO] {trial_id}: window {window_idx}/{total_windows} "
                f"({percent_complete:.0f}% complete)"
            )
            next_report += report_every

        end = start + window_size
        window = kinematics[start:end]

        win_labels = frame_labels.loc[start : end - 1, "gesture_label"].to_numpy(dtype=object)
        label, label_id, purity = dominant_label(win_labels)

        if purity < config.min_window_purity:
            continue

        feats = extract_window_features(
            window,
            sample_rate=config.sample_rate,
            include_frequency_features=config.include_frequency_features,
        )

        feats.update(
            {
                "trial_id": trial_id,
                "task": task_name,
                "surgeon_id": surgeon_id,
                "window_start_frame": int(start),
                "window_end_frame": int(end - 1),
                "window_center_frame": int(start + window_size // 2),
                "window_label": label,
                "window_label_id": int(label_id),
                "label_purity": float(purity),
            }
        )

        window_rows.append(feats)

    window_df = pd.DataFrame(window_rows)

    if save_per_trial_files:
        trial_out_dir = output_dir / trial_id
        trial_out_dir.mkdir(parents=True, exist_ok=True)

        frame_df.to_csv(trial_out_dir / f"{trial_id}_frame_level.csv", index=False)
        window_df.to_csv(trial_out_dir / f"{trial_id}_window_features.csv", index=False)

    return frame_df, window_df


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------

def _match_annotation_file(kinematics_file: Path, annotations_dir: Path) -> Optional[Path]:
    """
    Match annotation file by stem.
    """
    candidate = annotations_dir / f"{kinematics_file.stem}.txt"
    if candidate.exists():
        return candidate

    # Fall back to same stem with any extension.
    matches = list(annotations_dir.glob(f"{kinematics_file.stem}.*"))
    if matches:
        return matches[0]

    return None


def _infer_surgeon_from_trial_name(stem: str) -> Optional[str]:
    """
    Infer the surgeon letter from a JIGSAWS trial stem such as:
        Knot_Tying_B001
        Suturing_C015
        Needle_Passing_A002
    """

    match = re.search(r"([A-Z])\d{3}$", stem)
    if match:
        return match.group(1)

    # Conservative fallback for non-standard stems.
    parts = stem.split("_")
    if len(parts) > 1:
        suffix = parts[-1]
        if suffix and suffix[0].isalpha():
            return suffix[0].upper()

    return None


def _select_trials_balanced_by_surgeon(
    matched_trials: List[Tuple[Path, Path]],
    max_trials: int,
) -> List[Tuple[Path, Path]]:
    """
    When a smoke-test or development cap is used, prefer selecting trials from
    distinct surgeons so the prepared frame-level dataset can satisfy the LOUO
    contract (at least two surgeons) instead of taking the first max_trials files
    from a surgeon-sorted directory listing.
    """

    if max_trials is None or max_trials >= len(matched_trials):
        return matched_trials

    grouped: Dict[str, List[Tuple[Path, Path]]] = {}

    for kin_file, ann_file in matched_trials:
        surgeon = _infer_surgeon_from_trial_name(kin_file.stem)
        if surgeon is None:
            grouped.setdefault("unknown", []).append((kin_file, ann_file))
        else:
            grouped.setdefault(surgeon, []).append((kin_file, ann_file))

    # Prefer a fair first-pass: one trial from each surgeon as long as we can.
    selected: List[Tuple[Path, Path]] = []

    for surgeon in sorted(grouped.keys()):
        if len(selected) >= max_trials:
            break

        surgeon_trials = sorted(
            grouped[surgeon],
            key=lambda item: item[0].name,
        )

        if surgeon_trials:
            selected.append(surgeon_trials[0])

    if len(selected) < max_trials:
        # Round-robin fill from the remaining items across groups.
        order = sorted(
            matched_trials,
            key=lambda item: item[0].name,
        )

        for item in order:
            if item in selected:
                continue

            selected.append(item)

            if len(selected) >= max_trials:
                break

    if len(set(
        _infer_surgeon_from_trial_name(kin_file.stem)
        for kin_file, _ in selected
    ) - {None}) < 2:
        print(
            "[WARN] Smoke-test dataset selection produced fewer than two "
            "surgeons. Re-run with a broader source slice or use --skip-cv "
            "to bypass LOUO validation."
        )

    return selected


def process_folder(
    kinematics_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    config: PrepConfig,
    save_per_trial_files: bool = True,
    save_combined_files: bool = True,
    max_trials: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process every kinematic file in a folder and concatenate the outputs.
    """
    kin_files = sorted(
        [p for p in kinematics_dir.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".csv"}]
    )

    if not kin_files:
        raise ValueError(f"No kinematic files found in {kinematics_dir}")

    matched_trials: List[Tuple[Path, Path]] = []
    for kin_file in kin_files:
        ann_file = _match_annotation_file(kin_file, annotations_dir)
        if ann_file is None:
            print(f"[WARN] No annotation file found for {kin_file.name}; skipping")
            continue
        matched_trials.append((kin_file, ann_file))

    if max_trials is not None:
        matched_trials = _select_trials_balanced_by_surgeon(
            matched_trials,
            max_trials,
        )

    total_trials = len(matched_trials)
    if total_trials == 0:
        raise ValueError("No trials were processed successfully")

    all_frames: List[pd.DataFrame] = []
    all_windows: List[pd.DataFrame] = []
    trial_durations: List[float] = []

    for trial_idx, (kin_file, ann_file) in enumerate(matched_trials, start=1):
        percent_complete = trial_idx / total_trials * 100
        print(
            f"[INFO] Starting trial {trial_idx}/{total_trials} "
            f"({percent_complete:.1f}%): {kin_file.name}"
        )

        trial_start = time.time()
        frame_df, window_df = prepare_trial(
            kinematics_path=kin_file,
            annotations_path=ann_file,
            output_dir=output_dir,
            config=config,
            save_per_trial_files=save_per_trial_files,
        )
        trial_elapsed = time.time() - trial_start
        trial_durations.append(trial_elapsed)

        all_frames.append(frame_df)
        all_windows.append(window_df)

        total_elapsed = sum(trial_durations)
        avg_elapsed = total_elapsed / len(trial_durations)
        remaining_trials = total_trials - trial_idx
        estimated_remaining = avg_elapsed * remaining_trials

        print(
            f"[INFO] Completed trial {trial_idx}/{total_trials} in {format_duration(trial_elapsed)}."
        )
        print(
            f"[INFO] Elapsed: {format_duration(total_elapsed)}, "
            f"avg trial: {format_duration(avg_elapsed)}, "
            f"est remaining: {format_duration(estimated_remaining)}"
        )

    frames_out = pd.concat(all_frames, ignore_index=True)
    windows_out = pd.concat(all_windows, ignore_index=True)

    if save_combined_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_out.to_csv(output_dir / "all_frame_level.csv", index=False)
        windows_out.to_csv(output_dir / "all_window_features.csv", index=False)
        _print_combined_summary(frames_out, windows_out)

    return frames_out, windows_out


def _print_combined_summary(
    frames_out: pd.DataFrame,
    windows_out: pd.DataFrame,
) -> None:
    """
    Print validation statistics after combined datasets have been created.
    """
    trial_count = int(frames_out["trial_id"].nunique())
    frame_rows = len(frames_out)
    window_rows = len(windows_out)
    task_names = sorted(frames_out["task"].dropna().unique())
    gesture_labels = sorted(windows_out["window_label"].dropna().unique(), key=str)
    window_counts = windows_out["window_label"].value_counts().sort_index()

    print("[SUMMARY] Combined dataset validation:")
    print(f"  Trials processed: {trial_count}")
    print(f"  Total frame-level rows: {frame_rows}")
    print(f"  Total window-level rows: {window_rows}")
    print(
        f"  Unique task names: {', '.join(task_names) if task_names else '(none)'}"
    )
    print(
        f"  Unique gesture labels: {', '.join(gesture_labels) if gesture_labels else '(none)'}"
    )
    print("  Window counts by gesture:")
    for label, count in window_counts.items():
        print(f"    {label}: {count}")


def process_folder_pairs(
    kinematics_dirs: Sequence[Path],
    annotations_dirs: Sequence[Path],
    output_dir: Path,
    config: PrepConfig,
    save_per_trial_files: bool = True,
    max_trials: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process multiple kinematics/annotation directory pairs in batch.
    """
    if len(kinematics_dirs) != len(annotations_dirs):
        raise ValueError(
            "--kinematics-dir and --annotations-dir must be supplied the same number of times"
        )

    all_frames: List[pd.DataFrame] = []
    all_windows: List[pd.DataFrame] = []

    for kin_dir, ann_dir in zip(kinematics_dirs, annotations_dirs):
        frames_out, windows_out = process_folder(
            kinematics_dir=kin_dir,
            annotations_dir=ann_dir,
            output_dir=output_dir,
            config=config,
            save_per_trial_files=save_per_trial_files,
            save_combined_files=False,
            max_trials=max_trials,
        )
        all_frames.append(frames_out)
        all_windows.append(windows_out)

    if not all_frames:
        raise ValueError("No trials were processed successfully")

    frames_out = pd.concat(all_frames, ignore_index=True)
    windows_out = pd.concat(all_windows, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_out.to_csv(output_dir / "all_frame_level.csv", index=False)
    windows_out.to_csv(output_dir / "all_window_features.csv", index=False)
    _print_combined_summary(frames_out, windows_out)

    return frames_out, windows_out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare JIGSAWS kinematic and gesture data for gesture classification."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--kinematics",
        type=str,
        help="Path to one kinematic text file.",
    )
    input_group.add_argument(
        "--kinematics-dir",
        type=str,
        action="append",
        help="Directory containing one or more kinematic files. Can be supplied multiple times.",
    )

    parser.add_argument(
        "--annotations",
        type=str,
        help="Path to one annotation file (used with --kinematics).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        action="append",
        help="Directory containing annotation files (used with --kinematics-dir). Can be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where prepared CSV files will be saved.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=30.0,
        help="Sampling rate of the kinematic data in Hz.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Sliding window length in seconds.",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=0.5,
        help="Sliding window stride in seconds.",
    )
    parser.add_argument(
        "--frame-index-base",
        type=int,
        default=1,
        help="Use 1 if annotation frames start at 1, or 0 if they are zero-based.",
    )
    parser.add_argument(
        "--exclusive-end",
        action="store_true",
        help="Treat annotation end frames as exclusive instead of inclusive.",
    )
    parser.add_argument(
        "--min-window-purity",
        type=float,
        default=0.50,
        help="Minimum dominant-label fraction required to keep a window.",
    )
    parser.add_argument(
        "--no-frequency-features",
        action="store_true",
        help="Disable frequency-domain features for each kinematic channel.",
    )
    parser.add_argument(
        "--no-per-trial-files",
        action="store_true",
        help="Only save combined files in the output directory.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="If set, process at most this many matched trial pairs per dataset.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = PrepConfig(
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        frame_index_base=args.frame_index_base,
        inclusive_end=not args.exclusive_end,
        min_window_purity=args.min_window_purity,
        include_frequency_features=not args.no_frequency_features,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_per_trial_files = not args.no_per_trial_files

    if args.kinematics:
        if not args.annotations:
            raise ValueError("--annotations is required when using --kinematics")

        prepare_trial(
            kinematics_path=Path(args.kinematics),
            annotations_path=Path(args.annotations),
            output_dir=output_dir,
            config=config,
            save_per_trial_files=save_per_trial_files,
        )
        print(f"[DONE] Saved outputs to {output_dir}")

    else:
        if not args.annotations_dir:
            raise ValueError("--annotations-dir is required when using --kinematics-dir")

        kinematics_dirs = [Path(p) for p in args.kinematics_dir]
        annotations_dirs = [Path(p) for p in args.annotations_dir]

        if args.max_trials is not None:
            print(
                f"[PREP] Smoke-test mode: processing maximum {args.max_trials} trials from each dataset"
            )

        process_folder_pairs(
            kinematics_dirs=kinematics_dirs,
            annotations_dirs=annotations_dirs,
            output_dir=output_dir,
            config=config,
            save_per_trial_files=save_per_trial_files,
            max_trials=args.max_trials,
        )
        print(f"[DONE] Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()