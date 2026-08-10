r"""
ATARI-2: Gesture Classification / predict_gestures.py

Supports two backends:

- xgboost: uses the window-feature pipeline created for Train_xgboost.py
- pytorch: uses the raw-frame Transformer checkpoint created for Train_PyTorch.py

The XGBoost path preserves the existing behaviour.
The PyTorch path loads pytorch_gesture_model.pt and performs greedy autoregressive
gesture recognition on overlapping 30-frame windows, then merges framewise
predictions into contiguous gesture segments.

Output files:
- <trial>_window_predictions.csv
- <trial>_predicted_segments.csv
- <trial>_predicted_segments.txt
- <trial>_prediction_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# -----------------------------------------------------------------------------
# Make the sibling "Gesture Data Manipulation" folder importable.
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PREP_DIR = PROJECT_ROOT / "Gesture Data Manipulation"

if str(DATA_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PREP_DIR))

from data_prep import (  # type: ignore  # noqa: E402
    GESTURE_LABEL_MAP,
    extract_window_features,
    load_kinematics,
)


ID_TO_LABEL = {0: "BACKGROUND"}
for _k, _v in GESTURE_LABEL_MAP.items():
    if _k not in {"BACKGROUND", "BG", "O"}:
        ID_TO_LABEL[int(_v)] = _k


DEFAULT_SAMPLE_RATE = 30.0
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_STRIDE_SECONDS = 0.5
DEFAULT_SMOOTHING_WINDOW = 3
NUM_GESTURE_CLASSES = 16
KINEMATIC_DIM_PER_SOURCE = 38


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictConfig:
    kinematics: Path
    model_dir: Path
    output_dir: Path
    model_type: str = "xgboost"
    sample_rate: float = DEFAULT_SAMPLE_RATE
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    stride_seconds: float = DEFAULT_STRIDE_SECONDS
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW
    keep_background: bool = False
    min_segment_frames: int = 1
    include_frequency_features: bool = True


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

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


def _label_name(label_id: int) -> str:
    return ID_TO_LABEL.get(int(label_id), str(int(label_id)))


def _ensure_torch() -> None:
    if torch is None or nn is None or F is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is not installed. Install PyTorch to use --model-type pytorch."
        )


def _ensure_xgboost() -> None:
    if XGBClassifier is None:  # pragma: no cover
        raise ImportError(
            "xgboost is not installed. Install xgboost to use --model-type xgboost."
        )


def _select_source_columns(kinematics: np.ndarray, source: str) -> np.ndarray:
    if kinematics.ndim != 2 or kinematics.shape[1] != 76:
        raise ValueError(f"Expected kinematics shape (T, 76), got {kinematics.shape}")

    source = source.lower()
    if source == "mtm":
        return kinematics[:, :38]
    if source == "psm":
        return kinematics[:, 38:76]

    raise ValueError(f"Unknown kinematic source '{source}'. Expected 'mtm' or 'psm'.")


def _majority_label(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _smooth_sequence(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values.copy()
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    padded = np.pad(values, (half, half), mode="edge")
    out = np.empty_like(values)
    for i in range(values.size):
        chunk = padded[i : i + window_size]
        uniq, counts = np.unique(chunk, return_counts=True)
        out[i] = uniq[int(np.argmax(counts))]
    return out


# -----------------------------------------------------------------------------
# XGBoost backend
# -----------------------------------------------------------------------------

def load_xgb_artifacts(model_dir: Path) -> Tuple[XGBClassifier, Any, List[str]]:
    _ensure_xgboost()

    model_path = model_dir / "xgboost_gesture_model.json"
    encoder_path = model_dir / "label_encoder.pkl"
    features_path = model_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing label encoder file: {encoder_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature column file: {features_path}")

    model = XGBClassifier()
    model.load_model(str(model_path))

    label_encoder = joblib.load(encoder_path)

    with features_path.open("r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("feature_columns.json is invalid or empty")

    return model, label_encoder, feature_cols


def build_xgb_window_table(
    kinematics: np.ndarray,
    feature_cols: Sequence[str],
    sample_rate: float,
    window_seconds: float,
    stride_seconds: float,
    include_frequency_features: bool,
) -> pd.DataFrame:
    if kinematics.ndim != 2 or kinematics.shape[1] != 76:
        raise ValueError(f"Expected kinematics shape (T, 76), got {kinematics.shape}")

    window_size = max(1, int(round(window_seconds * sample_rate)))
    stride = max(1, int(round(stride_seconds * sample_rate)))

    rows: List[Dict[str, float]] = []
    n_frames = kinematics.shape[0]

    for start in range(0, n_frames - window_size + 1, stride):
        end = start + window_size
        window = kinematics[start:end]
        feats = extract_window_features(
            window,
            sample_rate=sample_rate,
            include_frequency_features=include_frequency_features,
        )

        row: Dict[str, float] = {
            "window_start_frame": float(start),
            "window_end_frame": float(end - 1),
            "window_center_frame": float(start + window_size // 2),
            "window_length": float(window_size),
            "window_duration_sec": float(window_size / sample_rate),
        }
        for col in feature_cols:
            row[col] = float(feats.get(col, 0.0))
        rows.append(row)

    if not rows:
        raise ValueError("No windows could be created from the input kinematic file")

    return pd.DataFrame(rows)


def predict_xgb_windows(
    model: XGBClassifier,
    label_encoder: Any,
    window_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    X = window_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    probs = model.predict_proba(X)
    pred_enc = np.argmax(probs, axis=1)

    pred_ids = label_encoder.inverse_transform(pred_enc)
    pred_df = window_df.copy()
    pred_df["pred_enc"] = pred_enc
    pred_df["pred_label_id"] = pred_ids.astype(int)
    pred_df["pred_label"] = [ _label_name(int(x)) for x in pred_ids ]
    pred_df["pred_confidence"] = np.max(probs, axis=1)

    return pred_df


# -----------------------------------------------------------------------------
# PyTorch backend
# -----------------------------------------------------------------------------

class SequenceBatchNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class SurgicalEncoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, feedforward_dimension: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = SequenceBatchNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, dimension),
        )
        self.norm2 = SequenceBatchNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn))
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class GestureDecoderLayer(nn.Module):
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
        self.norm1 = SequenceBatchNorm(decoder_dimension)
        self.norm2 = SequenceBatchNorm(decoder_dimension)
        self.norm3 = SequenceBatchNorm(decoder_dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(decoder_dimension, feedforward_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, decoder_dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn1, _ = self.self_attention(target, target, target, attn_mask=causal_mask, need_weights=False)
        target = self.norm1(target + self.dropout(attn1))
        attn2, _ = self.cross_attention(target, memory, memory, need_weights=False)
        target = self.norm2(target + self.dropout(attn2))
        ff = self.feed_forward(target)
        target = self.norm3(target + self.dropout(ff))
        return target


class GestureRecognitionTransformer(nn.Module):
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
            raise ValueError("encoder_dimension must be divisible by encoder_heads")
        if decoder_dimension % decoder_heads != 0:
            raise ValueError("decoder_dimension must be divisible by decoder_heads")

        self.encoder_input = nn.Linear(input_dimension, encoder_dimension)
        self.encoder = nn.ModuleList([
            SurgicalEncoderLayer(
                dimension=encoder_dimension,
                heads=encoder_heads,
                feedforward_dimension=encoder_ff_dimension,
                dropout=dropout,
            )
            for _ in range(encoder_layers)
        ])
        self.decoder_input = nn.Linear(num_classes, decoder_dimension)
        self.decoder = nn.ModuleList([
            GestureDecoderLayer(
                decoder_dimension=decoder_dimension,
                encoder_dimension=encoder_dimension,
                heads=decoder_heads,
                feedforward_dimension=decoder_ff_dimension,
                dropout=dropout,
            )
            for _ in range(decoder_layers)
        ])
        self.output_layer = nn.Linear(decoder_dimension, num_classes)

    @staticmethod
    def make_causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        memory = self.encoder_input(source)
        for layer in self.encoder:
            memory = layer(memory)
        return memory

    def decode(self, decoder_input: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        target = self.decoder_input(decoder_input)
        causal_mask = self.make_causal_mask(target.size(1), target.device)
        for layer in self.decoder:
            target = layer(target=target, memory=memory, causal_mask=causal_mask)
        return self.output_layer(target)

    def forward(self, source: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(decoder_input, self.encode(source))


def load_pytorch_artifacts(model_dir: Path) -> Dict[str, Any]:
    _ensure_torch()

    model_path = model_dir / "pytorch_gesture_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing PyTorch checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")  # type: ignore[arg-type]

    required = ["model_state_dict", "model_config"]
    for key in required:
        if key not in checkpoint:
            raise ValueError(f"PyTorch checkpoint is missing '{key}'")

    model_config = checkpoint["model_config"]
    model = GestureRecognitionTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint.get("mean", None)
    std = checkpoint.get("std", None)

    artifacts = {
        "model": model,
        "model_path": model_path,
        "kinematic_source": str(checkpoint.get("kinematic_source", "mtm")),
        "window_frames": int(checkpoint.get("window_frames", 30)),
        "stride_samples": int(checkpoint.get("stride_samples", 1)),
        "standardize": bool(checkpoint.get("standardize", False)),
        "mean": None if mean is None else np.asarray(mean, dtype=np.float32),
        "std": None if std is None else np.asarray(std, dtype=np.float32),
        "gesture_id_to_label": checkpoint.get("gesture_id_to_label", ID_TO_LABEL),
        "num_gesture_classes": int(checkpoint.get("num_gesture_classes", NUM_GESTURE_CLASSES)),
    }
    return artifacts


def _autoregressive_predict_batch(model: GestureRecognitionTransformer, source_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = source_batch.size(0)
    seq_len = source_batch.size(1)
    device = source_batch.device

    memory = model.encode(source_batch)

    decoder_sequence = torch.zeros(batch_size, 1, NUM_GESTURE_CLASSES, device=device)

    predicted_ids: List[torch.Tensor] = []
    predicted_confidences: List[torch.Tensor] = []

    for _ in range(seq_len):
        logits = model.decode(decoder_sequence, memory)
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=-1)
        predicted_ids.append(prediction)
        predicted_confidences.append(confidence)
        prediction_one_hot = F.one_hot(prediction, num_classes=NUM_GESTURE_CLASSES).float()
        decoder_sequence = torch.cat([decoder_sequence, prediction_one_hot.unsqueeze(1)], dim=1)

    return torch.stack(predicted_ids, dim=1), torch.stack(predicted_confidences, dim=1)


def _build_pytorch_windows(
    source_38: np.ndarray,
    window_frames: int,
    stride_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if source_38.ndim != 2 or source_38.shape[1] != KINEMATIC_DIM_PER_SOURCE:
        raise ValueError(f"Expected source kinematics shape (T, 38), got {source_38.shape}")
    if len(source_38) < window_frames:
        raise ValueError("Input sequence is shorter than the configured window length")

    starts = np.arange(0, len(source_38) - window_frames + 1, stride_samples, dtype=int)
    windows = np.stack([source_38[s:s + window_frames] for s in starts]).astype(np.float32)
    return windows, starts


def predict_pytorch_trial(
    raw_kinematics: np.ndarray,
    artifacts: Dict[str, Any],
    output_window_size: int,
    batch_size: int,
    smoothing_window: int,
    kinematic_source_override: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_torch()

    model: GestureRecognitionTransformer = artifacts["model"]
    device = next(model.parameters()).device

    source_name = str(artifacts["kinematic_source"]) if kinematic_source_override is None else kinematic_source_override
    source_38 = _select_source_columns(raw_kinematics, source_name)

    if artifacts["standardize"] and artifacts["mean"] is not None and artifacts["std"] is not None:
        source_38 = (source_38 - artifacts["mean"]) / artifacts["std"]

    window_frames = int(artifacts["window_frames"])
    stride_samples = int(artifacts["stride_samples"])

    windows, starts = _build_pytorch_windows(source_38, window_frames=window_frames, stride_samples=stride_samples)

    n_frames = len(source_38)
    frame_votes = np.zeros((n_frames, NUM_GESTURE_CLASSES), dtype=np.float32)
    frame_vote_counts = np.zeros(n_frames, dtype=np.float32)

    window_rows: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(windows), batch_size):
            batch_end = min(batch_start + batch_size, len(windows))
            batch_windows = torch.from_numpy(windows[batch_start:batch_end]).to(device)

            pred_ids, pred_conf = _autoregressive_predict_batch(model, batch_windows)
            pred_ids_np = pred_ids.cpu().numpy()
            pred_conf_np = pred_conf.cpu().numpy()

            for i in range(batch_end - batch_start):
                start = int(starts[batch_start + i])
                end = start + window_frames

                labels = pred_ids_np[i].astype(int)
                confs = pred_conf_np[i].astype(float)

                majority_label = _majority_label(labels)
                majority_label_name = _label_name(majority_label)
                mean_conf = float(np.mean(confs))

                window_rows.append({
                    "window_start_frame": start,
                    "window_end_frame": end - 1,
                    "window_center_frame": start + window_frames // 2,
                    "pred_label_id": majority_label,
                    "pred_label": majority_label_name,
                    "pred_confidence": mean_conf,
                })

                for offset, (label_id, conf) in enumerate(zip(labels, confs)):
                    frame_idx = start + offset
                    frame_votes[frame_idx, label_id] += float(conf)
                    frame_vote_counts[frame_idx] += float(conf)

    frame_pred_ids = np.argmax(frame_votes, axis=1).astype(int)
    frame_pred_conf = np.divide(
        frame_votes.max(axis=1),
        np.maximum(frame_votes.sum(axis=1), 1e-8),
    ).astype(np.float32)

    frame_pred_ids = _smooth_sequence(frame_pred_ids, smoothing_window)

    frame_df = pd.DataFrame({
        "frame_idx": np.arange(n_frames, dtype=int),
        "pred_label_id": frame_pred_ids,
        "pred_label": [_label_name(int(x)) for x in frame_pred_ids],
        "pred_confidence": frame_pred_conf,
    })

    window_df = pd.DataFrame(window_rows)
    return window_df, frame_df


# -----------------------------------------------------------------------------
# Segment merging
# -----------------------------------------------------------------------------

def merge_segments_from_frames(
    frame_df: pd.DataFrame,
    keep_background: bool,
    min_segment_frames: int,
) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame(columns=[
            "segment_id",
            "start_frame",
            "end_frame",
            "gesture_label_id",
            "gesture_label",
            "n_frames",
            "mean_confidence",
        ])

    labels = frame_df["pred_label_id"].to_numpy(dtype=int)
    confidences = frame_df["pred_confidence"].to_numpy(dtype=float)

    segments: List[Dict[str, Any]] = []
    seg_start = 0
    segment_id = 1

    while seg_start < len(labels):
        seg_end = seg_start
        while seg_end + 1 < len(labels) and labels[seg_end + 1] == labels[seg_start]:
            seg_end += 1

        label_id = int(labels[seg_start])
        start_frame = int(frame_df.iloc[seg_start]["frame_idx"])
        end_frame = int(frame_df.iloc[seg_end]["frame_idx"])
        n_frames = seg_end - seg_start + 1
        mean_conf = float(np.mean(confidences[seg_start:seg_end + 1]))

        if (label_id != 0 or keep_background) and n_frames >= min_segment_frames:
            segments.append({
                "segment_id": segment_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "gesture_label_id": label_id,
                "gesture_label": _label_name(label_id),
                "n_frames": n_frames,
                "mean_confidence": mean_conf,
            })
            segment_id += 1

        seg_start = seg_end + 1

    return pd.DataFrame(segments)


def merge_segments_from_windows(
    window_df: pd.DataFrame,
    keep_background: bool,
    min_segment_frames: int,
) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame(columns=[
            "segment_id",
            "start_frame",
            "end_frame",
            "gesture_label_id",
            "gesture_label",
            "n_windows",
            "mean_confidence",
        ])

    labels = window_df["pred_label_id"].to_numpy(dtype=int)
    confidences = window_df["pred_confidence"].to_numpy(dtype=float)
    start_frames = window_df["window_start_frame"].to_numpy(dtype=int)
    end_frames = window_df["window_end_frame"].to_numpy(dtype=int)

    segments: List[Dict[str, Any]] = []
    seg_start = 0
    segment_id = 1

    while seg_start < len(labels):
        seg_end = seg_start
        while seg_end + 1 < len(labels) and labels[seg_end + 1] == labels[seg_start]:
            seg_end += 1

        label_id = int(labels[seg_start])
        start_frame = int(start_frames[seg_start])
        end_frame = int(end_frames[seg_end])
        n_windows = seg_end - seg_start + 1
        mean_conf = float(np.mean(confidences[seg_start:seg_end + 1]))

        if (label_id != 0 or keep_background) and (end_frame - start_frame + 1) >= min_segment_frames:
            segments.append({
                "segment_id": segment_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "gesture_label_id": label_id,
                "gesture_label": _label_name(label_id),
                "n_windows": n_windows,
                "mean_confidence": mean_conf,
            })
            segment_id += 1

        seg_start = seg_end + 1

    return pd.DataFrame(segments)


# -----------------------------------------------------------------------------
# Prediction pipelines
# -----------------------------------------------------------------------------

def predict_xgboost(
    kinematics_path: Path,
    model_dir: Path,
    output_dir: Path,
    sample_rate: float,
    window_seconds: float,
    stride_seconds: float,
    include_frequency_features: bool,
    keep_background: bool,
    min_segment_frames: int,
) -> Dict[str, Any]:
    model, label_encoder, feature_cols = load_xgb_artifacts(model_dir)
    raw_kinematics = load_kinematics(kinematics_path)

    window_df = build_xgb_window_table(
        kinematics=raw_kinematics,
        feature_cols=feature_cols,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        include_frequency_features=include_frequency_features,
    )

    pred_window_df = predict_xgb_windows(
        model=model,
        label_encoder=label_encoder,
        window_df=window_df,
        feature_cols=feature_cols,
    )

    segments_df = merge_segments_from_windows(
        pred_window_df,
        keep_background=keep_background,
        min_segment_frames=min_segment_frames,
    )

    trial_id = kinematics_path.stem
    window_out = output_dir / f"{trial_id}_window_predictions.csv"
    segments_out = output_dir / f"{trial_id}_predicted_segments.csv"
    txt_out = output_dir / f"{trial_id}_predicted_segments.txt"
    summary_out = output_dir / f"{trial_id}_prediction_summary.json"

    pred_window_df.to_csv(window_out, index=False)
    segments_df.to_csv(segments_out, index=False)

    with txt_out.open("w", encoding="utf-8") as f:
        for _, row in segments_df.iterrows():
            f.write(f"{int(row['start_frame'])} {int(row['end_frame'])} {row['gesture_label']}\n")

    summary = {
        "model_type": "xgboost",
        "trial_id": trial_id,
        "input_file": str(kinematics_path),
        "model_dir": str(model_dir),
        "window_output": str(window_out),
        "segment_output": str(segments_out),
        "text_output": str(txt_out),
        "n_frames": int(raw_kinematics.shape[0]),
        "n_windows": int(len(pred_window_df)),
        "n_segments": int(len(segments_df)),
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def predict_pytorch(
    kinematics_path: Path,
    model_dir: Path,
    output_dir: Path,
    batch_size: int,
    smoothing_window: int,
    keep_background: bool,
    min_segment_frames: int,
    kinematic_source_override: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_torch()

    artifacts = load_pytorch_artifacts(model_dir)
    raw_kinematics = load_kinematics(kinematics_path)

    window_df, frame_df = predict_pytorch_trial(
        raw_kinematics=raw_kinematics,
        artifacts=artifacts,
        output_window_size=int(artifacts["window_frames"]),
        batch_size=batch_size,
        smoothing_window=smoothing_window,
        kinematic_source_override=kinematic_source_override,
    )

    segments_df = merge_segments_from_frames(
        frame_df=frame_df,
        keep_background=keep_background,
        min_segment_frames=min_segment_frames,
    )

    trial_id = kinematics_path.stem
    window_out = output_dir / f"{trial_id}_window_predictions.csv"
    frame_out = output_dir / f"{trial_id}_frame_predictions.csv"
    segments_out = output_dir / f"{trial_id}_predicted_segments.csv"
    txt_out = output_dir / f"{trial_id}_predicted_segments.txt"
    summary_out = output_dir / f"{trial_id}_prediction_summary.json"

    window_df.to_csv(window_out, index=False)
    frame_df.to_csv(frame_out, index=False)
    segments_df.to_csv(segments_out, index=False)

    with txt_out.open("w", encoding="utf-8") as f:
        for _, row in segments_df.iterrows():
            f.write(f"{int(row['start_frame'])} {int(row['end_frame'])} {row['gesture_label']}\n")

    summary = {
        "model_type": "pytorch",
        "trial_id": trial_id,
        "input_file": str(kinematics_path),
        "model_dir": str(model_dir),
        "window_output": str(window_out),
        "frame_output": str(frame_out),
        "segment_output": str(segments_out),
        "text_output": str(txt_out),
        "n_frames": int(raw_kinematics.shape[0]),
        "n_windows": int(len(window_df)),
        "n_segments": int(len(segments_df)),
        "kinematic_source": artifacts["kinematic_source"],
        "window_frames": artifacts["window_frames"],
        "stride_samples": artifacts["stride_samples"],
    }

    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict JIGSAWS gesture segments from raw kinematics using either XGBoost or PyTorch."
    )

    parser.add_argument("--kinematics", type=str, required=True, help="Path to one raw kinematic text file.")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing model artifacts.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where prediction outputs will be written.")
    parser.add_argument("--model-type", choices=["xgboost", "pytorch"], default="xgboost", help="Choose the inference backend.")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE, help="Kinematic sampling rate in Hz.")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS, help="Sliding window length in seconds.")
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS, help="Sliding window stride in seconds (XGBoost only).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PyTorch inference.")
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW, help="Majority-vote smoothing window.")
    parser.add_argument("--keep-background", action="store_true", help="Keep background segments instead of dropping label 0.")
    parser.add_argument("--min-segment-frames", type=int, default=1, help="Discard predicted segments shorter than this many frames.")
    parser.add_argument("--no-frequency-features", action="store_true", help="Disable frequency-domain feature extraction (XGBoost only).")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = PredictConfig(
        kinematics=Path(args.kinematics),
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        model_type=args.model_type,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        smoothing_window=args.smoothing_window,
        keep_background=args.keep_background,
        min_segment_frames=args.min_segment_frames,
        include_frequency_features=not args.no_frequency_features,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.model_type == "xgboost":
        summary = predict_xgboost(
            kinematics_path=config.kinematics,
            model_dir=config.model_dir,
            output_dir=config.output_dir,
            sample_rate=config.sample_rate,
            window_seconds=config.window_seconds,
            stride_seconds=config.stride_seconds,
            include_frequency_features=config.include_frequency_features,
            keep_background=config.keep_background,
            min_segment_frames=config.min_segment_frames,
        )
    else:
        summary = predict_pytorch(
            kinematics_path=config.kinematics,
            model_dir=config.model_dir,
            output_dir=config.output_dir,
            batch_size=args.batch_size,
            smoothing_window=config.smoothing_window,
            keep_background=config.keep_background,
            min_segment_frames=config.min_segment_frames,
        )

    print("[DONE] Prediction finished")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
