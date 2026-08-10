"""
ATARI-2: Gesture Classification / Train_xgboost.py

Purpose
-------

This script trains an XGBoost gesture classifier from the window-level feature
file produced by `data_prep.py`.

It expects a CSV where each row represents one sliding time window and contains:
- engineered numeric features from the kinematic signals
- the window gesture label
- the trial identifier
- optional metadata such as task name and label purity

What this script does
---------------------

1) Load the prepared window-feature table.
2) Select the numeric feature columns.
3) Select the gesture label as the prediction target.
4) Group the data by trial so windows from the same trial do not leak between
   train and validation splits.
5) Perform leave-one-group-out cross-validation when possible.
6) Train a final XGBoost model on all available training data.
7) Save:
   - the trained model
   - the label encoder
   - the selected feature column list
   - evaluation metrics
   - per-window prediction outputs

Why this script exists
----------------------

The preprocessing file should stay separate from the model script.

That separation matters because:
- preprocessing should be reused by all future gesture scripts
- model training should be easy to compare across algorithms
- debugging is much easier when data preparation is stable and shared

This script is specifically for the classical supervised baseline:
- XGBoost on engineered window features

Later you can reuse the same prepared CSV for:
- Random Forest
- SVM
- logistic regression
- other boosting methods
- comparison experiments

Input expectation
-----------------

Default input:
    all_window_features.csv

This should be the combined window-level output produced by `data_prep.py`.

Recommended columns in the input CSV:
- trial_id
- task
- window_start_frame
- window_end_frame
- window_center_frame
- window_label
- window_label_id
- label_purity
- many numeric feature columns

Important
---------

This script trains on windows, not on raw frames.

If you want frame-by-frame sequence modelling, use the frame-level output from
`data_prep.py` and a different script, most likely in PyTorch.

Usage examples
--------------

Train with default settings:
    python Train_xgboost.py --input-csv output/data/all_window_features.csv --output-dir output/xgb

Keep background windows:
    python Train_xgboost.py --input-csv output/data/all_window_features.csv --output-dir output/xgb --include-background

Change model settings:
    python Train_xgboost.py --input-csv output/data/all_window_features.csv --output-dir output/xgb --n-estimators 500 --max-depth 6

Outputs
-------

The script writes the following files into the output directory:
- xgboost_gesture_model.json
- label_encoder.pkl
- feature_columns.json
- metrics.json
- cv_predictions.csv
- feature_importance.csv

Notes
-----

- The JIGSAWS paper describes gesture annotations at frame level and recommends
  subject-independent evaluation setups such as leave-one-user-out and
  leave-one-trial-out. This script uses grouped cross-validation by trial to
  avoid train/test contamination across windows from the same trial.
- The gesture vocabulary is based on the JIGSAWS G1-G15 labels.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost is not installed. Install it with: pip install xgboost"
    ) from exc


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_RANDOM_STATE = 42

METADATA_COLUMNS = {
    "trial_id",
    "task",
    "window_start_frame",
    "window_end_frame",
    "window_center_frame",
    "window_label",
    "window_label_id",
    "label_purity",
}


@dataclass(frozen=True)
class TrainConfig:
    input_csv: Path
    output_dir: Path
    group_col: str = "trial_id"
    target_col: str = "window_label_id"
    drop_background: bool = True
    min_label_purity: float = 0.50
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    tree_method: str = "hist"
    random_state: int = DEFAULT_RANDOM_STATE


# ---------------------------------------------------------------------
# Data loading / cleaning
# ---------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    return df


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


def choose_target_column(df: pd.DataFrame, preferred: str = "window_label_id") -> str:
    if preferred in df.columns:
        return preferred
    if "window_label" in df.columns:
        return "window_label"
    raise ValueError(
        "Could not find a target column. Expected 'window_label_id' or 'window_label'."
    )


def filter_rows(df: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    out = df.copy()

    if "label_purity" in out.columns:
        out = out.loc[out["label_purity"] >= config.min_label_purity].copy()

    if config.drop_background:
        if "window_label_id" in out.columns:
            out = out.loc[out["window_label_id"] != 0].copy()
        elif "window_label" in out.columns:
            out = out.loc[out["window_label"].astype(str).str.upper() != "BACKGROUND"].copy()

    if out.empty:
        raise ValueError("No rows remain after filtering. Relax the thresholds or keep background windows.")

    return out


def get_feature_columns(df: pd.DataFrame, target_col: str, group_col: str) -> List[str]:
    excluded = set(METADATA_COLUMNS)
    excluded.add(target_col)
    excluded.add(group_col)

    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    return feature_cols


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in input data.")

    y = df[target_col].copy()
    groups = df[group_col].copy()
    X = df.loc[:, feature_cols].copy()

    # Replace non-finite values with column medians.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda s: s.fillna(s.median()) if pd.api.types.is_numeric_dtype(s) else s)

    if X.isna().any().any():
        X = X.fillna(0.0)

    return X, y, groups


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model(config: TrainConfig, n_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        gamma=config.gamma,
        tree_method=config.tree_method,
        random_state=config.random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def grouped_oof_predictions(
    model_factory,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    groups: pd.Series,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """
    Leave-one-group-out cross-validation predictions.

    Returns:
        oof_pred_labels
        oof_pred_probabilities
        fold_summaries
    """
    logo = LeaveOneGroupOut()
    n_samples = len(X)
    n_classes = len(class_names)
    n_folds = logo.get_n_splits(X, y_enc, groups)
    cv_start = time.time()
    print(f"[INFO] Starting leave-one-group-out cross-validation with {n_folds} folds.")

    oof_pred = np.full(n_samples, -1, dtype=int)
    oof_prob = np.zeros((n_samples, n_classes), dtype=float)
    fold_summaries: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y_enc, groups), start=1):
        X_train = X.iloc[train_idx]
        y_train = y_enc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y_enc[test_idx]
        held_out_groups = sorted(pd.unique(groups.iloc[test_idx]).tolist())

        percent_complete = (fold_idx - 1) / n_folds * 100
        print(
            f"[INFO] Starting fold {fold_idx}/{n_folds} "
            f"({percent_complete:.1f}% complete) - "
            f"held-out group(s): {held_out_groups} - "
            f"training windows: {len(train_idx)}, "
            f"test windows: {len(test_idx)}"
        )

        fold_start = time.time()
        model = model_factory()
        model.fit(X_train, y_train)

        fold_prob = model.predict_proba(X_test)
        fold_pred = np.argmax(fold_prob, axis=1)
        fold_elapsed = time.time() - fold_start

        oof_pred[test_idx] = fold_pred
        oof_prob[test_idx] = fold_prob

        fold_accuracy = accuracy_score(y_test, fold_pred)
        fold_macro_f1 = f1_score(y_test, fold_pred, average="macro", zero_division=0)
        total_elapsed = time.time() - cv_start
        average_elapsed = total_elapsed / fold_idx
        remaining_folds = n_folds - fold_idx
        estimated_remaining = average_elapsed * remaining_folds

        print(
            f"[INFO] Fold {fold_idx} complete - "
            f"runtime: {format_duration(fold_elapsed)}, "
            f"accuracy: {fold_accuracy:.3f}, "
            f"macro F1: {fold_macro_f1:.3f}"
        )
        print(
            f"[INFO] CV elapsed: {format_duration(total_elapsed)}, "
            f"avg fold: {format_duration(average_elapsed)}, "
            f"est remaining: {format_duration(estimated_remaining)}"
        )

        fold_summaries.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "group_values": sorted(pd.unique(groups.iloc[test_idx]).tolist()),
                "accuracy": float(fold_accuracy),
                "macro_f1": float(fold_macro_f1),
            }
        )

    if (oof_pred < 0).any():
        raise RuntimeError("Some samples did not receive an out-of-fold prediction.")

    return oof_pred, oof_prob, fold_summaries


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_precision": {class_names[i]: float(precision[i]) for i in range(len(class_names))},
        "per_class_recall": {class_names[i]: float(recall[i]) for i in range(len(class_names))},
        "per_class_f1": {class_names[i]: float(f1[i]) for i in range(len(class_names))},
        "per_class_support": {class_names[i]: int(support[i]) for i in range(len(class_names))},
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).tolist(),
    }


def make_feature_importance_table(model: XGBClassifier, feature_cols: Sequence[str]) -> pd.DataFrame:
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type="gain")

    rows = []
    for i, col in enumerate(feature_cols):
        key = f"f{i}"
        rows.append(
            {
                "feature": col,
                "importance_gain": float(score_dict.get(key, 0.0)),
            }
        )

    feat_df = pd.DataFrame(rows)
    feat_df = feat_df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    return feat_df


# ---------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------

def train_pipeline(config: TrainConfig) -> Dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(config.input_csv)
    target_col = choose_target_column(raw_df, preferred=config.target_col)
    filtered_df = filter_rows(raw_df, config)

    feature_cols = get_feature_columns(filtered_df, target_col=target_col, group_col=config.group_col)
    X, y_raw, groups = prepare_xy(filtered_df, feature_cols, target_col=target_col, group_col=config.group_col)

    # Encode labels to contiguous integers for XGBoost and save the mapping.
    label_encoder = LabelEncoder()
    if target_col == "window_label_id":
        # Preserve numeric order when labels are numeric gesture ids.
        y_values = y_raw.astype(int).to_numpy()
        label_encoder.fit(y_values)
        y_enc = label_encoder.transform(y_values)
        class_names = [str(c) for c in label_encoder.classes_]
    else:
        y_values = y_raw.astype(str).to_numpy()
        label_encoder.fit(y_values)
        y_enc = label_encoder.transform(y_values)
        class_names = list(label_encoder.classes_.astype(str))

    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError("Need at least two classes to train a classifier.")

    def model_factory() -> XGBClassifier:
        return build_model(config, n_classes=n_classes)

    print(f"[INFO] Rows used for training: {len(X)}")
    print(f"[INFO] Number of features: {len(feature_cols)}")
    print(f"[INFO] Number of classes: {n_classes}")
    print(f"[INFO] Unique groups: {groups.nunique()}")

    # Grouped out-of-fold predictions for honest evaluation.
    if groups.nunique() >= 2:
        oof_pred, oof_prob, fold_summaries = grouped_oof_predictions(
            model_factory=model_factory,
            X=X,
            y_enc=y_enc,
            groups=groups,
            class_names=class_names,
        )
        metrics = summarize_metrics(y_enc, oof_pred, class_names=class_names)
    else:
        warnings.warn(
            "Only one group found. Grouped cross-validation is not possible; "
            "metrics will be computed on the training data."
        )
        final_model = model_factory()
        final_model.fit(X, y_enc)
        oof_prob = final_model.predict_proba(X)
        oof_pred = np.argmax(oof_prob, axis=1)
        fold_summaries = []
        metrics = summarize_metrics(y_enc, oof_pred, class_names=class_names)

    # Train the final model on all data.
    print("[FINAL MODEL] Training model on all available windows...")
    final_start = time.time()
    final_model = model_factory()
    final_model.fit(X, y_enc)
    final_elapsed = time.time() - final_start
    print(
        f"[FINAL MODEL] Completed training in {format_duration(final_elapsed)} "
        f"on {len(X)} rows, {len(feature_cols)} features, {n_classes} classes."
    )

    # Save model and metadata.
    model_path = config.output_dir / "xgboost_gesture_model.json"
    label_encoder_path = config.output_dir / "label_encoder.pkl"
    features_path = config.output_dir / "feature_columns.json"
    metrics_path = config.output_dir / "metrics.json"
    preds_path = config.output_dir / "cv_predictions.csv"
    importance_path = config.output_dir / "feature_importance.csv"

    final_model.save_model(str(model_path))
    joblib.dump(label_encoder, label_encoder_path)

    with features_path.open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    pred_df = filtered_df.loc[:, [c for c in [config.group_col, "trial_id", "task", "window_start_frame", "window_end_frame", "window_label", "window_label_id", "label_purity"] if c in filtered_df.columns]].copy()

    pred_df["y_true_enc"] = y_enc
    pred_df["y_pred_enc"] = oof_pred
    pred_df["y_true"] = label_encoder.inverse_transform(y_enc)
    pred_df["y_pred"] = label_encoder.inverse_transform(oof_pred)
    pred_df["pred_confidence"] = np.max(oof_prob, axis=1)

    pred_df.to_csv(preds_path, index=False)

    importance_df = make_feature_importance_table(final_model, feature_cols)
    importance_df.to_csv(importance_path, index=False)

    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "n_rows": int(len(X)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(n_classes),
        "class_names": class_names,
        "metrics": metrics,
        "fold_summaries": fold_summaries,
        "artifacts": {
            "model_path": str(model_path),
            "label_encoder_path": str(label_encoder_path),
            "features_path": str(features_path),
            "metrics_path": str(metrics_path),
            "preds_path": str(preds_path),
            "importance_path": str(importance_path),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost gesture classifier from JIGSAWS window features."
    )

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the combined window feature CSV produced by data_prep.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where model and metrics will be saved.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="trial_id",
        help="Column used for grouped cross-validation.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="window_label_id",
        help="Target column to predict. Defaults to window_label_id.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Keep background windows instead of dropping label 0.",
    )
    parser.add_argument(
        "--min-label-purity",
        type=float,
        default=0.50,
        help="Discard windows with label purity below this threshold.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of boosting rounds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="XGBoost learning rate.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--min-child-weight",
        type=float,
        default=1.0,
        help="Minimum child weight.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsampling ratio.",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Column subsampling ratio per tree.",
    )
    parser.add_argument(
        "--reg-alpha",
        type=float,
        default=0.0,
        help="L1 regularization strength.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Minimum loss reduction required to make a split.",
    )
    parser.add_argument(
        "--tree-method",
        type=str,
        default="hist",
        help="XGBoost tree method. 'hist' is a good default.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = TrainConfig(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        group_col=args.group_col,
        target_col=args.target_col,
        drop_background=not args.include_background,
        min_label_purity=args.min_label_purity,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        gamma=args.gamma,
        tree_method=args.tree_method,
        random_state=args.random_state,
    )

    payload = train_pipeline(config)

    print("[DONE] Training finished")
    print(json.dumps(
        {
            "accuracy": payload["metrics"]["accuracy"],
            "macro_f1": payload["metrics"]["macro_f1"],
            "weighted_f1": payload["metrics"]["weighted_f1"],
            "model_path": payload["artifacts"]["model_path"],
        },
        indent=2
    ))


if __name__ == "__main__":
    main()