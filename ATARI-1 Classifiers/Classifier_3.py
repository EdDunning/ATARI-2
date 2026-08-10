"""
Task-specific classifier using XGBoost (Gradient Boosting).

This script keeps the same input and output variables as Task 3/Classifier_2.py
but trains separate models for each task (suturing, needle_passing, knot_tying)
using XGBoost instead of RandomForest. It produces a predictions CSV and a
validation summary CSV with the same metric structure as the RandomForest
implementation.

Notes:
- Requires `xgboost` to be installed in the environment (pip install xgboost).
- Uses Leave-One-Group-Out (surgeon grouping) for cross-validation, same as
  the reference implementation.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as e:
    raise ImportError(
        "xgboost is required for this script. Install with: pip install xgboost"
    ) from e


# ---------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
TASK2_CANDIDATE = HERE.parent / "Task 2"
TASK2_DIR = TASK2_CANDIDATE if TASK2_CANDIDATE.exists() else HERE

MERGED_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}

OUTPUT_DIR = HERE / "classifier_3_outputs"
PREDICTIONS_FILE = OUTPUT_DIR / "xgboost_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "xgboost_validation_summary.csv"


# ---------------------------------------------------------------------
# Targets and labels
# ---------------------------------------------------------------------
OSATS_TARGETS = [
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
]

EXPERIENCE_ORDER = ["N", "I", "E"]  # Novice, Intermediate, Expert

TASK_ORDER = ["suturing", "needle_passing", "knot_tying"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_merged_datasets() -> pd.DataFrame:
    frames = []

    for task_name, file_path in MERGED_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing merged file: {file_path}")

        df = pd.read_csv(file_path)
        df["task"] = task_name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def extract_surgeon_group(filename: str) -> str:
    """
    Extract the surgeon ID for leave-one-user-out grouping.

    Examples:
        Suturing_B001 -> B
        Knot_Tying_H005 -> H
    """
    text = str(filename).strip()

    match = re.search(r"_([A-Za-z]+)\d+$", text)
    if match:
        return match.group(1)[0].upper()

    if "_" in text:
        suffix = text.split("_", 1)[1]
        if suffix:
            return suffix[0].upper()

    return text[:1].upper()


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "filename" not in out.columns:
        if "file" in out.columns:
            out["filename"] = out["file"].astype(str)
        else:
            raise ValueError("Expected either a 'filename' or 'file' column.")

    if "experience_level" not in out.columns:
        raise ValueError("Expected an 'experience_level' column.")

    if "task" not in out.columns:
        raise ValueError("Expected a 'task' column.")

    out["experience_level"] = out["experience_level"].astype(str).str.strip().str.upper()
    out = out[out["experience_level"].isin(EXPERIENCE_ORDER)].copy()

    for col in OSATS_TARGETS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Keep task column but DO NOT add one-hot encodings here — script trains per-task.
    out["task"] = out["task"].astype(str).str.strip().str.lower()

    out["surgeon_group"] = out["filename"].apply(extract_surgeon_group)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Use motion features (numeric) only. Keep same exclusion logic as reference.
    """
    excluded = {
        "file",
        "filename",
        "task",
        "task_id",
        "surgeon_group",
        "experience_level",
        *OSATS_TARGETS,
        "grs",
    }

    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        # Keep numeric motion features only
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No usable feature columns were found.")

    return feature_cols


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_models(random_state: int = 42):
    """
    Build and return a multi-output regressor (XGBoost wrapped) and an
    XGBoost classifier for experience-level classification.
    """
    base_reg = XGBRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    # MultiOutputRegressor to support multi-target OSATS regression
    regressor = MultiOutputRegressor(base_reg, n_jobs=-1)

    classifier = XGBClassifier(
        n_estimators=500,
        random_state=random_state,
        use_label_encoder=False,
        n_jobs=-1,
        objective="multi:softprob",
        eval_metric="mlogloss",
    )

    return regressor, classifier


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_merged_datasets()
    df = prepare_data(df)

    # We'll store predictions for all held-out rows across tasks/folds
    all_pred_rows = []

    # For summary metrics across all tasks
    osats_true_all = []
    osats_pred_all = []
    grs_true_all = []
    grs_pred_all = []
    exp_true_all = []
    exp_pred_all = []

    for task_name in TASK_ORDER:
        print(f"Processing task: {task_name}")
        df_task = df[df["task"] == task_name].copy()
        if df_task.empty:
            print(f"  Skipping {task_name}: no data found.")
            continue

        feature_cols = get_feature_columns(df_task)

        X = df_task[feature_cols].copy()
        y_osats = df_task[OSATS_TARGETS].copy()
        y_grs = y_osats.sum(axis=1)
        y_exp = df_task["experience_level"].copy()
        groups = df_task["surgeon_group"].copy()

        logo = LeaveOneGroupOut()

        for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X, y_exp, groups=groups), start=1):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_osats_train = y_osats.iloc[train_idx]
            y_osats_val = y_osats.iloc[val_idx]

            y_exp_train = y_exp.iloc[train_idx]
            y_exp_val = y_exp.iloc[val_idx]

            y_grs_val = y_grs.iloc[val_idx]

            regressor, classifier = build_models(random_state=42)

            # Fit multi-output regressor
            regressor.fit(X_train, y_osats_train)

            # Encode experience labels to integers to make XGBoost stable
            le = LabelEncoder()
            y_exp_train_enc = le.fit_transform(y_exp_train)

            classifier.fit(X_train, y_exp_train_enc)

            # Predictions
            pred_osats = regressor.predict(X_val)

            pred_exp_enc = classifier.predict(X_val)
            try:
                pred_exp = le.inverse_transform(pred_exp_enc)
            except Exception:
                # As a fallback, convert encoded predictions to strings
                pred_exp = pd.Series(pred_exp_enc, index=X_val.index).astype(str).values

            pred_osats_df = pd.DataFrame(
                pred_osats,
                columns=[f"pred_{c}" for c in OSATS_TARGETS],
                index=df_task.iloc[val_idx].index,
            )
            pred_grs = pred_osats_df.sum(axis=1)

            # Collect fold-level data for final summary.
            osats_true_all.append(y_osats_val.reset_index(drop=True))
            osats_pred_all.append(pred_osats_df.reset_index(drop=True))
            grs_true_all.append(y_grs_val.reset_index(drop=True))
            grs_pred_all.append(pred_grs.reset_index(drop=True))
            exp_true_all.append(y_exp_val.reset_index(drop=True))
            exp_pred_all.append(pd.Series(pred_exp, name="experience_pred").reset_index(drop=True))

            fold_out = pd.DataFrame(index=df_task.iloc[val_idx].index)
            fold_out["filename"] = df_task.iloc[val_idx]["filename"].values
            fold_out["task"] = df_task.iloc[val_idx]["task"].values
            fold_out["surgeon_group"] = df_task.iloc[val_idx]["surgeon_group"].values
            fold_out["experience_true"] = y_exp_val.values
            fold_out["experience_pred"] = pred_exp
            fold_out["grs_true"] = y_grs_val.values
            fold_out["grs_pred"] = pred_grs.values

            for target in OSATS_TARGETS:
                fold_out[f"{target}_true"] = df_task.iloc[val_idx][target].values
                fold_out[f"{target}_pred"] = pred_osats_df[f"pred_{target}"].values

            all_pred_rows.append(fold_out)

            print(f"  {task_name}: completed fold {fold_idx}")

    if not all_pred_rows:
        raise RuntimeError("No predictions were generated. Check input files and data availability.")

    predictions = pd.concat(all_pred_rows, axis=0).sort_values(["filename"]).reset_index(drop=True)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    # Concatenate for summary metrics
    osats_true = pd.concat(osats_true_all, axis=0).reset_index(drop=True)
    osats_pred = pd.concat(osats_pred_all, axis=0).reset_index(drop=True)
    grs_true = pd.concat(grs_true_all, axis=0).reset_index(drop=True)
    grs_pred = pd.concat(grs_pred_all, axis=0).reset_index(drop=True)
    exp_true = pd.concat(exp_true_all, axis=0).reset_index(drop=True)
    exp_pred = pd.concat(exp_pred_all, axis=0).reset_index(drop=True)

    # Validation 1: OSATS RMSE
    osats_rmse = {}
    for i, target in enumerate(OSATS_TARGETS):
        osats_rmse[target] = rmse(osats_true[target].to_numpy(), osats_pred[f"pred_{target}"].to_numpy())

    mean_osats_rmse = float(np.mean(list(osats_rmse.values())))

    # Validation 2: GRS RMSE
    grs_rmse = rmse(grs_true.to_numpy(), grs_pred.to_numpy())

    # Validation 3: Experience classification accuracy
    exp_accuracy = accuracy_score(exp_true, exp_pred)

    summary_rows = [
        {"metric": f"osats_rmse_{k}", "value": v} for k, v in osats_rmse.items()
    ]
    summary_rows.append({"metric": "osats_rmse_mean", "value": mean_osats_rmse})
    summary_rows.append({"metric": "grs_rmse", "value": grs_rmse})
    summary_rows.append({"metric": "experience_accuracy", "value": exp_accuracy})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\nValidation results")
    print("------------------")
    for target, value in osats_rmse.items():
        print(f"OSATS RMSE ({target}): {value:.4f}")
    print(f"Mean OSATS RMSE: {mean_osats_rmse:.4f}")
    print(f"GRS RMSE: {grs_rmse:.4f}")
    print(f"Experience accuracy: {exp_accuracy:.4f}")

    print(f"\nSaved predictions to: {PREDICTIONS_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
