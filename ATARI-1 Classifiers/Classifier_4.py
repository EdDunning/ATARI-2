"""
==========================================================================
Linear Regression Skill Mapping
==========================================================================

Purpose
-------
This script trains linear regression models on the merged JIGSAWS motion-feature
data from Task 2. It uses only the extracted motion features as inputs and
predicts:

    - the six OSATS component scores (multi-output LinearRegression)
    - the surgeon experience level encoded as numeric values (1=Novice, 2=Intermediate, 3=Expert)

The model is evaluated on held-out validation data using a grouped 80/20 split
(grouped by surgeon). Outputs are written in the same format as the previous
classifiers so results can be compared easily.

Outputs
-------
The script writes two files to task_3/classifier_4_outputs:

    - linear_regression_4_predictions.csv
      Contains the true and predicted outputs for the validation set (same layout
      as Classifier_1 predictions file)

    - linear_regression_4_validation_summary.csv
      Contains RMSE metrics for each OSATS item, mean OSATS RMSE, GRS RMSE and
      experience accuracy (computed by rounding the regressed numeric experience
      to the nearest integer and mapping back to labels).
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GroupShuffleSplit


# File locations
HERE = Path(__file__).resolve().parent
TASK2_CANDIDATE = HERE.parent / "Task 2"
TASK2_DIR = TASK2_CANDIDATE if TASK2_CANDIDATE.exists() else HERE

MERGED_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}

OUTPUT_DIR = HERE / "classifier_4_outputs"
PREDICTIONS_FILE = OUTPUT_DIR / "linear_regression_4_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "linear_regression_4_validation_summary.csv"

# Targets
OSATS_TARGETS = [
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
]

EXPERIENCE_TARGET = "experience_level"
GRS_TARGET = "grs"
EXPERIENCE_ORDER = ["N", "I", "E"]  # Novice, Intermediate, Expert
EXP_TO_NUM = {"N": 1, "I": 2, "E": 3}
NUM_TO_EXP = {v: k for k, v in EXP_TO_NUM.items()}


# Helpers

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
    text = str(filename).strip()
    match = re.search(r"_([A-Za-z]+)\d+$", text)
    if match:
        return match.group(1)[0].upper()

    if "_" in text:
        suffix = text.split("_", 1)[1]
        if suffix:
            return suffix[0].upper()

    return text[:1].upper()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "file",
        "filename",
        "task",
        EXPERIENCE_TARGET,
        "experience_numeric",
        GRS_TARGET,
        *OSATS_TARGETS,
    }

    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric motion feature columns were found.")

    return feature_cols


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "filename" not in out.columns:
        if "file" in out.columns:
            out["filename"] = out["file"].astype(str)
        else:
            raise ValueError("Expected either a 'filename' or 'file' column.")

    if EXPERIENCE_TARGET not in out.columns:
        raise ValueError(f"Expected an '{EXPERIENCE_TARGET}' column.")

    if GRS_TARGET not in out.columns:
        raise ValueError(f"Expected a '{GRS_TARGET}' column.")

    for col in OSATS_TARGETS + [GRS_TARGET]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["experience_level"] = out[EXPERIENCE_TARGET].astype(str).str.strip().str.upper()
    out = out[out["experience_level"].isin(EXPERIENCE_ORDER)].copy()

    out["surgeon_group"] = out["filename"].apply(extract_surgeon_group)

    # numeric experience for regression
    out["experience_numeric"] = out["experience_level"].map(EXP_TO_NUM)

    return out


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# Main
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_merged_datasets()
    df = prepare_data(df)

    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y_osats = df[OSATS_TARGETS].copy()
    y_grs = df[GRS_TARGET].copy()
    y_exp_num = df["experience_numeric"].copy()
    y_exp_label = df["experience_level"].copy()
    groups = df["surgeon_group"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y_exp_label, groups=groups))

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_osats_train = y_osats.iloc[train_idx]
    y_osats_val = y_osats.iloc[val_idx]

    y_grs_val = y_grs.iloc[val_idx]
    y_exp_num_train = y_exp_num.iloc[train_idx]
    y_exp_num_val = y_exp_num.iloc[val_idx]
    y_exp_label_val = y_exp_label.iloc[val_idx]

    # Build and fit linear models
    osats_reg = LinearRegression()
    exp_reg = LinearRegression()

    osats_reg.fit(X_train, y_osats_train)
    exp_reg.fit(X_train, y_exp_num_train)

    # Predict on validation set
    pred_osats = osats_reg.predict(X_val)
    pred_exp_num = exp_reg.predict(X_val)

    # Store the experience-level regression coefficients for inspection
    exp_intercept = float(exp_reg.intercept_)
    exp_coeffs = np.asarray(exp_reg.coef_, dtype=float)

    # Convert predicted OSATS to DataFrame
    pred_osats_df = pd.DataFrame(pred_osats, columns=[f"pred_{c}" for c in OSATS_TARGETS], index=y_osats_val.index)
    pred_grs = pred_osats_df.sum(axis=1)

    # For experience, map predicted numeric -> nearest integer in 1..3 and back to labels
    pred_exp_rounded = np.rint(pred_exp_num).astype(int)
    pred_exp_rounded = np.clip(pred_exp_rounded, 1, 3)
    pred_exp_labels = [NUM_TO_EXP[int(v)] for v in pred_exp_rounded]

    # Validation metrics
    osats_rmse = {}
    for i, target in enumerate(OSATS_TARGETS):
        osats_rmse[target] = rmse(y_osats_val[target].to_numpy(), pred_osats[:, i])

    mean_osats_rmse = float(np.mean(list(osats_rmse.values())))
    grs_rmse = rmse(y_grs_val.to_numpy(), pred_grs.to_numpy())

    # Experience accuracy by comparing rounded mapped labels
    exp_accuracy = accuracy_score(y_exp_label_val.values, pred_exp_labels)
    exp_rmse = rmse(y_exp_num_val.to_numpy(), pred_exp_num)

    # Save predictions for inspection
    preds_out = pd.DataFrame(index=y_exp_label_val.index)
    preds_out["filename"] = df.loc[val_idx, "filename"].values
    preds_out["task"] = df.loc[val_idx, "task"].values
    preds_out["experience_true"] = y_exp_label_val.values
    preds_out["experience_pred"] = pred_exp_labels

    preds_out["grs_true"] = y_grs_val.values
    preds_out["grs_pred"] = pred_grs.values

    for target in OSATS_TARGETS:
        preds_out[f"{target}_true"] = df.loc[val_idx, target].values
    for target in OSATS_TARGETS:
        preds_out[f"{target}_pred"] = pred_osats_df[f"pred_{target}"].values

    preds_out.to_csv(PREDICTIONS_FILE, index=False)

    # Save summary metrics
    summary_rows = [
        {"metric": f"osats_rmse_{k}", "value": v} for k, v in osats_rmse.items()
    ]
    summary_rows.append({"metric": "osats_rmse_mean", "value": mean_osats_rmse})
    summary_rows.append({"metric": "grs_rmse", "value": grs_rmse})
    summary_rows.append({"metric": "experience_accuracy", "value": exp_accuracy})
    # Also include numeric experience RMSE for completeness
    summary_rows.append({"metric": "experience_rmse", "value": exp_rmse})
    summary_rows.append({"metric": "experience_intercept", "value": exp_intercept})
    for feature, coef in zip(feature_cols, exp_coeffs):
        summary_rows.append({"metric": f"experience_coef_{feature}", "value": float(coef)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    # Print summary
    print("\nLinear regression validation results")
    print("----------------------------------")
    for target, value in osats_rmse.items():
        print(f"OSATS RMSE ({target}): {value:.4f}")
    print(f"Mean OSATS RMSE: {mean_osats_rmse:.4f}")
    print(f"GRS RMSE: {grs_rmse:.4f}")
    print(f"Experience accuracy (rounded): {exp_accuracy:.4f}")
    print(f"Experience RMSE (numeric): {exp_rmse:.4f}")
    print(f"Experience intercept (b0): {exp_intercept:.6f}")
    for idx, feature in enumerate(feature_cols):
        print(f"{feature} (b{idx + 1}): {exp_coeffs[idx]:.6f}")

    print(f"\nSaved predictions to: {PREDICTIONS_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
