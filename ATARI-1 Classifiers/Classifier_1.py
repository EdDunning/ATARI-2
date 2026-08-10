"""
===========================================================================
Random Forest Skill Classification
===========================================================================

Purpose
-------
This script trains a Random Forest model on the merged JIGSAWS motion-feature
data from Task 2. It uses only the extracted motion features as inputs and
predicts:

    - the six OSATS component scores
    - the GRS (computed by summing the predicted OSATS scores)
    - the surgeon experience level (Novice / Intermediate / Expert)

The model is evaluated on held-out validation data to measure how well motion
features can predict surgical skill.

How the data are split
----------------------
The merged data from all three tasks are combined into one dataset and then
split into training and validation sets using a grouped 80/20 split.

The split is grouped by surgeon ID, extracted from the filename, so that data
from the same surgeon do not appear in both training and validation sets.
This reduces leakage and gives a more realistic estimate of generalisation
to unseen surgeons.

The split is random, but reproducible because a fixed random seed is used.

Model structure
---------------
Two Random Forest models are trained:

    1. RandomForestRegressor
        - Predicts all six OSATS scores at once
        - A single multi-output regressor is used, not six separate regressors

    2. RandomForestClassifier
        - Predicts experience level as a three-class label
        - Novice, Intermediate and Expert are treated as separate classes

The predicted GRS is then calculated as the sum of the six predicted OSATS
scores, rather than being predicted independently.

Number of estimators
--------------------
Both the regressor and classifier use 500 decision trees.

This was chosen as a practical compromise:
    - enough trees to reduce variance and improve stability
    - still computationally manageable for repeated testing
    - suitable for a relatively small tabular dataset

A single tree would be too unstable; many hundreds of trees usually improve
performance, and 500 is a common robust choice for this kind of task.

Validation
----------
The model is validated in three ways:

    1. OSATS prediction RMSE
        - Calculated separately for each of the six OSATS scores
        - Also reported as a mean RMSE across all six scores

    2. GRS prediction RMSE
        - Computed by comparing true GRS against the sum of the predicted
          OSATS scores

    3. Experience classification accuracy
        - Percentage of correctly predicted Novice / Intermediate / Expert
          labels

Outputs
-------
The script writes two output files:

    - random_forest_predictions.csv
        Contains the true and predicted outputs for the validation set

    - random_forest_validation_summary.csv
        Contains the RMSE and accuracy values for the validation run

In summary, this script tests whether motion features can predict surgical
skill both as a set of OSATS scores and as an overall experience category.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------
# This script will try to find Task 2 automatically. If your structure is
# different, adjust TASK2_DIR manually.
HERE = Path(__file__).resolve().parent
TASK2_CANDIDATE = HERE.parent / "Task 2"
TASK2_DIR = TASK2_CANDIDATE if TASK2_CANDIDATE.exists() else HERE

MERGED_FILES = {
    "suturing": TASK2_DIR / "suturing_merged.csv",
    "needle_passing": TASK2_DIR / "needle_passing_merged.csv",
    "knot_tying": TASK2_DIR / "knot_tying_merged.csv",
}

OUTPUT_DIR = HERE / "classifier_1_outputs"
PREDICTIONS_FILE = OUTPUT_DIR / "random_forest_1_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "random_forest_1_validation_summary.csv"


# ---------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------
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
    Extract the surgeon ID for grouped splitting.

    Examples:
        Suturing_B001 -> B
        NeedlePassing_H005 -> H
    """
    text = str(filename).strip()

    match = re.search(r"_([A-Za-z]+)\d+$", text)
    if match:
        return match.group(1)[0].upper()

    # Fallback: use the first letter after the underscore if present.
    if "_" in text:
        suffix = text.split("_", 1)[1]
        if suffix:
            return suffix[0].upper()

    return text[:1].upper()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric motion-feature columns only.
    Excludes metadata and target columns.
    """
    excluded = {
        "file",
        "filename",
        "task",
        EXPERIENCE_TARGET,
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

    return out


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_models(random_state: int = 42):
    """
    A random forest regressor for the six OSATS scores and a random forest
    classifier for experience level.
    """
    regressor = RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    classifier = RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
    )

    return regressor, classifier


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_merged_datasets()
    df = prepare_data(df)

    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y_osats = df[OSATS_TARGETS].copy()
    y_grs = df[GRS_TARGET].copy()
    y_exp = df["experience_level"].copy()
    groups = df["surgeon_group"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y_exp, groups=groups))

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_osats_train = y_osats.iloc[train_idx]
    y_osats_val = y_osats.iloc[val_idx]

    y_grs_val = y_grs.iloc[val_idx]
    y_exp_train = y_exp.iloc[train_idx]
    y_exp_val = y_exp.iloc[val_idx]

    regressor, classifier = build_models(random_state=42)

    # Train the OSATS regressor.
    regressor.fit(X_train, y_osats_train)

    # Train the experience classifier.
    classifier.fit(X_train, y_exp_train)

    # Predict on validation set.
    pred_osats = regressor.predict(X_val)
    pred_exp = classifier.predict(X_val)

    pred_osats_df = pd.DataFrame(pred_osats, columns=[f"pred_{c}" for c in OSATS_TARGETS], index=y_osats_val.index)
    pred_grs = pred_osats_df.sum(axis=1)

    # Validation 1: OSATS RMSE
    osats_rmse = {}
    for i, target in enumerate(OSATS_TARGETS):
        osats_rmse[target] = rmse(y_osats_val[target].to_numpy(), pred_osats[:, i])

    mean_osats_rmse = float(np.mean(list(osats_rmse.values())))

    # Validation 2: GRS RMSE
    grs_rmse = rmse(y_grs_val.to_numpy(), pred_grs.to_numpy())

    # Validation 3: Experience classification accuracy
    exp_accuracy = accuracy_score(y_exp_val, pred_exp)

    # Save predictions for inspection.
    preds_out = pd.DataFrame(index=y_exp_val.index)
    preds_out["filename"] = df.loc[val_idx, "filename"].values
    preds_out["task"] = df.loc[val_idx, "task"].values
    preds_out["experience_true"] = y_exp_val.values
    preds_out["experience_pred"] = pred_exp

    preds_out["grs_true"] = y_grs_val.values
    preds_out["grs_pred"] = pred_grs.values

    for target in OSATS_TARGETS:
        preds_out[f"{target}_true"] = df.loc[val_idx, target].values
    for target in OSATS_TARGETS:
        preds_out[f"{target}_pred"] = pred_osats_df[f"pred_{target}"].values

    preds_out.to_csv(PREDICTIONS_FILE, index=False)

    # Save summary metrics.
    summary_rows = [
        {"metric": f"osats_rmse_{k}", "value": v} for k, v in osats_rmse.items()
    ]
    summary_rows.append({"metric": "osats_rmse_mean", "value": mean_osats_rmse})
    summary_rows.append({"metric": "grs_rmse", "value": grs_rmse})
    summary_rows.append({"metric": "experience_accuracy", "value": exp_accuracy})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    # Print results.
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

