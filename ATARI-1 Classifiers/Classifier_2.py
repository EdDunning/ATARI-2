'''
Same method as Classifier 1 with the additional feature of task-ID one-hot encoding. 
This allows the model to learn task-specific patterns in the motion features.
'''
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut


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

OUTPUT_DIR = HERE / "classifier_2_outputs"
PREDICTIONS_FILE = OUTPUT_DIR / "random_forest_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "random_forest_validation_summary.csv"


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

    # Encode task as one-hot features.
    out["task"] = out["task"].astype(str).str.strip().str.lower()
    task_dummies = pd.get_dummies(out["task"], prefix="task")
    out = pd.concat([out, task_dummies], axis=1)

    out["surgeon_group"] = out["filename"].apply(extract_surgeon_group)

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Use motion features plus task-ID one-hot columns only.
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
        if col.startswith("task_"):
            feature_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No usable feature columns were found.")

    return feature_cols


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_models(random_state: int = 42):
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
    y_grs = df[OSATS_TARGETS].sum(axis=1)
    y_exp = df["experience_level"].copy()
    groups = df["surgeon_group"].copy()

    logo = LeaveOneGroupOut()

    # Store predictions for every held-out row across all folds.
    all_pred_rows = []

    # For summary metrics, keep all true/pred values.
    osats_true_all = []
    osats_pred_all = []
    grs_true_all = []
    grs_pred_all = []
    exp_true_all = []
    exp_pred_all = []

    for fold_idx, (train_idx, val_idx) in enumerate(logo.split(X, y_exp, groups=groups), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_osats_train = y_osats.iloc[train_idx]
        y_osats_val = y_osats.iloc[val_idx]

        y_exp_train = y_exp.iloc[train_idx]
        y_exp_val = y_exp.iloc[val_idx]

        y_grs_val = y_grs.iloc[val_idx]

        regressor, classifier = build_models(random_state=42)

        regressor.fit(X_train, y_osats_train)
        classifier.fit(X_train, y_exp_train)

        pred_osats = regressor.predict(X_val)
        pred_exp = classifier.predict(X_val)

        pred_osats_df = pd.DataFrame(
            pred_osats,
            columns=[f"pred_{c}" for c in OSATS_TARGETS],
            index=df.iloc[val_idx].index,
        )
        pred_grs = pred_osats_df.sum(axis=1)

        # Collect fold-level data for final summary.
        osats_true_all.append(y_osats_val.reset_index(drop=True))
        osats_pred_all.append(pred_osats_df.reset_index(drop=True))
        grs_true_all.append(y_grs_val.reset_index(drop=True))
        grs_pred_all.append(pred_grs.reset_index(drop=True))
        exp_true_all.append(y_exp_val.reset_index(drop=True))
        exp_pred_all.append(pd.Series(pred_exp, name="experience_pred"))

        fold_out = pd.DataFrame(index=df.iloc[val_idx].index)
        fold_out["filename"] = df.iloc[val_idx]["filename"].values
        fold_out["task"] = df.iloc[val_idx]["task"].values
        fold_out["surgeon_group"] = df.iloc[val_idx]["surgeon_group"].values
        fold_out["experience_true"] = y_exp_val.values
        fold_out["experience_pred"] = pred_exp
        fold_out["grs_true"] = y_grs_val.values
        fold_out["grs_pred"] = pred_grs.values

        for target in OSATS_TARGETS:
            fold_out[f"{target}_true"] = df.iloc[val_idx][target].values
            fold_out[f"{target}_pred"] = pred_osats_df[f"pred_{target}"].values

        all_pred_rows.append(fold_out)

        print(f"Completed fold {fold_idx}")

    predictions = pd.concat(all_pred_rows, axis=0).sort_values(["filename"]).reset_index(drop=True)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

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

