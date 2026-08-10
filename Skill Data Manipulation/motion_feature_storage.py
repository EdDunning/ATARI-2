"""
===========================================================================
Motion Feature Storage
===========================================================================

Purpose
-------
This script automates the extraction of motion features from all JIGSAWS
kinematic trials for the Suturing, Needle Passing and Knot Tying tasks.

For each kinematic file, the script calls the functions defined in
`motion_feature_extraction.py` to compute motion metrics for the Slave Left
and Slave Right robotic instruments. The corresponding metrics from the two
instruments are then averaged to produce a single feature vector describing
the overall performance for that surgical trial.

How it works
------------
1. Iterates through each of the three JIGSAWS task folders.
2. Loads every kinematic (.txt) file within each folder.
3. Calls the motion feature extraction functions for both slave arms.
4. Computes the mean value of each motion metric across the two instruments.
5. Stores the resulting feature vector for each trial.
6. Writes one CSV summary file for each surgical task.

Inputs
------
The script expects the following folders to exist in the same directory as
this file:

    - Suturing kinematics/AllGestures
    - Needle_passing kinematics/AllGestures
    - Knot_Tying kinematics/AllGestures

Each folder should contain the JIGSAWS kinematic (.txt) files for that task.

Outputs
-------
Three CSV files are produced, each containing one row per surgical trial and
one column for each extracted motion feature:

    - suturing_kinematics_summary.csv
    - needle_passing_kinematics_summary.csv
    - knot_tying_kinematics_summary.csv

These summary files provide the feature matrices used as inputs for the
subsequent machine learning stage of the project.
"""


from pathlib import Path

import pandas as pd

from motion_feature_kinematic_extraction import load_kinematic_file, analyse_arm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
JIGSAW_ROOT = ROOT_DIR / "JIGSAW"

DATASETS = {
    "Suturing kinematics": JIGSAW_ROOT / "Suturing" / "Suturing" / "Suturing kinematics" / "AllGestures",
    "Needle_Passing kinematics": JIGSAW_ROOT / "Needle_Passing" / "Needle_Passing" / "Needle_Passing kinematics" / "AllGestures",
    "Knot_Tying kinematics": JIGSAW_ROOT / "Knot_Tying" / "Knot_Tying" / "Knot_Tying kinematics" / "AllGestures",
}

OUTPUT_FILES = {
    "Suturing kinematics": SCRIPT_DIR / "suturing_kinematics_summary.csv",
    "Needle_Passing kinematics": SCRIPT_DIR / "needle_passing_kinematics_summary.csv",
    "Knot_Tying kinematics": SCRIPT_DIR / "knot_tying_kinematics_summary.csv",
}

# Metrics to store for each slave arm.
# These keys must match the names returned by analyse_arm(...).
METRICS = [
    "path_length",
    "economy_of_motion",
    "rms_jerk",
    "smoothness_score",
    "duration_s",
]


def analyse_file_both_arms(file_path: Path) -> dict:
    """
    Load one JIGSAWS kinematic file and compute features for both slave arms.

    Returns a row dictionary containing separate left and right arm metrics
    for each motion feature.
    """
    data = load_kinematic_file(file_path)

    left = analyse_arm(data, "slave_left")
    right = analyse_arm(data, "slave_right")

    row = {"file": file_path.name}

    for metric in METRICS:
        left_key = f"slave_left_{metric}"
        right_key = f"slave_right_{metric}"

        if left_key not in left or right_key not in right:
            raise KeyError(f"Missing metric keys: {left_key}, {right_key}")

        row[left_key] = left[left_key]
        row[right_key] = right[right_key]

    return row


def process_dataset(dataset_name: str, folder_path: Path, output_csv: Path) -> None:
    """
    Process all .txt files in one dataset folder and save a CSV summary.
    """
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(folder_path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {folder_path}")

    rows = []
    for file_path in files:
        try:
            rows.append(analyse_file_both_arms(file_path))
        except Exception as e:
            print(f"Skipping {file_path.name} in {dataset_name}: {e}")

    if not rows:
        raise RuntimeError(f"No files were processed successfully in {dataset_name}.")

    df = pd.DataFrame(rows)

    column_order = ["file"] + [
        f"slave_left_{metric}" for metric in METRICS
    ] + [
        f"slave_right_{metric}" for metric in METRICS
    ]
    df = df[column_order]

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows for {dataset_name} to {output_csv}")


def main() -> None:
    for dataset_name, folder_path in DATASETS.items():
        output_csv = OUTPUT_FILES[dataset_name]
        process_dataset(dataset_name, folder_path, output_csv)


if __name__ == "__main__":
    main()

