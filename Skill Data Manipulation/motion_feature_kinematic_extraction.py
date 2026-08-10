
"""
===========================================================================
Motion Feature Extraction from JIGSAWS Robotic Surgery Kinematic Data
===========================================================================

Purpose
-------
This script extracts objective motion-performance metrics from a single
JIGSAWS kinematic data file. The metrics quantify the quality and efficiency
of the movements performed by both robotic surgical instruments (Slave Left
and Slave Right) during a surgical training task.

The extracted features are intended for use in downstream machine learning
models for surgical skill assessment and classification.

Input
-----
The script expects one JIGSAWS kinematic (.txt) file containing a time series
of 76 kinematic variables sampled at 30 Hz.

Each row represents one time sample.

Relevant variables used in this script are:

    Slave Left tooltip position (x, y, z)
        Columns 39–41

    Slave Right tooltip position (x, y, z)
        Columns 58–60

Although the JIGSAWS files contain additional information such as orientation,
translational velocity, rotational velocity and gripper angle, only the
instrument tip positions are used for the motion metrics calculated here.

Metrics Calculated
------------------
For both the Slave Left and Slave Right instruments, the script computes:

    • Path Length
        Total distance travelled by the instrument tip.

    • Straight-Line Distance
        Euclidean distance between the start and end positions.

    • Economy of Motion
        A measure of the control effort required to execute the trajectory,
        calculated as the integral of squared acceleration over time.
        Lower values indicate more economical movement.

    • RMS Jerk
        Root-mean-square of the third derivative of position.
        Lower values indicate smoother motion.

    • Dimensionless Jerk Cost
        A normalised jerk-based smoothness metric that is independent of
        movement duration and path length.

    • Smoothness Score
        A logarithmic transformation of the dimensionless jerk cost.
        Larger values correspond to smoother instrument motion.

    • Duration
        Total duration of the recording.

In addition, combined metrics are produced across both slave instruments,
including:

    • Total Path Length
    • Mean Economy of Motion
    • Mean RMS Jerk
    • Mean Smoothness Score
    • Duration

Output
------
The script produces:

1. A summary table printed to the terminal.

2. A CSV file containing one row of extracted motion features for the input
   trial. The CSV is saved in the same directory as the input file and is
   named:

       <input_filename>_metrics.csv

These extracted features form the input feature vector for later stages of
the project, where they will be combined with additional metrics and used for
machine learning-based surgical skill classification.
"""



from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
JIGSAW_ROOT = ROOT_DIR / "JIGSAW"
KINEMATIC_FOLDERS = {
    "Suturing": JIGSAW_ROOT / "Suturing" / "Suturing" / "Suturing kinematics" / "AllGestures",
    "Needle_Passing": JIGSAW_ROOT / "Needle_Passing" / "Needle_Passing" / "Needle_Passing kinematics" / "AllGestures",
    "Knot_Tying": JIGSAW_ROOT / "Knot_Tying" / "Knot_Tying" / "Knot_Tying kinematics" / "AllGestures",
}

INPUT_FILE = "Suturing_B001.txt"
FPS = 30.0
DT = 1.0 / FPS

ARM_COLUMNS = {
    "slave_left": slice(38, 41),   # cols 39-41
    "slave_right": slice(57, 60),  # cols 58-60
}


# ============================================================
# LOADING
# ============================================================
def load_kinematic_file(file_path: str | Path) -> np.ndarray:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if df.shape[1] != 76:
        raise ValueError(
            f"Expected 76 columns, but found {df.shape[1]}."
        )

    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    if df.empty:
        raise ValueError("No valid numeric rows were found in the file.")

    return df.to_numpy(dtype=float)


def find_kinematic_file(file_name_or_path: str | Path) -> Path:
    """Resolve a kinematic file path from the project folders."""
    supplied_path = Path(file_name_or_path)

    if supplied_path.exists():
        return supplied_path.resolve()

    file_name = supplied_path.name
    if supplied_path.suffix.lower() != ".txt":
        file_name = supplied_path.with_suffix(".txt").name

    for folder in KINEMATIC_FOLDERS.values():
        if not folder.exists():
            continue

        exact_match = folder / file_name
        if exact_match.exists():
            return exact_match.resolve()

        for candidate in folder.glob("*.txt"):
            if candidate.name.lower() == file_name.lower():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find '{file_name_or_path}' in the Task 1 kinematic folders."
    )


def get_xyz_trajectory(data: np.ndarray, arm: str) -> np.ndarray:
    if arm not in ARM_COLUMNS:
        raise ValueError(f"Unknown arm '{arm}'. Choose from: {list(ARM_COLUMNS.keys())}")
    return data[:, ARM_COLUMNS[arm]]


# ============================================================
# METRICS
# ============================================================
def path_length(pos: np.ndarray) -> float:
    """
    Compute the total distance travelled by the instrument tip.

    The path length is calculated by summing the Euclidean distance between
    each pair of consecutive 3D position samples.
    """
    if len(pos) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def straight_line_distance(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(pos[-1] - pos[0]))


def economy_of_motion(pos: np.ndarray, dt: float = DT) -> float:
    """
    Compute the movement cost as the integral of squared acceleration.

    This serves as a proxy for the control effort (minimum-energy cost)
    required to execute the trajectory.

    Lower values indicate more economical movement.
    Units: (distance/s²)² × s
    """
    if len(pos) < 3:
        return 0.0

    acc = acceleration(pos, dt)
    acc_mag_sq = np.sum(acc**2, axis=1)

    movement_cost = np.sum(acc_mag_sq) * dt

    return float(movement_cost)


def _safe_gradient(arr: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute the numerical gradient of a time series.

    Uses a second-order finite difference scheme where sufficient data are
    available, but falls back to the default gradient calculation for very
    short trajectories to avoid numerical errors.
    """
    if arr.shape[0] < 3:
        return np.gradient(arr, dt, axis=0)
    return np.gradient(arr, dt, axis=0, edge_order=2)


def velocity(pos: np.ndarray, dt: float = DT) -> np.ndarray:
    return _safe_gradient(pos, dt)


def acceleration(pos: np.ndarray, dt: float = DT) -> np.ndarray:
    return _safe_gradient(velocity(pos, dt), dt)


def jerk_vector(pos: np.ndarray, dt: float = DT) -> np.ndarray:
    return _safe_gradient(acceleration(pos, dt), dt)


def rms_jerk(pos: np.ndarray, dt: float = DT) -> float:
    """
    Compute the root-mean-square (RMS) jerk of the trajectory.

    Jerk is the third derivative of position with respect to time. The RMS
    jerk is calculated from the magnitude of the jerk vector at each time
    sample and provides a measure of the abruptness of the motion. Lower
    values indicate smoother movements.
    """
    if len(pos) < 3:
        return 0.0
    j = jerk_vector(pos, dt)
    return float(np.sqrt(np.mean(np.linalg.norm(j, axis=1) ** 2)))


def duration_seconds(pos: np.ndarray, dt: float = DT) -> float:
    if len(pos) < 2:
        return 0.0
    return float((len(pos) - 1) * dt)


def dimensionless_jerk_cost(pos: np.ndarray, dt: float = DT) -> float:
    """
    Compute the dimensionless jerk cost of the trajectory.

    This metric normalizes the jerk integral by the path length and duration
    to provide a scale-invariant measure of motion smoothness.
    """
    if len(pos) < 3:
        return 0.0

    pl = path_length(pos)
    if pl == 0:
        return 0.0

    j = jerk_vector(pos, dt)
    jerk_integral = float(np.sum(np.sum(j ** 2, axis=1)) * dt)
    T = duration_seconds(pos, dt)

    return float((T ** 5 / (pl ** 2 + 1e-12)) * jerk_integral)


def smoothness_score(pos: np.ndarray, dt: float = DT) -> float: 
    """
    The log transformation of the dimensionless jerk cost provides a more interpretable
    smoothness score. Higher values correspond to smoother instrument motion.
    """
    return float(-np.log10(dimensionless_jerk_cost(pos, dt) + 1e-12))


def analyse_arm(data: np.ndarray, arm: str) -> dict:
    pos = get_xyz_trajectory(data, arm)

    return {
        f"{arm}_num_samples": int(len(pos)),
        f"{arm}_duration_s": duration_seconds(pos, DT),
        f"{arm}_path_length": path_length(pos),
        f"{arm}_straight_line_distance": straight_line_distance(pos),
        f"{arm}_economy_of_motion": economy_of_motion(pos),
        f"{arm}_rms_jerk": rms_jerk(pos, DT),
        f"{arm}_dimensionless_jerk_cost": dimensionless_jerk_cost(pos, DT),
        f"{arm}_smoothness_score": smoothness_score(pos, DT),
    }


def analyse_file(file_path: str | Path) -> dict:
    data = load_kinematic_file(file_path)

    left = analyse_arm(data, "slave_left")
    right = analyse_arm(data, "slave_right")

    combined = {
        "file": str(file_path),
        "total_path_length /m": left["slave_left_path_length"] + right["slave_right_path_length"],
        "mean_economy_of_motion /m^2s^-3": (
            left["slave_left_economy_of_motion"] + right["slave_right_economy_of_motion"]
        ) / 2.0,
        "mean_rms_jerk /ms^-3": (left["slave_left_rms_jerk"] + right["slave_right_rms_jerk"]) / 2.0,
        "mean_smoothness_score /(dimensionless)": (
            left["slave_left_smoothness_score"] + right["slave_right_smoothness_score"]
        ) / 2.0,
        "duration /s": max(left["slave_left_duration_s"], right["slave_right_duration_s"]),
    }

    return {**combined, **left, **right}


if __name__ == "__main__":
    input_path = find_kinematic_file(INPUT_FILE)
    results = analyse_file(input_path)
    results_df = pd.DataFrame([results])

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(results_df.to_string(index=False))

    output_path = input_path.with_suffix("").as_posix() + "_metrics.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved metrics to: {output_path}")

