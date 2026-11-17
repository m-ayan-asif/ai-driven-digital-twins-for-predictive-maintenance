"""C-MAPSS FD001 loading and causal rolling-window feature extraction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WINDOW_SIZE = 30
RUL_CAP = 125
INPUT_COLUMNS = 26
SENSOR_COUNT = 21
FEATURE_COUNT = 3 + SENSOR_COUNT * 3
DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)


@dataclass(frozen=True)
class CMapssDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_windows: np.ndarray
    test_engine_ids: np.ndarray


def _read_table(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Download the NASA C-MAPSS archive from "
            f"{DATASET_URL} and extract FD001 text files into {path.parent}."
        )
    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != INPUT_COLUMNS:
        raise ValueError(
            f"{path} must have {INPUT_COLUMNS} whitespace-separated columns; "
            f"found shape {table.shape}."
        )
    if not np.isfinite(table).all():
        raise ValueError(f"{path} contains non-finite values.")
    return table


def window_features(rows: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Return settings and sensor mean, standard deviation, and linear slope."""
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if rows.ndim != 2 or rows.shape[1] != INPUT_COLUMNS:
        raise ValueError(f"Expected a 2-D window with {INPUT_COLUMNS} columns.")
    if len(rows) == 0:
        raise ValueError("A feature window must contain at least one row.")

    window = rows[-window_size:]
    settings = window[-1, 2:5]
    sensors = window[:, 5:]
    positions = np.arange(len(sensors), dtype=np.float64)
    centered_positions = positions - positions.mean()
    denominator = np.dot(centered_positions, centered_positions)
    if denominator == 0:
        slopes = np.zeros(SENSOR_COUNT, dtype=np.float64)
    else:
        centered_sensors = sensors - sensors.mean(axis=0)
        slopes = centered_positions @ centered_sensors / denominator
    features = np.concatenate(
        (settings, sensors.mean(axis=0), sensors.std(axis=0), slopes)
    )
    if features.shape != (FEATURE_COUNT,):
        raise AssertionError(f"Unexpected feature shape: {features.shape}")
    return features


def _training_examples(
    table: np.ndarray, window_size: int, rul_cap: int
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[float] = []
    engine_ids = np.unique(table[:, 0]).astype(int)
    for engine_id in engine_ids:
        engine = table[table[:, 0] == engine_id]
        engine = engine[np.argsort(engine[:, 1])]
        terminal_cycle = engine[-1, 1]
        for end in range(len(engine)):
            start = max(0, end - window_size + 1)
            features.append(window_features(engine[start : end + 1], window_size))
            remaining = terminal_cycle - engine[end, 1]
            labels.append(float(min(rul_cap, remaining)))
    return np.vstack(features), np.asarray(labels, dtype=np.float64)


def load_fd001(
    input_dir: str | Path,
    window_size: int = WINDOW_SIZE,
    rul_cap: int = RUL_CAP,
) -> CMapssDataset:
    """Load official FD001 train/test splits and final-cycle test labels."""
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if rul_cap < 1:
        raise ValueError("rul_cap must be positive.")
    root = Path(input_dir)
    train = _read_table(root / "train_FD001.txt")
    test = _read_table(root / "test_FD001.txt")
    rul_path = root / "RUL_FD001.txt"
    if not rul_path.is_file():
        raise FileNotFoundError(f"Missing official test labels: {rul_path}")
    test_rul = np.loadtxt(rul_path, dtype=np.float64).reshape(-1)

    X_train, y_train = _training_examples(train, window_size, rul_cap)
    test_engine_ids = np.unique(test[:, 0]).astype(int)
    if len(test_engine_ids) != len(test_rul):
        raise ValueError(
            f"FD001 has {len(test_engine_ids)} test engines but "
            f"{len(test_rul)} final-cycle RUL labels."
        )

    test_features: list[np.ndarray] = []
    test_windows: list[np.ndarray] = []
    for engine_id in test_engine_ids:
        engine = test[test[:, 0] == engine_id]
        engine = engine[np.argsort(engine[:, 1])]
        window = engine[-window_size:]
        test_windows.append(window)
        test_features.append(window_features(window, window_size))

    X_test = np.vstack(test_features)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature extraction produced non-finite values.")
    return CMapssDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=test_rul,
        test_windows=np.stack(test_windows),
        test_engine_ids=test_engine_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    dataset = load_fd001(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fd001_features.npz"
    np.savez_compressed(
        output_path,
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        test_engine_ids=dataset.test_engine_ids,
    )
    print(
        f"Saved {output_path}: train={dataset.X_train.shape}, "
        f"test={dataset.X_test.shape}, features={FEATURE_COUNT}"
    )


if __name__ == "__main__":
    main()
