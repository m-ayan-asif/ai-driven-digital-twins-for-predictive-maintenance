"""Train, evaluate, and simulate all deployment scenarios on FD001."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from src.data_prep import DATASET_URL, load_fd001
from src.deployment_sim.simulator import (
    load_scenarios,
    measure_inference,
    simulate_scenarios,
)
from src.metrics import regression_metrics
from src.models.rul_model import make_mean_baseline, make_random_forest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (13, 37, 42)
TREE_COUNT = 100


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty results file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    scenarios = sorted({row["scenario"] for row in rows})
    summary: list[dict] = []
    for scenario in scenarios:
        group = [row for row in rows if row["scenario"] == scenario]
        result: dict[str, float | str] = {
            "scenario": scenario,
            "runs": len(group),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(result)
    return summary


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def run(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    seeds: tuple[int, ...],
    trees: int,
) -> None:
    dataset = load_fd001(input_dir)
    scenarios = load_scenarios(config_path)
    model_rows: list[dict] = []
    deployment_rows: list[dict] = []

    for seed in seeds:
        model = make_random_forest(seed=seed, trees=trees)
        start = time.perf_counter()
        model.fit(dataset.X_train, dataset.y_train)
        fit_seconds = time.perf_counter() - start
        prediction = model.predict(dataset.X_test)
        metrics = regression_metrics(dataset.y_test, prediction)
        model_rows.append(
            {
                "model": "random_forest",
                "seed": seed,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "fit_seconds": fit_seconds,
            }
        )

        baseline = make_mean_baseline()
        baseline.fit(dataset.X_train, dataset.y_train)
        baseline_prediction = baseline.predict(dataset.X_test)
        baseline_metrics = regression_metrics(dataset.y_test, baseline_prediction)
        model_rows.append(
            {
                "model": "training_mean",
                "seed": seed,
                "rmse": baseline_metrics["rmse"],
                "mae": baseline_metrics["mae"],
                "fit_seconds": 0.0,
            }
        )

        feature_times, inference_times = measure_inference(
            model,
            dataset.X_test,
            dataset.test_windows,
        )
        runs = simulate_scenarios(
            scenarios=scenarios,
            truth=dataset.y_test,
            prediction=prediction,
            feature_times_ms=feature_times,
            inference_times_ms=inference_times,
            window_size=dataset.test_windows.shape[1],
        )
        for run_row in runs:
            deployment_rows.append(
                {
                    "seed": seed,
                    **run_row,
                    "fit_seconds": fit_seconds,
                }
            )
        print(
            f"seed={seed} RF RMSE={metrics['rmse']:.3f}, "
            f"MAE={metrics['mae']:.3f}, fit={fit_seconds:.2f}s"
        )

    results_dir = output_dir
    raw_dir = results_dir / "raw"
    _write_csv(raw_dir / "model_runs.csv", model_rows)
    _write_csv(raw_dir / "deployment_runs.csv", deployment_rows)
    summary = _aggregate(
        deployment_rows,
        (
            "rmse",
            "mae",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "compute_mean_ms",
            "transfer_bytes_per_inference",
            "resource_cost_index",
        ),
    )
    _write_csv(results_dir / "summary.csv", summary)

    train_path = input_dir / "train_FD001.txt"
    test_path = input_dir / "test_FD001.txt"
    rul_path = input_dir / "RUL_FD001.txt"
    try:
        config_reference = config_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        config_reference = config_path.name
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "NASA C-MAPSS FD001",
        "dataset_source": DATASET_URL,
        "input_sha256": {
            path.name: _sha256(path) for path in (train_path, test_path, rul_path)
        },
        "training_rows": int(len(dataset.y_train)),
        "test_engines": int(len(dataset.y_test)),
        "feature_count": int(dataset.X_train.shape[1]),
        "window_size": int(dataset.test_windows.shape[1]),
        "rul_cap_cycles": 125,
        "seeds": list(seeds),
        "random_forest": {
            "trees": trees,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "max_features": 0.8,
        },
        "deployment_config": config_reference,
        "deployment_config_sha256": _sha256(config_path),
        "scenario_assumptions": scenarios,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "matplotlib": _package_version("matplotlib"),
        },
        "latency_note": (
            "Scenario latency combines measured local feature/model time with "
            "configured network and edge-compute assumptions; it is not measured "
            "cloud or edge infrastructure latency."
        ),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote results to {results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "deployment.json",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trees", type=int, default=TREE_COUNT)
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be positive")
    run(args.input_dir, args.output_dir, args.config, tuple(args.seeds), args.trees)


if __name__ == "__main__":
    main()
