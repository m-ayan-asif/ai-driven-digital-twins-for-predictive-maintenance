"""Reproduce the test-error breakdown by official FD001 RUL range."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from src.data_prep import load_fd001
from src.metrics import rul_stratified_metrics
from src.models.rul_model import make_mean_baseline, make_random_forest

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 37, 42)


def run() -> None:
    dataset = load_fd001(ROOT / "data" / "raw")
    run_rows: list[dict[str, float | int | str]] = []
    for seed in SEEDS:
        model = make_random_forest(seed)
        model.fit(dataset.X_train, dataset.y_train)
        prediction = model.predict(dataset.X_test)
        baseline = make_mean_baseline()
        baseline.fit(dataset.X_train, dataset.y_train)
        baseline_prediction = baseline.predict(dataset.X_test)
        run_rows.extend(
            {"seed": seed, **row}
            for row in rul_stratified_metrics(
                dataset.y_test, prediction, baseline_prediction
            )
        )

    output_dir = ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / "rul_range_runs.csv"
    with runs_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)

    summary_rows: list[dict[str, float | int | str]] = []
    for label in dict.fromkeys(str(row["rul_range"]) for row in run_rows):
        group = [row for row in run_rows if row["rul_range"] == label]
        result: dict[str, float | int | str] = {
            "rul_range": label,
            "engine_count": int(group[0]["engine_count"]),
        }
        for metric in ("rmse", "mae", "baseline_rmse", "baseline_mae"):
            values = [float(row[metric]) for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values)
        summary_rows.append(result)

    summary_path = output_dir / "rul_range_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {runs_path} and {summary_path}")


if __name__ == "__main__":
    run()
