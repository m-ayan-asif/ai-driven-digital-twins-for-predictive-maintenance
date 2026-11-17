"""Render deployment latency and normalized resource-cost charts."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_summary(summary_path: Path, output_dir: Path) -> Path:
    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No summary data found in {summary_path}")

    labels = [row["scenario"] for row in rows]
    latency = [float(row["latency_p95_ms_mean"]) for row in rows]
    latency_err = [float(row["latency_p95_ms_std"]) for row in rows]
    cost = [float(row["resource_cost_index_mean"]) for row in rows]
    cost_err = [float(row["resource_cost_index_std"]) for row in rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(labels))
    axes[0].bar(x, latency, yerr=latency_err, capsize=4, color="#4472C4")
    axes[0].set_ylabel("p95 latency (ms)")
    axes[0].set_title("Simulated response latency")
    for index, value in enumerate(latency):
        axes[0].text(
            index,
            value + latency_err[index] + 1.5,
            f"[{value:.1f}]",
            ha="center",
            va="bottom",
            color="#c00000",
            fontsize=9,
        )
    axes[0].set_ylim(top=max(latency + latency_err) * 1.12)
    axes[1].bar(x, cost, yerr=cost_err, capsize=4, color="#70AD47")
    axes[1].set_ylabel("Normalized resource-cost index")
    axes[1].set_title("Modeled compute + transmission")
    for index, value in enumerate(cost):
        axes[1].text(
            index,
            value + cost_err[index] + 0.025,
            f"[{value:.3f}]",
            ha="center",
            va="bottom",
            color="#c00000",
            fontsize=9,
        )
    axes[1].set_ylim(top=max(cost + cost_err) * 1.2)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_path = output_dir / "deployment_tradeoffs.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    print(plot_summary(args.input, args.output_dir))


if __name__ == "__main__":
    main()
