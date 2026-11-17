"""Generate the study's workflow and deployment-scenario diagram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(axis, x, y, width, height, text, color, fontsize=9):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#0f172a",
        wrap=True,
    )


def _arrow(axis, start, end, color="#475569", connection="arc3,rad=0"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=color,
            connectionstyle=connection,
        )
    )


def plot_architecture(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6.3))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    top_y, top_h, top_w = 0.71, 0.2, 0.19
    top_x = (0.025, 0.275, 0.525, 0.775)
    top_text = (
        "NASA C-MAPSS FD001\ntraining and test\nsensor cycles",
        "Causal feature\npipeline\n30-cycle windows\n66 features",
        "Random Forest\n100 trees\nseeds 13, 37, 42",
        "Shared test\npredictions\nsame model and\nfeatures",
    )
    top_colors = ("#dbeafe", "#e0f2fe", "#dcfce7", "#fef3c7")
    for x, text, color in zip(top_x, top_text, top_colors):
        _box(axis, x, top_y, top_w, top_h, text, color)
    for index in range(3):
        _arrow(
            axis,
            (top_x[index] + top_w + 0.008, top_y + top_h / 2),
            (top_x[index + 1] - 0.008, top_y + top_h / 2),
        )

    scenarios = (
        (
            0.025,
            "Cloud-only (simulated)\nassumed raw-window uplink\nfeatures + RF modeled in cloud",
            "#fee2e2",
        ),
        (
            0.355,
            "Edge-only (simulated)\nfeatures + RF modeled at edge\n0 B uplink",
            "#dcfce7",
        ),
        (
            0.685,
            "Hybrid edge-cloud\n(simulated)\nassumed feature-vector uplink\ninference modeled in cloud",
            "#ede9fe",
        ),
    )
    scenario_y, scenario_h, scenario_w = 0.32, 0.2, 0.29
    scenario_centers = []
    for x, text, color in scenarios:
        _box(axis, x, scenario_y, scenario_w, scenario_h, text, color)
        scenario_centers.append(x + scenario_w / 2)
    dispatch_x, dispatch_y, dispatch_w, dispatch_h = 0.33, 0.585, 0.34, 0.09
    _box(
        axis,
        dispatch_x,
        dispatch_y,
        dispatch_w,
        dispatch_h,
        "Apply configured placement formulas\nto shared predictions and\nmeasured local timings",
        "#f1f5f9",
        fontsize=8,
    )
    _arrow(
        axis,
        (top_x[3] + top_w / 2, top_y - 0.012),
        (dispatch_x + dispatch_w / 2, dispatch_y + dispatch_h + 0.012),
    )
    branch_y = 0.56
    axis.plot(
        (scenario_centers[0], scenario_centers[-1]),
        (branch_y, branch_y),
        color="#475569",
        linewidth=1.2,
    )
    _arrow(
        axis,
        (dispatch_x + dispatch_w / 2, dispatch_y - 0.012),
        (dispatch_x + dispatch_w / 2, branch_y),
    )
    for center in scenario_centers:
        _arrow(
            axis,
            (center, branch_y),
            (center, scenario_y + scenario_h + 0.012),
        )

    output_x, output_y, output_w, output_h = 0.245, 0.06, 0.51, 0.17
    _box(
        axis,
        output_x,
        output_y,
        output_w,
        output_h,
        "Score against test RUL; compare RMSE / MAE and simulated p95 latency\nplus local compute, uplink bytes, and normalized\nresource-cost index",
        "#e2e8f0",
        fontsize=8,
    )
    for center in scenario_centers:
        _arrow(
            axis,
            (center, scenario_y - 0.012),
            (output_x + output_w / 2, output_y + output_h + 0.012),
            connection="arc3,rad=0.08" if center < output_x + output_w / 2 else "arc3,rad=-0.08",
        )

    figure.suptitle(
        "Scenario model: placement costs are estimated; predictor outputs are fixed",
        y=0.98,
        fontsize=11,
        weight="bold",
        color="#0f172a",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "figures" / "deployment_architecture.png"
    plot_architecture(output)
    print(output)


if __name__ == "__main__":
    main()
