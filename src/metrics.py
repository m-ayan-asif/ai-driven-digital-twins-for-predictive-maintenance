"""Evaluation metrics shared by training and deployment runs."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(
    truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Truth and prediction shapes differ: {truth.shape} vs {prediction.shape}."
        )
    if len(truth) == 0 or not np.isfinite(prediction).all():
        raise ValueError("Metrics require non-empty finite predictions.")
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
    }


def rul_stratified_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline_prediction: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Compare model errors across fixed bins of official test RUL labels."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    baseline_prediction = np.asarray(baseline_prediction, dtype=np.float64).reshape(-1)
    if not (
        truth.shape == prediction.shape == baseline_prediction.shape
        and len(truth) > 0
        and np.isfinite(truth).all()
    ):
        raise ValueError("Stratified metrics require equal, non-empty finite arrays.")

    ranges = (
        ("0-30", (truth >= 0) & (truth <= 30)),
        ("31-60", (truth > 30) & (truth <= 60)),
        ("61-90", (truth > 60) & (truth <= 90)),
        (">90", truth > 90),
    )
    results: list[dict[str, float | int | str]] = []
    for label, mask in ranges:
        if not mask.any():
            continue
        model_metrics = regression_metrics(truth[mask], prediction[mask])
        baseline_metrics = regression_metrics(truth[mask], baseline_prediction[mask])
        results.append(
            {
                "rul_range": label,
                "engine_count": int(mask.sum()),
                "rmse": model_metrics["rmse"],
                "mae": model_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_mae": baseline_metrics["mae"],
            }
        )
    return results
