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
