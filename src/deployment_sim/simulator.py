"""Transparent single-request cloud/edge deployment scenario model."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data_prep import FEATURE_COUNT, window_features
from src.metrics import regression_metrics


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Deployment config must contain a non-empty scenarios list.")
    names = [item.get("name") for item in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("Scenario names must be unique.")
    for item in scenarios:
        required = {
            "name",
            "rtt_ms",
            "bandwidth_mbps",
            "edge_compute_factor",
            "placement",
        }
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Scenario {item.get('name')} missing {sorted(missing)}.")
        if item["bandwidth_mbps"] <= 0 or item["edge_compute_factor"] <= 0:
            raise ValueError(f"Scenario {item['name']} has invalid capacity settings.")
        if item["rtt_ms"] < 0:
            raise ValueError(f"Scenario {item['name']} has a negative RTT.")
        if item["placement"] not in {"cloud", "edge", "hybrid"}:
            raise ValueError(
                f"Scenario {item['name']} has unsupported placement "
                f"{item['placement']!r}."
            )
    return scenarios


def _timed_ms(function) -> float:
    start = time.perf_counter()
    function()
    return (time.perf_counter() - start) * 1000.0


def measure_inference(
    model,
    X_test: np.ndarray,
    test_windows: np.ndarray,
    repeats: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure per-sample local feature and model time for use as simulator inputs."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    feature_times = np.empty(len(test_windows), dtype=np.float64)
    inference_times = np.empty(len(X_test), dtype=np.float64)
    for index, window in enumerate(test_windows):
        measurements = [
            _timed_ms(lambda: window_features(window))
            for _ in range(repeats)
        ]
        feature_times[index] = float(np.median(measurements))
    for index, row in enumerate(X_test):
        sample = row.reshape(1, -1)
        measurements = [
            _timed_ms(lambda: model.predict(sample))
            for _ in range(repeats)
        ]
        inference_times[index] = float(np.median(measurements))
    return feature_times, inference_times


def simulate_scenarios(
    scenarios: list[dict[str, Any]],
    truth: np.ndarray,
    prediction: np.ndarray,
    feature_times_ms: np.ndarray,
    inference_times_ms: np.ndarray,
    window_size: int,
) -> list[dict[str, float | str]]:
    """Calculate scenario latency and resource index; this is not a live deployment."""
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if not (
        len(truth)
        == len(prediction)
        == len(feature_times_ms)
        == len(inference_times_ms)
    ):
        raise ValueError("Scenario inputs must have equal request counts.")

    base_metrics = regression_metrics(truth, prediction)
    raw_payload_bytes = window_size * 24 * np.dtype(np.float32).itemsize
    feature_payload_bytes = FEATURE_COUNT * np.dtype(np.float32).itemsize
    cloud_compute = feature_times_ms + inference_times_ms
    cloud_bytes = float(raw_payload_bytes)
    rows: list[dict[str, float | str]] = []

    for scenario in scenarios:
        placement = scenario["placement"]
        edge_factor = float(scenario["edge_compute_factor"])
        rtt = float(scenario["rtt_ms"])
        bandwidth = float(scenario["bandwidth_mbps"])
        if placement == "cloud":
            compute_ms = cloud_compute
            payload_bytes = raw_payload_bytes
            transfer_ms = payload_bytes * 8.0 / (bandwidth * 1000.0)
            latency = rtt + transfer_ms + compute_ms
        elif placement == "edge":
            compute_ms = edge_factor * cloud_compute
            payload_bytes = 0
            transfer_ms = 0.0
            latency = rtt + compute_ms
        elif placement == "hybrid":
            compute_ms = edge_factor * feature_times_ms + inference_times_ms
            payload_bytes = feature_payload_bytes
            transfer_ms = payload_bytes * 8.0 / (bandwidth * 1000.0)
            latency = rtt + transfer_ms + compute_ms
        else:
            raise ValueError(f"Unsupported placement: {placement}")

        cpu_ratio = float(np.mean(compute_ms / cloud_compute))
        network_ratio = payload_bytes / cloud_bytes
        resource_cost_index = 0.5 * cpu_ratio + 0.5 * network_ratio
        rows.append(
            {
                "scenario": str(scenario["name"]),
                "rmse": base_metrics["rmse"],
                "mae": base_metrics["mae"],
                "latency_mean_ms": float(np.mean(latency)),
                "latency_p50_ms": float(np.percentile(latency, 50)),
                "latency_p95_ms": float(np.percentile(latency, 95)),
                "compute_mean_ms": float(np.mean(compute_ms)),
                "transfer_bytes_per_inference": float(payload_bytes),
                "transfer_ms_per_inference": float(transfer_ms),
                "resource_cost_index": float(resource_cost_index),
            }
        )
    return rows
