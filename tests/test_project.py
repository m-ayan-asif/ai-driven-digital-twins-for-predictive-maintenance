import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data_prep import FEATURE_COUNT, load_fd001, window_features
from src.deployment_sim.simulator import load_scenarios, simulate_scenarios
from src.metrics import rul_stratified_metrics


def make_row(engine: int, cycle: int, value: float) -> list[float]:
    return [engine, cycle, 0.1, 0.2, 0.3] + [value + sensor for sensor in range(21)]


class DataPrepTests(unittest.TestCase):
    def test_window_features_are_finite_and_have_expected_shape(self):
        rows = np.asarray([make_row(1, cycle, float(cycle * 2)) for cycle in range(1, 5)])
        features = window_features(rows, window_size=3)
        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0:3].tolist(), [0.1, 0.2, 0.3])
        self.assertAlmostEqual(features[-21], 2.0)

    def test_window_size_must_be_positive(self):
        rows = np.asarray([make_row(1, 1, 1.0)])
        with self.assertRaises(ValueError):
            window_features(rows, window_size=0)

    def test_loader_pairs_terminal_test_labels_to_engines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = np.asarray(
                [make_row(engine, cycle, float(cycle)) for engine in (1, 2) for cycle in range(1, 5)]
            )
            test = np.asarray(
                [make_row(engine, cycle, float(cycle)) for engine in (1, 2) for cycle in range(1, 4)]
            )
            np.savetxt(root / "train_FD001.txt", train)
            np.savetxt(root / "test_FD001.txt", test)
            np.savetxt(root / "RUL_FD001.txt", np.asarray([5, 9]))
            dataset = load_fd001(root, window_size=3, rul_cap=2)
        self.assertEqual(dataset.X_train.shape, (8, FEATURE_COUNT))
        self.assertEqual(dataset.X_test.shape, (2, FEATURE_COUNT))
        self.assertEqual(dataset.test_windows.shape, (2, 3, 26))
        np.testing.assert_array_equal(dataset.y_test, [5, 9])
        self.assertEqual(float(dataset.y_train.max()), 2.0)


class MetricsTests(unittest.TestCase):
    def test_rul_stratification_uses_disjoint_inclusive_upper_bounds(self):
        rows = rul_stratified_metrics(
            truth=np.asarray([0, 30, 31, 60, 61, 90, 91]),
            prediction=np.asarray([0, 29, 33, 60, 60, 88, 100]),
            baseline_prediction=np.asarray([10, 10, 10, 10, 10, 10, 10]),
        )
        self.assertEqual(
            [(row["rul_range"], row["engine_count"]) for row in rows],
            [("0-30", 2), ("31-60", 2), ("61-90", 2), (">90", 1)],
        )
        self.assertAlmostEqual(rows[1]["rmse"], np.sqrt(2.0))

    def test_rul_stratification_rejects_mismatched_arrays(self):
        with self.assertRaises(ValueError):
            rul_stratified_metrics(
                np.asarray([10.0]),
                np.asarray([11.0, 12.0]),
                np.asarray([10.0]),
            )


class DeploymentTests(unittest.TestCase):
    def test_scenarios_preserve_accuracy_and_reduce_hybrid_payload(self):
        scenarios = [
            {
                "name": "cloud-only",
                "placement": "cloud",
                "rtt_ms": 80,
                "bandwidth_mbps": 20,
                "edge_compute_factor": 1,
            },
            {
                "name": "edge-only",
                "placement": "edge",
                "rtt_ms": 1,
                "bandwidth_mbps": 20,
                "edge_compute_factor": 2,
            },
            {
                "name": "hybrid-edge-cloud",
                "placement": "hybrid",
                "rtt_ms": 80,
                "bandwidth_mbps": 20,
                "edge_compute_factor": 2,
            },
        ]
        results = simulate_scenarios(
            scenarios,
            truth=np.asarray([12.0, 20.0]),
            prediction=np.asarray([10.0, 22.0]),
            feature_times_ms=np.asarray([1.0, 1.0]),
            inference_times_ms=np.asarray([2.0, 2.0]),
            window_size=30,
        )
        self.assertEqual(len({result["rmse"] for result in results}), 1)
        self.assertEqual(results[0]["transfer_bytes_per_inference"], 30 * 24 * 4)
        self.assertEqual(results[1]["transfer_bytes_per_inference"], 0)
        self.assertLess(
            results[2]["transfer_bytes_per_inference"],
            results[0]["transfer_bytes_per_inference"],
        )
        self.assertGreater(results[0]["latency_p50_ms"], results[1]["latency_p50_ms"])

    def test_deployment_config_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            scenario = {
                "name": "same",
                "placement": "cloud",
                "rtt_ms": 10,
                "bandwidth_mbps": 20,
                "edge_compute_factor": 1,
            }
            path.write_text(
                json.dumps({"scenarios": [scenario, scenario]}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_scenarios(path)

    def test_deployment_config_rejects_negative_round_trip_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            scenario = {
                "name": "invalid",
                "placement": "cloud",
                "rtt_ms": -1,
                "bandwidth_mbps": 20,
                "edge_compute_factor": 1,
            }
            path.write_text(
                json.dumps({"scenarios": [scenario]}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_scenarios(path)


if __name__ == "__main__":
    unittest.main()
