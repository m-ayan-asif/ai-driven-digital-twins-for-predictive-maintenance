"""Deterministic model definitions for the C-MAPSS comparison."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor


def make_random_forest(seed: int, trees: int = 100) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=trees,
        max_depth=16,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=seed,
    )


def make_mean_baseline() -> DummyRegressor:
    return DummyRegressor(strategy="mean")
