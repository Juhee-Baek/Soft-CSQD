"""Fixed-K K-Modes clusterer implementation."""

from __future__ import annotations

import numpy as np
from kmodes.kmodes import KModes

from .._random_state import derive_seed, make_rng
from .._types import ClusterResult
from ..config import ClusteringConfig
from .base import make_cluster_result


class KModesClusterer:
    """Hard K-Modes clusterer used as the fixed-K CSQD baseline."""

    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config

    def fit(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> ClusterResult:
        """Fit a fixed-K K-Modes model and return hard cluster labels."""

        seed = self.config.random_seed
        if seed is None:
            seed = derive_seed(rng)

        km = KModes(
            n_clusters=self.config.n_clusters,
            init=self.config.method_params.get("init", "Huang"),
            n_init=self.config.n_init,
            max_iter=self.config.method_params.get("max_iter", 100),
            verbose=self.config.method_params.get("verbose", 0),
            random_state=seed,
            n_jobs=self.config.n_jobs,
        )
        labels = km.fit_predict(spin_strings, sample_weight=weights)
        return make_cluster_result(
            bitstrings=spin_strings,
            probabilities=weights,
            labels=labels,
            n_clusters=self.config.n_clusters,
            model_name="kmodes",
            seed=seed,
            diagnostics={
                "cost": float(km.cost_) if hasattr(km, "cost_") else None,
                "n_iter": getattr(km, "n_iter_", None),
                "n_init": self.config.n_init,
            },
            fitted_model_metadata={
                "cluster_centroids": getattr(km, "cluster_centroids_", None),
            },
        )


def assign_clusters_kmodes(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 100,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Legacy helper for fixed-K K-Modes clustering."""

    config = ClusteringConfig(
        method="kmodes",
        n_clusters=k,
        n_init=n_init,
        random_seed=random_state,
    )
    result = KModesClusterer(config).fit(bitstrings, probabilities, make_rng(random_state))
    return result.as_legacy_tuple()
