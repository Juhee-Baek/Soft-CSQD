"""Fixed-K Bernoulli mixture model clusterer."""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Tuple, cast

import numpy as np
from joblib import Parallel, delayed
from stepmix.stepmix import StepMix

from .._random_state import derive_seed, make_rng
from .._types import ClusterResult
from ..config import ClusteringConfig
from .base import hard_membership, make_cluster_result


@dataclass(frozen=True)
class BMMFitState:
    """Internal fixed-K BMM fit state."""

    labels: np.ndarray
    reference_vectors: np.ndarray
    representatives: np.ndarray
    metadata: dict


class BMMClusterer:
    """Bernoulli mixture model clusterer with optional soft responsibilities."""

    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config
        self._best_model: StepMix | None = None
        self._fitted_n_clusters = config.n_clusters

    @staticmethod
    def _fit_single_stepmix(
        spin_strings: np.ndarray,
        weights: np.ndarray,
        n_clusters: int,
        worker_seed: int,
        measurement: str,
        verbose: int,
    ) -> Tuple[StepMix, float]:
        """Fit one StepMix initialization and return its lower bound."""

        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                model = StepMix(
                    n_components=n_clusters,
                    measurement=measurement,
                    verbose=verbose,
                    random_state=worker_seed,
                    n_init=1,
                )
                model.fit(spin_strings, sample_weight=weights)
        return model, float(model.lower_bound_)

    def _fit_best_model(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        n_clusters: int,
        seed: int,
    ) -> Tuple[StepMix, float]:
        """Fit multiple BMM initializations and keep the best lower bound."""

        n_init = max(1, int(self.config.n_init))
        seeds = [int(seed) + offset for offset in range(n_init)]
        measurement = self.config.method_params.get("measurement", "binary")
        verbose = int(self.config.method_params.get("verbose", 0))
        parallel_results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._fit_single_stepmix)(
                spin_strings,
                weights,
                n_clusters,
                worker_seed,
                measurement,
                verbose,
            )
            for worker_seed in seeds
        )
        results = cast(list[Tuple[StepMix, float]], parallel_results)
        return max(results, key=lambda item: item[1])

    @staticmethod
    def _reference_vectors_from_labels(
        spin_strings: np.ndarray,
        weights: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute hard-label weighted reference vectors for each component."""

        n_features = spin_strings.shape[1]
        refs = np.zeros((n_clusters, n_features), dtype=float)
        representatives = np.zeros((n_clusters, n_features), dtype=bool)

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                fallback = rng.choice(len(spin_strings), p=weights)
                refs[cluster_id] = spin_strings[fallback].astype(float)
                representatives[cluster_id] = spin_strings[fallback]
                continue

            local_weights = weights[mask]
            refs[cluster_id] = (
                local_weights[:, None] * spin_strings[mask].astype(float)
            ).sum(axis=0) / np.sum(local_weights)
            representatives[cluster_id] = refs[cluster_id] >= 0.5

        return refs, representatives

    @staticmethod
    def _model_reference_vectors(
        model: StepMix,
        expected_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Return Bernoulli component probabilities when StepMix exposes them."""

        params = model.get_parameters()
        measurement = params.get("measurement", {})
        pis = measurement.get("pis") if isinstance(measurement, dict) else None
        if pis is None:
            return None

        refs = np.asarray(pis, dtype=float)
        if refs.shape != expected_shape:
            return None
        refs = np.nan_to_num(refs, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(refs, 0.0, 1.0)

    def _state_from_model(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        model: StepMix,
        lower_bound: float,
        rng: np.random.Generator,
    ) -> BMMFitState:
        """Convert a fitted StepMix model into a CSQD clustering state."""

        labels = np.asarray(model.predict(spin_strings), dtype=int)
        n_clusters = int(model.n_components)
        refs, representatives = self._reference_vectors_from_labels(
            spin_strings,
            weights,
            labels,
            n_clusters,
            rng,
        )
        model_refs = self._model_reference_vectors(
            model,
            expected_shape=(n_clusters, spin_strings.shape[1]),
        )
        if model_refs is not None:
            refs = model_refs
            representatives = refs >= 0.5

        return BMMFitState(
            labels=labels,
            reference_vectors=refs,
            representatives=representatives,
            metadata={
                "model": model,
                "lower_bound": lower_bound,
                "component_weights": getattr(model, "weights_", None),
            },
        )

    def fit(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> ClusterResult:
        """Fit a fixed-K BMM and return hard or soft cluster memberships."""

        seed = self.config.random_seed
        if seed is None:
            seed = derive_seed(rng)

        best_model, best_lower_bound = self._fit_best_model(
            spin_strings,
            weights,
            self.config.n_clusters,
            seed,
        )
        state = self._state_from_model(
            spin_strings,
            weights,
            best_model,
            best_lower_bound,
            rng,
        )
        self._best_model = best_model
        self._fitted_n_clusters = self.config.n_clusters

        responsibilities = None
        if self.config.soft_assignment and hasattr(best_model, "predict_proba"):
            responsibilities = np.asarray(best_model.predict_proba(spin_strings), dtype=float)

        return make_cluster_result(
            bitstrings=spin_strings,
            probabilities=weights,
            labels=state.labels,
            n_clusters=self.config.n_clusters,
            model_name="bmm",
            seed=seed,
            diagnostics={
                "best_lower_bound": best_lower_bound,
                "n_init": self.config.n_init,
                "soft_assignment": bool(self.config.soft_assignment),
            },
            responsibilities=responsibilities,
            fitted_model_metadata={
                "component_weights": state.metadata.get("component_weights"),
                "reference_vectors": state.reference_vectors,
                "representatives": state.representatives,
            },
        )

    def predict_membership(self, spin_strings: np.ndarray) -> np.ndarray:
        """Infer membership rows for new spin strings after fitting."""

        if self._best_model is None:
            raise RuntimeError("BMMClusterer must be fit before predict_membership.")
        bitstrings = np.asarray(spin_strings, dtype=bool)
        if self.config.soft_assignment and hasattr(self._best_model, "predict_proba"):
            return np.asarray(self._best_model.predict_proba(bitstrings), dtype=float)
        labels = self._best_model.predict(bitstrings)
        return hard_membership(labels, self._fitted_n_clusters).astype(float)


def assign_clusters_bmm(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 100,
    soft_assignment: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Legacy helper for fixed-K BMM clustering."""

    config = ClusteringConfig(
        method="bmm",
        n_clusters=k,
        n_init=n_init,
        random_seed=random_state,
        soft_assignment=soft_assignment,
    )
    result = BMMClusterer(config).fit(bitstrings, probabilities, make_rng(random_state))
    return result.as_legacy_tuple()
