"""Fixed-K fuzzy K-Modes clusterer for binary spin strings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from joblib import Parallel, delayed

from .._random_state import derive_seed, make_rng
from .._types import ClusterResult
from ..config import ClusteringConfig
from ..counts import normalize_probabilities
from .base import make_cluster_result


@dataclass(frozen=True)
class FuzzyKModesState:
    labels: np.ndarray
    membership: np.ndarray
    representatives: np.ndarray
    reference_vectors: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    objective_history: list[float]


class FuzzyKModesClusterer:
    """Fuzzy K-Modes clusterer using weighted Hamming dissimilarity."""

    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config
        self._fitted_representatives: np.ndarray | None = None
        self._fitted_fuzzifier: float | None = None

    @staticmethod
    def _hamming_distance(spin_strings: np.ndarray, representatives: np.ndarray) -> np.ndarray:
        return np.sum(
            spin_strings[:, None, :] != representatives[None, :, :],
            axis=2,
        ).astype(float)

    @staticmethod
    def _sampling_probabilities(weights: np.ndarray) -> np.ndarray | None:
        probs = np.asarray(weights, dtype=float)
        total = float(np.sum(probs))
        if total <= 0.0:
            return None
        return probs / total

    def _initial_representatives(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        n_clusters: int,
        init_representatives: np.ndarray | None = None,
    ) -> np.ndarray:
        if init_representatives is not None:
            representatives = np.asarray(init_representatives, dtype=bool)
            expected_shape = (n_clusters, spin_strings.shape[1])
            if representatives.shape != expected_shape:
                raise ValueError(f"fuzzy-kmodes init must have shape {expected_shape}.")
            return representatives

        init = self.config.method_params.get("init", "random")
        if not isinstance(init, str):
            representatives = np.asarray(init, dtype=bool)
            expected_shape = (n_clusters, spin_strings.shape[1])
            if representatives.shape != expected_shape:
                raise ValueError(f"fuzzy-kmodes init must have shape {expected_shape}.")
            return representatives

        replace = n_clusters > len(spin_strings)
        indices = rng.choice(
            len(spin_strings),
            size=n_clusters,
            replace=replace,
            p=self._sampling_probabilities(weights),
        )
        return np.asarray(spin_strings[indices], dtype=bool).copy()

    @staticmethod
    def _compute_membership(distances: np.ndarray, fuzzifier: float) -> np.ndarray:
        n_samples, n_clusters = distances.shape
        membership = np.zeros((n_samples, n_clusters), dtype=float)
        zero_distance = distances <= 0.0
        zero_rows = np.any(zero_distance, axis=1)

        if np.any(zero_rows):
            zero_counts = np.sum(zero_distance[zero_rows], axis=1, keepdims=True)
            membership[zero_rows] = zero_distance[zero_rows] / zero_counts

        nonzero_rows = ~zero_rows
        if np.any(nonzero_rows):
            power = 1.0 / (fuzzifier - 1.0)
            inverse_distances = distances[nonzero_rows] ** (-power)
            row_sums = np.sum(inverse_distances, axis=1, keepdims=True)
            membership[nonzero_rows] = inverse_distances / row_sums

        return membership

    def _update_representatives(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        membership: np.ndarray,
        fuzzifier: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_clusters = membership.shape[1]
        n_features = spin_strings.shape[1]
        representatives = np.zeros((n_clusters, n_features), dtype=bool)
        reference_vectors = np.zeros((n_clusters, n_features), dtype=float)
        bit_values = np.asarray(spin_strings, dtype=float)
        fuzzy_weights = weights[:, None] * (membership**fuzzifier)

        for cluster_id in range(n_clusters):
            cluster_weights = fuzzy_weights[:, cluster_id]
            total = float(np.sum(cluster_weights))
            if total <= 0.0:
                fallback = rng.choice(len(spin_strings), p=self._sampling_probabilities(weights))
                representatives[cluster_id] = spin_strings[fallback]
                reference_vectors[cluster_id] = bit_values[fallback]
                continue

            refs = np.sum(cluster_weights[:, None] * bit_values, axis=0) / total
            reference_vectors[cluster_id] = refs
            representatives[cluster_id] = refs >= 0.5

        return representatives, reference_vectors

    @staticmethod
    def _objective(
        distances: np.ndarray,
        membership: np.ndarray,
        weights: np.ndarray,
        fuzzifier: float,
    ) -> float:
        return float(np.sum(weights[:, None] * (membership**fuzzifier) * distances))

    def _fit_single(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        worker_seed: int,
        n_clusters: int,
        *,
        init_representatives: np.ndarray | None = None,
        max_iter: int | None = None,
    ) -> FuzzyKModesState:
        rng = make_rng(worker_seed)
        fuzzifier = float(self.config.method_params.get("fuzzifier", 2.0))
        if fuzzifier <= 1.0:
            raise ValueError("fuzzy-kmodes fuzzifier must be greater than 1.0.")

        max_iter = max(
            1,
            int(self.config.method_params.get("max_iter", 100) if max_iter is None else max_iter),
        )
        tol = float(self.config.method_params.get("tol", 1e-6))
        representatives = self._initial_representatives(
            spin_strings,
            weights,
            rng,
            n_clusters,
            init_representatives=init_representatives,
        )
        objective_history: list[float] = []
        previous_objective: float | None = None
        converged = False
        membership = np.zeros((len(spin_strings), n_clusters), dtype=float)
        reference_vectors = representatives.astype(float)

        for n_iter in range(1, max_iter + 1):
            distances = self._hamming_distance(spin_strings, representatives)
            membership = self._compute_membership(distances, fuzzifier)
            updated_representatives, reference_vectors = self._update_representatives(
                spin_strings,
                weights,
                membership,
                fuzzifier,
                rng,
            )
            updated_distances = self._hamming_distance(spin_strings, updated_representatives)
            updated_membership = self._compute_membership(updated_distances, fuzzifier)
            objective = self._objective(
                updated_distances,
                updated_membership,
                weights,
                fuzzifier,
            )
            objective_history.append(objective)

            representatives_unchanged = np.array_equal(updated_representatives, representatives)
            objective_stable = previous_objective is not None and abs(
                previous_objective - objective
            ) <= tol * max(1.0, abs(previous_objective))
            representatives = updated_representatives
            membership = updated_membership
            if representatives_unchanged or objective_stable:
                converged = True
                break
            previous_objective = objective
        else:
            n_iter = max_iter

        final_representatives, reference_vectors = self._update_representatives(
            spin_strings,
            weights,
            membership,
            fuzzifier,
            rng,
        )
        if not np.array_equal(final_representatives, representatives):
            representatives = final_representatives
            distances = self._hamming_distance(spin_strings, representatives)
            membership = self._compute_membership(distances, fuzzifier)
            _, reference_vectors = self._update_representatives(
                spin_strings,
                weights,
                membership,
                fuzzifier,
                rng,
            )
        distances = self._hamming_distance(spin_strings, representatives)
        objective = self._objective(distances, membership, weights, fuzzifier)
        labels = np.argmax(membership, axis=1).astype(int)

        return FuzzyKModesState(
            labels=labels,
            membership=membership,
            representatives=representatives,
            reference_vectors=reference_vectors,
            objective=objective,
            n_iter=n_iter,
            converged=converged,
            objective_history=objective_history,
        )

    def _fit_best_state(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        *,
        n_clusters: int,
        seed: int,
        init_representatives: np.ndarray | None = None,
        max_iter: int | None = None,
        seed_offset: int = 0,
    ) -> FuzzyKModesState:
        n_init = max(1, int(self.config.n_init))
        seeds = [int(seed) + int(seed_offset) * n_init + offset for offset in range(n_init)]
        parallel_results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(self._fit_single)(
                spin_strings,
                weights,
                worker_seed,
                n_clusters,
                init_representatives=init_representatives,
                max_iter=max_iter,
            )
            for worker_seed in seeds
        )
        states = cast(list[FuzzyKModesState], list(parallel_results))
        return min(states, key=lambda state: state.objective)

    def _state_from_membership(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        membership: np.ndarray,
        rng: np.random.Generator,
    ) -> FuzzyKModesState:
        fuzzifier = float(self.config.method_params.get("fuzzifier", 2.0))
        membership = np.asarray(membership, dtype=float)
        representatives, reference_vectors = self._update_representatives(
            spin_strings,
            weights,
            membership,
            fuzzifier,
            rng,
        )
        distances = self._hamming_distance(spin_strings, representatives)
        objective = self._objective(distances, membership, weights, fuzzifier)
        labels = np.argmax(membership, axis=1).astype(int)
        return FuzzyKModesState(
            labels=labels,
            membership=membership,
            representatives=representatives,
            reference_vectors=reference_vectors,
            objective=objective,
            n_iter=0,
            converged=True,
            objective_history=[],
        )

    @staticmethod
    def _cluster_weights(membership: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return normalize_probabilities(
            np.sum(weights[:, None] * membership, axis=0),
            size=membership.shape[1],
        )

    def _store_fitted_state(
        self, state: FuzzyKModesState, fuzzifier: float
    ) -> None:
        self._fitted_representatives = np.asarray(state.representatives, dtype=bool).copy()
        self._fitted_fuzzifier = float(fuzzifier)

    def predict_membership(self, spin_strings: np.ndarray) -> np.ndarray:
        """Infer fuzzy posterior rows for new spin strings using fitted representatives."""

        if self._fitted_representatives is None or self._fitted_fuzzifier is None:
            raise RuntimeError("FuzzyKModesClusterer must be fit before predict_membership.")
        bitstrings = np.asarray(spin_strings, dtype=bool)
        distances = self._hamming_distance(bitstrings, self._fitted_representatives)
        return self._compute_membership(distances, self._fitted_fuzzifier)

    def fit(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> ClusterResult:
        seed = self.config.random_seed
        if seed is None:
            seed = derive_seed(rng)

        n_init = max(1, int(self.config.n_init))
        fuzzifier = float(self.config.method_params.get("fuzzifier", 2.0))

        best_state = self._fit_best_state(
            spin_strings,
            weights,
            rng,
            n_clusters=self.config.n_clusters,
            seed=seed,
        )
        self._store_fitted_state(best_state, fuzzifier)

        return make_cluster_result(
            bitstrings=spin_strings,
            probabilities=weights,
            labels=best_state.labels,
            n_clusters=self.config.n_clusters,
            model_name="fuzzy_kmodes",
            seed=seed,
            diagnostics={
                "objective": best_state.objective,
                "n_iter": best_state.n_iter,
                "n_init": n_init,
                "converged": best_state.converged,
                "fuzzifier": fuzzifier,
            },
            responsibilities=best_state.membership,
            fitted_model_metadata={
                "cluster_centroids": best_state.representatives,
                "reference_vectors": best_state.reference_vectors,
                "objective_history": best_state.objective_history,
            },
        )


def assign_clusters_fuzzy_kmodes(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    k: int,
    random_state: int = 42,
    n_init: int = 100,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    config = ClusteringConfig(
        method="fuzzy_kmodes",
        n_clusters=k,
        n_init=n_init,
        random_seed=random_state,
    )
    result = FuzzyKModesClusterer(config).fit(bitstrings, probabilities, make_rng(random_state))
    return result.as_legacy_tuple()
