"""Stable clustering interface for CSQD."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from ..counts import normalize_probabilities
from .._types import ClusterResult


class Clusterer(Protocol):
    """Protocol implemented by all clustering plug-ins."""

    def fit(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> ClusterResult: ...


def compute_cluster_weights(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    weights = np.zeros(n_clusters, dtype=float)
    for cluster_id in range(n_clusters):
        weights[cluster_id] = float(np.sum(probabilities[labels == cluster_id]))
    return normalize_probabilities(weights, size=n_clusters)


def compute_soft_cluster_weights(
    membership: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    weighted_membership = np.asarray(probabilities, dtype=float)[:, None] * membership
    return normalize_probabilities(
        np.sum(weighted_membership, axis=0),
        size=membership.shape[1],
    )


def assign_clusters(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    clusters: list[np.ndarray] = []
    cluster_probabilities: list[np.ndarray] = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        clusters.append(bitstrings[mask])
        cluster_probabilities.append(probabilities[mask])
    return tuple(clusters), tuple(cluster_probabilities)


def assign_soft_clusters(
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    membership: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    clusters: list[np.ndarray] = []
    cluster_probabilities: list[np.ndarray] = []
    for cluster_id in range(membership.shape[1]):
        cluster_membership = membership[:, cluster_id]
        mask = cluster_membership > 0.0
        clusters.append(bitstrings[mask])
        cluster_probabilities.append(probabilities[mask] * cluster_membership[mask])
    return tuple(clusters), tuple(cluster_probabilities)


def hard_membership(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    membership = np.zeros((len(labels), n_clusters), dtype=bool)
    if len(labels) > 0:
        membership[np.arange(len(labels)), labels.astype(int)] = True
    return membership


def normalize_membership(
    responsibilities: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    membership = np.asarray(responsibilities, dtype=float)
    if membership.shape != (len(labels), n_clusters):
        raise ValueError("responsibilities must have shape (n_samples, n_clusters).")

    membership = np.nan_to_num(membership, nan=0.0, posinf=0.0, neginf=0.0)
    membership = np.maximum(membership, 0.0)
    row_sums = np.sum(membership, axis=1, keepdims=True)

    empty_rows = np.ravel(row_sums <= 0.0)
    if np.any(empty_rows):
        membership[empty_rows] = hard_membership(labels[empty_rows], n_clusters).astype(float)
        row_sums = np.sum(membership, axis=1, keepdims=True)

    return np.divide(
        membership,
        row_sums,
        out=np.zeros_like(membership),
        where=row_sums > 0.0,
    )


def make_cluster_result(
    *,
    bitstrings: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    model_name: str,
    seed: int | None,
    diagnostics: dict,
    responsibilities: np.ndarray | None = None,
    fitted_model_metadata: dict | None = None,
) -> ClusterResult:
    labels = np.asarray(labels, dtype=int)
    if responsibilities is None:
        membership = hard_membership(labels, n_clusters)
        clusters, cluster_probabilities = assign_clusters(
            bitstrings,
            probabilities,
            labels,
            n_clusters,
        )
        weights = compute_cluster_weights(labels, probabilities, n_clusters)
    else:
        membership = normalize_membership(responsibilities, labels, n_clusters)
        clusters, cluster_probabilities = assign_soft_clusters(
            bitstrings,
            probabilities,
            membership,
        )
        weights = compute_soft_cluster_weights(membership, probabilities)

    return ClusterResult(
        labels=labels,
        cluster_weights=weights,
        n_clusters=n_clusters,
        diagnostics=diagnostics,
        model_name=model_name,
        seed=seed,
        clusters=clusters,
        cluster_probabilities=cluster_probabilities,
        responsibilities=responsibilities,
        membership=membership,
        fitted_model_metadata=fitted_model_metadata or {},
    )


class LegacyClusteringAdapter:
    """Adapter for old callables returning `(clusters, cluster_probs, weights)`."""

    def __init__(
        self,
        clustering_func: Callable[[np.ndarray, np.ndarray, int], tuple],
        n_clusters: int,
    ) -> None:
        self.clustering_func = clustering_func
        self.n_clusters = n_clusters

    def fit(
        self,
        spin_strings: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> ClusterResult:
        clusters, cluster_probabilities, cluster_weights = self.clustering_func(
            spin_strings,
            weights,
            self.n_clusters,
        )
        label_lookup: dict[bytes, int] = {}
        for cluster_id, cluster in enumerate(clusters):
            for block in cluster:
                label_lookup[block.tobytes()] = cluster_id

        labels = np.array(
            [label_lookup[block.tobytes()] for block in spin_strings],
            dtype=int,
        )
        normalized_weights = normalize_probabilities(
            np.asarray(cluster_weights, dtype=float),
            size=self.n_clusters,
        )
        return ClusterResult(
            labels=labels,
            cluster_weights=normalized_weights,
            n_clusters=self.n_clusters,
            diagnostics={"adapter": "legacy_callable"},
            model_name=getattr(self.clustering_func, "__name__", "legacy_callable"),
            seed=None,
            clusters=tuple(clusters),
            cluster_probabilities=tuple(cluster_probabilities),
            membership=hard_membership(labels, self.n_clusters),
        )


def fit_clusters(
    clusterer: Clusterer,
    spin_strings: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> ClusterResult:
    result = clusterer.fit(spin_strings, weights, rng)
    if len(result.labels) != len(spin_strings):
        raise ValueError("ClusterResult.labels length must match spin_strings length.")
    if result.cluster_weights.shape != (result.n_clusters,):
        raise ValueError("ClusterResult.cluster_weights has the wrong shape.")
    return result
