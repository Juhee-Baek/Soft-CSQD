"""Clustering plug-in factory."""

from __future__ import annotations

from typing import Any

from ..config import ClusteringConfig


def clusterer_from_config(config: ClusteringConfig) -> Any:
    method = config.method.lower()
    if method in {"kmodes", "k-modes", "k_modes"}:
        from .kmodes import KModesClusterer

        return KModesClusterer(config)
    if method in {"bmm", "bernoulli_mixture", "bernoulli-mixture"}:
        from .bmm import BMMClusterer

        return BMMClusterer(config)
    if method in {"fuzzy_kmodes", "fuzzy-kmodes", "fuzzykmodes", "fuzzy_k_modes", "fkmodes"}:
        from .fuzzy_kmodes import FuzzyKModesClusterer

        return FuzzyKModesClusterer(config)
    raise ValueError(f"Unknown clustering method: {config.method}")


__all__ = [
    "Clusterer",
    "LegacyClusteringAdapter",
    "clusterer_from_config",
    "fit_clusters",
]


def __getattr__(name: str) -> Any:
    if name in {"Clusterer", "LegacyClusteringAdapter", "fit_clusters"}:
        from . import base

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
