"""Internal typed containers for the CSQD workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np


class SCIResult(NamedTuple):
    """Tuple-compatible selected-CI result."""

    ci_strings: tuple[np.ndarray, np.ndarray]
    coefficients: np.ndarray
    energy: float
    membership: np.ndarray


@dataclass(frozen=True)
class ClusterResult:
    """Algorithm-agnostic clustering output consumed by the CSQD core."""

    labels: np.ndarray
    cluster_weights: np.ndarray
    n_clusters: int
    diagnostics: dict[str, Any]
    model_name: str
    seed: int | None
    clusters: tuple[np.ndarray, ...]
    cluster_probabilities: tuple[np.ndarray, ...]
    responsibilities: np.ndarray | None = None
    membership: np.ndarray | None = None
    fitted_model_metadata: dict[str, Any] = field(default_factory=dict)

    def as_legacy_tuple(self) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        return (
            list(self.clusters),
            list(self.cluster_probabilities),
            self.cluster_weights,
        )


@dataclass(frozen=True)
class BasisBatch:
    """A spin-string basis candidate plus aligned weights and memberships."""

    bitstrings: np.ndarray
    probabilities: np.ndarray
    membership: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_legacy_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.bitstrings, self.probabilities, self.membership


@dataclass(frozen=True)
class FinalizedBatch:
    """A PySCF-ready basis with membership rows aligned to CI integers."""

    ci_integers: np.ndarray
    probabilities: np.ndarray
    membership: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
