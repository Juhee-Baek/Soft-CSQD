"""Configuration objects for fixed-K soft-CSQD runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration shared by fixed-K clustering implementations.

    The paper-ready code intentionally uses a fixed number of clusters.  It does
    not include auto-K, active-K, pruning, or merge-based cluster-number search.
    """

    method: str = "kmodes"
    n_clusters: int = 2
    n_init: int = 100
    random_seed: int | None = None
    n_jobs: int = -1
    soft_assignment: bool = False
    precluster_sector_distance_cutoff: int | None = None
    method_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryConfig:
    """Configuration for particle-number correction and synthetic sampling."""

    flip_epsilon: float = 0.01
    generation_attempt_factor: int = 20


@dataclass(frozen=True)
class DiagonalizationConfig:
    """Selected-CI diagonalization and convergence settings."""

    threshold: float
    max_dim: int | None = None
    energy_tol: float = 1e-8
    occupancies_tol: float = 1e-5
    spin_sq: float | None = 0.0


@dataclass(frozen=True)
class LoggingConfig:
    """Diagnostics and metadata settings."""

    enabled: bool = True
    schema_version: str = "soft-csqd-v1"


@dataclass(frozen=True)
class CSQDConfig:
    """Top-level fixed-K soft-CSQD workflow configuration."""

    n_orb: int
    n_elec_per_spin: int
    n_batch: int
    samples_per_batch: int
    max_iterations: int
    random_seed: int = 42
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    diagonalization: DiagonalizationConfig = field(
        default_factory=lambda: DiagonalizationConfig(threshold=0.0)
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return asdict(self)

    @classmethod
    def from_legacy_args(
        cls,
        *,
        n_orb: int,
        n_elec: int,
        n_clusters: int,
        n_batch: int,
        threshold: float,
        samples_per_batch: int,
        max_iterations: int,
        max_dim: int | None,
        energy_tol: float,
        occupancies_tol: float,
        random_seed: int = 42,
        clustering_method: str = "kmodes",
        clustering_seed: int | None = 42,
        n_init: int = 100,
        soft_assignment: bool = False,
        method_params: dict[str, Any] | None = None,
    ) -> "CSQDConfig":
        """Build a fixed-K config from the legacy CSQD argument list."""

        return cls(
            n_orb=n_orb,
            n_elec_per_spin=n_elec,
            n_batch=n_batch,
            samples_per_batch=samples_per_batch,
            max_iterations=max_iterations,
            random_seed=random_seed,
            clustering=ClusteringConfig(
                method=clustering_method,
                n_clusters=n_clusters,
                n_init=n_init,
                random_seed=clustering_seed,
                soft_assignment=soft_assignment,
                method_params={} if method_params is None else dict(method_params),
            ),
            diagonalization=DiagonalizationConfig(
                threshold=threshold,
                max_dim=max_dim,
                energy_tol=energy_tol,
                occupancies_tol=occupancies_tol,
            ),
        )
