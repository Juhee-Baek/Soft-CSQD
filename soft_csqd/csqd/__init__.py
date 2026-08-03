"""Fixed-K soft cluster-adaptive sample-based quantum diagonalization."""

from typing import Any

from .config import (
    CSQDConfig,
    ClusteringConfig,
    DiagonalizationConfig,
    LoggingConfig,
    RecoveryConfig,
)

__all__ = [
    "CSQDConfig",
    "CSQDRunResult",
    "ClusteringConfig",
    "DiagonalizationConfig",
    "LoggingConfig",
    "RecoveryConfig",
    "diagonalize_fermionic_hamiltonian_with_clustering",
    "run_csqd",
    "save_result_fields",
    "select_result_fields",
]


def __getattr__(name: str) -> Any:
    if name in {
        "CSQDRunResult",
        "diagonalize_fermionic_hamiltonian_with_clustering",
        "run_csqd",
    }:
        from . import fermion

        return getattr(fermion, name)
    if name in {"save_result_fields", "select_result_fields"}:
        from . import io

        return getattr(io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
