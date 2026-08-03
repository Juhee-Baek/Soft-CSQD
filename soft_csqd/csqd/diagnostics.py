"""Structured diagnostics collected during CSQD runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiagnosticsStore:
    """Simple append-only diagnostics container."""

    metadata: dict[str, Any] = field(default_factory=dict)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    batches: list[dict[str, Any]] = field(default_factory=list)

    def add_cluster(self, **payload: Any) -> None:
        self.clusters.append(dict(payload))

    def add_iteration(self, **payload: Any) -> None:
        self.iterations.append(dict(payload))

    def add_batch(self, **payload: Any) -> None:
        self.batches.append(dict(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "clusters": self.clusters,
            "iterations": self.iterations,
            "batches": self.batches,
        }
