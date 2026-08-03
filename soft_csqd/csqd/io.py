"""Result serialization helpers for selected CSQD outputs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .fermion import CSQDRunResult

DEFAULT_RESULT_FIELDS = (
    "method",
    "best_energy",
    "energy_history",
    "basis_sizes",
    "cluster_weights",
    "diagnostics",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def result_field_map(result: CSQDRunResult) -> dict[str, Any]:
    """Return supported serializable fields for a CSQD run result."""

    metadata = result.diagnostics.get("metadata", {})
    return {
        "method": result.cluster_result.model_name,
        "config": metadata.get("config"),
        "metadata": metadata,
        "best_energy": result.best_result.energy,
        "energy_history": result.energy_history,
        "basis_sizes": [len(basis) for basis in result.basis_history],
        "basis_history": result.basis_history,
        "cluster_weights": result.cluster_result.cluster_weights,
        "cluster_labels": result.cluster_result.labels,
        "cluster_spin_strings": result.cluster_spin_strings,
        "n_vec_history": result.n_vec_history,
        "diagnostics": result.diagnostics,
        "iteration_diagnostics": result.diagnostics.get("iterations", []),
        "batch_diagnostics": result.diagnostics.get("batches", []),
        "cluster_diagnostics": result.diagnostics.get("clusters", []),
        "best_ci_strings": result.best_result.ci_strings,
        "best_coefficients": result.best_result.coefficients,
        "best_membership": result.best_result.membership,
    }


def select_result_fields(
    result: CSQDRunResult,
    fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a payload containing only the requested result fields."""

    available = result_field_map(result)
    selected_fields = tuple(DEFAULT_RESULT_FIELDS if fields is None else fields)
    unknown = sorted(set(selected_fields) - set(available))
    if unknown:
        raise KeyError(f"Unknown CSQD result field(s): {unknown}")
    return {field: available[field] for field in selected_fields}


def save_result_fields(
    result: CSQDRunResult,
    path: str | Path,
    fields: Iterable[str] | None = None,
    *,
    file_format: str | None = None,
) -> Path:
    """Save selected CSQD result fields as JSON or pickle.

    JSON is preferred for summaries and diagnostics because it is readable and
    language-neutral. Pickle is useful when preserving NumPy arrays exactly.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = select_result_fields(result, fields)

    resolved_format = (file_format or output_path.suffix.lstrip(".") or "json").lower()
    if resolved_format == "json":
        output_path.write_text(json.dumps(_json_ready(payload), indent=2))
    elif resolved_format in {"pkl", "pickle"}:
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError("file_format must be 'json' or 'pickle'.")

    return output_path
