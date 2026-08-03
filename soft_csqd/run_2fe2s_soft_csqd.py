#!/usr/bin/env python
"""Run fixed-K soft-CSQD 2Fe-2S experiments from an HPC batch job."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from soft_csqd.csqd.config import CSQDConfig, ClusteringConfig, DiagonalizationConfig
from soft_csqd.csqd.fermion import CSQDRunResult, run_csqd
from soft_csqd.csqd.io import save_result_fields, select_result_fields


DEFAULT_COUNTS_FILE = Path("2Fe2S/2023-12-22T19-07-32.839270_results.npy")
DEFAULT_HCORE_FILE = Path("2Fe2S/h1e_Fe2S2_MO.npy")
DEFAULT_ERI_FILE = Path("2Fe2S/h2e_Fe2S2_MO.npy")


def _json_ready(value: Any) -> Any:
    """Convert NumPy containers into JSON-ready Python objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _bitstring_key(raw_key: Any) -> str:
    """Normalize a count-dictionary key into a binary bitstring."""

    bitstring = str(raw_key).replace(" ", "")
    if not bitstring:
        raise ValueError("Counts contain an empty bitstring key.")
    if any(char not in {"0", "1"} for char in bitstring):
        raise ValueError(f"Counts key {raw_key!r} is not a binary bitstring.")
    return bitstring


def concat_count_dicts(counts_list: Any) -> dict[str, float]:
    """Flatten an array or list of count dictionaries into one count dictionary."""

    counts_dict: dict[str, float] = {}
    for counts in np.ravel(counts_list):
        if hasattr(counts, "item") and not isinstance(counts, dict):
            counts = counts.item()
        if not isinstance(counts, Mapping):
            raise TypeError(f"Counts data must contain dictionaries, got {type(counts)!r}.")
        for key, value in counts.items():
            bitstring = _bitstring_key(key)
            counts_dict[bitstring] = counts_dict.get(bitstring, 0.0) + float(value)
    return counts_dict


def counts_to_probability_arrays(
    counts: Mapping[Any, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert count weights into CSQD bitstring and probability arrays."""

    if not counts:
        raise ValueError("Counts dictionary is empty.")
    keys = [_bitstring_key(key) for key in counts]
    widths = {len(key) for key in keys}
    if len(widths) != 1:
        raise ValueError(f"All bitstrings must have the same length, got {sorted(widths)}.")

    weights = np.array([float(counts[key]) for key in counts], dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Total count/probability must be positive.")

    bitstrings = np.array([[char == "1" for char in key] for key in keys], dtype=bool)
    return bitstrings, weights / total


def _parse_method_params(raw_items: list[str]) -> dict[str, Any]:
    """Parse KEY=VALUE CLI options into a method parameter dictionary."""

    params: dict[str, Any] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"method-param must be KEY=VALUE, got {item!r}.")
        key, raw_value = item.split("=", 1)
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value: Any = lowered == "true"
        elif lowered in {"none", "null"}:
            value = None
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        params[key] = value
    return params


def _pickle_summary(result: CSQDRunResult, output_path: Path, wall_time: float) -> None:
    """Persist selected result fields in a pickle file."""

    payload = select_result_fields(
        result,
        fields=[
            "method",
            "config",
            "metadata",
            "best_energy",
            "energy_history",
            "basis_sizes",
            "basis_history",
            "cluster_weights",
            "cluster_labels",
            "cluster_spin_strings",
            "n_vec_history",
            "diagnostics",
            "best_ci_strings",
            "best_coefficients",
            "best_membership",
        ],
    )
    payload["wall_time"] = wall_time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_parser() -> argparse.ArgumentParser:
    """Create the fixed-K 2Fe-2S runner CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["kmodes", "bmm", "fuzzy_kmodes"],
        required=True,
    )
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--counts_file", type=Path, default=DEFAULT_COUNTS_FILE)
    parser.add_argument("--hcore_file", type=Path, default=DEFAULT_HCORE_FILE)
    parser.add_argument("--eri_file", type=Path, default=DEFAULT_ERI_FILE)
    parser.add_argument("--output_dir", type=Path, default=Path("results/2fe2s"))

    parser.add_argument("--n_elec_per_spin", type=int, default=15)
    parser.add_argument("--nuclear_repulsion_energy", type=float, default=0.0)
    parser.add_argument("--max_dim", type=int, default=1000)
    parser.add_argument("--n_samples_per_batch", type=int, default=1000)
    parser.add_argument("--n_batch", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--energy_tol", type=float, default=1e-8)
    parser.add_argument("--occupancies_tol", type=float, default=1e-5)
    parser.add_argument("--spin_sq", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_init", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--precluster_sector_distance_cutoff", type=int, default=None)
    parser.add_argument(
        "--soft_assignment",
        action="store_true",
        help="Keep BMM posterior responsibilities instead of collapsing them to hard labels.",
    )
    parser.add_argument("--fuzzifier", type=float, default=2.0)
    parser.add_argument("--method-param", action="append", default=[])
    parser.add_argument("--no_json", action="store_true")
    return parser


def main() -> None:
    """Run one fixed-K 2Fe-2S soft-CSQD job."""

    args = build_parser().parse_args()
    for path in (args.counts_file, args.hcore_file, args.eri_file):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path.resolve()}")

    load_start = time.time()
    counts_data = np.load(args.counts_file, allow_pickle=True)
    counts = concat_count_dicts(counts_data)
    bitarray = counts_to_probability_arrays(counts)
    hcore = np.load(args.hcore_file)
    eri = np.load(args.eri_file)
    n_orb = int(hcore.shape[0])
    if bitarray[0].shape[1] != 2 * n_orb:
        raise ValueError(
            f"Bitstring length ({bitarray[0].shape[1]}) must equal 2*n_orb ({2 * n_orb})."
        )

    method_params = _parse_method_params(args.method_param)
    if args.method == "fuzzy_kmodes":
        method_params.setdefault("fuzzifier", args.fuzzifier)
    soft_assignment = bool(args.soft_assignment or args.method == "fuzzy_kmodes")

    config = CSQDConfig(
        n_orb=n_orb,
        n_elec_per_spin=args.n_elec_per_spin,
        n_batch=args.n_batch,
        samples_per_batch=args.n_samples_per_batch,
        max_iterations=args.max_iterations,
        random_seed=args.seed,
        clustering=ClusteringConfig(
            method=args.method,
            n_clusters=args.n_clusters,
            n_init=args.n_init,
            random_seed=args.seed,
            n_jobs=args.n_jobs,
            soft_assignment=soft_assignment,
            precluster_sector_distance_cutoff=args.precluster_sector_distance_cutoff,
            method_params=method_params,
        ),
        diagonalization=DiagonalizationConfig(
            threshold=args.threshold,
            max_dim=args.max_dim,
            energy_tol=args.energy_tol,
            occupancies_tol=args.occupancies_tol,
            spin_sq=args.spin_sq,
        ),
    )

    print(
        "[LOAD]",
        json.dumps(
            _json_ready(
                {
                    "system": "2fe2s",
                    "counts_file": str(args.counts_file),
                    "hcore_file": str(args.hcore_file),
                    "eri_file": str(args.eri_file),
                    "unique_bitstrings": int(len(bitarray[0])),
                    "probability_sum": float(np.sum(bitarray[1])),
                    "bitstring_shape": list(bitarray[0].shape),
                    "hcore_shape": list(hcore.shape),
                    "eri_shape": list(eri.shape),
                    "nuclear_repulsion_energy": float(args.nuclear_repulsion_energy),
                    "seconds": round(time.time() - load_start, 3),
                }
            )
        ),
        flush=True,
    )
    print("[CONFIG]", json.dumps(config.to_dict(), default=str), flush=True)

    start = time.time()
    result = run_csqd(
        bitarray=bitarray,
        hcore=hcore,
        eri=eri,
        nuclear_repulsion_energy=args.nuclear_repulsion_energy,
        config=config,
    )
    wall_time = time.time() - start

    soft_tag = "_soft" if soft_assignment else ""
    precluster_tag = (
        ""
        if args.precluster_sector_distance_cutoff is None
        else f"_pcut{args.precluster_sector_distance_cutoff}"
    )
    stem = (
        f"csqd_ref_{args.method}{soft_tag}{precluster_tag}_k{args.n_clusters}_"
        f"2fe2s_{args.max_dim}_nb{args.n_batch}_ns{args.n_samples_per_batch}_it{args.max_iterations}"
    )
    pickle_path = args.output_dir / f"{stem}.pkl"
    json_path = args.output_dir / f"{stem}.json"
    _pickle_summary(result, pickle_path, wall_time)
    if not args.no_json:
        save_result_fields(
            result,
            json_path,
            fields=[
                "method",
                "config",
                "best_energy",
                "energy_history",
                "basis_sizes",
                "cluster_weights",
                "diagnostics",
                "best_membership",
            ],
            file_format="json",
        )

    summary = {
        "method": result.cluster_result.model_name,
        "best_energy": result.best_result.energy,
        "energy_history": result.energy_history,
        "basis_sizes": [len(basis) for basis in result.basis_history],
        "cluster_weights": result.cluster_result.cluster_weights.tolist(),
        "n_clusters": result.cluster_result.n_clusters,
        "soft_assignment": soft_assignment,
        "wall_time": wall_time,
        "pickle_path": str(pickle_path),
        "json_path": None if args.no_json else str(json_path),
    }
    print("[DONE]", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
