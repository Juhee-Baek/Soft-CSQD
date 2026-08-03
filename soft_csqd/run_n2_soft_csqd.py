#!/usr/bin/env python
"""Run fixed-K soft-CSQD N2 experiments from an HPC batch job."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from soft_csqd.csqd.config import CSQDConfig, ClusteringConfig, DiagonalizationConfig
from soft_csqd.csqd.fermion import CSQDRunResult, run_csqd
from soft_csqd.csqd.io import save_result_fields, select_result_fields
from soft_csqd.examples.chemistry import define_n2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = PROJECT_ROOT / "n2" / "data" / "n2_sample_fez_dd_mt"


def _parse_method_params(raw_items: list[str]) -> dict[str, Any]:
    """Parse KEY=VALUE CLI options into a method parameter dictionary."""

    params: dict[str, Any] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"method-param must be KEY=VALUE, got {item!r}.")
        key, raw_value = item.split("=", 1)
        value: Any
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
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


def _load_distance_sample(sample_path: Path, distance: float):
    """Load a sampled N2 BitArray for the requested bond distance."""

    with sample_path.open("rb") as handle:
        n2_samples = pickle.load(handle)

    rounded_distance = round(float(distance), 1)
    for key in (np.float64(rounded_distance), rounded_distance):
        if key in n2_samples:
            return rounded_distance, n2_samples[key]
    available = sorted(float(key) for key in n2_samples.keys())
    raise KeyError(
        f"Distance {rounded_distance} not found in {sample_path}. Available distances: {available}"
    )


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
    """Create the fixed-K N2 runner CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["kmodes", "bmm", "fuzzy_kmodes"],
        required=True,
    )
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--sample_path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output_dir", type=Path, default=Path("results/n2"))

    parser.add_argument("--max_dim", type=int, default=2000)
    parser.add_argument("--n_samples_per_batch", type=int, default=2000)
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
    parser.add_argument(
        "--method-param",
        action="append",
        default=[],
        help="Extra clustering method parameter as KEY=VALUE. Can be repeated.",
    )
    parser.add_argument("--no_json", action="store_true", help="Skip JSON summary output.")
    return parser


def main() -> None:
    """Run one fixed-K N2 soft-CSQD job."""

    args = build_parser().parse_args()

    distance, bitarray = _load_distance_sample(args.sample_path, args.distance)
    print(
        f"[LOAD] sample={args.sample_path} distance={distance} "
        f"shots={getattr(bitarray, 'num_shots', 'unknown')}",
        flush=True,
    )

    hamiltonian_start = time.time()
    hcore, eri, nuclear_repulsion_energy = define_n2(distance)
    print(
        f"[HAMILTONIAN] hcore={hcore.shape} eri={eri.shape} "
        f"seconds={time.time() - hamiltonian_start:.3f}",
        flush=True,
    )

    method_params = _parse_method_params(args.method_param)
    if args.method == "fuzzy_kmodes":
        method_params.setdefault("fuzzifier", args.fuzzifier)

    soft_assignment = bool(args.soft_assignment or args.method == "fuzzy_kmodes")
    config = CSQDConfig(
        n_orb=int(hcore.shape[0]),
        n_elec_per_spin=5,
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
    print("[CONFIG]", json.dumps(config.to_dict(), default=str), flush=True)

    start = time.time()
    result = run_csqd(
        bitarray=bitarray,
        hcore=hcore,
        eri=eri,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        config=config,
    )
    wall_time = time.time() - start

    dist_str = f"{int(round(distance * 10)):02}"
    soft_tag = "_soft" if soft_assignment else ""
    precluster_tag = (
        ""
        if args.precluster_sector_distance_cutoff is None
        else f"_pcut{args.precluster_sector_distance_cutoff}"
    )
    stem = (
        f"csqd_ref_{args.method}{soft_tag}{precluster_tag}_k{args.n_clusters}_"
        f"n2_d{dist_str}_maxdim{args.max_dim}_spb{args.n_samples_per_batch}"
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
        "distance": distance,
        "best_energy": result.best_result.energy,
        "energy_history": result.energy_history,
        "basis_sizes": [len(basis) for basis in result.basis_history],
        "cluster_weights": result.cluster_result.cluster_weights.tolist(),
        "n_clusters": result.cluster_result.n_clusters,
        "soft_assignment": soft_assignment,
        "precluster_sector_distance_cutoff": args.precluster_sector_distance_cutoff,
        "wall_time": wall_time,
        "pickle_path": str(pickle_path),
        "json_path": None if args.no_json else str(json_path),
    }
    print("[DONE]", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
