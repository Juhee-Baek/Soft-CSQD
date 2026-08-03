"""Fermionic CSQD workflow and projected diagonalization entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .subsampling import (
    MembershipPredictor,
    allocate_samples_per_cluster,
    finalize_batch_for_diagonalization,
    make_batches,
    make_initial_batches,
)
from .counts import (
    filter_samples_by_sector_distance,
    normalize_input_samples,
    pool_spin_blocks,
    sort_correct_samples_by_existing_clusters,
)
from .clustering import LegacyClusteringAdapter, clusterer_from_config, fit_clusters
from .clustering.base import Clusterer
from .config import CSQDConfig
from .configuration_recovery import cal_n_vecs, calc_raw_n_vectors, refine_a_cluster
from .diagnostics import DiagnosticsStore
from ._diagonalization import solve_sci_cluster
from ._random_state import derive_seed, make_rng
from ._types import ClusterResult, SCIResult


@dataclass(frozen=True)
class CSQDRunResult:
    """Full modular CSQD result with legacy tuple conversion."""

    best_result: SCIResult
    energy_history: list[float]
    basis_history: list[np.ndarray]
    n_vec_history: list[tuple[np.ndarray, ...]]
    cluster_result: ClusterResult
    cluster_spin_strings: np.ndarray
    diagnostics: dict

    def as_legacy_tuple(
        self,
    ) -> tuple[
        SCIResult,
        list[float],
        list[np.ndarray],
        list[tuple[np.ndarray, ...]],
        np.ndarray,
    ]:
        return (
            self.best_result,
            self.energy_history,
            self.basis_history,
            self.n_vec_history,
            self.cluster_result.cluster_weights,
        )


def _diagonalize_batch(
    *,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    hcore: np.ndarray,
    eri: np.ndarray,
    n_orb: int,
    n_elec: int,
    nuclear_repulsion_energy: float,
    max_dim: int | None,
    spin_sq: float | None,
) -> tuple[SCIResult, np.ndarray, dict]:
    batch_bool, batch_probs, membership = batch
    finalized = finalize_batch_for_diagonalization(
        batch_bool,
        batch_probs,
        membership,
        max_dim,
    )
    current_ci_strs = (finalized.ci_integers, finalized.ci_integers)
    result = solve_sci_cluster(
        current_ci_strs,
        hcore,
        eri,
        n_orb,
        (n_elec, n_elec),
        nuclear_repulsion_energy,
        finalized.membership,
        spin_sq=spin_sq,
    )
    return result, finalized.ci_integers, finalized.diagnostics


def _make_membership_predictor(
    clusterer: Clusterer,
    *,
    enabled: bool,
) -> MembershipPredictor | None:
    if not enabled:
        return None
    predictor = getattr(clusterer, "predict_membership", None)
    if not callable(predictor):
        return None

    def predict(bitstrings: np.ndarray) -> np.ndarray:
        return np.asarray(predictor(bitstrings), dtype=float)

    return predict


def run_csqd(
    *,
    bitarray: Tuple[np.ndarray, np.ndarray] | np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
    nuclear_repulsion_energy: float,
    config: CSQDConfig,
    clusterer: Clusterer | None = None,
    rng: np.random.Generator | None = None,
) -> CSQDRunResult:
    """Run the fixed-K soft-CSQD workflow without printing side effects."""

    run_rng = make_rng(config.random_seed) if rng is None else rng
    iteration_batch_seed_trace: list[dict[str, object]] = []
    seed_trace: dict[str, object] = {
        "run_seed": config.random_seed,
        "configured_clustering_seed": config.clustering.random_seed,
        "initial_batch_base_seed": None,
        "initial_batch_seeds": [],
        "iteration_batch_seeds": iteration_batch_seed_trace,
    }
    diagnostics = DiagnosticsStore(
        metadata={
            "schema_version": config.logging.schema_version,
            "config": config.to_dict(),
            "seed": config.random_seed,
            "clustering_seed": config.clustering.random_seed,
            "clustering_method": config.clustering.method,
            "seed_trace": seed_trace,
            "hamiltonian_info": {
                "hcore_shape": tuple(hcore.shape),
                "eri_shape": tuple(eri.shape),
                "nuclear_repulsion_energy": float(nuclear_repulsion_energy),
            },
        }
    )

    input_data = normalize_input_samples(bitarray)
    cutoff = config.clustering.precluster_sector_distance_cutoff
    if cutoff is None:
        clustering_input_data = input_data
        precluster_filter_diagnostics: dict[str, object] = {
            "enabled": False,
            "cutoff": None,
            "input_count": int(len(input_data[0])),
            "kept_count": int(len(input_data[0])),
            "removed_count": 0,
            "total_probability_mass": float(np.sum(input_data[1])),
            "kept_probability_mass": float(np.sum(input_data[1])),
            "removed_probability_mass": 0.0,
            "distance_summary": None,
            "kept_distance_summary": None,
            "removed_distance_summary": None,
        }
    else:
        clustering_input_data, precluster_filter_diagnostics = (
            filter_samples_by_sector_distance(
                input_data,
                config.n_orb,
                config.n_elec_per_spin,
                cutoff,
            )
        )
    diagnostics.metadata["precluster_sector_filter"] = precluster_filter_diagnostics

    spin_strings, spin_probs = pool_spin_blocks(config.n_orb, clustering_input_data)

    active_clusterer = (
        clusterer if clusterer is not None else clusterer_from_config(config.clustering)
    )
    cluster_result = fit_clusters(active_clusterer, spin_strings, spin_probs, run_rng)
    diagnostics.metadata["cluster_result_seed"] = cluster_result.seed
    seed_trace["cluster_result_seed"] = cluster_result.seed
    clusters = cluster_result.clusters
    cluster_probs = cluster_result.cluster_probabilities
    soft_membership = cluster_result.membership is not None and not np.issubdtype(
        cluster_result.membership.dtype,
        np.bool_,
    )
    membership_predictor = _make_membership_predictor(
        active_clusterer,
        enabled=soft_membership,
    )
    diagnostics.metadata["soft_membership"] = bool(soft_membership)
    diagnostics.metadata["membership_predictor"] = membership_predictor is not None

    correct_samples, correct_probs = sort_correct_samples_by_existing_clusters(
        input_data,
        config.n_elec_per_spin,
        clusters,
        spin_strings=spin_strings if soft_membership else None,
        membership=cluster_result.membership if soft_membership else None,
    )
    raw_n_vectors = calc_raw_n_vectors(clusters, cluster_probs, config.n_orb)
    samples_per_cluster = allocate_samples_per_cluster(
        cluster_result.cluster_weights,
        config.samples_per_batch,
    )

    for cluster_id, (cluster, weight, raw_n_vec) in enumerate(
        zip(clusters, cluster_result.cluster_weights, raw_n_vectors)
    ):
        diagnostics.add_cluster(
            cluster_id=cluster_id,
            cluster_weight=float(weight),
            cluster_size=int(len(cluster)),
            allocated_samples=int(samples_per_cluster[cluster_id]),
            reference_vector=raw_n_vec.tolist(),
            clustering_diagnostics=cluster_result.diagnostics,
        )

    energy_history: list[float] = []
    basis_history: list[np.ndarray] = []
    n_vec_history: list[tuple[np.ndarray, ...]] = []

    initial_seed = derive_seed(run_rng)
    seed_trace["initial_batch_base_seed"] = initial_seed
    seed_trace["initial_batch_seeds"] = [
        initial_seed + batch_idx for batch_idx in range(config.n_batch)
    ]
    initial_batches = make_initial_batches(
        config.n_batch,
        config.n_elec_per_spin,
        config.n_orb,
        raw_n_vectors,
        samples_per_cluster,
        correct_samples,
        correct_probs,
        base_seed=initial_seed,
        soft_membership=soft_membership,
        membership_predictor=membership_predictor,
        generation_attempt_factor=config.recovery.generation_attempt_factor,
    )

    initial_results: list[SCIResult] = []
    for batch_idx, batch in enumerate(initial_batches):
        result, basis, batch_diag = _diagonalize_batch(
            batch=batch,
            hcore=hcore,
            eri=eri,
            n_orb=config.n_orb,
            n_elec=config.n_elec_per_spin,
            nuclear_repulsion_energy=nuclear_repulsion_energy,
            max_dim=config.diagonalization.max_dim,
            spin_sq=config.diagonalization.spin_sq,
        )
        initial_results.append(result)
        basis_history.append(basis)
        diagnostics.add_batch(
            iteration_index=1,
            batch_index=batch_idx,
            batch_energy=float(result.energy),
            pool_size=int(len(basis)),
            subspace_dimension=int(len(basis) ** 2),
            solver_status="ok",
            **batch_diag,
        )

    if not initial_results:
        raise ValueError("At least one batch is required for CSQD diagonalization.")

    best_result = min(initial_results, key=lambda item: item.energy)
    energy_history.append(float(best_result.energy))
    n_vec_history.append(raw_n_vectors)
    diagnostics.add_iteration(
        iteration_index=1,
        best_energy=float(best_result.energy),
        energy_improvement=None,
        reference_change=None,
        convergence_flag=False,
    )

    prev_energy = float(best_result.energy) + 10.0
    prev_n_vec = tuple(np.asarray(vec, dtype=float).copy() for vec in raw_n_vectors)

    for iteration_offset in range(config.max_iterations - 1):
        iteration_index = iteration_offset + 2
        n_vec = cal_n_vecs(
            best_result,
            prev_n_vec,
            config.n_elec_per_spin,
            config.n_orb,
            cluster_result.n_clusters,
        )
        n_vec_history.append(n_vec)

        energy_diff = abs(float(best_result.energy) - prev_energy)
        occupancy_diff = float(np.max(np.abs(np.asarray(n_vec) - np.asarray(prev_n_vec))))
        converged = (
            energy_diff < config.diagonalization.energy_tol
            and occupancy_diff < config.diagonalization.occupancies_tol
        )
        diagnostics.add_iteration(
            iteration_index=iteration_index,
            best_energy=float(best_result.energy),
            energy_improvement=energy_diff,
            reference_change=occupancy_diff,
            convergence_flag=converged,
        )
        if converged:
            break

        prev_energy = float(best_result.energy)
        refined_clusters = [
            refine_a_cluster(
                clusters[cluster_id],
                cluster_probs[cluster_id],
                n_vec[cluster_id],
                config.n_elec_per_spin,
                config.n_orb,
                run_rng,
                flip_epsilon=config.recovery.flip_epsilon,
            )
            for cluster_id in range(cluster_result.n_clusters)
        ]

        batch_base_seed = derive_seed(run_rng)
        iteration_batch_seed_trace.append(
            {
                "iteration_index": iteration_index,
                "batch_base_seed": batch_base_seed,
                "batch_seeds": [batch_base_seed + batch_idx for batch_idx in range(config.n_batch)],
            }
        )
        batches = make_batches(
            config.n_batch,
            best_result,
            refined_clusters,
            samples_per_cluster,
            config.n_orb,
            config.diagonalization.threshold,
            config.diagonalization.max_dim,
            base_seed=batch_base_seed,
            soft_membership=soft_membership,
            membership_predictor=membership_predictor,
        )

        results: list[SCIResult] = []
        for batch_idx, batch in enumerate(batches):
            result, basis, batch_diag = _diagonalize_batch(
                batch=batch,
                hcore=hcore,
                eri=eri,
                n_orb=config.n_orb,
                n_elec=config.n_elec_per_spin,
                nuclear_repulsion_energy=nuclear_repulsion_energy,
                max_dim=config.diagonalization.max_dim,
                spin_sq=config.diagonalization.spin_sq,
            )
            results.append(result)
            basis_history.append(basis)
            diagnostics.add_batch(
                iteration_index=iteration_index,
                batch_index=batch_idx,
                batch_energy=float(result.energy),
                pool_size=int(len(basis)),
                subspace_dimension=int(len(basis) ** 2),
                solver_status="ok",
                **batch_diag,
            )

        current_gen_best = min(results, key=lambda item: item.energy)
        if current_gen_best.energy < best_result.energy:
            best_result = current_gen_best

        energy_history.append(float(best_result.energy))
        prev_n_vec = tuple(np.asarray(vec, dtype=float).copy() for vec in n_vec)

    return CSQDRunResult(
        best_result=best_result,
        energy_history=energy_history,
        basis_history=basis_history,
        n_vec_history=n_vec_history,
        cluster_result=cluster_result,
        cluster_spin_strings=spin_strings,
        diagnostics=diagnostics.as_dict(),
    )


def diagonalize_fermionic_hamiltonian_with_clustering(
    bitarray: Tuple[np.ndarray, np.ndarray] | np.ndarray,
    n_orb: int,
    n_elec: int,
    n_clusters: int,
    n_batch: int,
    threshold: float,
    samples_per_batch: int,
    max_iterations: int,
    hcore: np.ndarray,
    eri: np.ndarray,
    nuclear_repulsion_energy: float,
    max_dim: Optional[int] = None,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    clustering_func: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
    return_result: bool = False,
) -> (
    CSQDRunResult
    | tuple[
        SCIResult,
        list[float],
        list[np.ndarray],
        list[tuple[np.ndarray, ...]],
        np.ndarray,
    ]
):
    """Legacy-compatible entrypoint backed by the refactored workflow."""

    config = CSQDConfig.from_legacy_args(
        n_orb=n_orb,
        n_elec=n_elec,
        n_clusters=n_clusters,
        n_batch=n_batch,
        threshold=threshold,
        samples_per_batch=samples_per_batch,
        max_iterations=max_iterations,
        max_dim=max_dim,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
    )
    clusterer = (
        LegacyClusteringAdapter(clustering_func, n_clusters)
        if clustering_func is not None
        else None
    )
    result = run_csqd(
        bitarray=bitarray,
        hcore=hcore,
        eri=eri,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        config=config,
        clusterer=clusterer,
        rng=rng,
    )
    if return_result:
        return result
    return result.as_legacy_tuple()
