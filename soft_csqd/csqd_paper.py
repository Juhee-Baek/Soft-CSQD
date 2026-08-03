"""Legacy-compatible facade for the soft-CSQD implementation.

This module preserves the original paper-style API while delegating the
implementation to the modular code under `soft_csqd/csqd/`.
"""

from __future__ import annotations

if __package__:
    from .csqd.subsampling import (
        finalize_batch_for_diagonalization,
        initial_subsample_cluster,
        make_batches,
        make_initial_batches,
        subsample_cluster,
    )
    from .csqd.config import (
        CSQDConfig,
        ClusteringConfig,
        DiagonalizationConfig,
        LoggingConfig,
        RecoveryConfig,
    )
    from .csqd.counts import (
        pool_spin_blocks,
        sort_correct_samples_by_existing_clusters,
    )
    from .csqd.configuration_recovery import (
        _bipartite_bitstring_correcting_cluster,
        cal_n_vecs,
        calc_raw_n_vectors,
        refine_a_cluster,
    )
    from .csqd.fermion import (
        CSQDRunResult,
        diagonalize_fermionic_hamiltonian_with_clustering,
        run_csqd,
    )
    from .csqd._diagonalization import solve_sci_cluster
    from .examples.chemistry import define_n2
else:
    from csqd.subsampling import (
        finalize_batch_for_diagonalization,
        initial_subsample_cluster,
        make_batches,
        make_initial_batches,
        subsample_cluster,
    )
    from csqd.config import (
        CSQDConfig,
        ClusteringConfig,
        DiagonalizationConfig,
        LoggingConfig,
        RecoveryConfig,
    )
    from csqd.counts import pool_spin_blocks, sort_correct_samples_by_existing_clusters
    from csqd.configuration_recovery import (
        _bipartite_bitstring_correcting_cluster,
        cal_n_vecs,
        calc_raw_n_vectors,
        refine_a_cluster,
    )
    from csqd.fermion import (
        CSQDRunResult,
        diagonalize_fermionic_hamiltonian_with_clustering,
        run_csqd,
    )
    from csqd._diagonalization import solve_sci_cluster
    from examples.chemistry import define_n2


def assign_clusters_kmodes(*args, **kwargs):
    if __package__:
        from .csqd.clustering.kmodes import assign_clusters_kmodes as impl
    else:
        from csqd.clustering.kmodes import assign_clusters_kmodes as impl
    return impl(*args, **kwargs)


def assign_clusters_bmm(*args, **kwargs):
    if __package__:
        from .csqd.clustering.bmm import assign_clusters_bmm as impl
    else:
        from csqd.clustering.bmm import assign_clusters_bmm as impl
    return impl(*args, **kwargs)


def assign_clusters_fuzzy_kmodes(*args, **kwargs):
    if __package__:
        from .csqd.clustering.fuzzy_kmodes import assign_clusters_fuzzy_kmodes as impl
    else:
        from csqd.clustering.fuzzy_kmodes import assign_clusters_fuzzy_kmodes as impl
    return impl(*args, **kwargs)


__all__ = [
    "CSQDConfig",
    "CSQDRunResult",
    "ClusteringConfig",
    "DiagonalizationConfig",
    "LoggingConfig",
    "RecoveryConfig",
    "_bipartite_bitstring_correcting_cluster",
    "assign_clusters_bmm",
    "assign_clusters_fuzzy_kmodes",
    "assign_clusters_kmodes",
    "cal_n_vecs",
    "calc_raw_n_vectors",
    "define_n2",
    "diagonalize_fermionic_hamiltonian_with_clustering",
    "finalize_batch_for_diagonalization",
    "initial_subsample_cluster",
    "make_batches",
    "make_initial_batches",
    "pool_spin_blocks",
    "refine_a_cluster",
    "run_csqd",
    "solve_sci_cluster",
    "sort_correct_samples_by_existing_clusters",
    "subsample_cluster",
]
