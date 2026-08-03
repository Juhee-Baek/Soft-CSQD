"""Cluster-aware configuration recovery and reference updates."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit_addon_sqd.configuration_recovery import _p_flip_0_to_1, _p_flip_1_to_0

from .counts import ci_integers_to_spin_bitstrings, normalize_probabilities
from ._types import SCIResult


def _choice_probs(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values[mask], dtype=float)
    return normalize_probabilities(selected, size=len(selected))


def _bipartite_bitstring_correcting_cluster(
    bitstring: np.ndarray,
    n_vec: np.ndarray,
    n_elec: int,
    n_orb: int,
    rng: np.random.Generator,
    flip_epsilon: float = 0.01,
) -> np.ndarray:
    """Correct one spin block to the target particle number."""

    bit_array = bitstring.copy().astype(bool)
    probs_vec = np.zeros(n_orb, dtype=float)

    for orbital_idx in range(n_orb):
        if bit_array[orbital_idx]:
            probs_vec[orbital_idx] = _p_flip_1_to_0(
                n_elec / n_orb,
                n_vec[orbital_idx],
                flip_epsilon,
            )
        else:
            probs_vec[orbital_idx] = _p_flip_0_to_1(
                n_elec / n_orb,
                n_vec[orbital_idx],
                flip_epsilon,
            )

    probs_vec = np.abs(np.nan_to_num(probs_vec, nan=0.0, posinf=0.0, neginf=0.0))
    n_diff = int(np.sum(bit_array) - n_elec)

    if n_diff > 0:
        occupied = np.where(bit_array)[0]
        p_choice = _choice_probs(probs_vec, bit_array)
        indices_to_flip = rng.choice(
            occupied,
            size=n_diff,
            replace=False,
            p=p_choice,
        )
        bit_array[indices_to_flip] = False
    elif n_diff < 0:
        empty_mask = np.logical_not(bit_array)
        empty = np.where(empty_mask)[0]
        p_choice = _choice_probs(probs_vec, empty_mask)
        indices_to_flip = rng.choice(
            empty,
            size=abs(n_diff),
            replace=False,
            p=p_choice,
        )
        bit_array[indices_to_flip] = True

    return bit_array


def refine_a_cluster(
    cluster_bits: np.ndarray,
    cluster_probs: np.ndarray,
    n_vec: np.ndarray,
    n_elec: int,
    n_orb: int,
    rng: np.random.Generator,
    flip_epsilon: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover every bitstring in a cluster and aggregate duplicate outputs."""

    if len(cluster_bits) == 0:
        return np.empty((0, n_orb), dtype=bool), np.array([], dtype=float)

    refined_matrix = np.array(
        [
            _bipartite_bitstring_correcting_cluster(
                cluster_bits[row_idx],
                n_vec,
                n_elec,
                n_orb,
                rng,
                flip_epsilon=flip_epsilon,
            )
            for row_idx in range(len(cluster_bits))
        ],
        dtype=bool,
    )

    unique_bits, inverse_indices = np.unique(
        refined_matrix,
        axis=0,
        return_inverse=True,
    )
    unique_probs = np.zeros(len(unique_bits), dtype=float)
    np.add.at(unique_probs, inverse_indices, cluster_probs)
    return unique_bits, unique_probs


def rescale_reference_vector(n_vec: np.ndarray, n_elec: int, n_orb: int) -> np.ndarray:
    """Return a finite non-negative vector whose sum is `n_elec`."""

    arr = np.asarray(n_vec, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(n_orb, n_elec / n_orb, dtype=float)
    return (arr / total) * n_elec


def calc_raw_n_vectors(
    clusters: list[np.ndarray] | tuple[np.ndarray, ...],
    cluster_probs: list[np.ndarray] | tuple[np.ndarray, ...],
    n_orb: int,
) -> tuple[np.ndarray, ...]:
    """Compute initial weighted occupancy profiles for each cluster."""

    raw_n_vectors: list[np.ndarray] = []
    for bitstrings, probs in zip(clusters, cluster_probs):
        if len(bitstrings) == 0:
            raw_n_vectors.append(np.zeros(n_orb, dtype=float))
            continue
        weighted_sum = np.dot(probs, bitstrings.astype(float))
        raw_n_vectors.append(np.asarray(weighted_sum, dtype=float))
    return tuple(raw_n_vectors)


def cal_n_vecs(
    best_result: SCIResult | Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, float, np.ndarray],
    old_n_vectors: tuple[np.ndarray, ...],
    n_elec: int,
    n_orb: int,
    n_clusters: int,
) -> tuple[np.ndarray, ...]:
    """Update per-cluster occupancy vectors from the selected-CI wavefunction."""

    ci_strs, coef, _, membership = best_result
    alpha_strs = ci_strs[0]
    probs_matrix = np.abs(coef) ** 2

    alpha_contribution = np.sum(probs_matrix, axis=1)
    beta_contribution = np.sum(probs_matrix, axis=0)
    bit_mat = ci_integers_to_spin_bitstrings(alpha_strs, n_orb).astype(float)

    new_n_vecs: list[np.ndarray] = []
    for cluster_id in range(n_clusters):
        membership_weights = np.asarray(membership[:, cluster_id], dtype=float)
        if not np.any(membership_weights > 0.0):
            new_n_vecs.append(rescale_reference_vector(old_n_vectors[cluster_id], n_elec, n_orb))
            continue

        string_weights = (alpha_contribution + beta_contribution) * membership_weights
        n_vec_k = np.dot(string_weights, bit_mat)
        new_n_vecs.append(rescale_reference_vector(n_vec_k, n_elec, n_orb))

    return tuple(new_n_vecs)
