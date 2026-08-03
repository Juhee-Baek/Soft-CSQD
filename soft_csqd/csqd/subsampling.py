"""Cluster-aware subsampling and batch construction."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .counts import (
    ci_integers_to_spin_bitstrings,
    normalize_probabilities,
    spin_bitstrings_to_ci_integers,
)
from ._random_state import make_rng
from ._types import FinalizedBatch

MembershipPredictor = Callable[[np.ndarray], np.ndarray]


def allocate_samples_per_cluster(
    cluster_weights: np.ndarray,
    samples_per_batch: int,
) -> tuple[int, ...]:
    """Use the original ceil-based allocation policy."""

    return tuple(int(np.ceil(weight * samples_per_batch)) for weight in cluster_weights)


def _one_hot_membership(
    cluster_ids: np.ndarray,
    n_clusters: int,
    *,
    soft_membership: bool,
) -> np.ndarray:
    dtype = float if soft_membership else bool
    membership = np.zeros((len(cluster_ids), n_clusters), dtype=dtype)
    if len(cluster_ids) > 0:
        membership[np.arange(len(cluster_ids)), cluster_ids.astype(int)] = (
            1.0 if soft_membership else True
        )
    return membership


def _predict_membership(
    bitstrings: np.ndarray,
    n_clusters: int,
    membership_predictor: MembershipPredictor | None,
) -> np.ndarray | None:
    if membership_predictor is None:
        return None
    if len(bitstrings) == 0:
        return np.empty((0, n_clusters), dtype=float)

    predicted = np.asarray(membership_predictor(bitstrings.astype(bool)), dtype=float)
    expected_shape = (len(bitstrings), n_clusters)
    if predicted.shape != expected_shape:
        raise ValueError(f"membership_predictor must return shape {expected_shape}.")

    predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
    predicted = np.maximum(predicted, 0.0)
    if n_clusters == 0:
        return predicted

    row_sums = np.sum(predicted, axis=1, keepdims=True)
    empty_rows = np.ravel(row_sums <= 0.0)
    if np.any(empty_rows):
        predicted[empty_rows] = 1.0 / n_clusters
        row_sums = np.sum(predicted, axis=1, keepdims=True)

    return np.divide(
        predicted,
        row_sums,
        out=np.zeros_like(predicted),
        where=row_sums > 0.0,
    )


def _merge_duplicate_membership(
    inverse_indices: np.ndarray,
    row_probabilities: np.ndarray,
    row_membership: np.ndarray,
    n_unique: int,
    n_clusters: int,
) -> np.ndarray:
    if np.issubdtype(row_membership.dtype, np.bool_):
        unique_membership = np.zeros((n_unique, n_clusters), dtype=bool)
        for original_idx, target_idx in enumerate(inverse_indices):
            unique_membership[target_idx] |= row_membership[original_idx]
        return unique_membership

    unique_membership = np.zeros((n_unique, n_clusters), dtype=float)
    row_weight_totals = np.zeros(n_unique, dtype=float)
    row_membership = np.nan_to_num(
        np.asarray(row_membership, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    row_membership = np.maximum(row_membership, 0.0)

    for original_idx, target_idx in enumerate(inverse_indices):
        row_weight = max(float(row_probabilities[original_idx]), 0.0)
        unique_membership[target_idx] += row_membership[original_idx] * row_weight
        row_weight_totals[target_idx] += row_weight

    nonzero_rows = row_weight_totals > 0.0
    unique_membership[nonzero_rows] /= row_weight_totals[nonzero_rows, None]
    for row_idx in np.where(nonzero_rows)[0]:
        unique_membership[row_idx] = normalize_probabilities(
            unique_membership[row_idx],
            size=n_clusters,
        )
    return unique_membership


def _sample_spin_string(
    probability_vector: np.ndarray,
    n_elec: int,
    n_orb: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    probs = normalize_probabilities(probability_vector, size=n_orb)
    sample = np.zeros(n_orb, dtype=bool)
    if n_elec > 0:
        occ_idx = rng.choice(n_orb, size=n_elec, replace=False, p=probs)
        sample[occ_idx] = True
        score = float(np.prod(probs[occ_idx]))
    else:
        score = 1.0
    return sample, score


def initial_subsample_cluster(
    n_elec: int,
    n_orb: int,
    raw_n_vectors: tuple[np.ndarray, ...],
    samples_per_cluster: tuple[int, ...],
    correct_samples: tuple[np.ndarray, ...],
    correct_probs: tuple[np.ndarray, ...],
    seed: Optional[int] = None,
    soft_membership: bool = False,
    membership_predictor: MembershipPredictor | None = None,
    generation_attempt_factor: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one initial batch with membership rows aligned to bitstrings."""

    rng = make_rng(seed)
    attempt_factor = max(1, int(generation_attempt_factor))
    n_clusters = len(raw_n_vectors)
    total_target_count = int(sum(samples_per_cluster))
    if total_target_count <= 0:
        membership_dtype = float if soft_membership else bool
        return (
            np.empty((0, n_orb), dtype=bool),
            np.array([], dtype=float),
            np.empty((0, n_clusters), dtype=membership_dtype),
        )

    all_sampled_strs: list[np.ndarray] = []
    all_sampled_probs: list[np.ndarray] = []
    all_sampled_cluster_ids: list[np.ndarray] = []

    for cluster_id in range(n_clusters):
        target_count = int(samples_per_cluster[cluster_id])
        if target_count <= 0:
            continue

        raw_n_vec = raw_n_vectors[cluster_id]
        real_samples = correct_samples[cluster_id]
        real_probs = correct_probs[cluster_id]
        num_real = len(real_samples)
        cluster_global_weight = target_count / total_target_count

        if num_real >= target_count:
            p_choice = normalize_probabilities(real_probs, size=num_real)
            indices = rng.choice(
                num_real,
                size=target_count,
                replace=False,
                p=p_choice,
            )
            cluster_strs = real_samples[indices]
            cluster_probs = real_probs[indices]
        else:
            cluster_strs_list: list[np.ndarray] = []
            cluster_probs_list: list[np.ndarray] = []
            if num_real > 0:
                cluster_strs_list.append(real_samples)
                cluster_probs_list.append(real_probs)

            deficit = target_count - num_real
            hashes = {sample.tobytes() for sample in real_samples}
            generated_samples: list[np.ndarray] = []
            generated_probs: list[float] = []
            max_attempts = max(deficit, deficit * attempt_factor)
            attempts = 0
            while len(generated_samples) < deficit and attempts < max_attempts:
                attempts += 1
                sample, score = _sample_spin_string(raw_n_vec, n_elec, n_orb, rng)
                sample_hash = sample.tobytes()
                if sample_hash in hashes:
                    continue
                hashes.add(sample_hash)
                generated_samples.append(sample)
                generated_probs.append(score)

            if generated_samples:
                cluster_strs_list.append(np.array(generated_samples, dtype=bool))
                cluster_probs_list.append(np.array(generated_probs, dtype=float))

            if not cluster_strs_list:
                continue

            cluster_strs = np.vstack(cluster_strs_list)
            cluster_probs = np.concatenate(cluster_probs_list)

        cluster_probs = normalize_probabilities(cluster_probs, size=len(cluster_probs))
        cluster_probs = cluster_probs * cluster_global_weight
        all_sampled_strs.append(cluster_strs.astype(bool))
        all_sampled_probs.append(cluster_probs)
        all_sampled_cluster_ids.append(np.full(len(cluster_strs), cluster_id))

    if not all_sampled_strs:
        membership_dtype = float if soft_membership else bool
        return (
            np.empty((0, n_orb), dtype=bool),
            np.array([], dtype=float),
            np.empty((0, n_clusters), dtype=membership_dtype),
        )

    merged_strs = np.vstack(all_sampled_strs)
    merged_probs = np.concatenate(all_sampled_probs)
    merged_ids = np.concatenate(all_sampled_cluster_ids)

    unique_batch, inverse_indices = np.unique(
        merged_strs,
        axis=0,
        return_inverse=True,
    )
    unique_probs = np.zeros(len(unique_batch), dtype=float)
    np.add.at(unique_probs, inverse_indices, merged_probs)

    predicted_membership = _predict_membership(
        unique_batch,
        n_clusters,
        membership_predictor,
    )
    if predicted_membership is None:
        merged_membership = _one_hot_membership(
            merged_ids,
            n_clusters,
            soft_membership=soft_membership,
        )
        membership_matrix = _merge_duplicate_membership(
            inverse_indices,
            merged_probs,
            merged_membership,
            len(unique_batch),
            n_clusters,
        )
    else:
        membership_matrix = predicted_membership

    return unique_batch, unique_probs, membership_matrix


def make_initial_batches(
    n_batch: int,
    n_elec: int,
    n_orb: int,
    raw_n_vectors: tuple[np.ndarray, ...],
    samples_per_cluster: tuple[int, ...],
    correct_samples: tuple[np.ndarray, ...],
    correct_probs: tuple[np.ndarray, ...],
    base_seed: Optional[int] = None,
    soft_membership: bool = False,
    membership_predictor: MembershipPredictor | None = None,
    generation_attempt_factor: int = 20,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate deterministic initial batches from a base seed."""

    seed_base = 42 if base_seed is None else base_seed
    return [
        initial_subsample_cluster(
            n_elec,
            n_orb,
            raw_n_vectors,
            samples_per_cluster,
            correct_samples,
            correct_probs,
            seed=seed_base + batch_idx,
            soft_membership=soft_membership,
            membership_predictor=membership_predictor,
            generation_attempt_factor=generation_attempt_factor,
        )
        for batch_idx in range(n_batch)
    ]


def subsample_cluster(
    refined_clusters: list[Tuple[np.ndarray, np.ndarray]],
    samples_per_cluster: Tuple[int, ...],
    n_orb: int,
    rng: np.random.Generator,
    soft_membership: bool = False,
    membership_predictor: MembershipPredictor | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample from refined clusters and aggregate duplicate basis rows."""

    n_clusters = len(refined_clusters)
    total_target_count = int(sum(samples_per_cluster))
    if total_target_count <= 0:
        membership_dtype = float if soft_membership else bool
        return (
            np.empty((0, n_orb), dtype=bool),
            np.array([], dtype=float),
            np.empty((0, n_clusters), dtype=membership_dtype),
        )

    all_sampled_strs: list[np.ndarray] = []
    all_sampled_probs: list[np.ndarray] = []
    all_sampled_cluster_ids: list[np.ndarray] = []

    for cluster_id in range(n_clusters):
        ref_bits, ref_probs = refined_clusters[cluster_id]
        target_count = int(samples_per_cluster[cluster_id])
        if target_count <= 0 or len(ref_bits) == 0:
            continue

        cluster_global_weight = target_count / total_target_count
        p_intra = normalize_probabilities(ref_probs, size=len(ref_bits))
        replace = target_count > len(ref_bits)
        indices = rng.choice(
            len(ref_bits),
            size=target_count,
            replace=replace,
            p=p_intra,
        )
        if replace:
            sampled_probs = normalize_probabilities(p_intra[indices], size=target_count)
            scaled_probs = sampled_probs * cluster_global_weight
        else:
            scaled_probs = p_intra[indices] * cluster_global_weight

        all_sampled_strs.append(ref_bits[indices])
        all_sampled_probs.append(scaled_probs)
        all_sampled_cluster_ids.append(np.full(target_count, cluster_id))

    if not all_sampled_strs:
        membership_dtype = float if soft_membership else bool
        return (
            np.empty((0, n_orb), dtype=bool),
            np.array([], dtype=float),
            np.empty((0, n_clusters), dtype=membership_dtype),
        )

    merged_strs = np.vstack(all_sampled_strs)
    merged_probs = np.concatenate(all_sampled_probs)
    merged_ids = np.concatenate(all_sampled_cluster_ids)

    unique_batch, inverse_indices = np.unique(
        merged_strs,
        axis=0,
        return_inverse=True,
    )
    unique_probs = np.zeros(len(unique_batch), dtype=float)
    np.add.at(unique_probs, inverse_indices, merged_probs)

    predicted_membership = _predict_membership(
        unique_batch,
        n_clusters,
        membership_predictor,
    )
    if predicted_membership is None:
        merged_membership = _one_hot_membership(
            merged_ids,
            n_clusters,
            soft_membership=soft_membership,
        )
        membership_matrix = _merge_duplicate_membership(
            inverse_indices,
            merged_probs,
            merged_membership,
            len(unique_batch),
            n_clusters,
        )
    else:
        membership_matrix = predicted_membership

    return unique_batch, unique_probs, membership_matrix


def make_batches(
    n_batch: int,
    best_result: Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, float, np.ndarray],
    refined_clusters: list[Tuple[np.ndarray, np.ndarray]],
    samples_per_cluster: Tuple[int, ...],
    n_orb: int,
    threshold: float,
    max_dim: Optional[int],
    base_seed: Optional[int] = None,
    soft_membership: bool = False,
    membership_predictor: MembershipPredictor | None = None,
) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Merge carry-over states with sampled refined states for each batch."""

    ci_strs, coef, _, prev_membership = best_result
    alpha_ints = ci_strs[0]
    abs_coef = np.abs(coef)

    important_alpha = np.any(abs_coef > threshold, axis=1)
    important_beta = np.any(abs_coef > threshold, axis=0)
    important_mask = important_alpha | important_beta

    carry_over_ints = alpha_ints[important_mask]
    carry_over_membership = prev_membership[important_mask]
    coefficient_probs = np.abs(coef) ** 2
    alpha_weights = np.sum(coefficient_probs, axis=1)
    beta_weights = np.sum(coefficient_probs, axis=0)
    carry_over_weights = 0.5 * (alpha_weights + beta_weights)[important_mask]
    importance_scores = (
        np.sum(abs_coef, axis=1)[important_mask] + np.sum(abs_coef, axis=0)[important_mask]
    )

    if max_dim is not None and len(carry_over_ints) > max_dim:
        top_indices = np.argsort(importance_scores)[-max_dim:]
        carry_over_ints = carry_over_ints[top_indices]
        carry_over_membership = carry_over_membership[top_indices]
        carry_over_weights = carry_over_weights[top_indices]

    carry_over_bits = ci_integers_to_spin_bitstrings(carry_over_ints, n_orb)
    seed_base = 42 if base_seed is None else base_seed
    all_batches: list[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_clusters = len(refined_clusters)

    for batch_idx in range(n_batch):
        rng = make_rng(seed_base + batch_idx)
        new_bits, new_probs, new_membership = subsample_cluster(
            refined_clusters,
            samples_per_cluster,
            n_orb,
            rng,
            soft_membership=soft_membership,
            membership_predictor=membership_predictor,
        )

        combined_bits = np.vstack([carry_over_bits, new_bits])
        combined_probs = np.concatenate([carry_over_weights, new_probs])
        combined_membership = np.vstack([carry_over_membership, new_membership])

        if len(combined_bits) == 0:
            membership_dtype = float if soft_membership else bool
            all_batches.append(
                (
                    np.empty((0, n_orb), dtype=bool),
                    np.array([], dtype=float),
                    np.empty((0, n_clusters), dtype=membership_dtype),
                )
            )
            continue

        unique_batch, inverse_indices = np.unique(
            combined_bits,
            axis=0,
            return_inverse=True,
        )
        unique_probs = np.zeros(len(unique_batch), dtype=float)
        np.add.at(unique_probs, inverse_indices, combined_probs)

        predicted_membership = _predict_membership(
            unique_batch,
            n_clusters,
            membership_predictor,
        )
        if predicted_membership is None:
            unique_membership = _merge_duplicate_membership(
                inverse_indices,
                combined_probs,
                combined_membership,
                len(unique_batch),
                n_clusters,
            )
        else:
            unique_membership = predicted_membership

        sorted_indices = np.argsort(unique_probs)[::-1]
        unique_batch = unique_batch[sorted_indices]
        unique_probs = unique_probs[sorted_indices]
        unique_membership = unique_membership[sorted_indices]

        if max_dim is not None and len(unique_batch) > max_dim:
            unique_batch = unique_batch[:max_dim]
            unique_probs = unique_probs[:max_dim]
            unique_membership = unique_membership[:max_dim]

        all_batches.append((unique_batch, unique_probs, unique_membership))

    return all_batches


def finalize_batch_for_diagonalization(
    batch_bool: np.ndarray,
    batch_probs: np.ndarray,
    membership: np.ndarray,
    max_dim: Optional[int],
) -> FinalizedBatch:
    """Sort, deduplicate, truncate, and PySCF-sort a batch in one aligned step."""

    if len(batch_bool) == 0:
        return FinalizedBatch(
            ci_integers=np.array([], dtype=np.int64),
            probabilities=np.array([], dtype=float),
            membership=membership,
            diagnostics={"duplicate_count": 0, "truncated_count": 0},
        )

    batch_ints = spin_bitstrings_to_ci_integers(batch_bool)
    sorted_idx = np.argsort(batch_probs)[::-1]
    sorted_ints = batch_ints[sorted_idx]
    sorted_probs = batch_probs[sorted_idx]
    sorted_membership = membership[sorted_idx]

    position_by_int: dict[int, int] = {}
    unique_ints: list[int] = []
    unique_probs: list[float] = []
    unique_membership: list[np.ndarray] = []
    duplicate_count = 0

    for row_idx, ci_int in enumerate(sorted_ints):
        key = int(ci_int)
        if key in position_by_int:
            target_idx = position_by_int[key]
            unique_probs[target_idx] += float(sorted_probs[row_idx])
            if np.issubdtype(sorted_membership.dtype, np.bool_):
                unique_membership[target_idx] |= sorted_membership[row_idx]
            else:
                existing_weight = unique_probs[target_idx] - float(sorted_probs[row_idx])
                current_weight = float(sorted_probs[row_idx])
                total_weight = unique_probs[target_idx]
                if total_weight > 0.0:
                    unique_membership[target_idx] = (
                        unique_membership[target_idx] * existing_weight
                        + sorted_membership[row_idx] * current_weight
                    ) / total_weight
            duplicate_count += 1
            continue

        position_by_int[key] = len(unique_ints)
        unique_ints.append(key)
        unique_probs.append(float(sorted_probs[row_idx]))
        unique_membership.append(sorted_membership[row_idx].copy())

    truncated_count = 0
    if max_dim is not None and len(unique_ints) > max_dim:
        truncated_count = len(unique_ints) - max_dim
        unique_ints = unique_ints[:max_dim]
        unique_probs = unique_probs[:max_dim]
        unique_membership = unique_membership[:max_dim]

    ci_array = np.asarray(unique_ints, dtype=batch_ints.dtype)
    probs_array = np.asarray(unique_probs, dtype=float)
    membership_array = np.asarray(
        unique_membership,
        dtype=bool if np.issubdtype(sorted_membership.dtype, np.bool_) else float,
    )
    if not np.issubdtype(membership_array.dtype, np.bool_):
        membership_array = np.vstack(
            [
                normalize_probabilities(row, size=membership_array.shape[1])
                for row in membership_array
            ]
        )

    final_sort_idx = np.argsort(ci_array)
    return FinalizedBatch(
        ci_integers=ci_array[final_sort_idx],
        probabilities=probs_array[final_sort_idx],
        membership=membership_array[final_sort_idx],
        diagnostics={
            "duplicate_count": duplicate_count,
            "truncated_count": truncated_count,
        },
    )
