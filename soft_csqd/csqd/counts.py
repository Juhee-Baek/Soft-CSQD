"""Count, BitArray, and bitstring conversion utilities."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:
    from qiskit_addon_sqd.counts import (
        bit_array_to_arrays as _bit_array_to_arrays,
        bitstring_matrix_to_integers as _bitstring_matrix_to_integers,
    )
    from qiskit_addon_sqd.subsampling import (
        postselect_by_hamming_right_and_left as _postselect_by_hamming_right_and_left,
    )
except ImportError:
    _bit_array_to_arrays = None
    _bitstring_matrix_to_integers = None
    _postselect_by_hamming_right_and_left = None


def _require_qiskit_addon_sqd(name: str) -> None:
    raise ImportError(
        f"{name} requires the optional 'qiskit_addon_sqd' package."
    )


def normalize_input_samples(
    bitarray: Tuple[np.ndarray, np.ndarray] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(full_bitstrings, probabilities)` for either accepted input format."""

    if isinstance(bitarray, tuple):
        return bitarray
    if _bit_array_to_arrays is None:
        _require_qiskit_addon_sqd("normalize_input_samples for BitArray inputs")
    return _bit_array_to_arrays(bitarray)


def split_beta_alpha(
    full_bitstrings: np.ndarray,
    n_orb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split full strings ordered as `[beta spin block | alpha spin block]`."""

    beta_strs = full_bitstrings[:, :n_orb]
    alpha_strs = full_bitstrings[:, n_orb:]
    return beta_strs, alpha_strs


def sector_particle_number_distances(
    full_bitstrings: np.ndarray,
    n_orb: int,
    n_elec_per_spin: int,
) -> np.ndarray:
    """Return spin-resolved distance from the target particle-number sector."""

    bitstrings = np.asarray(full_bitstrings, dtype=bool)
    if bitstrings.ndim != 2:
        raise ValueError("full_bitstrings must be a 2D bitstring matrix.")
    expected_width = 2 * n_orb
    if bitstrings.shape[1] != expected_width:
        raise ValueError(
            f"Bitstring length ({bitstrings.shape[1]}) must equal 2*n_orb ({expected_width})."
        )

    beta_strs, alpha_strs = split_beta_alpha(bitstrings, n_orb)
    beta_counts = np.sum(beta_strs, axis=1)
    alpha_counts = np.sum(alpha_strs, axis=1)
    return np.abs(beta_counts - n_elec_per_spin) + np.abs(alpha_counts - n_elec_per_spin)


def _distance_summary(distances: np.ndarray) -> dict[str, int | float | None]:
    if len(distances) == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    return {
        "min": int(np.min(distances)),
        "max": int(np.max(distances)),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
    }


def filter_samples_by_sector_distance(
    input_data: tuple[np.ndarray, np.ndarray],
    n_orb: int,
    n_elec_per_spin: int,
    cutoff: int,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, object]]:
    """Drop full samples whose sector distance is greater than or equal to `cutoff`."""

    if cutoff <= 0:
        raise ValueError("precluster_sector_distance_cutoff must be a positive integer.")

    bitstring_matrix, probs = input_data
    distances = sector_particle_number_distances(
        bitstring_matrix,
        n_orb,
        n_elec_per_spin,
    )
    keep_mask = distances < cutoff
    if not np.any(keep_mask):
        raise ValueError(
            "Pre-clustering sector-distance filter removed all samples. "
            "Use a larger cutoff or disable the filter."
        )

    probabilities = np.asarray(probs, dtype=float)
    total_probability_mass = float(np.sum(probabilities))
    kept_probability_mass = float(np.sum(probabilities[keep_mask]))
    removed_mask = ~keep_mask

    diagnostics: dict[str, object] = {
        "enabled": True,
        "cutoff": int(cutoff),
        "criterion": "abs(n_beta - target) + abs(n_alpha - target) < cutoff",
        "input_count": int(len(bitstring_matrix)),
        "kept_count": int(np.sum(keep_mask)),
        "removed_count": int(np.sum(removed_mask)),
        "total_probability_mass": total_probability_mass,
        "kept_probability_mass": kept_probability_mass,
        "removed_probability_mass": float(total_probability_mass - kept_probability_mass),
        "distance_summary": _distance_summary(distances),
        "kept_distance_summary": _distance_summary(distances[keep_mask]),
        "removed_distance_summary": _distance_summary(distances[removed_mask]),
    }
    filtered = (bitstring_matrix[keep_mask], probabilities[keep_mask])
    return filtered, diagnostics


def ci_integers_to_spin_bitstrings(ints: np.ndarray, n_orb: int) -> np.ndarray:
    """Convert PySCF CI integer strings to big-endian spin-block bitstrings."""

    if len(ints) == 0:
        return np.empty((0, n_orb), dtype=bool)
    return np.array(
        [[(int(value) >> p) & 1 for p in range(n_orb - 1, -1, -1)] for value in ints],
        dtype=bool,
    )


def spin_bitstrings_to_ci_integers(bitstrings: np.ndarray) -> np.ndarray:
    if _bitstring_matrix_to_integers is not None:
        return _bitstring_matrix_to_integers(bitstrings)

    matrix = np.asarray(bitstrings, dtype=bool)
    powers = np.arange(matrix.shape[1] - 1, -1, -1, dtype=object)
    weights = np.array([1 << int(power) for power in powers], dtype=object)
    return np.asarray(matrix.astype(object) @ weights, dtype=object)


def normalize_probabilities(
    weights: np.ndarray,
    *,
    size: int | None = None,
) -> np.ndarray:
    """Normalize finite non-negative weights, falling back to uniform weights."""

    arr = np.asarray(weights, dtype=float)
    if size is None:
        size = len(arr)
    if size == 0:
        return np.array([], dtype=float)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(size, 1.0 / size, dtype=float)
    return arr / total


def pool_spin_blocks(
    n_orb: int,
    input_data: tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pool beta and alpha spin blocks into one normalized spin-string distribution.

    Full bitstrings are assumed to be ordered as `[beta block | alpha block]`.
    """

    bitstring_matrix, probs = input_data
    beta_strs, alpha_strs = split_beta_alpha(bitstring_matrix, n_orb)

    combined_strs = np.vstack([beta_strs, alpha_strs])
    combined_probs = np.concatenate([probs, probs])

    unique_strs, inverse_str_indices = np.unique(
        combined_strs,
        axis=0,
        return_inverse=True,
    )
    unique_probs = np.zeros(len(unique_strs), dtype=float)
    np.add.at(unique_probs, inverse_str_indices, combined_probs)

    total = float(np.sum(unique_probs))
    normalized_probs = unique_probs / total if total > 0.0 else unique_probs
    return unique_strs, normalized_probs


def sort_correct_samples_by_existing_clusters(
    input_data: tuple[np.ndarray, np.ndarray],
    n_elec_per_spin: int,
    clusters: list[np.ndarray] | tuple[np.ndarray, ...],
    spin_strings: np.ndarray | None = None,
    membership: np.ndarray | None = None,
) -> Tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Postselect valid samples and map them to already-fitted clusters."""

    n_orb = input_data[0].shape[1] // 2
    if _postselect_by_hamming_right_and_left is None:
        full_bitstrings = np.asarray(input_data[0], dtype=bool)
        probs = np.asarray(input_data[1], dtype=float)
        beta_strs, alpha_strs = split_beta_alpha(full_bitstrings, n_orb)
        keep = (
            np.sum(beta_strs, axis=1) == n_elec_per_spin
        ) & (
            np.sum(alpha_strs, axis=1) == n_elec_per_spin
        )
        correct_samples_combined = full_bitstrings[keep]
        correct_probs_combined = probs[keep]
    else:
        correct_samples_combined, correct_probs_combined = (
            _postselect_by_hamming_right_and_left(
                input_data[0],
                input_data[1],
                hamming_right=n_elec_per_spin,
                hamming_left=n_elec_per_spin,
            )
        )

    correct_samples, correct_probs = pool_spin_blocks(
        n_orb,
        (correct_samples_combined, correct_probs_combined),
    )

    n_clusters = len(clusters)
    block_to_cluster_map: dict[bytes, int] = {}
    block_to_membership_map: dict[bytes, np.ndarray] = {}
    if membership is None:
        for cluster_id, cluster_data in enumerate(clusters):
            for block in cluster_data:
                block_to_cluster_map[block.tobytes()] = cluster_id
    else:
        if spin_strings is None:
            raise ValueError("spin_strings must be provided with soft membership.")
        membership = np.asarray(membership, dtype=float)
        if membership.shape != (len(spin_strings), n_clusters):
            raise ValueError("membership must have shape (n_spin_strings, n_clusters).")
        for sample, membership_row in zip(spin_strings, membership):
            block_to_membership_map[sample.tobytes()] = membership_row

    sorted_samples_list: list[list[np.ndarray]] = [[] for _ in range(n_clusters)]
    sorted_probs_list: list[list[float]] = [[] for _ in range(n_clusters)]

    for sample, prob in zip(correct_samples, correct_probs):
        sample_key = sample.tobytes()
        if membership is None:
            cluster_id = block_to_cluster_map[sample_key]
            sorted_samples_list[cluster_id].append(sample)
            sorted_probs_list[cluster_id].append(float(prob))
            continue

        membership_row = block_to_membership_map[sample_key]
        for cluster_id, responsibility in enumerate(membership_row):
            if responsibility <= 0.0:
                continue
            sorted_samples_list[cluster_id].append(sample)
            sorted_probs_list[cluster_id].append(float(prob) * float(responsibility))

    final_samples: list[np.ndarray] = []
    final_probs: list[np.ndarray] = []
    for cluster_id in range(n_clusters):
        samples = np.array(sorted_samples_list[cluster_id], dtype=bool)
        probs = np.array(sorted_probs_list[cluster_id], dtype=float)

        if len(samples) == 0:
            samples = samples.reshape(0, n_orb)
            conditional_probs = np.array([], dtype=float)
        else:
            total = float(np.sum(probs))
            conditional_probs = (
                probs / total
                if total > 0.0
                else np.full(
                    len(probs),
                    1.0 / len(probs),
                )
            )

        final_samples.append(samples)
        final_probs.append(conditional_probs)

    return tuple(final_samples), tuple(final_probs)
