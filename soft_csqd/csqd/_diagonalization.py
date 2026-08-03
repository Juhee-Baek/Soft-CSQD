"""Internal projected selected-CI diagonalization."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pyscf import fci

from ._types import SCIResult


def solve_sci_cluster(
    ci_strings: Tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    nuclear_repulsion_energy: float,
    membership_matrix: np.ndarray,
    spin_sq: float | None = 0.0,
) -> SCIResult:
    """Diagonalize the Hamiltonian in a fixed selected-CI subspace."""

    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)

    _, coef = fci.selected_ci.kernel_fixed_space(
        myci,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        ci_strs=ci_strings,
    )
    coef_array = np.asarray(coef)

    dm1 = myci.make_rdm1(coef_array, norb, nelec)
    dm2 = myci.make_rdm2(coef_array, norb, nelec)
    e_elec = np.einsum("pr,pr->", dm1, one_body_tensor) + 0.5 * np.einsum(
        "prqs,prqs->",
        dm2,
        two_body_tensor,
    )
    total_energy = float(e_elec + nuclear_repulsion_energy)
    if not np.isfinite(total_energy):
        raise FloatingPointError("Projected diagonalization returned non-finite energy.")

    return SCIResult(ci_strings, coef_array, total_energy, membership_matrix)
