"""Quantum-chemistry setup helpers used by examples and tutorials."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pyscf
import pyscf.mcscf


def define_n2(distance: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute active-space Hamiltonian tensors for N2 at a given distance."""

    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (distance, 0, 0)]],
        basis="cc-pvdz",
        symmetry="Dooh",
    )

    n_frozen = 2
    active_space = range(n_frozen, mol.nao_nr())
    scf = pyscf.scf.RHF(mol).run()

    num_orbitals = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    num_elec_a = (n_electrons + mol.spin) // 2
    num_elec_b = (n_electrons - mol.spin) // 2

    cas = pyscf.mcscf.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
    mo = cas.sort_mo(active_space, base=0)

    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)

    return hcore, eri, nuclear_repulsion_energy
