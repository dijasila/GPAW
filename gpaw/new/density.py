from __future__ import annotations
import numpy as np
from gpaw.lfc import BasisFunctions
from gpaw.typing import ArrayLike1D
from gpaw.core.atom_centered_functions import AtomArraysLayout
from gpaw.utilities import unpack2


def magmoms2dims(magmoms):
    if magmoms is None:
        return 1, 0
    if magmoms.shape[1] == 1:
        return 2, 0
    return 1, 3


class Density:
    def __init__(self, density, density_matrices, core_density, core_acf):
        self.density = density
        self.density_matrices = density_matrices
        self.core_density = core_density
        self.core_acf = core_acf

    @classmethod
    def from_superposition(self,
                           layout,
                           atoms,
                           setups,
                           magmoms=None,
                           charge=0.0,
                           hund=False):

        ndens, nmag = magmoms2dims(magmoms)
        grid = layout.grid if hasattr(layout, 'grid') else layout

        basis_functions = BasisFunctions(grid._gd,
                                         [setup.phit_j for setup in setups],
                                         cut=True)
        basis_functions.set_positions(atoms.get_scaled_positions())

        if magmoms is None:
            magmoms = [None] * len(atoms)

        f_asi = {a: atomic_occupation_numbers(setup, magmom, hund,
                                              charge / len(atoms))
                 for a, (setup, magmom) in enumerate(zip(setups, magmoms))}
        density = grid.zeros(ndens + nmag)
        basis_functions.add_to_density(density.data, f_asi)

        core_acf = create_pseudo_core_densities(setups, layout,
                                                atoms.get_scaled_positions())
        core_density = grid.zeros()
        core_acf.add_to(core_density, 1.0 / ndens)
        density.data[:ndens] += core_density.data

        atom_array_layout = AtomArraysLayout([(setup.ni, setup.ni)
                                              for setup in setups],
                                             atomdist=grid.comm)
        density_matrices = atom_array_layout.empty(ndens + nmag)
        for a, D in density_matrices.items():
            D[:] = unpack2(setups[a].initialize_density_matrix(f_asi[a])).T

        return Density(density, density_matrices, core_density, core_acf)

    def from_wave_functions(self, ibz):
        ...


def create_pseudo_core_densities(setups, layout, positions):
    spline_aj = []
    for setup in setups:
        if setup.nct is None:
            spline_aj.append([])
        else:
            spline_aj.append([setup.nct])
    return layout.atom_centered_functions(spline_aj, positions)


def atomic_occupation_numbers(setup,
                              magmom: float | ArrayLike1D = None,
                              hund: bool = False,
                              charge: float = 0.0):
    if magmom is None:
        M = 0.0
        nspins = 1
    elif isinstance(magmom, float):
        M = abs(M)
        nspins = 2
    else:
        M = np.linalg.norm(magmom)
        nspins = 2

    f_si = setup.calculate_initial_occupation_numbers(
        M, hund, charge=charge, nspins=nspins)

    if magmom is None:
        pass
    elif isinstance(magmom, float):
        if magmom < 0:
            f_si = f_si[::-1].copy()
    else:
        f_i = f_si.sum(0)
        fm_i = f_si[0] - f_si[1]
        f_si = np.zeros((4, len(f_i)))
        f_si[0] = f_i
        if M > 0:
            f_si[1:] = np.asarray(magmom)[:, np.newaxis] / M * fm_i

    return f_si
