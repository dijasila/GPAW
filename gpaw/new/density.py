from __future__ import annotations
from math import sqrt, pi
import numpy as np
from gpaw.typing import ArrayLike1D
from gpaw.core.atom_centered_functions import AtomArraysLayout
from gpaw.utilities import unpack2
from typing import Union


def magmoms2dims(magmoms):
    if magmoms is None:
        return 1, 0
    if magmoms.shape[1] == 1:
        return 2, 0
    return 1, 3


class Density:
    def __init__(self, density, density_matrices, core_density, core_acf,
                 setups, charge):
        self.density = density
        self.density_matrices = density_matrices
        self.core_density = core_density
        self.core_acf = core_acf
        self.setups = setups
        self.charge = charge

        self.ndensities = {1: 1, 2: 2, 4: 1}[density.shape[0]]
        self.collinear = density.shape[0] != 4

    def calculate_compensation_charge_coefficients(self):
        coefs = AtomArraysLayout(
            [setup.Delta_iiL.shape[2] for setup in self.setups],
            atomdist=self.density_matrices.layout.atomdist).empty()

        for a, D in self.density_matrices.items():
            setup = self.setups[a]
            Q = np.einsum('ijs, ijL -> L',
                          D[:, :, :self.ndensities], setup.Delta_iiL)
            Q[0] += setup.Delta0
            coefs[a] = Q

        return coefs

    def normalize(self):
        comp_charge = self.charge
        for a, D in self.density_matrices.items():
            setup = self.setups[a]
            comp_charge += np.einsum('ijs, ij ->',
                                     D[:, :, :self.ndensities],
                                     setup.Delta_iiL[:, :, 0])
            comp_charge += setup.Delta0
        comp_charge = self.density.grid.comm.sum(comp_charge * sqrt(4 * pi))
        charge = comp_charge + self.charge
        pseudo_charge = self.density.integrate().sum()
        x = -charge / pseudo_charge
        self.density.data *= x

    def move(self, fracpos):
        self.core_acf.positions = fracpos
        core_density = self.core_acf.to_uniform_grid(1.0 / self.ndensities)
        self.density.data[:self.ndensities] += (core_density.data -
                                                self.core_density.data)
        self.core_density = core_density

    @classmethod
    def from_superposition(cls,
                           wf_grid,
                           grid,
                           setups,
                           magmoms,
                           fracpos,
                           basis_set,
                           charge=0.0,
                           hund=False):
        # density and magnitization components:
        ndens, nmag = magmoms2dims(magmoms)
        setups = setups

        if magmoms is None:
            magmoms = [None] * len(setups)
        f_asi = {a: atomic_occupation_numbers(setup, magmom, hund,
                                              charge / len(setups))
                 for a, (setup, magmom) in enumerate(zip(setups, magmoms))}

        density = grid.zeros(ndens + nmag)
        basis_set.add_to_density(density.data, f_asi)

        core_acf = setups.create_pseudo_core_densities(wf_grid, fracpos)
        core_density = core_acf.to_uniform_grid(1.0 / ndens)
        density.data[:ndens] += core_density.data

        atom_array_layout = AtomArraysLayout([(setup.ni, setup.ni)
                                              for setup in setups],
                                             atomdist=grid.comm)
        density_matrices = atom_array_layout.empty(ndens + nmag)
        for a, D in density_matrices.items():
            D[:] = unpack2(setups[a].initialize_density_matrix(f_asi[a])).T

        return cls(density, density_matrices, core_density, core_acf,
                   setups, charge)


def atomic_occupation_numbers(setup,
                              magmom: Union[float, ArrayLike1D] = None,
                              hund: bool = False,
                              charge: float = 0.0):
    if magmom is None:
        M = 0.0
        nspins = 1
    elif isinstance(magmom, float):
        M = abs(magmom)
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
