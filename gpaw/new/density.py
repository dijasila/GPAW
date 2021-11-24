from __future__ import annotations
from math import sqrt, pi
import numpy as np
from gpaw.typing import ArrayLike1D
from gpaw.core.atom_centered_functions import AtomArraysLayout
from gpaw.utilities import unpack2
from typing import Union
from gpaw.core.atom_arrays import AtomArrays


def magmoms2dims(magmoms):
    """Convert magmoms input to number of density and magnetization components.

    >>> magmoms2dims(None)
    (1, 0)
    """
    if magmoms is None:
        return 1, 0
    if magmoms.shape[1] == 1:
        return 2, 0
    return 1, 3


class Density:
    def __init__(self,
                 nt_sR,
                 nct_R,
                 D_asii,
                 delta_aiiL,
                 delta0_a,
                 charge=0.0):
        self.nt_sR = nt_sR
        self.nct_R = nct_R
        self.D_asii = D_asii
        self.delta_aiiL = delta_aiiL
        self.delta0_a = delta0_a
        self.charge = charge

        self.ncomponents = nt_sR.dims[0]
        self.ndensities = {1: 1,
                           2: 2,
                           4: 1}[self.ncomponents]
        self.collinear = nt_sR.dims[0] != 4

    def calculate_compensation_charge_coefficients(self) -> AtomArrays:
        ccc_aL = AtomArraysLayout(
            [delta_iiL.shape[2] for delta_iiL in self.delta_aiiL],
            atomdist=self.D_asii.layout.atomdist).empty()

        for a, D_sii in self.D_asii.items():
            Q_L = np.einsum('sij, ijL -> L',
                            D_sii[:self.ndensities], self.delta_aiiL[a])
            Q_L[0] += self.delta0_a[a]
            ccc_aL[a] = Q_L

        return ccc_aL

    def normalize(self):
        comp_charge = self.charge
        for a, D_sii in self.D_asii.items():
            comp_charge += np.einsum('sij, ij ->',
                                     D_sii[:self.ndensities],
                                     self.delta_aiiL[a][:, :, 0])
            comp_charge += self.delta0_a[a]
        comp_charge = self.nt_sR.desc.comm.sum(comp_charge * sqrt(4 * pi))
        charge = comp_charge + self.charge
        pseudo_charge = self.nt_sR.integrate().sum()
        x = -charge / pseudo_charge
        self.nt_sR.data *= x

    def overlap_correction(self,
                           P_ain: AtomArrays,
                           out: AtomArrays) -> AtomArrays:
        x = (4 * np.pi)**0.5
        for a, I1, I2 in P_ain.layout.myindices:
            ds = self.delta_aiiL[a][:, :, 0] * x
            # use mmm ?????
            out.data[I1:I2] = ds @ P_ain.data[I1:I2]
        return out

    def move(self, fracpos_ac):
        self.nt_sR.data[:self.ndensities] -= self.nct_R.data
        self.nct_acf.move(fracpos_ac)
        self.nct_acf.to_uniform_grid(out=self.nct_R,
                                     scale=1.0 / self.ndensities)
        self.nt_sR.data[:self.ndensities] += self.nct_R.data

    @classmethod
    def from_superposition(cls,
                           grid,
                           nct,
                           setups,
                           basis_set,
                           magmoms=None,
                           charge=0.0,
                           hund=False):
        # density and magnitization components:
        ndens, nmag = magmoms2dims(magmoms)

        nct_R = grid.empty()
        nct.to_uniform_grid(out=nct_R, scale=1.0 / ndens)

        if magmoms is None:
            magmoms = [None] * len(setups)

        f_asi = {a: atomic_occupation_numbers(setup, magmom, hund,
                                              charge / len(setups))
                 for a, (setup, magmom) in enumerate(zip(setups, magmoms))}

        nt_sR = nct_R.desc.zeros(ndens + nmag)
        basis_set.add_to_density(nt_sR.data, f_asi)
        nt_sR.data[:ndens] += nct_R.data

        atom_array_layout = AtomArraysLayout([(setup.ni, setup.ni)
                                              for setup in setups],
                                             atomdist=nct.layout.atomdist)
        D_asii = atom_array_layout.empty(ndens + nmag)
        for a, D_sii in D_asii.items():
            D_sii[:] = unpack2(setups[a].initialize_density_matrix(f_asi[a]))

        return cls(nt_sR,
                   nct_R,
                   D_asii,
                   [setup.Delta_iiL for setup in setups],
                   [setup.Delta0 for setup in setups],
                   charge)


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
