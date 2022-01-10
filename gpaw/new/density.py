from __future__ import annotations
from math import sqrt, pi
import numpy as np
from gpaw.typing import ArrayLike1D
from gpaw.core.atom_centered_functions import AtomArraysLayout
from gpaw.utilities import unpack2, unpack
from typing import Union
from gpaw.core.atom_arrays import AtomArrays


def magmoms2dims(magmoms: np.ndarray | None) -> tuple[int, int]:
    """Convert magmoms input to number of density and magnetization components.

    >>> magmoms2dims(None)
    (1, 0)
    """
    if magmoms is None:
        return 1, 0
    if magmoms.ndim == 1:
        return 2, 0
    return 1, 3


class Density:
    def __init__(self,
                 nt_sR,
                 D_asii,
                 charge,
                 delta_aiiL,
                 delta0_a,
                 N0_aii,
                 l_aj):
        self.nt_sR = nt_sR
        self.D_asii = D_asii
        self.delta_aiiL = delta_aiiL
        self.delta0_a = delta0_a
        self.N0_aii = N0_aii
        self.l_aj = l_aj
        self.charge = charge

        self.ncomponents = nt_sR.dims[0]
        self.ndensities = {1: 1,
                           2: 2,
                           4: 1}[self.ncomponents]
        self.collinear = self.ncomponents != 4
        self.natoms = len(delta0_a)

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

    def update(self, nct_R, ibzwfs):
        self.nt_sR.data[:] = 0.0
        self.D_asii.data[:] = 0.0
        ibzwfs.add_to_density(self.nt_sR, self.D_asii)
        self.nt_sR.data[:] += nct_R.data
        self.symmetrize(ibzwfs.ibz.symmetries)

    def symmetrize(self, symmetries):
        self.nt_sR.symmetrize(symmetries.rotation_scc,
                              symmetries.translation_sc)

        D_asii = self.D_asii.gather(broadcast=True, copy=True)
        for a1, D_sii in self.D_asii.items():
            D_sii[:] = 0.0
            for a2, rotation_ii in zip(symmetries.a_sa[:, a1],
                                       symmetries.rotations(self.l_aj[a1])):
                D_sii += np.einsum('ij, sjk, lk -> sil',
                                   rotation_ii, D_asii[a2], rotation_ii)
        self.D_asii.data *= 1.0 / len(symmetries)

    def overlap_correction(self,
                           P_ain: AtomArrays,
                           out: AtomArrays) -> AtomArrays:
        x = (4 * np.pi)**0.5
        for a, I1, I2 in P_ain.layout.myindices:
            ds = self.delta_aiiL[a][:, :, 0] * x
            # use mmm ?????
            out.data[I1:I2] = ds @ P_ain.data[I1:I2]
        return out

    def move(self, delta_nct_R):
        self.nt_sR.data[:self.ndensities] += delta_nct_R.data

    @classmethod
    def from_superposition(cls,
                           grid,
                           nct_R,
                           atomdist,
                           setups,
                           basis_set,
                           magmoms=None,
                           charge=0.0,
                           hund=False):
        # density and magnitization components:
        ndens, nmag = magmoms2dims(magmoms)

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
                                             atomdist=atomdist)
        D_asii = atom_array_layout.empty(ndens + nmag)
        for a, D_sii in D_asii.items():
            D_sii[:] = unpack2(setups[a].initialize_density_matrix(f_asi[a]))

        return cls.from_data_and_setups(nt_sR,
                                        D_asii,
                                        charge,
                                        setups)

    @classmethod
    def from_data_and_setups(cls,
                             nt_sR,
                             D_asii,
                             charge,
                             setups):
        return cls(nt_sR,
                   D_asii,
                   charge,
                   [setup.Delta_iiL for setup in setups],
                   [setup.Delta0 for setup in setups],
                   [unpack(setup.N0_p) for setup in setups],
                   [setup.l_j for setup in setups])

    def calculate_magnetic_moments(self):
        magmom_av = np.zeros((self.natoms, 3))
        magmom_v = np.zeros(3)
        domain_comm = self.nt_sR.desc.comm

        if self.ncomponents == 2:
            for a, D_sii in self.D_asii.items():
                M_ii = D_sii[0] - D_sii[1]
                magmom_av[a, 2] = np.einsum('ij, ij ->', M_ii, self.N0_aii[a])
                magmom_v[2] += (np.einsum('ij, ij ->', M_ii,
                                          self.delta_aiiL[a][:, :, 0]) *
                                sqrt(4 * pi))
            domain_comm.sum(magmom_av)
            domain_comm.sum(magmom_v)

            M_s = self.nt_sR.integrate()
            magmom_v[2] += M_s[0] - M_s[1]

        elif self.ncomponents == 4:
            for a, D_sii in self.D_asii.items():
                M_vii = D_sii[1:4]
                magmom_av[a] = np.einsum('vij, ij -> v',
                                         M_vii, self.N0_aii[a])
                magmom_v += (np.einsum('vij, ij ->', M_vii,
                                       self.delta_aiiL[a][:, :, 0]) *
                             sqrt(4 * pi))
            domain_comm.sum(magmom_av)
            domain_comm.sum(magmom_v)

            magmom_v += self.nt_sR.integrate()[1:]

        return magmom_v, magmom_av


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
        M = np.linalg.norm(magmom)  # type: ignore
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
