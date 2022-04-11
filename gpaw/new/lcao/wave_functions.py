from __future__ import annotations

from typing import Callable

import numpy as np
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   AtomDistribution)
from gpaw.core.matrix import Matrix
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import cached_property
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups
from gpaw.typing import Array2D, Array3D


class LCAOWaveFunctions(WaveFunctions):
    def __init__(self,
                 *,
                 setups: Setups,
                 density_adder: Callable[[Array2D, Array3D], None],
                 C_nM: Matrix,
                 S_MM: Matrix,
                 T_MM: Array2D,
                 P_aMi,
                 fracpos_ac: Array2D,
                 atomdist: AtomDistribution,
                 kpt_c=(0.0, 0.0, 0.0),
                 domain_comm: MPIComm = serial_comm,
                 spin: int = 0,
                 q: int = 0,
                 k: int = 0,
                 weight: float = 1.0,
                 ncomponents: int = 1):
        super().__init__(setups=setups,
                         nbands=C_nM.shape[0],
                         spin=spin,
                         q=q,
                         k=k,
                         kpt_c=kpt_c,
                         weight=weight,
                         fracpos_ac=fracpos_ac,
                         atomdist=atomdist,
                         ncomponents=ncomponents,
                         dtype=C_nM.dtype,
                         domain_comm=domain_comm,
                         band_comm=C_nM.dist.comm)
        self.density_adder = density_adder
        self.C_nM = C_nM
        self.T_MM = T_MM
        self.S_MM = S_MM
        self.P_aMi = P_aMi

        # This is for TB-mode (and MYPY):
        self.V_MM: Matrix

    @cached_property
    def L_MM(self):
        S_MM = self.S_MM.copy()
        S_MM.invcholesky()
        return S_MM

    def array_shape(self, global_shape=False):
        if global_shape:
            return self.C_nM.shape[1:]
        1 / 0

    @property
    def P_ain(self):
        if self._P_ain is None:
            atomdist = AtomDistribution.from_atom_indices(
                list(self.P_aMi),
                self.domain_comm,
                natoms=len(self.setups))
            layout = AtomArraysLayout([setup.ni for setup in self.setups],
                                      atomdist=atomdist,
                                      dtype=self.dtype)
            self._P_ain = layout.empty(self.nbands,
                                       comm=self.C_nM.dist.comm,
                                       transposed=True)
            for a, P_Mi in self.P_aMi.items():
                self._P_ain[a][:] = (self.C_nM.data @ P_Mi).T
        return self._P_ain

    def add_to_density(self,
                       nt_sR,
                       D_asii: AtomArrays) -> None:
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n
        C_nM = self.C_nM.data
        rho_MM = (C_nM.T.conj() * occ_n) @ C_nM
        self.density_adder(rho_MM, nt_sR.data[self.spin])
        self.add_to_atomic_density_matrices(occ_n, D_asii)

    def gather_wave_function_coefficients(self) -> np.ndarray:
        C_nM = self.C_nM.gather()
        if C_nM is not None:
            return C_nM.data
        return None

    def to_uniform_grid_wave_functions(self,
                                       grid,
                                       basis):
        grid = grid.new(kpt=self.kpt_c, dtype=self.dtype)
        psit_nR = grid.zeros(self.nbands, self.band_comm)
        basis.lcao_to_grid(self.C_nM.data, psit_nR.data, self.q)

        return PWFDWaveFunctions(
            psit_nR,
            self.spin,
            self.q,
            self.k,
            self.setups,
            self.fracpos_ac,
            self.atomdist,
            self.weight,
            self.ncomponents)
