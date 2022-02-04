from __future__ import annotations

import numpy as np
from gpaw.core.atom_arrays import AtomArrays
from gpaw.setup import Setups
from gpaw.typing import Array1D
from gpaw.mpi import MPIComm, serial_comm


class WaveFunctions:
    def __init__(self,
                 spin: int | None,
                 setups: Setups,
                 nbands: int,
                 weight: float = 1.0,
                 spin_degeneracy: int = 2,
                 dtype=float,
                 domain_comm: MPIComm = serial_comm,
                 band_comm: MPIComm = serial_comm):
        self.spin = spin
        self.setups = setups
        self.weight = weight
        self.spin_degeneracy = spin_degeneracy
        self.dtype = dtype
        self.domain_comm = domain_comm
        self.band_comm = band_comm

        self._P_ain = None

        self._eig_n: Array1D | None = None
        self._occ_n: Array1D | None = None

    @property
    def eig_n(self) -> Array1D:
        if self._eig_n is None:
            raise ValueError
        return self._eig_n

    @property
    def occ_n(self) -> Array1D:
        if self._occ_n is None:
            raise ValueError
        return self._occ_n

    @property
    def myeig_n(self):
        assert self.band_comm.size == 1
        return self.eig_n

    @property
    def myocc_n(self):
        assert self.band_comm.size == 1
        return self.occ_n

    @property
    def P_ain(self) -> AtomArrays:
        return self._P_ain

    def add_to_atomic_density_matrices(self,
                                       occ_n,
                                       D_asii: AtomArrays) -> None:
        for D_sii, P_in in zip(D_asii.values(), self.P_ain.values()):
            D_sii[self.spin] += np.einsum('in, n, jn -> ij',
                                          P_in.conj(), occ_n, P_in).real
