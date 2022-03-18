from __future__ import annotations

import numpy as np
from ase.io.ulm import Writer
from gpaw.core.atom_arrays import AtomArrays
from gpaw.setup import Setups
from gpaw.typing import Array1D
from gpaw.mpi import MPIComm, serial_comm


class WaveFunctions:
    def __init__(self,
                 setups: Setups,
                 nbands: int,
                 spin: int = 0,
                 q: int = 0,
                 k: int = 0,
                 kpt_c=(0.0, 0.0, 0.0),
                 weight: float = 1.0,
                 ncomponents: int = 1,
                 dtype=float,
                 domain_comm: MPIComm = serial_comm,
                 band_comm: MPIComm = serial_comm):
        """"""
        assert spin < ncomponents

        self.spin = spin
        self.q = q
        self.k = k
        self.setups = setups
        self.weight = weight
        self.ncomponents = ncomponents
        self.dtype = dtype
        self.kpt_c = kpt_c
        self.domain_comm = domain_comm
        self.band_comm = band_comm
        self.nbands = nbands

        self.nspins = ncomponents % 3
        self.spin_degeneracy = ncomponents % 2 + 1

        self._P_ain: AtomArrays | None = None

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
        assert self._P_ain is not None
        return self._P_ain

    def add_to_atomic_density_matrices(self,
                                       occ_n,
                                       D_asii: AtomArrays) -> None:
        for D_sii, P_in in zip(D_asii.values(), self.P_ain.values()):
            D_sii[self.spin] += np.einsum('in, n, jn -> ij',
                                          P_in.conj(), occ_n, P_in).real

    def add_wave_functions_array(self,
                                 writer: Writer,
                                 spin_k_shape: tuple[int, int]):
        """ Write the array header for the wave functions

        Parameters
        ----------
        writer:
            Ulm writer

        spin_k_shape:
            Shape of the spin and k-point dimensions
        """
        raise NotImplementedError

    def fill_wave_functions(self, writer: Writer):
        """ Fill the wave function array using this wave function

        Parameters
        ----------
        writer:
            Ulm writer
        """
        raise NotImplementedError

    def receive(self, kpt_comm, rank):
        raise NotImplementedError
