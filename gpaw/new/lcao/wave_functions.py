from __future__ import annotations
import numpy as np
from ase.io.ulm import Writer
from ase.units import Bohr
from gpaw.core.atom_arrays import AtomArrays, AtomArraysLayout
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups


class LCAOWaveFunctions(WaveFunctions):
    def __init__(self,
                 *,
                 setups: Setups,
                 density_adder,
                 C_nM,
                 S_MM,
                 T_MM,
                 P_aMi,
                 kpt_c=(0.0, 0.0, 0.0),
                 domain_comm: MPIComm = serial_comm,
                 spin: int = 0,
                 q=0,
                 k=0,
                 weight: float = 1.0,
                 ncomponents: int = 1):
        super().__init__(setups,
                         nbands=C_nM.shape[0],
                         spin=spin,
                         q=q,
                         k=k,
                         kpt_c=kpt_c,
                         weight=weight,
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
        self.V_MM: np.ndarray

    @property
    def P_ain(self):
        if self._P_ain is None:
            layout = AtomArraysLayout([P_Mi.shape[1]
                                       for P_Mi in self.P_aMi.values()],
                                      dtype=self.dtype)
            self._P_ain = layout.empty(self.nbands, transposed=True)
            for a, P_Mi in self.P_aMi.items():
                self._P_ain[a][:] = (self.C_nM.data @ P_Mi).T
        return self._P_ain

    def add_to_density(self,
                       nt_sR,
                       D_asii: AtomArrays) -> None:
        """ Compute density from wave functions and add to nt_sR and D_asii """
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n
        C_nM = self.C_nM.data
        rho_MM = (C_nM.T.conj() * occ_n) @ C_nM
        self.density_adder(rho_MM, nt_sR.data[self.spin])
        self.add_to_atomic_density_matrices(occ_n, D_asii)

    def add_wave_functions_array(self,
                                 writer: Writer,
                                 spin_k_shape: tuple[int, int]):
        shape = spin_k_shape + self.C_nM.shape
        if self.domain_comm.rank == 0 and self.band_comm.rank == 0:
            writer.add_array('coefficients', shape, dtype=self.dtype)

    def fill_wave_functions(self, writer: Writer):
        C_nM = self.C_nM.gather()
        if self.domain_comm.rank == 0 and self.band_comm.rank == 0:
            writer.fill(C_nM.data * Bohr**-1.5)

    def calculate_density_matrix(self) -> np.ndarray:
        """ Calculate the density matrix

        The density matrix is

                 *
        ρ   = Σ C  C   f
         μν   n  nμ nν  n

        Returns
        -------
        The density matrix in the LCAO basis
        """
        f_n = self.weight * self.spin_degeneracy * self.myocc_n
        C_nM = self.C_nM.data
        rho_MM = (C_nM.T.conj() * f_n) @ C_nM

        return rho_MM
