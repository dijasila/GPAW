from __future__ import annotations
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups
from gpaw.typing import Array2D
from gpaw.core.atom_arrays import AtomArrays, AtomArraysLayout


class LCAOWaveFunctions(WaveFunctions):
    def __init__(self,
                 kpt_c,
                 density_adder,
                 C_nM,
                 S_MM,
                 T_MM,
                 P_aMi,
                 domain_comm,
                 spin: int | None,
                 setups: Setups,
                 fracpos_ac: Array2D,
                 weight: float = 1.0,
                 spin_degeneracy: int = 2):
        super().__init__(spin, setups, fracpos_ac, weight, spin_degeneracy,
                         dtype=C_nM.dtype)
        self.kpt_c = kpt_c
        self.density_adder = density_adder
        self.C_nM = C_nM
        self.T_MM = T_MM
        self.S_MM = S_MM
        self.P_aMi = P_aMi
        self.domain_comm = domain_comm
        self.band_comm = C_nM.dist.comm
        self.nbands = C_nM.shape[0]

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
        occ_n = self.weight * self.spin_degeneracy * self.myocc_n
        C_nM = self.C_nM.data
        rho_MM = (C_nM.T.conj() * occ_n) @ C_nM
        self.density_adder(rho_MM, nt_sR.data[self.spin])
        self.add_to_atomic_density_matrices(occ_n, D_asii)
