from __future__ import annotations
from gpaw.new.wave_functions import WaveFunctions
from gpaw.setup import Setups
from gpaw.typing import Array2D


class LCAOWaveFunctions(WaveFunctions):
    def __init__(self,
                 kpt_c,
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
        super().__init__(spin, setups, fracpos_ac, weight, spin_degeneracy)
        self.kpt_c = kpt_c
        self.C_nM = C_nM
        self.T_MM = T_MM
        self.P_aMi = P_aMi
        self.domain_comm = domain_comm
        self.band_comm = C_nM.dist.comm
        self.nbands = C_nM.shape[0]
