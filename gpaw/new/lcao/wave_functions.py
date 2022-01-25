from __future__ import annotations
from gpaw.new.wave_functions import WaveFunctions
from gpaw.core.atom_arrays import AtomArrays
from gpaw.setup import Setups
from gpaw.typing import Array2D


class LCAOWaveFunctions(WaveFunctions):
    def __init__(self,
                 basis,
                 nbands,
                 band_comm,
                 spin: int | None,
                 setups: Setups,
                 fracpos_ac: Array2D,
                 weight: float = 1.0,
                 spin_degeneracy: int = 2):
        super().__init__(spin, setups, fracpos_ac, weight, spin_degeneracy)
        self.basis = basis
        self.domain_comm = basis.grid.comm
        self.band_comm = band_comm
        self.nbands = nbands
        self.coef_ani = basis.empty(nbands, comm=band_comm)
