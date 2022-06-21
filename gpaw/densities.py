from __future__ import annotations

from typing import TYPE_CHECKING

from ase.units import Bohr
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.setup import Setups
from gpaw.typing import ArrayLike2D

if TYPE_CHECKING:
    from gpaw.new.calculation import DFTCalculation


class Densities:
    def __init__(self,
                 nt_sR: UniformGridFunctions,
                 D_asii: AtomArrays,
                 fracpos_ac: ArrayLike2D,
                 setups: Setups):
        self.nt_sR = nt_sR
        self.D_asii = D_asii
        self.fracpos_ac = fracpos_ac
        self.setups = setups

    @classmethod
    def from_calculation(cls, calculation: DFTCalculation):
        density = calculation.state.density
        return cls(density.nt_sR,
                   density.D_asii,
                   calculation.fracpos_ac,
                   calculation.setups)

    def pseudo_densities(self,
                         grid_spacing: float = None,  # Ang
                         ) -> UniformGridFunctions:
        nt_sR = self.nt_sR.to_pbc_grid()
        if grid_spacing is not None:
            grid = nt_sR.desc.uniform_grid_with_grid_spacing(
                grid_spacing / Bohr)
            nt_sR = nt_sR.interpolate(grid=grid)
        return nt_sR.scaled(Bohr, Bohr**-3)
