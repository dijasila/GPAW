from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.units import Bohr
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.setup import Setups
from gpaw.spline import Spline
from gpaw.typing import Array3D, ArrayLike2D, Vector
from gpaw.spherical_harmonics import Y

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
                         grid_refinement: int = None,
                         ) -> UniformGridFunctions:
        nt_sR = self._pseudo_densities(grid_spacing, grid_refinement)
        return nt_sR.scaled(Bohr, Bohr**-3)

    def _pseudo_densities(self,
                          grid_spacing: float = None,  # Ang
                          grid_refinement: int = None,
                          ) -> UniformGridFunctions:
        nt_sR = self.nt_sR.to_pbc_grid()
        grid = nt_sR.desc
        if grid_spacing is not None:
            assert grid_refinement is None
            grid = grid.uniform_grid_with_grid_spacing(
                grid_spacing / Bohr)
        elif grid_refinement is not None and grid_refinement > 1:
            grid = grid.new(size=grid.size * grid_refinement)
        else:
            return nt_sR.copy()

        return nt_sR.interpolate(grid=grid)

    def all_electron_densities(self,
                               *,
                               grid_spacing: float = None,  # Ang
                               grid_refinement: int = None,
                               ) -> UniformGridFunctions:
        n_sR = self._pseudo_densities(grid_spacing, grid_refinement)

        splines = {}
        for R_v, setup, D_sii in zip(self.fracpos_ac @ n_sR.desc.cell_cv,
                                     self.setups,
                                     self.D_asii.values()):
            if setup not in splines:
                phi_j, phit_j, nc, nct = setup.get_partial_waves()[:4]
                rcut = max(setup.rcut_j)
                splines[setup] = (rcut, phi_j, phit_j, nc, nct)
            rcut, phi_j, phit_j, nc, nct = splines[setup]

            add(R_v, n_sR, phi_j, phit_j, nc, nct, rcut, D_sii)

        return n_sR.scaled(Bohr, Bohr**-3)


def add(R_v: Vector,
        a_sR: UniformGridFunctions,
        phi_j: list[Spline],
        phit_j: list[Spline], nc, nct,
        rcut: float,
        D_sii: Array3D):
    ug = a_sR.desc
    R_Rv = ug.xyz()
    lmax = max(phi.l for phi in phi_j)
    start_c = 0 - ug.pbc
    stop_c = 1 + ug.pbc
    for u0 in range(start_c[0], stop_c[0]):
        for u1 in range(start_c[1], stop_c[1]):
            for u2 in range(start_c[2], stop_c[2]):
                d_Rv = R_Rv - (R_v + (u0, u1, u2) @ ug.cell_cv)
                d_R = (d_Rv**2).sum(3)**0.5
                mask_R = d_R < rcut
                if not mask_R.any():
                    continue
                d_rv = d_Rv[mask_R]
                d_r = d_R[mask_R]
                Y_Lr = [Y(L, *d_rv.T) for L in range((lmax + 1)**2)]
                phi_jr = [phi.map(d_r) for phi in phi_j]
                phit_jr = [phit.map(d_r) for phit in phit_j]
                l_j = [phi.l for phi in phi_j]
                i1 = 0
                for l1, phi1_r, phit1_r in zip(l_j, phi_jr, phit_jr):
                    i2 = 0
                    i1b = i1 + 2 * l1 + 1
                    D_smi = D_sii[:, i1:i1b]
                    for l2, phi2_r, phit2_r in zip(l_j, phi_jr, phit_jr):
                        i2b = i2 + 2 * l2 + 1
                        D_smm = D_smi[:, :, i2:i2b]
                        a_sr = np.einsum(
                            'smn, mr, nr -> sr',
                            D_smm,
                            Y_Lr[l1**2:(l1 + 1)**2],
                            Y_Lr[l2**2:(l2 + 1)**2]) * (
                            phi1_r * phi2_r - phit1_r * phit2_r)
                        a_sR.data[:, mask_R] += a_sr
                        i2 = i2b
                    i1 = i1b
                dn_r = nc.map(d_r) - nct.map(d_r)
                a_sR.data[:, mask_R] += dn_r * (4 * np.pi)**-0.5
