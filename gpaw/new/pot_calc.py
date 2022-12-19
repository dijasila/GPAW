"""
==  ==========
R
r
G
g
h
x   r or h
==  ==========

"""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

import numpy as np
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.new.potential import Potential
from gpaw.new.xc import XCFunctional
from gpaw.setup import Setup
from gpaw.spinorbit import soc as soc_terms
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.utilities import pack, pack2, unpack
from gpaw.yml import indent


class PotentialCalculator:
    def __init__(self,
                 xc: XCFunctional,
                 poisson_solver,
                 setups: list[Setup],
                 nct_R: UniformGridFunctions,
                 fracpos_ac: Array2D,
                 soc: bool = False):
        self.poisson_solver = poisson_solver
        self.xc = xc
        self.setups = setups
        self.nct_R = nct_R
        self.fracpos_ac = fracpos_ac
        self.soc = soc

    def __str__(self):
        return (f'{self.poisson_solver}\n'
                f'xc functional:\n{indent(self.xc)}\n')

    def calculate(self, density, vHt_x=None):
        energies, vt_sR, vHt_x = self._calculate(density, vHt_x)

        Q_aL = self.calculate_charges(vHt_x)
        dH_asii, corrections = calculate_non_local_potential(
            self.setups, density, self.xc, Q_aL, self.soc)

        for key, e in corrections.items():
            print(f'{key:10} {e:15.9f} {energies[key]:15.9f}')
            energies[key] += e

        return Potential(vt_sR, dH_asii, energies), vHt_x, Q_aL

    def move(self, fracpos_ac, atomdist, ndensities):
        delta_nct_R = self.nct_R.new()
        delta_nct_R.data = -self.nct_R.data
        self._move(fracpos_ac, atomdist, ndensities)
        delta_nct_R.data += self.nct_R.data
        return delta_nct_R


def calculate_non_local_potential(setups,
                                  density,
                                  xc,
                                  Q_aL,
                                  soc: bool):
    dtype = float if density.ncomponents < 4 else complex
    dH_asii = density.D_asii.layout.new(dtype=dtype).empty(density.ncomponents)
    energy_corrections: DefaultDict[str, float] = defaultdict(float)
    for a, D_sii in density.D_asii.items():
        Q_L = Q_aL[a]
        setup = setups[a]
        dH_sii, corrections = calculate_non_local_potential1(
            setup, xc, D_sii, Q_L, soc)
        dH_asii[a][:] = dH_sii
        for key, e in corrections.items():
            energy_corrections[key] += e

    # Sum over domain:
    names = ['kinetic',
             'coulomb',
             'zero',
             'xc',
             'external']
    energies = np.array([energy_corrections[name] for name in names])
    density.D_asii.layout.atomdist.comm.sum(energies)
    return dH_asii, {name: e for name, e in zip(names, energies)}


def calculate_non_local_potential1(setup: Setup,
                                   xc: XCFunctional,
                                   D_sii: Array3D,
                                   Q_L: Array1D,
                                   soc: bool) -> tuple[Array3D,
                                                       dict[str, float]]:
    ncomponents = len(D_sii)
    ndensities = 2 if ncomponents == 2 else 1
    D_sp = np.array([pack(D_ii) for D_ii in D_sii])

    D_p = D_sp[:ndensities].sum(0)

    dH_p = (setup.K_p + setup.M_p +
            setup.MB_p + 2.0 * setup.M_pp @ D_p +
            setup.Delta_pL @ Q_L)
    e_kinetic = setup.K_p @ D_p + setup.Kc
    e_zero = setup.MB + setup.MB_p @ D_p
    e_coulomb = setup.M + D_p @ (setup.M_p + setup.M_pp @ D_p)

    dH_sp = np.zeros_like(D_sp, dtype=float if ncomponents < 4 else complex)
    if soc:
        dH_sp[1:4] = pack2(soc_terms(setup, xc.xc, D_sp))
    dH_sp[:ndensities] = dH_p
    e_xc = xc.calculate_paw_correction(setup, D_sp, dH_sp)
    e_kinetic -= (D_sp * dH_sp).sum().real
    e_external = 0.0

    dH_sii = unpack(dH_sp)

    return dH_sii, {'kinetic': e_kinetic,
                    'coulomb': e_coulomb,
                    'zero': e_zero,
                    'xc': e_xc,
                    'external': e_external}
