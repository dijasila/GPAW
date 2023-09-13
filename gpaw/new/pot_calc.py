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
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.uniform_grid import UGArray
from gpaw.new import zips
from gpaw.new.potential import Potential
from gpaw.new.xc import Functional
from gpaw.setup import Setup
from gpaw.spinorbit import soc as soc_terms
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.utilities import pack, pack2, unpack
from gpaw.yml import indent


class PotentialCalculator:
    def __init__(self,
                 xc: Functional,
                 poisson_solver,
                 setups: list[Setup],
                 *,
                 fracpos_ac: Array2D,
                 soc: bool = False):
        self.poisson_solver = poisson_solver
        self.xc = xc
        self.setups = setups
        self.fracpos_ac = fracpos_ac
        self.soc = soc

    def __str__(self):
        return (f'{self.poisson_solver}\n'
                f'xc functional:\n{indent(self.xc)}\n')

    def calculate_pseudo_potential(self,
                                   density,
                                   ibzwfs,
                                   vHt_x: DistributedArrays | None
                                   ) -> tuple[dict[str, float],
                                              UGArray,
                                              UGArray,
                                              DistributedArrays]:
        raise NotImplementedError

    def calculate_charges(self, vHt_x):
        raise NotImplementedError

    def restrict(self, a_r, a_R=None):
        raise NotImplementedError

    def calculate(self,
                  density,
                  ibzwfs=None,
                  vHt_x: DistributedArrays | None = None,
                  ) -> tuple[Potential, AtomArrays]:
        energies, vt_sR, dedtaut_sr, vHt_x = self.calculate_pseudo_potential(
            density, ibzwfs, vHt_x)

        e_kinetic = 0.0
        for spin, (vt_R, nt_R) in enumerate(zips(vt_sR, density.nt_sR)):
            e_kinetic -= vt_R.integrate(nt_R)
            if spin < density.ndensities:
                e_kinetic += vt_R.integrate(density.nct_R)

        if dedtaut_sr is not None:
            dedtaut_sR = self.restrict(dedtaut_sr)
            for dedtaut_R, taut_R in zips(dedtaut_sR,
                                          density.taut_sR):
                e_kinetic -= dedtaut_R.integrate(taut_R)
                e_kinetic += dedtaut_R.integrate(density.tauct_R)
        else:
            dedtaut_sR = None

        energies['kinetic'] = e_kinetic

        Q_aL = self.calculate_charges(vHt_x)
        dH_asii, corrections = calculate_non_local_potential(
            self.setups, density, self.xc, Q_aL, self.soc)

        for key, e in corrections.items():
            if 0:
                print(f'{key:10} {energies[key]:15.9f} {e:15.9f}')
            energies[key] += e

        return Potential(vt_sR, dH_asii, dedtaut_sR, energies, vHt_x), Q_aL


def calculate_non_local_potential(setups,
                                  density,
                                  xc,
                                  Q_aL,
                                  soc: bool) -> tuple[AtomArrays,
                                                      dict[str, float]]:
    dtype = float if density.ncomponents < 4 else complex
    D_asii = density.D_asii.to_xp(np)
    dH_asii = D_asii.layout.new(dtype=dtype).empty(density.ncomponents)
    Q_aL = Q_aL.to_xp(np)
    energy_corrections: DefaultDict[str, float] = defaultdict(float)
    for a, D_sii in D_asii.items():
        Q_L = Q_aL[a]
        setup = setups[a]
        dH_sii, corrections = calculate_non_local_potential1(
            setup, xc, D_sii, Q_L, soc)
        dH_asii[a][:] = dH_sii
        for key, e in corrections.items():
            energy_corrections[key] += e

    # Sum over domain:
    names = ['kinetic', 'coulomb', 'zero', 'xc', 'external']
    energies = np.array([energy_corrections[name] for name in names])
    density.D_asii.layout.atomdist.comm.sum(energies)

    return (dH_asii.to_xp(density.D_asii.layout.xp),
            dict(zips(names, energies)))


def calculate_non_local_potential1(setup: Setup,
                                   xc: Functional,
                                   D_sii: Array3D,
                                   Q_L: Array1D,
                                   soc: bool) -> tuple[Array3D,
                                                       dict[str, float]]:
    ncomponents = len(D_sii)
    ndensities = 2 if ncomponents == 2 else 1
    D_sp = np.array([pack(D_ii.real) for D_ii in D_sii])

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
    e_external = 0.0
    if setup.hubbard_u is not None:
        eU, dHU_sp = setup.hubbard_u.calculate(setup, D_sp)
        e_xc += eU
        dH_sp += dHU_sp

    dH_sii = unpack(dH_sp)
    e_kinetic -= (D_sii * dH_sii).sum().real

    return dH_sii, {'kinetic': e_kinetic,
                    'coulomb': e_coulomb,
                    'zero': e_zero,
                    'xc': e_xc,
                    'external': e_external}
