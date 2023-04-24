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
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.new import zip
from gpaw.new.potential import Potential
from gpaw.new.xc import XCFunctional
from gpaw.setup import Setup
from gpaw.spinorbit import soc as soc_terms
from gpaw.typing import Array1D, Array2D, Array3D
from gpaw.utilities import pack, pack2, unpack
from gpaw.yml import indent
from gpaw.mpi import serial_comm


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

    def calculate_pseudo_potential(self,
                                   density,
                                   vHt_x: DistributedArrays | None
                                   ) -> tuple[dict[str, float],
                                              UniformGridFunctions,
                                              DistributedArrays]:
        raise NotImplementedError

    def calculate_charges(self, vHt_x):
        raise NotImplementedError

    def calculate(self,
                  density,
                  vHt_x: DistributedArrays | None = None,
                  kpt_comm=serial_comm
                  ) -> tuple[Potential, DistributedArrays, AtomArrays]:
        energies, vt_sR, vHt_x = self.calculate_pseudo_potential(
            density, vHt_x)

        Q_aL = self.calculate_charges(vHt_x)
        dH_asii, corrections = calculate_non_local_potential(
            self.setups, density, self.xc, Q_aL, self.soc, kpt_comm)

        for key, e in corrections.items():
            # print(f'{key:10} {energies[key]:15.9f} {e:15.9f}')
            energies[key] += e

        return Potential(vt_sR, dH_asii, energies), vHt_x, Q_aL

    def move(self, fracpos_ac, atomdist, ndensities) -> UniformGridFunctions:
        """Move things and return change in pseudo core density."""
        delta_nct_R = self.nct_R.new()
        delta_nct_R.data[:] = self.nct_R.data
        delta_nct_R.data *= -1
        self._move(fracpos_ac, atomdist, ndensities)
        delta_nct_R.data += self.nct_R.data
        return delta_nct_R

    def _move(self, fracpos_ac, atomdist, ndensities) -> None:
        raise NotImplementedError


def calculate_non_local_potential(setups,
                                  density,
                                  xc,
                                  Q_aL,
                                  soc: bool,
                                  comm) -> tuple[AtomArrays,
                                                 dict[str, float]]:
    dtype = float if density.ncomponents < 4 else complex
    D_asii = density.D_asii.to_xp(np)
    dH_asii = D_asii.layout.new(dtype=dtype).empty(density.ncomponents)
    Q_aL = Q_aL.to_xp(np)
    energy_corrections: DefaultDict[str, float] = defaultdict(float)
    for a, D_sii in D_asii.items():
        if a % comm.size != comm.rank:
            dH_asii[a][:] = 0.0
            continue
        Q_L = Q_aL[a]
        setup = setups[a]
        dH_sii, corrections = calculate_non_local_potential1(
            setup, xc, D_sii, Q_L, soc)
        dH_asii[a][:] = dH_sii
        for key, e in corrections.items():
            energy_corrections[key] += e

    comm.sum(dH_asii.data)
    # Sum over domain:
    names = ['kinetic', 'coulomb', 'zero', 'xc', 'external']
    energies = np.array([energy_corrections[name] for name in names])
    density.D_asii.layout.atomdist.comm.sum(energies)
    comm.sum(energies)

    return (dH_asii.to_xp(density.D_asii.layout.xp),
            {name: e for name, e in zip(names, energies)})


def calculate_non_local_potential1(setup: Setup,
                                   xc: XCFunctional,
                                   D_sii: Array3D,
                                   Q_L: Array1D,
                                   soc: bool) -> tuple[Array3D,
                                                       dict[str, float]]:
    ncomponents = len(D_sii)
    ndensities = 2 if ncomponents == 2 else 1

    D_ii = D_sii[:ndensities].sum(0)

    K_ii     = unpack(setup.K_p)
    M_ii     = unpack(setup.M_p)
    MB_ii    = unpack(setup.MB_p)
    Delta_ii = unpack(np.dot(setup.Delta_pL, Q_L))
    M_pii_D_ii = np.zeros_like(setup.K_p, dtype=complex)
    for p_index in range(M_pii_D_ii.size):
        M_pii_D_ii[p_index] = np.sum(unpack(setup.M_pp[p_index, :])*D_ii)
    M_iiii_D_ii = unpack(M_pii_D_ii)

    dH_ii = (K_ii + M_ii + MB_ii + Delta_ii +
            2.0 * M_iiii_D_ii)
    e_kinetic = np.sum(K_ii * D_ii).real + setup.Kc
    e_zero = setup.MB + np.sum(MB_ii * D_ii).real
    e_coulomb = setup.M + np.sum(M_ii * D_ii).real +  np.sum(D_ii.conj() * M_iiii_D_ii).real
    
    dH_sii = np.zeros_like(D_sii, dtype=float if ncomponents < 4 else complex)
    if soc:
        dH_sii[1:4] = soc_terms(setup, xc.xc, D_sii)
    dH_sii[:ndensities] = dH_ii
    e_xc = xc.calculate_paw_correction(setup, D_sii, dH_sii)
    e_kinetic -= (D_sii * dH_sii).sum().real
    print([(D_sii.real * dH_sii.real).sum(), (D_sii.real * dH_sii).sum(), (D_sii * dH_sii.real).sum(), (D_sii * dH_sii).sum()])

    crash

    e_external = 0.0

    return dH_sii, {'kinetic': e_kinetic,
                    'coulomb': e_coulomb,
                    'zero': e_zero,
                    'xc': e_xc,
                    'external': e_external}
