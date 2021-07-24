import functools
from collections import defaultdict
from math import pi
from typing import List

import numpy as np
from ase.units import Bohr, Ha

from gpaw import GPAW
from gpaw.grid_descriptor import GridDescriptor
from gpaw.typing import Array1D, Array2D, Array4D
from gpaw.hyperfine import alpha  # fine-structure constant: ~ 1 / 137
from gpaw.projections import Projections
from gpaw.setup import Setup
from gpaw.utilities.ps2ae import PS2AE
from gpaw.wavefunctions.pw import PWLFC, PWDescriptor
from .paw import coulomb, coulomb_integrals


def zfs(calc: GPAW,
        method: int = 1,
        with_paw_correction: bool = True,
        grid_spacing: float = -1.0) -> Array2D:
    """Zero-field splitting.

    Calculate magnetic dipole-coupling tensor in eV.
    """
    compensation_charge = None

    if grid_spacing == -1.0:
        wf1, wf2 = (WaveFunctions.from_calc(calc, spin)
                    for spin in [0, 1])
        compensation_charge = create_compensation_charge(wf1, calc.spos_ac)
    else:
        with_paw_correction = False
        converter = PS2AE(calc, grid_spacing)
        wf1, wf2 = (WaveFunctions.from_calc_ae(calc, spin, converter)
                    for spin in [0, 1])

    if method == 1:
        n1 = len(wf1)
        wf1 = wf1.view(n1 - 2, n1)
        wf2 = wf2.view(0, 0)

    D_vv = np.zeros((3, 3))
    for wfa in [wf1, wf2]:
        for wfb in [wf1, wf2]:
            d_vv = zfs1(wfa, wfb, compensation_charge, with_paw_correction)
            D_vv += d_vv

    return D_vv


class WaveFunctions:
    def __init__(self,
                 psit_nR: Array4D,
                 projections: Projections,
                 spin: int,
                 setups: List[Setup],
                 gd: GridDescriptor = None,
                 pd: PWDescriptor = None):
        """Container for wave functions in real-space and projections."""
        self.pd = pd or PWDescriptor(ecut=None, gd=gd)
        self.psit_nR = psit_nR
        self.projections = projections
        self.spin = spin
        self.setups = setups

    def __len__(self) -> int:
        return len(self.psit_nR)

    def view(self, n1: int, n2: int) -> 'WaveFunctions':
        """Create WaveFuntions object with view of data."""
        return WaveFunctions(self.psit_nR[n1:n2],
                             self.projections.view(n1, n2),
                             self.spin,
                             self.setups,
                             pd=self.pd)

    @staticmethod
    def from_calc(calc: GPAW,
                  spin: int) -> 'WaveFunctions':
        """Create WaveFunctions object from GPAW object."""
        kpt = calc.wfs.kpt_qs[0][spin]
        nocc = (kpt.f_n > 0.5).sum()
        gd = calc.wfs.gd.new_descriptor(pbc_c=np.ones(3, bool))
        psit_nR = gd.empty(nocc)
        for band, psit_R in enumerate(psit_nR):
            psit_R[:] = calc.get_pseudo_wave_function(band,
                                                      spin=spin,
                                                      pad=True) * Bohr**1.5
        return WaveFunctions(psit_nR,
                             kpt.projections.view(0, nocc),
                             spin,
                             calc.setups,
                             gd=gd)

    @staticmethod
    def from_calc_ae(calc: GPAW,
                     spin: int,
                     converter: PS2AE) -> 'WaveFunctions':
        """Create all-electron WaveFunctions object from GPAW object.

        The wave-functions are interpolated to a fine grid with
        PAW-corrections added.
        """
        kpt = calc.wfs.kpt_qs[0][spin]
        nocc = (kpt.f_n > 0.5).sum()
        psit_nR = converter.gd.empty(nocc)
        for band, psit_R in enumerate(psit_nR):
            psit_R[:] = converter.get_wave_function(n=band,
                                                    s=spin,
                                                    ae=True) * Bohr**1.5
        projections = Projections(nocc, [])
        return WaveFunctions(psit_nR,
                             projections,
                             spin,
                             calc.setups,
                             converter.gd)


def create_compensation_charge(wf: WaveFunctions,
                               spos_ac: Array2D) -> PWLFC:
    compensation_charge = PWLFC([data.ghat_l for data in wf.setups], wf.pd)
    compensation_charge.set_positions(spos_ac)
    return compensation_charge


def zfs1(wf1: WaveFunctions,
         wf2: WaveFunctions,
         compensation_charge: PWLFC = None,
         with_paw_correction: bool = False) -> Array2D:
    """Compute dipole coupling."""
    sign = +1 if wf1.spin == wf2.spin else -1

    pd = wf1.pd
    setups = wf1.setups
    N2 = len(wf2)

    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0
    G_Gv = pd.get_reciprocal_vectors() / G_G[:, np.newaxis]

    n_sG = pd.zeros(2)
    D_asii = defaultdict(list)
    for n_G, wf in zip(n_sG, [wf1, wf2]):
        for psit_R in wf.psit_nR:
            n_G += pd.fft(psit_R**2)

        Q_aL = {}
        for a, P_ni in wf.projections.items():
            D_ii = np.einsum('ni, nj -> ij', P_ni, P_ni)
            D_asii[a].append(D_ii)
            Q_aL[a] = np.einsum('ij, ijL -> L', D_ii, setups[a].Delta_iiL)

        if compensation_charge:
            compensation_charge.add(n_G, Q_aL)

    nn_G = (n_sG[0] * n_sG[1].conj()).real
    D_vv = zfs2(pd, G_Gv, nn_G)

    if with_paw_correction:
        for a, D_sii in D_asii.items():
            D_vv += zfs2paw(D_sii[0], D_sii[1], setups[a])

    n_nG = pd.empty(N2)
    # D_naii = ...
    for n1, psit1_R in enumerate(wf1.psit_nR):
        for n_G, psit2_R in zip(n_nG, wf2.psit_nR):
            n_G[:] = pd.fft(psit1_R * psit2_R)

        D_anii = {}
        Q_anL = {}
        for a, P1_ni in wf1.projections.items():
            D_nii = np.einsum('i, nj -> nij', P1_ni[n1], wf2.projections[a])
            D_anii[a] = D_nii
            Q_anL[a] = np.einsum('nij, ijL -> nL',
                                 D_nii, setups[a].Delta_iiL)

        if compensation_charge:
            compensation_charge.add(n_nG, Q_anL)

        nn_G = (n_nG * n_nG.conj()).sum(axis=0).real
        D_vv -= zfs2(pd, G_Gv, nn_G)

        if with_paw_correction:
            for a, D_sii in D_asii.items():
                D_vv += zfs2paw(D_sii[0], D_sii[1], setups[a])

    # Remove trace:
    D_vv -= np.trace(D_vv) / 3 * np.eye(3)

    return sign * alpha**2 * pi * D_vv * Ha


def zfs2(pd: PWDescriptor,
         G_Gv: Array2D,
         nn_G: Array1D) -> Array2D:
    """Integral."""
    D_vv = np.einsum('gv, gw, g -> vw', G_Gv, G_Gv, nn_G)
    # D_vv -= np.eye(3) * nn_G.sum() / 3
    D_vv *= 2 * pd.gd.dv / pd.gd.N_c.prod()
    return D_vv


@functools.lru_cache
def zfs_tensor(setup):
    things = coulomb(setup.rgd,
                     np.array(setup.data.phi_jg),
                     np.array(setup.data.phit_jg),
                     setup.l_j,
                     setup.g_lg)
    return coulomb_integrals(setup.rgd, setup.l_j, *things)


def zfs2paw(D1_ii, D2_ii, setup):
    """PAW correction."""
    C_iiiivv = zfs_tensor(setup)
    return np.einsum('ij, ijklab, kl -> ab', D1_ii, C_iiiivv, D2_ii)


def integral_lesser(rho12, rho34, l, r, dr):
    a34 = r**l * rho34 * dr
    v34 = np.add.accumulate(a34) - a34
    return (rho12 * v34 * dr)[1:] @ r[1:]**(1 - l)


def integral_greater(rho12, rho34, l, r, dr):
    a12 = r**(l + 2) * rho12 * dr
    v12 = np.add.accumulate(a12) - a12
    return (v12 * rho34 * dr)[1:] @ r[1:]**(-1 - l)
