"""Zero-field splitting.

See::

    Spin decontamination for magnetic dipolar coupling calculations:
    Application to high-spin molecules and solid-state spin qubits

    Timur Biktagirov, Wolf Gero Schmidt, and Uwe Gerstmann

    Phys. Rev. Research 2, 022024(R) – Published 30 April 2020

"""
from math import pi
from typing import List, Tuple

import numpy as np
from ase.units import Bohr, Ha, _c, _e, _hplanck

from gpaw import GPAW
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hints import Array1D, Array2D, Array4D
from gpaw.hyperfine import alpha  # fine-structure constant: ~ 1 / 137
from gpaw.projections import Projections
from gpaw.setup import Setup
from gpaw.wavefunctions.pw import PWLFC, PWDescriptor


def zfs(calc: GPAW,
        method: int = 1) -> Array2D:
    """Zero-field splitting.

    Calculate magnetic dipole coupling tensor in eV.
    """
    wf1, wf2 = (WaveFunctions.from_calc(calc, spin) for spin in [0, 1])

    compensation_charge = create_compensation_charge(wf1, calc.spos_ac)

    if method == 1:
        n1 = len(wf1)
        wf = wf1.view(n1 - 2, n1)
        return zfs1(wf, wf, compensation_charge)

    D_vv = np.zeros((3, 3))
    for wfa in [wf1, wf2]:
        for wfb in [wf1, wf2]:
            d_vv = zfs1(wfa, wfb, compensation_charge)
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
        """Container for wave function in real-space and projections."""

        self.pd = pd or PWDescriptor(ecut=None, gd=gd)
        self.psit_nR = psit_nR
        self.projections = projections
        self.spin = spin
        self.setups = setups

    def view(self, n1: int, n2: int) -> 'WaveFunctions':
        """Create WaveFuntions object with view of data."""
        return WaveFunctions(self.psit_nR[n1:n2],
                             self.projections.view(n1, n2),
                             self.spin,
                             self.setups,
                             pd=self.pd)

    @staticmethod
    def from_calc(calc, spin) -> 'WaveFunctions':
        """Create WaveFunctions object from PW-mode representation."""
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

    def __len__(self) -> int:
        return len(self.psit_nR)


def create_compensation_charge(wf: WaveFunctions,
                               spos_ac: Array2D) -> PWLFC:
    compensation_charge = PWLFC([data.ghat_l for data in wf.setups], wf.pd)
    compensation_charge.set_positions(spos_ac)
    return compensation_charge


def zfs1(wf1: WaveFunctions,
         wf2: WaveFunctions,
         compensation_charge: PWLFC) -> Array2D:
    """Compute dipole coupling."""
    pd = wf1.pd
    setups = wf1.setups
    N2 = len(wf2)

    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0
    G_Gv = pd.get_reciprocal_vectors() / G_G[:, np.newaxis]

    n_sG = pd.zeros(2)
    for n_G, wf in zip(n_sG, [wf1, wf2]):
        D_aii = {}
        Q_aL = {}
        for a, P_ni in wf.projections.items():
            D_ii = np.einsum('ni, nj -> ij', P_ni, P_ni)
            D_aii[a] = D_ii
            Q_aL[a] = np.einsum('ij, ijL -> L', D_ii, setups[a].Delta_iiL)

        for psit_R in wf.psit_nR:
            n_G += pd.fft(psit_R**2)

        compensation_charge.add(n_G, Q_aL)

    nn_G = (n_sG[0] * n_sG[1].conj()).real
    D_vv = zfs2(pd, G_Gv, nn_G)

    n_nG = pd.empty(N2)
    for n1, psit1_R in enumerate(wf1.psit_nR):
        D_anii = {}
        Q_anL = {}
        for a, P1_ni in wf1.projections.items():
            D_nii = np.einsum('i, nj -> nij', P1_ni[n1], wf2.projections[a])
            D_anii[a] = D_nii
            Q_anL[a] = np.einsum('nij, ijL -> nL',
                                 D_nii, setups[a].Delta_iiL)

        for n_G, psit2_R in zip(n_nG, wf2.psit_nR):
            n_G[:] = pd.fft(psit1_R * psit2_R)

        compensation_charge.add(n_nG, Q_anL)

        nn_G = (n_nG * n_nG.conj()).sum(axis=0).real
        D_vv -= zfs2(pd, G_Gv, nn_G)

    D_vv -= np.trace(D_vv) / 3 * np.eye(3)  # should be traceless

    sign = 1.0 if wf1.spin == wf2.spin else -1.0

    return sign * alpha**2 * pi * D_vv * Ha


def zfs2(pd: PWDescriptor,
         G_Gv: Array2D,
         nn_G: Array1D) -> Array2D:
    """Integral."""
    D_vv = np.einsum('gv, gw, g -> vw', G_Gv, G_Gv, nn_G)
    D_vv *= 2 * pd.gd.dv / pd.gd.N_c.prod()
    return D_vv


def convert_tensor(D_vv: Array2D,
                   unit: str = 'eV') -> Tuple[float, float, Array1D, Array2D]:
    """Convert 3x3 tensor to D, E and easy axis.

    Input tensor must be in eV and the result can be returned in
    eV, μeV, MHz or 1/cm acording to the value uf *unit*
    (must be one of "eV", "ueV", "MHz", "1/cm").

    >>> D_vv = np.diag([1, 2, 3])
    >>> D, E, axis, _ = convert_tensor(D_vv)
    >>> D
    4.5
    >>> E
    0.5
    >>> axis
    array([0., 0., 1.])
    """
    if unit == 'ueV':
        scale = 1e6
    elif unit == 'MHz':
        scale = _e / _hplanck * 1e-6
    elif unit == '1/cm':
        scale = _e / _hplanck / _c / 100
    elif unit == 'eV':
        scale = 1.0
    else:
        raise ValueError(f'Unknown unit: {unit}')

    (e1, e2, e3), U = np.linalg.eigh(D_vv * scale)

    if abs(e1) > abs(e3):
        D = 1.5 * e1
        E = 0.5 * (e2 - e3)
        axis = U[:, 0]
    else:
        D = 1.5 * e3
        E = 0.5 * (e2 - e1)
        axis = U[:, 2]

    return D, E, axis, D_vv * scale


def main(argv: List[str] = None) -> Array2D:
    """CLI interface."""
    import argparse

    from gpaw import GPAW
    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.zero_field_splitting',
        description='...')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='GPW-file with wave functions.')
    add('-u', '--unit', default='ueV', choices=['ueV', 'MHz', '1/cm'],
        help='Unit.  Must be "ueV" (micro-eV, default), "MHz" or "1/cm".')
    add('-m', '--method', type=int, default=1)

    if hasattr(parser, 'parse_intermixed_args'):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    calc = GPAW(args.file)

    D_vv = zfs(calc, args.method)
    D, E, axis, D_vv = convert_tensor(D_vv, args.unit)

    unit = args.unit
    if unit == 'ueV':
        unit = 'μeV'

    print('D_ij = (' +
          ',\n        '.join('(' + ', '.join(f'{d:10.3f}' for d in D_v) + ')'
                             for D_v in D_vv) +
          ') ', unit)
    print('i, j = x, y, z')
    print()
    print(f'D = {D:.3f} {unit}')
    print(f'E = {E:.3f} {unit}')
    x, y, z = axis
    print(f'axis = ({x:.3f}, {y:.3f}, {z:.3f})')

    return D_vv


if __name__ == '__main__':
    main()
