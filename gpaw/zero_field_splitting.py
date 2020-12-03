"""Zero-field splitting.

See::

    Spin decontamination for magnetic dipolar coupling calculations:
    Application to high-spin molecules and solid-state spin qubits

    Timur Biktagirov, Wolf Gero Schmidt, and Uwe Gerstmann

    Phys. Rev. Research 2, 022024(R) – Published 30 April 2020

"""
from math import pi
from typing import List

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.wavefunctions.pw import PWLFC
from gpaw.hints import Array2D
from gpaw.hyperfine import alpha  # fine-structure constant: ~ 1 / 137


def zfs(calc: GPAW,
        method: int = 1) -> Array2D:
    """"""
    wfs = calc.wfs
    kpt_s, = wfs.kpt_qs

    wf1, wf2 = (WaveFunctions.from_kpt(kpt, wfs.setups)
                for kpt in kpt_s)

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
    def __init__(self, pd, psit_nR, projections, spin, setups):
        self.pd = pd
        self.psit_nR = psit_nR
        self.projections = projections
        self.spin = spin
        self.setups = setups

    def view(self, n1: int, n2: int) -> 'WaveFunctions':
        return WaveFunctions(self.pd,
                             self.psit_nR[n1:n2],
                             self.projections.view(n1, n2),
                             self.spin,
                             self.setups)

    @staticmethod
    def from_kpt(kpt, setups) -> 'WaveFunctions':
        nocc = (kpt.f_n > 0.5).sum()
        print(nocc)
        psit = kpt.psit
        pd = psit.pd
        psit_nR = pd.gd.empty(nocc)
        for psit_R, psit_G in zip(psit_nR, psit.array):
            psit_R[:] = pd.ifft(psit_G)
        return WaveFunctions(pd,
                             psit_nR,
                             kpt.projections.view(0, nocc),
                             psit.spin,
                             setups)

    def __len__(self):
        return len(self.psit_nR)


def create_compensation_charge(wf: WaveFunctions, spos_ac: Array2D) -> PWLFC:
    compensation_charge = PWLFC([data.ghat_l for data in wf.setups], wf.pd)
    compensation_charge.set_positions(spos_ac)
    return compensation_charge


def zfs1(wf1: WaveFunctions,
         wf2: WaveFunctions,
         compensation_charge: PWLFC) -> Array2D:
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

        print(n_G[0] * pd.gd.dv)
        compensation_charge.add(n_G, Q_aL)
        print(n_G[0] * pd.gd.dv)

    nn_G = (n_sG[0] * n_sG[1].conj()).real
    D_vv = zfs2(G_Gv, nn_G)

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
        print(N2, n_nG[:, 0] * pd.gd.dv)

        compensation_charge.add(n_nG, Q_anL)
        print(N2, n_nG[:, 0] * pd.gd.dv)

        nn_G = (n_nG * n_nG.conj()).sum(axis=0).real
        D_vv -= zfs2(G_Gv, nn_G)

    D_vv *= 2 * pd.gd.dv / pd.gd.N_c.prod()

    print(np.trace(D_vv))

    D_vv -= np.trace(D_vv) / 3 * np.eye(3)  # should be traceless

    sign = 1.0 if wf1.spin == wf2.spin else -1.0

    return sign * alpha**2 * pi * D_vv * Ha


def zfs2(G_Gv, nn_G):
    return np.einsum('gv, gw, g -> vw', G_Gv, G_Gv, nn_G)


def main(argv: List[str] = None) -> Array2D:
    import argparse
    import ase.units as units
    from gpaw import GPAW
    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.zero_field_splitting',
        description='...')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='GPW-file with wave functions.')
    add('-u', '--units', default='ueV', choices=['ueV', 'MHz', 'cm-1'],
        help='Units.  Must be "ueV" (micro-eV, default), "MHz" or "cm-1".')
    add('-m', '--method', type=int, default=1)

    if hasattr(parser, 'parse_intermixed_args'):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    calc = GPAW(args.file)

    if args.units == 'ueV':
        scale = 1e6
        unit = 'μeV'
    elif args.units == 'MHz':
        scale = units._e / units._hplanck * 1e-6
        unit = 'MHz'
    else:
        scale = units._e / units._hplanck / units._c / 100
        unit = '1/cm'

    D_vv = zfs(calc, args.method) * scale

    print('D_ij = (' +
          ',\n        '.join('(' + ', '.join(f'{d:10.3f}' for d in D_v) + ')'
                             for D_v in D_vv) +
          ') ', unit)
    print('i, j = x, y, z')

    (e1, e2, e3), U = np.linalg.eigh(D_vv)

    if abs(e1) > abs(e3):
        D = 1.5 * e1
        E = 0.5 * (e2 - e3)
        axis = U[:, 0]
    else:
        D = 1.5 * e3
        E = 0.5 * (e2 - e1)
        axis = U[:, 2]

    print()
    print(f'D = {D:.3f} {unit}')
    print(f'E = {E:.3f} {unit}')
    x, y, z = axis
    print(f'axis = ({x:.3f}, {y:.3f}, {z:.3f})')

    return D_vv


if __name__ == '__main__':
    main()
