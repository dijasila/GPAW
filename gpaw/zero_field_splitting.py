"""Zero-field splitting.

See::

    Spin decontamination for magnetic dipolar coupling calculations:
    Application to high-spin molecules and solid-state spin qubits

    Timur Biktagirov, Wolf Gero Schmidt, and Uwe Gerstmann

    Phys. Rev. Research 2, 022024(R) – Published 30 April 2020

"""
from typing import List

import numpy as np
from ase.units import Bohr, Ha

from gpaw.wavefunctions.pw import PWLFC
from gpaw.hints import Array2D
from gpaw.hyperfine import alpha  # alpha ~= 1/137


def zfs(calc,
        # n1, n2
        ) -> Array2D:
    """"""
    wfs = calc.wfs
    kpt_s = wfs.kpt_qs[0]

    wf1, wf2 = (WaveFunctions.from_kpt(kpt, wfs.setups)
                for kpt in kpt_s)

    compensation_charge = PWLFC([data.ghat_l for data in wfs.setups], wfs.pd)
    compensation_charge.set_positions(calc.spos_ac)

    if 0:
        n1 = len(wf1)
        wf = wf1.view(n1 - 2, n1)
        return zfs1(wf, wf, compensation_charge)

    D_vv = np.zeros((3, 3))
    for wfa in [wf1, wf2]:
        for wfb in [wf1, wf2]:
            d_vv = zfs1(wfa, wfb, compensation_charge)
            D_vv += d_vv
            print(d_vv)
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


def zfs1(wf1, wf2, compensation_charge) -> Array2D:
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
    D_vv = np.einsum('gv, gw, g -> vw', G_Gv, G_Gv, nn_G)

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
        D_vv -= np.einsum('gv, gw, g -> vw', G_Gv, G_Gv, nn_G)

    D_vv -= np.trace(D_vv) / 3 * np.eye(3)

    sign = 1.0 if wf1.spin == wf2.spin else -1.0

    return sign * alpha**2 / 4 * D_vv * Ha * Bohr**2
    return sign * alpha**2 / 4 * D_vv / pd.gd.dv * Ha * Bohr**2


def main(argv: List[str] = None) -> None:
    import argparse
    import ase.units as units
    from gpaw import GPAW
    parser = argparse.ArgumentParser(
        prog='python3 -m gpaw.zero_field_splitting',
        description='...')
    add = parser.add_argument
    add('file', metavar='input-file',
        help='GPW-file with wave functions.')
    add('-u', '--units', default='ueV', choices=['ueV', 'MHz'],
        help='Units.  Must be "ueV" (micro-eV, default) or "MHz".')

    if hasattr(parser, 'parse_intermixed_args'):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    calc = GPAW(args.file)

    if args.units == 'ueV':
        scale = 1e6
        unit = 'μeV'
    else:
        scale = units._e / units._hplanck * 1e-6
        unit = 'MHz'

    D_vv = zfs(calc) * scale

    print(D_vv, unit)

    d1, d2, d3 = np.linalg.eigvalsh(D_vv)
    print(np.linalg.eigh(D_vv))
    return D_vv


if __name__ == '__main__':
    main()
