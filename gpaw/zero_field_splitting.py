"""Zero-field splitting.

See::

    Spin decontamination for magnetic dipolar coupling calculations:
    Application to high-spin molecules and solid-state spin qubits

    Timur Biktagirov, Wolf Gero Schmidt, and Uwe Gerstmann

    Phys. Rev. Research 2, 022024(R) – Published 30 April 2020

"""
from typing import List

import numpy as np

from gpaw.wavefunctions.pw import PWLFC
from gpaw.hints import Array2D


def zfs(calc,
        # n1, n2
        ) -> Array2D:
    """"""
    wfs = calc.wfs
    kpt_s = wfs.kpt_qs[0]

    wf_s = [WaveFunctions.from_kpt(kpt, wfs.setups)
            for kpt in kpt_s]

    compensation_charge = PWLFC([data.ghat_l for data in wfs.setups], wfs.pd)
    compensation_charge.set_positions(calc.spos_ac)

    return zfs1(*wf_s, compensation_charge)


class WaveFunctions:
    def __init__(self, psit, projections, setups):
        self.pd = psit.pd
        N = len(psit.array)
        self.psit_nR = self.pd.gd.empty(N)
        for n, psit_G in enumerate(psit.array):
            self.psit_nR[n] = self.pd.ifft(psit_G)
        self.projections = projections
        self.setups = setups

    @staticmethod
    def from_kpt(kpt, setups):
        return WaveFunctions(kpt.psit, kpt.projections, setups)

    def __len__(self):
        return len(self.psit_nR)


def zfs1(wf1, wf2, compensation_charge) -> Array2D:
    pd = wf1.pd
    setups = wf1.setups
    N = len(wf2)

    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0
    G_Gv = pd.get_reciprocal_vectors() / G_G[:, np.newaxis]

    n_nG = pd.empty(N)
    D_vv = np.zeros((3, 3))

    for n, psit1_R in enumerate(wf1.psit_nR):
        D_anii = {}
        Q_anL = {}
        for a, P1_ni in wf1.projections.items():
            D_nii = np.einsum('i, nj -> nij', P1_ni[n], wf2.projections[a])
            D_anii[a] = D_nii
            Q_anL[a] = np.einsum('nij, ijL -> nL',
                                 D_nii, setups[a].Delta_iiL)

        for n_G, psit2_R in zip(n_nG, wf2.psit_nR):
            n_G[:] = pd.fft(psit1_R * psit2_R)

        compensation_charge.add(n_nG, Q_anL)

        nn_G = (n_nG * n_nG.conj()).sum(axis=0).real

        D_vv += np.einsum('gv, gw, g -> vw', G_Gv, G_Gv, nn_G)

    # / gd.dv

    return D_vv


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


if __name__ == '__main__':
    main()
