# from typing import Tuple

import numpy as np
from ase.units import Ha, Bohr

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import serial_comm
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from .kpts import get_kpt


def calculate_forces(wfs, coulomb, sym, paw_s, ftol=1e-9) -> np.ndarray:
    kd = wfs.kd
    nspins = wfs.nspins

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)
    nocc = kd.comm.max(int(nocc))

    dPdR_skaniv = {(kpt.s, kpt.k): wfs.pt.derivative(kpt.psit_nG, q=kpt.k)
                   for kpt in wfs.mykpts}

    for spin in range(nspins):
        kpts = [get_kpt(wfs, k, spin, 0, nocc) for k in range(kd.nibzkpts)]
        dPdR_kaniv = [dPdR_skaniv[(spin, k)] for k in range(kd.nibzkpts)]
        forces(kpts, dPdR_kaniv, paw_s[spin],
               wfs, sym, coulomb)
    return 0.0


def forces(kpts, dPdR_kaniv, paw, wfs, sym, coulomb):
    pd = kpts[0].psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    comm = wfs.world

    """
    for i, kpt in enumerate(kpts):
        for a, VV_ii in paw.VV_aii.items():
            P_ni = kpt.proj[a]
            vv_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), VV_ii, P_ni).real
            vc_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), paw.VC_aii[a], P_ni).real
            exxvv -= vv_n.dot(kpt.f_n) * kpt.weight
            exxvc -= vc_n.dot(kpt.f_n) * kpt.weight
    """

    for i1, i2, s, k1, k2, count in sym.pairs(kpts, wfs, wfs.spos_ac):
        q_c = k2.k_c - k1.k_c
        qd = KPointDescriptor([-q_c])

        pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
        ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
        ghat.set_positions(wfs.spos_ac)

        v_G = coulomb.get_potential(pd12)
        calculate_exx_for_pair(k1, k2,
                               dPdR_kaniv[i1], dPdR_kaniv[i2],
                               ghat, v_G, comm,
                               paw.Delta_aiiL)


def calculate_exx_for_pair(k1,
                           k2,
                           dPdR1_aniv,
                           dPdR2_aniv,
                           ghat,
                           v_G,
                           comm,
                           Delta_aiiL):

    N1 = len(k1.u_nR)
    N2 = len(k2.u_nR)

    size = comm.size
    rank = comm.rank

    Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                        k1.proj[a],
                        Delta_iiL,
                        k2.proj[a].conj())
              for a, Delta_iiL in enumerate(Delta_aiiL)]

    if k1 is k2:
        n2max = (N1 + size - 1) // size
    else:
        n2max = N2

    rho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)
    vrho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)

    for n1, u1_R in enumerate(k1.u_nR):
        if k1 is k2:
            B = (N1 - n1 + size - 1) // size
            n2a = min(n1 + rank * B, N2)
            n2b = min(n2a + B, N2)
        else:
            B = (N1 + size - 1) // size
            n2a = 0
            n2b = N2

        for n2, rho_G in enumerate(rho_nG[:n2b - n2a], n2a):
            rho_G[:] = ghat.pd.fft(u1_R * k2.u_nR[n2].conj())

        ghat.add(rho_nG[:n2b - n2a],
                 {a: Q_nnL[n1, n2a:n2b]
                  for a, Q_nnL in enumerate(Q_annL)})

        for n2, rho_G in enumerate(rho_nG[:n2b - n2a], n2a):
            vrho_G = v_G * rho_G
            vrho_nG[n2 - n2a] = vrho_G
        for a, v_nLv in ghat.derivative(vrho_nG).items():
            F_v = k1.f_n[n1] * np.einsum('n, nL, nLv -> v',
                                         k2.f_n,
                                         Q_annL[a][n1].conj(),
                                         v_nLv)
            print(a, F_v[2] * Ha / Bohr)

        for a, v_nL in ghat.integrate(vrho_nG[:n2b - n2a]).items():
            v_iin = Delta_aiiL[a].dot(v_nL.T)
            F_v = k1.f_n[n1] * np.einsum('ijn, iv, nj, n -> v',
                                         v_iin,
                                         dPdR1_aniv[a][n1].conj(),
                                         k2.proj[a][n2a:n2b],
                                         k2.f_n[n2a:n2b])
            print(a, F_v[2] * Ha / Bohr)
