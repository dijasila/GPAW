from pathlib import Path
from typing import Union, Tuple

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.mpi import serial_comm
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.xc import XC
from . import parse_name
from .coulomb import coulomb_inteaction
from .kpts import KPoint
from .paw import calculate_paw_stuff
from .symmetry import Symmetry


def non_self_consistent_energy(calc: Union[GPAW, str, Path],
                               xcname: str,
                               ftol=1e-9) -> Tuple[float, float, float, float]:
    """Calculate non self-consistent energy for Hybrid functional.

    Based on a self-consistent DFT calculation (calc).  EXX integrals involving
    states with occupation numbers less than ftol are skipped.

    >>> ?????????????????????????????????????
    >>> eig_dft, vxc_dft, vxc_hyb = non_self_consistent_eigenvalues(...)
    >>> eig_hyb = eig_dft - vxc_dft + vxc_hyb
    """

    if not isinstance(calc, GPAW):
        calc = GPAW(calc, txt=None, parallel={'band': 1, 'kpt': 1})

    wfs = calc.wfs
    dens = calc.density
    kd = wfs.kd
    setups = wfs.setups
    nspins = wfs.nspins

    nocc = max(((kpt.f_n / kpt.weight) > ftol).sum()
               for kpt in wfs.mykpts)

    xcname, exx_fraction, omega = parse_name(xcname)

    xc = XC(xcname)
    exc = xc.calculate(dens.finegd, dens.nt_sg)
    for a, D_sp in dens.D_asp.items():
        exc += xc.calculate_paw_correction(setups[a], D_sp)

    coulomb = coulomb_inteaction(omega, wfs.gd, kd)
    sym = Symmetry(kd)

    paw_s = calculate_paw_stuff(dens, setups)

    ecc = sum(setup.ExxC for setup in setups) * exx_fraction
    evc = 0.0
    evv = 0.0
    for spin in range(nspins):
        K = kd.nibzkpts
        k1 = spin * K
        k2 = k1 + K
        kpts = [KPoint(kpt.psit.view(0, nocc),
                       kpt.projections.view(0, nocc),
                       kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                       kd.ibzk_kc[kpt.k],
                       kd.weight_k[kpt.k])
                for kpt in wfs.mykpts[k1:k2]]
        e1, e2 = calculate_energy(kpts, paw_s[spin],
                                  wfs, sym, coulomb)
        evc += e1 * 2 / wfs.nspins
        evv += e2 * 2 / wfs.nspins

    return exc * Ha, ecc * Ha, evc * Ha, evv * Ha


def calculate_energy(kpts, paw, wfs, sym, coulomb):
    pd = kpts[0].psit.pd
    gd = pd.gd.new_descriptor(comm=serial_comm)
    comm = wfs.world

    exxvv = 0.0
    for i1, i2, s, k1, k2, count in sym.pairs(kpts, wfs):
        q_c = k2.k_c - k1.k_c
        qd = KPointDescriptor([-q_c])

        pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
        ghat = PWLFC([data.ghat_l for data in wfs.setups], pd12)
        ghat.set_positions(wfs.spos_ac)

        v_G = coulomb.get_potential(pd12)
        e_nn = calculate_exx_for_pair(k1, k2, ghat, v_G, comm,
                                      paw.Delta_aiiL)

        e_nn *= count
        e = k1.f_n.dot(e_nn).dot(k2.f_n) / sym.kd.nbzkpts**2
        exxvv -= 0.5 * e

    exxvc = 0.0
    for i, kpt in enumerate(kpts):
        for a, VV_ii in paw.VV_aii.items():
            P_ni = kpt.proj[a]
            vv_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), VV_ii, P_ni).real
            vc_n = np.einsum('ni, ij, nj -> n',
                             P_ni.conj(), paw.VC_aii[a], P_ni).real
            exxvv -= vv_n.dot(kpt.f_n) * kpt.weight
            exxvc -= vc_n.dot(kpt.f_n) * kpt.weight

    return comm.sum(exxvv), comm.sum(exxvc)


def calculate_exx_for_pair(k1,
                           k2,
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

    e_nn = np.zeros((N1, N2))
    rho_nG = ghat.pd.empty(n2max, k1.u_nR.dtype)

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
            e = ghat.pd.integrate(rho_G, vrho_G).real
            e_nn[n1, n2] = e
            if k1 is k2:
                e_nn[n2, n1] = e

    return e_nn
