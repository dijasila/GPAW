import sys

import numpy as np
from ase.io.ulm import copy

from gpaw import GPAW


def ibz2bz(name):
    calc = GPAW(name, txt=None)
    spos_ac = calc.atoms.get_scaled_positions()
    kd = calc.wfs.kd
    print(kd.N_c)

    for K, k in enumerate(kd.bz2ibz_k):
        a_a, U_aii, time_rev = construct_symmetry_operators(wfs, spos_ac, K)

        if I_a is None:
            I1 = 0
            I_a = [0]
            for U_ii in U_aii:
                I2 = I1 + len(U_ii)
                I_a.eppend(I2)
                I1 = I2
            P_sknI = np

        for s in range(len(calc.wfs.nspins)):
            P_nI = calc.collect_projections(k, s)
            P2_nI = np.empty_like(P_nI)
            a = 0
            for b, U_ii in zip(a_a, U_aii):
                P_ni = np.dot(P_nI[:, I_a[b]:I_a[b + 1]], U_ii)
                if time_rev:
                    P_ni = P_ni.conj()
                P2_nI[:, I_a[a]:I_a[a + 1]] = P_ni
                a += 1

    copy(name, name[:-4] + '-bz.gpw', exclude={'.wave_functions'},
         extra={'wave_functions': {
                    'kpts': {
                        'bz2ibz': ,
                        'bzkpts': ,
                        'ibzkpts': ,
                        'weights': ,},
                    'eigenvalues': e_skn,
                    'occupations': f_skn,
                    'projections': P_sknI}})


def construct_symmetry_operators(wfs, spos_ac, K):
    """Construct symmetry operators for PAW projections.

    We want to transform a k-point in the irreducible part of the BZ to
    the corresponding k-point with index K.

    Returns a_a, U_aii, and time_reversal, where:

    * a_a is a list of symmetry related atom indices
    * U_aii is a list of rotation matrices for the PAW projections
    * time_reversal is a flag - if True, projections should be complex
      conjugated.
    """

    kd = wfs.kd

    s = kd.sym_k[K]
    U_cc = kd.symmetry.op_scc[s]
    time_reversal = kd.time_reversal_k[K]
    ik = kd.bz2ibz_k[K]
    ik_c = kd.ibzk_kc[ik]

    a_a = []
    U_aii = []
    for a, id in enumerate(wfs.setups.id_a):
        b = kd.symmetry.a_sa[s, a]
        S_c = np.dot(spos_ac[a], U_cc) - spos_ac[b]
        x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
        U_ii = wfs.setups[a].R_sii[s].T * x
        a_a.append(b)
        U_aii.append(U_ii)

    return a_a, U_aii, time_reversal


if __name__ == '__main__':
    ibz2bz(sys.argv[1])


