from functools import partial

import numpy as np


def construct_symmetry_operators(kd, gd, K, *, spos_ac, R_asii):
    """Construct symmetry operators for wave function and PAW projections.

    We want to transform a k-point in the irreducible part of the BZ to
    the corresponding k-point with index K.

    Returns U_cc, T, a_a, U_aii, k_c and time_reversal, where:

    * U_cc is a rotation matrix.
    * T() is a function that transforms the periodic part of the wave
      function.
    * a_a is a list of symmetry related atom indices
    * U_aii is a list of rotation matrices for the PAW projections
    * k_c is an array of the relative k-point coordinates of the k-point to
      which the wave function is mapped. NB: This can lie outside the 1st BZ.
    * time_reversal is a flag - if True, projections should be complex
      conjugated.
    """
    s = kd.sym_k[K]
    U_cc = kd.symmetry.op_scc[s]
    time_reversal = kd.time_reversal_k[K]
    ik = kd.bz2ibz_k[K]
    ik_c = kd.ibzk_kc[ik]

    # Apply symmetry operations to the irreducible k-point
    sign = 1 - 2 * time_reversal
    k_c = sign * U_cc @ ik_c

    if (U_cc == np.eye(3)).all():
        def T(f_R):
            return f_R
    else:
        N_c = gd.N_c
        i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
        i = np.ravel_multi_index(i_cr, N_c, 'wrap')

        def T(f_R):
            return f_R.ravel()[i].reshape(N_c)

    if time_reversal:
        T0 = T

        def T(f_R):
            return T0(f_R).conj()

    a_a = []
    U_aii = []
    for a, R_sii in enumerate(R_asii):
        b = kd.symmetry.a_sa[s, a]
        S_c = np.dot(spos_ac[a], U_cc) - spos_ac[b]
        x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
        U_ii = R_sii[s].T * x
        a_a.append(b)
        U_aii.append(U_ii)

    return U_cc, T, a_a, U_aii, k_c, time_reversal


def construct_symmetrizers(gs, K):
    """Construct functions to symmetrize ut_hR and Ph."""
    _, T, a_a, U_aii, k_c, time_reversal = \
        gs.construct_symmetry_operators(K)

    ut_symmetrizer = T
    Ph_symmetrizer = partial(symmetrize_projections,
                             a1_a2=a_a, U_aii=U_aii,
                             time_reversal=time_reversal)

    return ut_symmetrizer, Ph_symmetrizer, k_c


def symmetrize_projections(Ph, a1_a2, U_aii, time_reversal):
    """Symmetrize the PAW projections.

    NB: The projections of atom a1 are mapped onto a *different* atom a2
    according to the input map of atomic indices a1_a2."""
    # First, we apply the symmetry operations to the projections one at a time
    P_a2hi = []
    for a1, U_ii in zip(a1_a2, U_aii):
        P_hi = Ph[a1].copy(order='C')
        np.dot(P_hi, U_ii, out=P_hi)
        if time_reversal:
            np.conj(P_hi, out=P_hi)
        P_a2hi.append(P_hi)

    # Then, we store the symmetry mapped projectors in the projections object
    for a2, P_hi in enumerate(P_a2hi):
        I1, I2 = Ph.map[a2]
        Ph.array[..., I1:I2] = P_hi

    return Ph
