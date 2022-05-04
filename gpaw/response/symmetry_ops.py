import numpy as np


"""This module exists to reduce code duplication.

The costruct_symmetry_operators method was duplicated on two classes.
Now it only exists in one place, at the cost of a nasty fake "self".
"""


def construct_symmetry_operators(self, K, k_c=None, *, apply_strange_shift,
                                 spos_ac):
    """Construct symmetry operators for wave function and PAW projections.

    We want to transform a k-point in the irreducible part of the BZ to
    the corresponding k-point with index K.

    Returns U_cc, T, a_a, U_aii, shift_c and time_reversal, where:

    * U_cc is a rotation matrix.
    * T() is a function that transforms the periodic part of the wave
      function.
    * a_a is a list of symmetry related atom indices
    * U_aii is a list of rotation matrices for the PAW projections
    * shift_c is three integers: see code below.
    * time_reversal is a flag - if True, projections should be complex
      conjugated.

    See the get_k_point() method for how to use these tuples.
    """

    wfs = self.calc.wfs

    R_asii = [setup.R_sii for setup in wfs.setups]
    return _construct_symmetry_operators(
        self, K, k_c=k_c,
        apply_strange_shift=apply_strange_shift,
        spos_ac=spos_ac,
        R_asii=R_asii)


def _construct_symmetry_operators(self, K, k_c=None, *, apply_strange_shift,
                                  spos_ac, R_asii):
    wfs = self.calc.wfs
    kd = wfs.kd

    s = kd.sym_k[K]
    U_cc = kd.symmetry.op_scc[s]
    time_reversal = kd.time_reversal_k[K]
    ik = kd.bz2ibz_k[K]
    if k_c is None:
        k_c = kd.bzk_kc[K]
    ik_c = kd.ibzk_kc[ik]

    sign = 1 - 2 * time_reversal
    shift_c = np.dot(U_cc, ik_c) - k_c * sign

    try:
        assert np.allclose(shift_c.round(), shift_c)
    except AssertionError:
        print('shift_c ' + str(shift_c), file=self.fd)
        print('k_c ' + str(k_c), file=self.fd)
        print('kd.bzk_kc[K] ' + str(kd.bzk_kc[K]), file=self.fd)
        print('ik_c ' + str(ik_c), file=self.fd)
        print('U_cc ' + str(U_cc), file=self.fd)
        print('sign ' + str(sign), file=self.fd)
        raise AssertionError

    shift_c = shift_c.round().astype(int)

    if (U_cc == np.eye(3)).all():
        def T(f_R):
            return f_R
    else:
        N_c = self.calc.wfs.gd.N_c
        i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
        i = np.ravel_multi_index(i_cr, N_c, 'wrap')

        def T(f_R):
            return f_R.ravel()[i].reshape(N_c)

    if time_reversal:
        T0 = T

        def T(f_R):
            return T0(f_R).conj()

        shift_c *= -1

    a_a = []
    U_aii = []
    for a, R_sii in enumerate(R_asii):
        b = kd.symmetry.a_sa[s, a]
        S_c = np.dot(spos_ac[a], U_cc) - spos_ac[b]
        x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
        U_ii = R_sii[s].T * x
        a_a.append(b)
        U_aii.append(U_ii)

    if apply_strange_shift:
        shift0_c = (kd.bzk_kc[K] - k_c).round().astype(int)
        shift_c += -shift0_c

    return U_cc, T, a_a, U_aii, shift_c, time_reversal
