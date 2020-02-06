from math import pi

import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from .kpts import RSKPoint


def create_symmetry_map(kd: KPointDescriptor):  # -> List[List[int]]
    sym = kd.symmetry
    U_scc = sym.op_scc
    nsym = len(U_scc)
    compconj_s = np.zeros(nsym, bool)
    if sym.time_reversal and not sym.has_inversion:
        U_scc = np.concatenate([U_scc, -U_scc])
        compconj_s = np.zeros(nsym * 2, bool)
        compconj_s[nsym:] = True
        nsym *= 2

    map_ss = np.zeros((nsym, nsym), int)
    for s1 in range(nsym):
        for s2 in range(nsym):
            diff_s = abs(U_scc[s1].dot(U_scc).transpose((1, 0, 2)) -
                         U_scc[s2]).sum(2).sum(1)
            indices = (diff_s == 0).nonzero()[0]
            assert len(indices) == 1
            s = indices[0]
            assert compconj_s[s1] ^ compconj_s[s2] == compconj_s[s]
            map_ss[s1, s2] = s

    return map_ss


class Symmetry:
    def __init__(self, kd: KPointDescriptor):
        self.symmetry_map_ss = create_symmetry_map(kd)

        U_scc = kd.symmetry.op_scc
        is_identity_s = (U_scc == np.eye(3, dtype=int)).all(2).all(1)
        self.s0 = is_identity_s.nonzero()[0][0]
        self.inverse_s = self.symmetry_map_ss[:, self.s0]

    def symmetry_operation(self, s: int):
        U_scc = self.kd.symmetry.op_scc
        nsym = len(U_scc)
        time_reversal = s >= nsym
        s %= nsym
        U_cc = U_scc[s]

        if (U_cc == np.eye(3, dtype=int)).all():
            def T0(a_R):
                return a_R
        else:
            N_c = self.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')

            def T0(a_R):
                return a_R.ravel()[i].reshape(N_c)

        if time_reversal:
            def T(a_R):
                return T0(a_R).conj()
        else:
            T = T0

        T_a = []
        for a, id in enumerate(self.setups.id_a):
            b = self.kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            U_ii = self.setups[a].R_sii[s].T
            T_a.append((b, S_c, U_ii))

        return T, T_a, time_reversal

    def apply_symmetry(self, s: int, rsk):
        U_scc = self.kd.symmetry.op_scc
        nsym = len(U_scc)
        time_reversal = s >= nsym
        s %= nsym
        sign = 1 - 2 * int(time_reversal)
        U_cc = U_scc[s]

        if (U_cc == np.eye(3)).all() and not time_reversal:
            return rsk

        u1_nR, proj1, f_n, k1_c, weight = rsk

        u2_nR = np.empty_like(u1_nR)
        proj2 = proj1.new()

        k2_c = sign * U_cc.dot(k1_c)

        N_c = u1_nR.shape[1:]
        i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
        i = np.ravel_multi_index(i_cr, N_c, 'wrap')
        for u1_R, u2_R in zip(u1_nR, u2_nR):
            u2_R[:] = u1_R.ravel()[i].reshape(N_c)

        for a, id in enumerate(self.setups.id_a):
            b = self.kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * pi * np.dot(k1_c, S_c))
            U_ii = self.setups[a].R_sii[s].T * x
            proj2[a][:] = proj1[b].dot(U_ii)

        if time_reversal:
            np.conj(u2_nR, out=u2_nR)
            np.conj(proj2.array, out=proj2.array)

        return RSKPoint(u2_nR, proj2, f_n, k2_c, weight)
