from collections import namedtuple, defaultdict
from math import pi
from typing import List, Tuple, Dict

import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.utilities import unpack
import gpaw.mpi as mpi


KPoint = namedtuple(
    'KPoint',
    ['psit',   # plane-wave expansion of wfs
     'proj',   # projections
     'f_n',    # occupations numbers between 0 and 1
     'k_c',    # k-vector in units of reciprocal cell
     'weight'  # weight of k-point
     ])

RSKPoint = namedtuple(
    'RealSpaceKPoint',
    ['u_nR',  # wfs on a real-space grid
     'proj',  # same as above
     'f_n',   # ...
     'k_c',
     'weight',
     # 'index'  # IBZ k-point index
     ])


class EXX:
    def __init__(self,
                 kd: KPointDescriptor,
                 setups: List['Setup'],
                 pt,
                 coulomb,
                 spos_ac):
        """Exact exchange operator."""
        self.kd = kd
        self.setups = setups
        self.pt = pt
        self.coulomb = coulomb
        self.spos_ac = spos_ac

        self.comm = self.pt.comm

        # PAW-correction stuff:
        self.Delta_aiiL = []
        self.VC_aii = {}
        for a, data in enumerate(setups):
            self.Delta_aiiL.append(data.Delta_iiL)
            self.VC_aii[a] = unpack(data.X_p)

        self.symmetry_map_ss = create_symmetry_map(kd)
        self.inverse_s = self.symmetry_map_ss[:, 0]
        U_scc = kd.symmetry.op_scc
        assert (U_scc[0] == np.eye(3, dtype=int)).all()

    def calculate(self, kpts1, kpts2, VV_aii, derivatives=False, e_kn=None):
        pd = kpts1[0].psit.pd
        gd = pd.gd.new_descriptor(comm=mpi.serial_comm)
        comm = self.comm
        self.N_c = gd.N_c

        if derivatives:
            nbands = len(kpts1[0].psit.array)
            shapes = [(nbands, len(Delta_iiL))
                      for Delta_iiL in self.Delta_aiiL]
            v_kani = [{a: np.zeros(shape, complex)
                       for a, shape in enumerate(shapes)}
                      for _ in range(len(kpts1))]
            v_knG = [k.psit.pd.zeros(nbands, global_array=True, q=k.psit.kpt)
                     for k in kpts1]

        exxvv = 0.0
        ekin = 0.0
        for i1, i2, s, k1, k2, count in self.ipairs(kpts1, kpts2):
            q_c = k2.k_c - k1.k_c
            qd = KPointDescriptor([-q_c])

            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)

            v1_nG = None
            v1_ani = None
            v2_nG = None
            v2_ani = None
            if derivatives:
                v1_nG = v_knG[i1]
                v1_ani = v_kani[i1]
                if kpts1 is kpts2:
                    v2_nG = v_knG[i2]
                    v2_ani = v_kani[i2]

            v_G = self.coulomb.get_potential(pd12)
            e_nn = self.calculate_exx_for_pair(k1, k2, ghat, v_G,
                                               kpts1[i1].psit.pd,
                                               kpts2[i2].psit.pd,
                                               kpts1[i1].psit.kpt,
                                               kpts2[i2].psit.kpt,
                                               k1.f_n,
                                               k2.f_n,
                                               s,
                                               count,
                                               v1_nG, v1_ani,
                                               v2_nG, v2_ani)

            if 0:
                print(i1, i2, s,
                      k1.k_c[2], k2.k_c[2], kpts1 is kpts2, count,
                      e_nn[0, 0])
            e_nn *= count
            e = k1.f_n.dot(e_nn).dot(k2.f_n) / self.kd.nbzkpts
            exxvv -= 0.5 * e
            ekin += e
            if e_kn is not None:
                e_kn[i1] -= e_nn.dot(k2.f_n)

        exxvc = 0.0
        for i, kpt in enumerate(kpts1):
            for a, VV_ii in VV_aii.items():
                P_ni = kpt.proj[a]
                vv_n = np.einsum('ni, ij, nj -> n',
                                 P_ni.conj(), VV_ii, P_ni).real
                vc_n = np.einsum('ni, ij, nj -> n',
                                 P_ni.conj(), self.VC_aii[a], P_ni).real
                exxvv -= vv_n.dot(kpt.f_n) * kpt.weight
                exxvc -= vc_n.dot(kpt.f_n) * kpt.weight
                if e_kn is not None:
                    e_kn[i] -= (2 * vv_n + vc_n)

        w_knG = {}
        if derivatives:
            G1 = comm.rank * pd.maxmyng
            G2 = (comm.rank + 1) * pd.maxmyng
            for v_nG, v_ani, kpt in zip(v_knG, v_kani, kpts1):
                comm.sum(v_nG)
                w_nG = v_nG[:, G1:G2].copy()
                w_knG[len(w_knG)] = w_nG
                for v_ni in v_ani.values():
                    comm.sum(v_ni)
                v1_ani = {}
                for a, VV_ii in VV_aii.items():
                    P_ni = kpt.proj[a]
                    v_ni = P_ni.dot(self.VC_aii[a] + 2 * VV_ii)
                    v1_ani[a] = v_ani[a] - v_ni
                    ekin += np.einsum('n, ni, ni',
                                      kpt.f_n, P_ni.conj(), v_ni).real
                self.pt.add(w_nG, v1_ani, kpt.psit.kpt)

        return comm.sum(exxvv), comm.sum(exxvc), comm.sum(ekin), w_knG

    def calculate_exx_for_pair(self,
                               k1,
                               k2,
                               ghat,
                               v_G,
                               pd1, pd2,
                               index1, index2,
                               f1_n, f2_n,
                               s,
                               count,
                               v1_nG=None,
                               v1_ani=None,
                               v2_nG=None,
                               v2_ani=None):
        Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                            k1.proj[a],
                            Delta_iiL,
                            k2.proj[a].conj())
                  for a, Delta_iiL in enumerate(self.Delta_aiiL)]

        if v2_nG is not None:
            T, T_a, cc = self.symmetry_operation(self.inverse_s[s])

        factor = 1.0 / self.kd.nbzkpts
        N1 = len(k1.u_nR)
        N2 = len(k2.u_nR)
        exx_nn = np.zeros((N1, N2))
        rho_nG = ghat.pd.empty(N2, k1.u_nR.dtype)
        S = self.comm.size
        for n1, u1_R in enumerate(k1.u_nR):
            n0 = n1 if k1 is k2 else 0
            n0 = 0
            n2a = min(n0 + (N2 - n0 + S - 1) // S * self.comm.rank, N2)
            n2b = min(n2a + (N2 - n0 + S - 1) // S, N2)
            for n2, rho_G in enumerate(rho_nG[n2a:n2b], n2a):
                rho_G[:] = ghat.pd.fft(u1_R * k2.u_nR[n2].conj())

            ghat.add(rho_nG[n2a:n2b],
                     {a: Q_nnL[n1, n2a:n2b]
                      for a, Q_nnL in enumerate(Q_annL)})

            for n2, rho_G in enumerate(rho_nG[n2a:n2b], n2a):
                vrho_G = v_G * rho_G
                e = ghat.pd.integrate(rho_G, vrho_G).real
                exx_nn[n1, n2] = e
                if k1 is k2:
                    exx_nn[n2, n1] = e

                if v1_nG is not None:
                    vrho_R = ghat.pd.ifft(vrho_G)
                    if v2_nG is None:
                        v1_nG[n1] -= f2_n[n2] * factor * pd1.fft(
                            vrho_R * k2.u_nR[n2], index1, local=True)
                        for a, v_xL in ghat.integrate(vrho_G).items():
                            v_ii = self.Delta_aiiL[a].dot(v_xL[0])
                            v1_ani[a][n1] -= (v_ii.dot(k2.proj[a][n2]) *
                                              f2_n[n2] * factor)
                    else:
                        x = factor * count / 2
                        x1 = x / (self.kd.weight_k[index1] * self.kd.nbzkpts)
                        x2 = x / (self.kd.weight_k[index2] * self.kd.nbzkpts)
                        v1_nG[n1] -= f2_n[n2] * x1 * pd1.fft(
                            vrho_R * k2.u_nR[n2], index1, local=True)
                        v2_nG[n2] -= f1_n[n1] * x2 * pd2.fft(
                            T(vrho_R.conj() * u1_R), index2,
                            local=True)
                        for a, v_xL in ghat.integrate(vrho_G).items():
                            v_ii = self.Delta_aiiL[a].dot(v_xL[0])
                            v1_ani[a][n1] -= (v_ii.dot(k2.proj[a][n2]) *
                                              f2_n[n2] * x1)
                            b, S_c, U_ii = T_a[a]
                            v_i = v_ii.conj().dot(k1.proj[b][n1])
                            v_i = v_i.dot(U_ii)
                            v_i *= np.exp(2j * pi * k2.k_c.dot(S_c))
                            if cc:
                                v_i = v_i.conj()
                            v2_ani[a][n2] -= v_i * f1_n[n1] * x2

        return exx_nn * factor

    def calculate_eigenvalues(self, kpts1, kpts2, coulomb, VV_aii,
                              e_kn, v_nG=None):
        pd = kpts1[0].psit.pd

        for i1, k1, k2, count in self.ipairs(kpts1, kpts2):
            q_c = k2.k_c - k1.k_c
            qd = KPointDescriptor([q_c])

            pd12 = PWDescriptor(pd.ecut, pd.gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)

            v_G = coulomb.get_potential(pd12)
            e_nn = self.calculate_exx_for_pair(k1, k2, ghat, v_G,
                                               pd, i1, k2.f_n, v_nG)

            e_nn *= count / self.kd.nbzkpts
            e_kn[i1] -= e_nn.dot(k2.f_n)

        for i, kpt in enumerate(kpts1):
            for a, P_ni in kpt.proj.items():
                vv_n = np.einsum('ni, ij, nj -> n',
                                 P_ni.conj(), VV_aii[a], P_ni).real
                vc_n = np.einsum('ni,ij,nj->n',
                                 P_ni.conj(), self.VC_aii[a], P_ni).real
                e_kn[i] -= (2 * vv_n + vc_n)

    def ipairs(self, kpts1, kpts2):
        kd = self.kd
        nsym = len(kd.symmetry.op_scc)

        assert len(kpts2) == kd.nibzkpts

        symmetries_k = []
        for k in range(kd.nibzkpts):
            indices = np.where(kd.bz2ibz_k == k)[0]
            sindices = (kd.sym_k[indices] +
                        kd.time_reversal_k[indices] * nsym)
            symmetries_k.append(sindices)

        pairs: Dict[Tuple[int, int, int], int]

        if kpts1 is kpts2:
            pairs1 = defaultdict(int)
            for i1 in range(kd.nibzkpts):
                for s1 in symmetries_k[i1]:
                    for i2 in range(kd.nibzkpts):
                        for s2 in symmetries_k[i2]:
                            s3 = self.symmetry_map_ss[s1, s2]
                            if i1 < i2:
                                pairs1[(i1, i2, s3)] += 1
                            else:
                                s4 = self.inverse_s[s3]
                                if i1 == i2:
                                    pairs1[(i1, i1, min(s3, s4))] += 1
                                else:
                                    pairs1[(i2, i1, s4)] += 1
            pairs = {}
            seen = {}
            for (i1, i2, s), count in pairs1.items():
                k2 = kd.bz2bz_ks[kd.ibz2bz_k[i2], s]
                if (i1, k2) in seen:
                    pairs[seen[(i1, k2)]] += count
                else:
                    pairs[(i1, i2, s)] = count
                    seen[(i1, k2)] = (i1, i2, s)
        else:
            pairs = {}
            for i1 in range(len(kpts1)):
                for i2 in range(kd.nibzkpts):
                    for s2 in symmetries_k[i2]:
                        pairs[(i1, i2, s2)] = 1

        if 0:
            for (i1, i2, s), count in sorted(pairs.items()):
                print(i1, i2, s, count)
            print(kd.nibzkpts)
            print(self.symmetry_map_ss)
            print(symmetries_k)

        lasti1 = -1
        lasti2 = -1
        for (i1, i2, s), count in sorted(pairs.items()):
            if i1 != lasti1:
                k1 = kpts1[i1]
                u1_nR = to_real_space(k1.psit)
                rsk1 = RSKPoint(u1_nR, k1.proj.broadcast(),
                                k1.f_n, k1.k_c,
                                k1.weight)
                lasti1 = i1
            if i2 == i1 and kpts1 is kpts2:
                rsk2 = rsk1
                lasti2 = i2
            elif i2 != lasti2:
                k2 = kpts2[i2]
                u2_nR = to_real_space(k2.psit)
                rsk2 = RSKPoint(u2_nR, k2.proj.broadcast(),
                                k2.f_n, k2.k_c,
                                k2.weight)
                lasti2 = i2

            yield i1, i2, s, rsk1, self.apply_symmetry(s, rsk2), count

    def symmetry_operation(self, s: int):
        U_scc = self.kd.symmetry.op_scc
        nsym = len(U_scc)
        time_reversal = s >= nsym
        s %= nsym
        U_cc = U_scc[s]
        # print(s, U_cc, time_reversal)

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


def to_real_space(psit):
    pd = psit.pd
    comm = pd.comm
    S = comm.size
    q = psit.kpt
    nbands = len(psit.array)
    u_nR = pd.gd.empty(nbands, pd.dtype, global_array=True)
    for n1 in range(0, nbands, S):
        n2 = min(n1 + S, nbands)
        u_G = pd.alltoall1(psit.array[n1:n2], q)
        if u_G is not None:
            n = n1 + comm.rank
            u_nR[n] = pd.ifft(u_G, local=True, safe=False, q=q)
        for n in range(n1, n2):
            comm.broadcast(u_nR[n], n - n1)

    return u_nR


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
