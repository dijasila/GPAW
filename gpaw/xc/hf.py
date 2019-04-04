from collections import namedtuple, defaultdict
from math import pi

import numpy as np
from ase.units import Ha

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.xc.exx import pawexxvv
from gpaw.utilities import unpack, unpack2


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
     'index'  # IBZ k-point index
     ])


class EXX:
    def __init__(self,
                 kd: KPointDescriptor,
                 setups):  # List[Setup]
        """Exact exchange operator."""
        self.kd = kd
        self.setups = setups

        self.spos_ac = None  # will be set later

        # PAW-correction stuff:
        self.Delta_aiiL = []
        self.VC_aii = []
        self.exxcc = 0.0
        for a, data in enumerate(setups):
            self.Delta_aiiL.append(data.Delta_iiL)
            self.VC_aii.append(unpack(data.X_p))
            self.exxcc += data.ExxC

        self.symmetry_map_ss = create_symmetry_map(kd)
        self.inverse_s = self.symmetry_map_ss[:, 0]

    def one_shot_calculation(self, wfs, D_asp, coulomb):
        self.spos_ac = wfs.spos_ac

        kd = self.kd

        assert kd.comm.size == 1 or kd.comm.size == 2 and wfs.nspins == 2
        assert wfs.bd.comm.size == 1

        VV_asii = self.calculate_valence_valence_paw_corrections(D_asp)

        exxvv = 0.0
        exxvc = 0.0
        eig_skn = np.zeros((wfs.nspins, kd.nibzkpts, wfs.bd.nbands))

        K = kd.nibzkpts
        deg = 2 / wfs.nspins

        spin = kd.comm.rank  # 0 or 1
        k1 = 0
        k2 = K
        while True:
            kpts = [KPoint(kpt.psit,
                           kpt.projections,
                           kpt.f_n / kpt.weight,  # scale to [0:1] range
                           kd.ibzk_kc[kpt.k],
                           kd.weight_k[kpt.k])
                    for kpt in wfs.mykpts[k1:k2]]

            VV_aii = [VV_sii[spin] for VV_sii in VV_asii]  # PAW corrections

            e1, e2 = self.calculate(kpts, kpts, coulomb, VV_aii,
                                    eig_skn[spin])
            exxvv += e1 * deg
            exxvc += e2 * deg

            if len(wfs.mykpts) == k2:
                break

            # Next spin:
            k1 = K
            k2 = 2 * K
            spin = 1

        exxvv = kd.comm.sum(exxvv)
        exxvc = kd.comm.sum(exxvc)
        kd.comm.sum(eig_skn)

        return exxvv, exxvc, eig_skn

    def calculate(self, kpts1, kpts2, coulomb, VV_aii, eig_kn=None):
        exxvv = 0.0
        exxvc = 0.0

        pd = kpts1[0].psit.pd

        for k1, k2, count in self.ipairs(kpts1, kpts2):
            q_c = k2.k_c - k1.k_c
            qd = KPointDescriptor([q_c])

            pd12 = PWDescriptor(pd.ecut, pd.gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)

            v_G = coulomb.get_potential(pd12)

            e_nn = self.calculate_exx_for_pair(k1, k2, ghat, v_G)
            # e_nn *= k1.weight * count / self.kd.nbzkpts
            e_nn *= count / self.kd.nbzkpts**2
            print(k1.index, k2.index, count, q_c, e_nn)
            exxvv -= 0.5 * k1.f_n.dot(e_nn).dot(k2.f_n)

            if eig_kn is not None:
                eig_kn[k1.index] -= 0.5 * e_nn.dot(k2.f_n)
                eig_kn[k2.index] -= 0.5 * k1.f_n.dot(e_nn)

        for psit, proj, f_n, k_c, weight in kpts1:
            for a, P_ni in proj.items():
                vv_n = np.einsum('ni,ij,nj->n',
                                 P_ni.conj(), VV_aii[a], P_ni).real
                vc_n = np.einsum('ni,ij,nj->n',
                                 P_ni.conj(), self.VC_aii[a], P_ni).real
                exxvv -= vv_n.dot(f_n) * weight
                exxvc -= vc_n.dot(f_n) * weight
                eig_kn[psit.kpt] -= (2 * vv_n + vc_n) * weight

        eig_kn /= self.kd.weight_k[:, np.newaxis]

        return exxvv, exxvc

    def calculate_exx_for_pair(self,
                               k1,
                               k2,
                               ghat,
                               v_G):  # -> float
        Q_annL = [np.einsum('mi,ijL,nj->mnL',
                            k1.proj[a].conj(),
                            Delta_iiL,
                            k2.proj[a])
                  for a, Delta_iiL in enumerate(self.Delta_aiiL)]

        N1 = len(k1.u_nR)
        N2 = len(k2.u_nR)
        exx_nn = np.empty((N1, N2))
        rho_nG = ghat.pd.empty(N2, k1.u_nR.dtype)

        for n1, u1_R in enumerate(k1.u_nR):
            u1cc_R = u1_R.conj()

            n0 = n1 if k1 is k2 else 0

            for n2, rho_G in enumerate(rho_nG[n0:], n0):
                rho_G[:] = ghat.pd.fft(u1cc_R * k2.u_nR[n2])

            ghat.add(rho_nG[n0:],
                     {a: Q_nnL[n1, n0:]
                      for a, Q_nnL in enumerate(Q_annL)})

            for n2, rho_G in enumerate(rho_nG[n0:], n0):
                e = ghat.pd.integrate(rho_G, v_G * rho_G).real
                exx_nn[n1, n2] = e
                if k1 is k2:
                    exx_nn[n2, n1] = e

        return exx_nn

    def ipairs(self, kpts1, kpts2):
        kd = self.kd
        nsym = len(kd.symmetry.op_scc)

        symmetries_k = []
        for k in range(kd.nibzkpts):
            indices = np.where(kd.bz2ibz_k == k)[0]
            sindices = (kd.sym_k[indices] +
                        kd.time_reversal_k[indices] * nsym)
            symmetries_k.append(sindices)

        pairs = defaultdict(int)
        for i1 in range(kd.nibzkpts):
            for s1 in symmetries_k[i1]:
                for i2 in range(kd.nibzkpts):
                    for s2 in symmetries_k[i2]:
                        s3 = self.symmetry_map_ss[s1, s2]
                        if i1 < i2:
                            pairs[(i1, i2, s3)] += 1
                        else:
                            s4 = self.inverse_s[s3]
                            if i1 == i2:
                                pairs[(i1, i1, min(s3, s4))] += 1
                            else:
                                pairs[(i2, i1, s4)] += 1

        lasti1 = None
        lasti2 = None
        for (i1, i2, s), count in sorted(pairs.items()):
            if i1 is not lasti1:
                k1 = kpts1[i1]
                u1_nR = to_real_space(k1.psit)
                rsk1 = RSKPoint(u1_nR, k1.proj, k1.f_n, k1.k_c,
                                k1.weight, k1.psit.kpt)
            if i2 is not lasti2:
                k2 = kpts2[i2]
                u2_nR = to_real_space(k2.psit)
                rsk2 = RSKPoint(u2_nR, k2.proj, k2.f_n, k2.k_c,
                                k2.weight, k2.psit.kpt)

            yield rsk1, self.apply_symmetry(s, rsk2), count

            lasti1 = i1
            lasti2 = i2

    def calculate_valence_valence_paw_corrections(self, D_asp):
        VV_asii = []
        for a, data in enumerate(self.setups):
            VV_sii = []
            D_sp = D_asp[a]
            for D_p in D_sp:
                D_ii = unpack2(D_p) * (len(D_sp) / 2)
                VV_ii = pawexxvv(data, D_ii)
                VV_sii.append(VV_ii)
            VV_asii.append(VV_sii)
        return VV_asii

    def apply_symmetry(self, s: int, rsk):
        if s == 0:
            return rsk

        u1_nR, proj1, f_n, k1_c, weight, ibz_index = rsk

        u2_nR = np.empty_like(u1_nR)
        proj2 = proj1.new()

        nsym = len(self.kd.symmetry.op_scc)
        time_reversal = s > nsym
        sign = 1 - 2 * int(time_reversal)
        U_cc = self.kd.symmetry.op_scc[s % nsym]

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

        return RSKPoint(u2_nR, proj2, f_n, k2_c, weight, ibz_index)


def to_real_space(psit):
    pd = psit.pd
    nbands = len(psit.array)
    u_nR = pd.gd.empty(nbands, pd.dtype, psit.kpt)
    for psit_G, u_R in zip(psit.array, u_nR):
        u_R[:] = pd.ifft(psit_G, psit.kpt)
    return u_nR


def create_symmetry_map(kd: KPointDescriptor):  # -> List[List[int]]
    sym = kd.symmetry
    U_scc = sym.op_scc
    assert (U_scc[0] == np.eye(3)).all()
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


class Hybrid:
    def calculate(self, calc):
        from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
        # print('Using Wigner-Seitz truncated Coulomb interaction.')

        wfs = calc.wfs
        wstc = WignerSeitzTruncatedCoulomb(wfs.gd.cell_cv,
                                           wfs.kd.N_c)
        x = EXX(wfs.kd, wfs.setups)
        evv, evc, e_skn = x.one_shot_calculation(wfs, calc.density.D_asp, wstc)
        return evv, evc, x.exxcc, e_skn


if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW, PW
    h = Atoms('Li', cell=(3, 3, 7), pbc=(1, 1, 0))
    h.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  kpts=(3, 1, 1),
                  # spinpol=True,
                  txt=None)
    h.get_potential_energy()

    evv, evc, ecc, e_kn = Hybrid().calculate(h.calc)
    e = evv + evc + ecc
    print(e * Ha, e_kn * Ha)

    from gpaw.xc.exx import EXX as EXX0
    xx = EXX0(h.calc, bands=(0, 4))
    xx.calculate()
    e0 = xx.get_exx_energy()
    eps0 = xx.get_eigenvalue_contributions()
    print(e0, eps0)
    print(e * Ha - e0, e_kn * Ha - eps0)
