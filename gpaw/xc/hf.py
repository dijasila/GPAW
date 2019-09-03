from collections import namedtuple, defaultdict
from math import pi
from typing import List, Optional, Tuple
import numpy as np
from ase.units import Ha

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.xc import XC
from gpaw.xc.exx import pawexxvv
from gpaw.xc.tools import _vxc
from gpaw.utilities import unpack, unpack2
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb as WSTC


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

        # PAW-correction stuff:
        self.Delta_aiiL = []
        self.VC_aii = []
        self.ecc = 0.0
        for a, data in enumerate(setups):
            self.Delta_aiiL.append(data.Delta_iiL)
            self.VC_aii.append(unpack(data.X_p))
            self.ecc += data.ExxC

        self.symmetry_map_ss = create_symmetry_map(kd)
        self.inverse_s = self.symmetry_map_ss[:, 0]

    def calculate_energy(self, kpts, VV_aii, v_knG=None):
        pd = kpts[0].psit.pd

        if v_knG is not None:
            nbands = len(v_knG[0])
            shapes = [(nbands, len(VV_ii)) for VV_ii in VV_aii]
            v_kani = [{a: np.zeros(shape, complex)
                       for a, shape in enumerate(shapes)}
                      for _ in range(len(v_knG))]
        else:
            v_nG = None
            v_kani = None

        exxvv = 0.0
        for i1, k1, k2, count in self.ipairs(kpts, kpts):
            q_c = k2.k_c - k1.k_c
            qd = KPointDescriptor([q_c])

            pd12 = PWDescriptor(pd.ecut, pd.gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)

            if v_knG is not None:
                v_nG = v_knG[i1]
                v_ani = v_kani[i1]

            v_G = self.coulomb.get_potential(pd12)
            e_nn = self.calculate_exx_for_pair(k1, k2, ghat, v_G,
                                               pd, i1, k2.f_n, v_nG, v_ani)
            e_nn *= count / self.kd.nbzkpts**2
            exxvv -= 0.5 * k1.f_n.dot(e_nn).dot(k2.f_n)

        exxvc = 0.0
        for i, kpt in enumerate(kpts):
            for a, P_ni in kpt.proj.items():
                vv_n = np.einsum('ni,ij,nj->n',
                                 P_ni.conj(), VV_aii[a], P_ni).real
                vc_n = np.einsum('ni,ij,nj->n',
                                 P_ni.conj(), self.VC_aii[a], P_ni).real
                exxvv -= vv_n.dot(kpt.f_n) * kpt.weight
                exxvc -= vc_n.dot(kpt.f_n) * kpt.weight

        if v_knG is not None:
            for v_nG, v_ani in zip(v_knG, v_kani):
                self.pt.add(v_nG, v_ani)

        return exxvv, exxvc

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
                vv_n = np.einsum('ni,ij,nj->n',
                                 P_ni.conj(), VV_aii[a], P_ni).real
                vc_n = np.einsum('ni,ij,nj->n',
                                 P_ni.conj(), self.VC_aii[a], P_ni).real
                e_kn[i] -= (2 * vv_n + vc_n)

    def calculate_exx_for_pair(self,
                               k1,
                               k2,
                               ghat,
                               v_G,
                               pd,
                               index,
                               f2_n,
                               vpsit_nG=None,
                               v_ani=None):
        Q_annL = [np.einsum('mi, ijL, nj -> mnL',
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
                vrho_G = v_G * rho_G
                if vpsit_nG is not None:
                    for a, v_L in ghat.integrate(vrho_G).items():
                        v_ii = f2_n[n1] * self.Delta_aiiL[a].dot(v_L[0])
                        v_ani[a][n1] += v_ii.dot(k2.proj[a][n2])
                    vrho_R = ghat.pd.ifft(vrho_G)
                    vpsit_nG[n1] -= f2_n[n2] * pd.fft(
                        vrho_R.conj() * k2.u_nR[n2], index)
                e = ghat.pd.integrate(rho_G, vrho_G).real
                exx_nn[n1, n2] = e
                if k1 is k2:
                    exx_nn[n2, n1] = e

        return exx_nn

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

        # pairs: Dict[Tuple[int, int, int], int]

        if kpts1 is kpts2:
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
        else:
            pairs = {}
            for i1 in range(len(kpts1)):
                for i2 in range(kd.nibzkpts):
                    for s2 in symmetries_k[i2]:
                        pairs[(i1, i2, s2)] = 1

        lasti1 = -1
        lasti2 = -1
        for (i1, i2, s), count in sorted(pairs.items()):
            if i1 is not lasti1:
                k1 = kpts1[i1]
                u1_nR = to_real_space(k1.psit)
                rsk1 = RSKPoint(u1_nR, k1.proj, k1.f_n, k1.k_c,
                                k1.weight)  # , k1.psit.kpt)
            if i2 is not lasti2:
                k2 = kpts2[i2]
                u2_nR = to_real_space(k2.psit)
                rsk2 = RSKPoint(u2_nR, k2.proj, k2.f_n, k2.k_c,
                                k2.weight)  # , k2.psit.kpt)

            yield i1, rsk1, self.apply_symmetry(s, rsk2), count

            lasti1 = i1
            lasti2 = i2

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
    nbands = len(psit.array)
    u_nR = pd.gd.empty(nbands, pd.dtype, psit.kpt)
    for psit_G, u_R in zip(psit.array, u_nR):
        u_R[:] = pd.ifft(psit_G, psit.kpt)
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


def parse_name(name: str) -> Tuple[str, float, float]:
    if name == 'EXX':
        return '', 1.0, np.nan
    if name == 'PBE0':
        return 'HYB_GGA_XC_PBEH', 0.25, np.nan
    if name == 'HSE03':
        return 'HYB_GGA_XC_HSE03', 0.25, 0.106
    if name == 'HSE06':
        return 'HYB_GGA_XC_HSE06', 0.25, 0.11
    if name == 'B3LYP':
        return 'HYB_GGA_XC_B3LYP', 0.2, np.nan


class Hybrid:
    orbital_dependent = True
    type = 'HYB'
    ftol = 1e-9

    def __init__(self,
                 name: str = None,
                 xc=None,
                 exx_fraction: float = None,
                 omega: float = None):

        if name is not None:
            assert xc is None and exx_fraction is None and omega is None
            xc, exx_fraction, omega = parse_name(name)
            self.name = name
        else:
            assert xc is not None and exx_fraction is not None

        if isinstance(xc, (str, dict)):
            xc = XC(xc)

        self.xc = xc
        self.exx_fraction = exx_fraction
        self.omega = omega

        self.initialized = False

        self.dens = None
        self.ham = None
        self.wfs = None

        self.xx = None
        self.coulomb = None
        self.cache = {}

        self.evv = np.nan
        self.evc = np.nan

        self.vt_sR = None
        self.spos_ac = None

    def get_setup_name(self):
        return 'LDA'

    def initialize(self, dens, ham, wfs, occupations):
        self.dens = dens
        self.ham = ham
        self.wfs = wfs

    def get_description(self):
        return '*****\n' * 4

    def set_grid_descriptor(self, gd):
        pass

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate(self, gd, nt_sr, vt_sr):
        if self.xc is None:
            return self.evv + self.evc
        e_r = gd.empty()
        self.xc.calculate(gd, nt_sr, vt_sr, e_r)
        return self.evv + self.evc + gd.integrate(e_r)

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        return 0.0

    def get_kinetic_energy_correction(self):
        return -2 * self.evv

    def _initialize(self):
        if self.initialized:
            return

        wfs = self.wfs
        kd = wfs.kd

        assert kd.comm.size == 1 or kd.comm.size == 2 and wfs.nspins == 2
        assert wfs.bd.comm.size == 1

        # print('Using Wigner-Seitz truncated Coulomb interaction.')
        self.xx = EXX(wfs.kd, wfs.setups, self.spos_ac)

        self.initialized = True

        if self.omega is None:
            # Wigner-Seitz truncated Coulomb:
            self.coulomb = WSTC(wfs.gd.cell_cv, wfs.kd.N_c)
        else:
            def coulomb(pd):
                G2_G = pd.G2_qG[0]
                x_G = 1 - np.exp(-G2_G / (4 * self.omega**2))
                return 4 * np.pi * x_G / G2_G
            self.coulomb = coulomb

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG=None, dH_asp=None):
        self._initialize()

        kd = self.wfs.kd
        spin = kpt.s

        if kpt.f_n is None:
            print(f'F0 {spin}')
            if self.vt_sR is None:
                from gpaw.xc import XC
                lda = XC('LDA')
                nt_sr = self.dens.nt_sg
                vt_sr = np.zeros_like(nt_sr)
                self.vt_sR = self.dens.gd.zeros(self.wfs.nspins)
                lda.calculate(self.dens.finegd, nt_sr, vt_sr)
                for vt_R, vt_r in zip(self.vt_sR, vt_sr):
                    vt_R[:], _ = self.dens.pd3.restrict(vt_r, self.dens.pd2)

            pd = kpt.psit.pd
            for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
                Htpsit_G += pd.fft(self.vt_sR[kpt.s] *
                                   pd.ifft(psit_G, kpt.k), kpt.q)
            return

        self.vt_sR = None
        deg = 2 / self.wfs.nspins

        if kpt.psit.array.base is psit_xG.base:
            print(f'1 {spin}')
            if (spin, kpt.k) not in self.cache:
                if spin == 0:
                    self.evv = 0.0
                    self.evc = 0.0
                VV_aii = self.calculate_valence_valence_paw_corrections(spin)
                K = kd.nibzkpts
                k1 = (spin - kd.comm.rank) * K
                k2 = k1 + K
                kpts = self.wfs.mykpts[k1:k2]
                for k in kpts:
                    self.cache[(spin, k.k)] = np.zeros_like(k.psit.array)
                kpts = [KPoint(kpt.psit,
                               kpt.projections,
                               kpt.f_n / kpt.weight,  # scale to [0, 1] range
                               kd.ibzk_kc[kpt.k],
                               kd.weight_k[kpt.k])
                        for kpt in kpts]
                evv, evc = self.xx.calculate_energy(
                    kpts,
                    self.coulomb,
                    VV_aii,
                    v_knG=[self.cache[(spin, k)] for k in range(len(kpts))])
                self.evv += evv * deg
                self.evc += evc * deg
            Htpsit_xG += self.cache.pop((spin, kpt.k))
        else:
            assert len(self.cache) == 0
            print(f'2 {spin}')
            VV_aii = self.calculate_valence_valence_paw_corrections(spin)
            K = kd.nibzkpts
            k1 = (spin - kd.comm.rank) * K
            k2 = k1 + K
            kpts2 = [KPoint(kpt.psit,
                            kpt.projections,
                            kpt.f_n / kpt.weight,  # scale to [0, 1] range
                            kd.ibzk_kc[kpt.k],
                            kd.weight_k[kpt.k])
                     for kpt in self.wfs.mykpts[k1:k2]]

            psit = kpt.psit.new(buf=psit_xG)
            kpts1 = [KPoint(psit,
                            kpt.projections,
                            None,
                            kd.ibzk_kc[kpt.k],
                            None)]
            self.xx.calculate_eigenvalues(
                kpts1, kpts2,
                self.coulomb,
                VV_aii,
                np.zeros((1, self.wfs.bd.nbands)),
                v_nG=Htpsit_xG)

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        pass  # 1 / 0

    def rotate(self, kpt, U_nn):
        pass  # 1 / 0

    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        pass  # 1 / 0

    def calculate_valence_valence_paw_corrections(self, spin):
        VV_aii = []
        for a, data in enumerate(self.wfs.setups):
            D_p = self.dens.D_asp[a][spin]
            D_ii = unpack2(D_p) * (self.wfs.nspins / 2)
            VV_ii = pawexxvv(data, D_ii)
            VV_aii.append(VV_ii)
        return VV_aii

    def calculate_energy(self):
        self._initialize()

        wfs = self.wfs
        kd = wfs.kd
        # rank = kd.comm.rank
        # size = kd.comm.size

        nocc = max(((kpt.f_n / kpt.weight) > self.ftol).sum()
                   for kpt in wfs.mykpts)

        evv = 0.0
        evc = 0.0

        for spin in range(wfs.nspins):
            VV_aii = self.calculate_valence_valence_paw_corrections(spin)
            K = kd.nibzkpts
            k1 = spin * K
            k2 = k1 + K
            kpts = [KPoint(kpt.psit.view(0, nocc),
                           kpt.projections.view(0, nocc),
                           kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                           kd.ibzk_kc[kpt.k],
                           kd.weight_k[kpt.k])
                    for kpt in wfs.mykpts[k1:k2]]
            e1, e2 = self.xx.calculate_energy(kpts,
                                              self.coulomb,
                                              VV_aii)
            evv += e1
            evc += e2

        deg = 2 / wfs.nspins
        evv = kd.comm.sum(evv) * deg
        evc = kd.comm.sum(evc) * deg

        if self.xc:
            pass

        return evv * Ha, evc * Ha

    def calculate_eigenvalues(self, n1, n2, kpts):
        self._initialize()

        wfs = self.wfs
        kd = wfs.kd
        # rank = kd.comm.rank
        # size = kd.comm.size

        nocc = max(((kpt.f_n / kpt.weight) > self.ftol).sum()
                   for kpt in wfs.mykpts)

        self.e_skn = np.zeros((wfs.nspins, len(kpts), n2 - n1))

        for spin in range(wfs.nspins):
            VV_aii = self.calculate_valence_valence_paw_corrections(spin)
            K = kd.nibzkpts
            k1 = spin * K
            k2 = k1 + K
            kpts1 = [KPoint(kpt.psit.view(n1, n2),
                            kpt.projections.view(n1, n2),
                            kpt.f_n[n1:n2] / kpt.weight,  # scale to [0, 1]
                            kd.ibzk_kc[kpt.k],
                            kd.weight_k[kpt.k])
                     for kpt in (wfs.mykpts[k] for k in kpts)]
            kpts2 = [KPoint(kpt.psit.view(0, nocc),
                            kpt.projections.view(0, nocc),
                            kpt.f_n[:nocc] / kpt.weight,  # scale to [0, 1]
                            kd.ibzk_kc[kpt.k],
                            kd.weight_k[kpt.k])
                     for kpt in wfs.mykpts[k1:k2]]
            self.xx.calculate_eigenvalues(
                kpts1, kpts2,
                self.coulomb,
                VV_aii,
                self.e_skn[spin])

        kd.comm.sum(self.e_skn)
        self.e_skn *= self.exx_fraction

        if self.xc:
            vxc_skn = _vxc(self.xc, self.ham, self.dens, self.wfs) / Ha
            self.e_skn += vxc_skn[:, kpts, n1:n2]

        return self.e_skn * Ha

    def summary(self, log):
        log('????????????\n' * 4)


if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW, PW
    h = Atoms('H', cell=(3, 3, 3), pbc=(1, 1, 1))
    h.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  setups='ae',
                  kpts=(1, 1, 1),
                  spinpol=True,
                  txt=None)
    h.get_potential_energy()
    exx = Hybrid()
    h.calc.get_xc_difference(exx)
    e = exx.evv + exx.evc + exx.exx.ecc
    print(e * Ha, exx.e_skn * Ha)

    from gpaw.xc.exx import EXX as EXX0
    xx = EXX0(h.calc, bands=(0, 1))
    xx.calculate()
    e0 = xx.get_exx_energy()
    eps0 = xx.get_eigenvalue_contributions()
    print(e0, eps0)
    print(e * Ha - e0, exx.e_skn * Ha - eps0)
