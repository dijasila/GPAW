from collections import namedtuple, defaultdict
from math import pi, nan
from typing import List, Tuple
from io import StringIO

import numpy as np
from ase.units import Ha

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.xc import XC
from gpaw.xc.exx import pawexxvv
from gpaw.xc.tools import _vxc
from gpaw.utilities import unpack, unpack2
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb as WSTC
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


class ShortRangeCoulomb:
    def __init__(self, omega):
        self.omega = omega

    def get_potential(self, pd):
        G2_G = pd.G2_qG[0]
        x_G = 1 - np.exp(-G2_G / (4 * self.omega**2))
        with np.errstate(invalid='ignore'):
            v_G = 4 * pi * x_G / G2_G
        if pd.kd.gamma:
            v_G[0] = pi / self.omega**2
        return v_G


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

    def calculate(self, kpts1, kpts2, VV_aii, derivatives=False, e_kn=None):
        pd = kpts1[0].psit.pd
        gd = pd.gd.new_descriptor(comm=mpi.serial_comm)
        comm = self.comm

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
        for i1, k1, k2, count in self.ipairs(kpts1, kpts2):
            q_c = k2.k_c - k1.k_c
            qd = KPointDescriptor([-q_c])

            pd12 = PWDescriptor(pd.ecut, gd, pd.dtype, kd=qd)
            ghat = PWLFC([data.ghat_l for data in self.setups], pd12)
            ghat.set_positions(self.spos_ac)

            if derivatives:
                v_nG = v_knG[i1]
                v_ani = v_kani[i1]
            else:
                v_nG = None
                v_ani = None

            v_G = self.coulomb.get_potential(pd12)
            e_nn = self.calculate_exx_for_pair(k1, k2, ghat, v_G,
                                               kpts1[i1].psit.pd,
                                               kpts1[i1].psit.kpt,
                                               k2.f_n, v_nG, v_ani)

            print(k1.k_c[2], k2.k_c[2], kpts1 is kpts2, count, e_nn, k1.f_n)
            e_nn *= count / self.kd.nbzkpts
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
                               pd,
                               index,
                               f2_n,
                               vpsit_nG=None,
                               v_ani=None):
        Q_annL = [np.einsum('mi, ijL, nj -> mnL',
                            k1.proj[a],
                            Delta_iiL,
                            k2.proj[a].conj())
                  for a, Delta_iiL in enumerate(self.Delta_aiiL)]

        N1 = len(k1.u_nR)
        N2 = len(k2.u_nR)
        exx_nn = np.zeros((N1, N2))
        rho_nG = ghat.pd.empty(N2, k1.u_nR.dtype)
        S = self.comm.size
        for n1, u1_R in enumerate(k1.u_nR):
            n0 = n1 if k1 is k2 else 0
            n2a = min(n0 + (N2 - n0 + S - 1) // S * self.comm.rank, N2)
            n2b = min(n2a + (N2 - n0 + S - 1) // S, N2)
            for n2, rho_G in enumerate(rho_nG[n2a:n2b], n2a):
                rho_G[:] = ghat.pd.fft(u1_R * k2.u_nR[n2].conj())

            ghat.add(rho_nG[n2a:n2b],
                     {a: Q_nnL[n1, n2a:n2b]
                      for a, Q_nnL in enumerate(Q_annL)})

            for n2, rho_G in enumerate(rho_nG[n2a:n2b], n2a):
                vrho_G = v_G * rho_G
                if vpsit_nG is not None:
                    for a, v_xL in ghat.integrate(vrho_G).items():
                        v_ii = self.Delta_aiiL[a].dot(v_xL[0])
                        v_ani[a][n1] -= v_ii.dot(k2.proj[a][n2]) * f2_n[n1]
                        if k1 is k2 and n1 != n2:
                            v_ani[a][n2] -= (v_ii.conj().dot(k2.proj[a][n1]) *
                                             f2_n[n2])
                    vrho_R = ghat.pd.ifft(vrho_G)
                    vpsit_nG[n1] -= f2_n[n2] * pd.fft(
                        vrho_R * k2.u_nR[n2], index, local=True)
                    if k1 is k2 and n1 != n2:
                        vpsit_nG[n2] -= f2_n[n1] * pd.fft(
                            vrho_R * k2.u_nR[n1], index, local=True)

                e = ghat.pd.integrate(rho_G, vrho_G).real
                exx_nn[n1, n2] = e
                if k1 is k2:
                    exx_nn[n2, n1] = e

        return exx_nn

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
            if i1 != lasti1:
                k1 = kpts1[i1]
                u1_nR = to_real_space(k1.psit)
                rsk1 = RSKPoint(u1_nR, k1.proj.broadcast(),
                                k1.f_n, k1.k_c,
                                k1.weight)  # , k1.psit.kpt)
                lasti1 = i1
            if i2 == i1 and kpts1 is kpts2:
                rsk2 = rsk1
            elif i2 != lasti2:
                k2 = kpts2[i2]
                u2_nR = to_real_space(k2.psit)
                rsk2 = RSKPoint(u2_nR, k2.proj.broadcast(),
                                k2.f_n, k2.k_c,
                                k2.weight)  # , k2.psit.kpt)
                lasti2 = i2

            yield i1, rsk1, self.apply_symmetry(s, rsk2), count

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


def parse_name(name: str) -> Tuple[str, float, float]:
    if name == 'EXX':
        return 'null', 1.0, 0.0
    if name == 'PBE0':
        return 'HYB_GGA_XC_PBEH', 0.25, 0.0
    if name == 'HSE03':
        return 'HYB_GGA_XC_HSE03', 0.25, 0.106
    if name == 'HSE06':
        return 'HYB_GGA_XC_HSE06', 0.25, 0.11
    if name == 'B3LYP':
        return 'HYB_GGA_XC_B3LYP', 0.2, 0.0


class Hybrid:
    orbital_dependent = True
    type = 'HYB'
    ftol = 1e-9

    def __init__(self,
                 name: str = None,
                 xc=None,
                 exx_fraction: float = None,
                 omega: float = None,
                 mix_all: bool = True):
        if name is not None:
            assert xc is None and exx_fraction is None and omega is None
            xc, exx_fraction, omega = parse_name(name)
            self.name = name
        else:
            assert xc is not None and exx_fraction is not None
            self.name = '???'

        if xc:
            xc = XC(xc)

        self.xc = xc
        self.exx_fraction = exx_fraction
        self.omega = omega
        self.mix_all = mix_all

        self.initialized = False

        self.dens = None
        self.ham = None
        self.wfs = None

        self.xx = None
        self.coulomb = None
        self.v_knG = {}
        self.spin = -1

        self.evv = np.nan
        self.evc = np.nan
        self.ecc = np.nan
        self.ekin = np.nan

        self.vt_sR = None
        self.spos_ac = None

        self.description = ''

    def get_setup_name(self):
        return 'PBE'
        return 'LDA'

    def initialize(self, dens, ham, wfs, occupations):
        self.dens = dens
        self.ham = ham
        self.wfs = wfs
        assert wfs.world.size == wfs.gd.comm.size

    def get_description(self):
        return self.description

    def set_grid_descriptor(self, gd):
        pass

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate(self, gd, nt_sr, vt_sr):
        if not self.xc:
            return self.evv + self.evc
        e_r = gd.empty()
        self.xc.calculate(gd, nt_sr, vt_sr, e_r)
        print(self.ecc, self.evv, self.evc, gd.integrate(e_r))
        return self.ecc + self.evv + self.evc + gd.integrate(e_r)

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        if not self.xc:
            return 0.0
        return self.xc.calculate_paw_correction(setup, D_sp, dH_sp, a=a)

    def get_kinetic_energy_correction(self):
        print(self.ekin)
        return self.ekin

    def _initialize(self):
        if self.initialized:
            return

        wfs = self.wfs
        kd = wfs.kd

        assert kd.comm.size == 1 or kd.comm.size == 2 and wfs.nspins == 2
        assert wfs.bd.comm.size == 1

        if self.omega:
            coulomb = ShortRangeCoulomb(self.omega)
        else:
            # Wigner-Seitz truncated Coulomb:
            output = StringIO()
            coulomb = WSTC(wfs.gd.cell_cv, wfs.kd.N_c, txt=output)
            self.description += output.getvalue()

        self.xx = EXX(wfs.kd, wfs.setups, wfs.pt, coulomb, self.spos_ac)

        self.ecc = sum(setup.ExxC for setup in wfs.setups) * self.exx_fraction

        self.initialized = True

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG=None, dH_asp=None):
        self._initialize()

        kd = self.wfs.kd
        spin = kpt.s

        if kpt.f_n is None:
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
            if not self.v_knG:
                self.spin = spin
                if spin == 0:
                    self.evv = 0.0
                    self.evc = 0.0
                    self.ekin = 0.0
                VV_aii = self.calculate_valence_valence_paw_corrections(spin)
                K = kd.nibzkpts
                k1 = (spin - kd.comm.rank) * K
                k2 = k1 + K
                kpts = [KPoint(kpt.psit,
                               kpt.projections,
                               kpt.f_n / kpt.weight,  # scale to [0, 1] range
                               kd.ibzk_kc[kpt.k],
                               kd.weight_k[kpt.k])
                        for kpt in self.wfs.mykpts[k1:k2]]
                evv, evc, ekin, self.v_knG = self.xx.calculate(
                    kpts, kpts,
                    VV_aii,
                    derivatives=True)
                self.evv += evv * deg * self.exx_fraction
                self.evc += evc * deg * self.exx_fraction
                self.ekin += ekin * deg * self.exx_fraction
            assert self.spin == spin
            Htpsit_xG += self.v_knG.pop(kpt.k) * self.exx_fraction
        else:
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
            P = kpt.projections.new()
            psit.matrix_elements(self.wfs.pt, out=P)

            kpts1 = [KPoint(psit,
                            P,
                            kpt.f_n + nan,
                            kd.ibzk_kc[kpt.k],
                            nan)]
            _, _, _, v_1xG = self.xx.calculate(
                kpts1, kpts2,
                VV_aii,
                derivatives=True)
            Htpsit_xG += self.exx_fraction * v_1xG[0]

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        if self.mix_all or kpt.f_n is None:
            return

        n = (kpt.f_n > kpt.weight * self.ftol).sum()
        H_nn[n:, :n] = 0.0
        H_nn[:n, n:] = 0.0

    def rotate(self, kpt, U_nn):
        pass  # 1 / 0

    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        pass  # 1 / 0

    def calculate_valence_valence_paw_corrections(self, spin):
        VV_aii = {}
        for a, D_sp in self.dens.D_asp.items():
            data = self.wfs.setups[a]
            D_p = D_sp[spin]
            D_ii = unpack2(D_p) * (self.wfs.nspins / 2)
            VV_ii = pawexxvv(data, D_ii)
            VV_aii[a] = VV_ii
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
            e1, e2, _, _ = self.xx.calculate(kpts, kpts, VV_aii)
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
            _, _, _, v_knG = self.xx.calculate(
                kpts1, kpts2,
                VV_aii,
                e_kn=self.e_skn[spin],
                derivatives=True)

        kd.comm.sum(self.e_skn)
        self.e_skn *= self.exx_fraction

        if self.xc:
            vxc_skn = _vxc(self.xc, self.ham, self.dens, self.wfs) / Ha
            self.e_skn += vxc_skn[:, kpts, n1:n2]

        print(self.e_skn)
        for kpt, v_nG in zip(kpts1, v_knG.values()):
            for v_G, psit_G in zip(v_nG, kpt.psit.array):
                print(kpt.psit.pd.integrate(v_G, psit_G)*0.5)

        return self.e_skn * Ha

    def summary(self, log):
        log(self.description)


if __name__ == '__main__':
    from ase import Atoms
    from gpaw import GPAW, PW
    h = Atoms('H', cell=(3, 3, 3), pbc=(1, 1, 1))
    h = Atoms('H2', cell=(3, 3, 3), pbc=1, positions=[[0, 0, 0], [0, 0, 0.75]])
    h.calc = GPAW(mode=PW(100, force_complex_dtype=True),
                  setups='ae',
                  kpts=(1, 1, 2),
                  #spinpol=True,
                  txt=None
                  )
    h.get_potential_energy()
    x = Hybrid('EXX')

    # h.calc.get_xc_difference(exx)
    # e = exx.evv + exx.evc + exx.exx.ecc
    # print(e * Ha, exx.e_skn * Ha)

    c = h.calc
    x.initialize(c.density, c.hamiltonian, c.wfs, c.occupations)
    x.set_positions(c.spos_ac)
    e = x.calculate_energy()
    print(e)
    x.calculate_eigenvalues(0, 1, [0])
    print(x.e_skn * Ha)

    from gpaw.xc.exx import EXX as EXX0
    xx = EXX0(c, bands=(0, 1))
    xx.calculate()
    e0 = xx.get_exx_energy()
    eps0 = xx.get_eigenvalue_contributions()
    print(e0, eps0)
    print(e0 - e[0] - e[1])
    print(eps0 - x.e_skn * Ha)
    #print(e * Ha - e0, xx.e_skn * Ha - eps0)
    print(x.description)
