"""Todo: optical limit, Hilbert transform"""

import sys
from math import pi

import numpy as np
from ase.units import Hartree
from ase.utils import devnull

import gpaw.mpi as mpi
from gpaw.utilities.blas import gemm, rk
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.math_func import two_phi_planewave_integrals


class KPoint:
    def __init__(self, K, n1, n2, ut_nR, eps_n, f_n, P_ani, shift_c):
        self.K = K  # BZ k-point index
        self.n1 = n1  # first band
        self.n2 = n2  # first band not included
        self.ut_nR = ut_nR  # periodic part of wave functions in real-space
        self.eps_n = eps_n  # eigenvalues
        self.f_n = f_n  # occupation numbers
        self.P_ani = P_ani  # PAW projections
        self.shift_c = shift_c  # long story. See construct_symmetry_operators


class Chi0:
    def __init__(self, calc, omega_w, ecut=50 / Hartree, hilbert=False,
                 eta=0.2 / Hartree, blocksize=50, spin=0, ftol=1e-6,
                 world=mpi.world, txt=sys.stdout):
        self.calc = calc
        self.omega_w = np.asarray(omega_w)
        self.ecut = ecut
        self.hilbert = hilbert
        self.eta = eta
        self.blocksize = blocksize
        self.spin = spin
        self.ftol = ftol
        self.world = world
        
        if world.rank != 0:
            txt = devnull
        self.fd = txt
        
        if eta == 0.0:
            assert not hilbert
            assert not self.omega_w.real.any()
            
        self.spos_ac = calc.atoms.get_scaled_positions()
        
        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()
        
        self.Kn1n2 = None  # my occupied k-points and bands
        self.distribute_k_points_and_bands()
        
        self.mykpts = [self.get_k_point(K, n1, n2)
                       for K, n1, n2 in self.Kn1n2_k]
        
        vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))
        self.prefactor = 2 / vol / calc.wfs.kd.nbzkpts

        self.ut_KnvR = None  # gradient of wave functions for optical limit
        
    def count_occupied_bands(self):
        self.nocc1 = 9999999
        self.nocc2 = 0
        for kpt in self.calc.wfs.kpt_u:
            f_n = kpt.f_n / kpt.weight
            self.nocc1 = min((f_n > 1 - self.ftol).sum(), self.nocc1)
            self.nocc2 = max((f_n > self.ftol).sum(), self.nocc2)
        self.fd.write('Number of completely filled bands: %d\n' % self.nocc1)
        self.fd.write('Number of non-empty bands: %d\n' % self.nocc2)
        self.fd.write('Total number of bands: %d\n' % self.calc.wfs.bd.nbands)
        
    def distribute_k_points_and_bands(self):
        world = self.world
        nk = self.calc.wfs.kd.nbzkpts
        n = (nk * self.nocc2 + world.size - 1) // world.size
        i1 = world.rank * n
        i2 = min((world.rank + 1) * n, nk * self.nocc2)
        K1, n1 = divmod(i1, self.nocc2)
        K2, n2 = divmod(i2, self.nocc2)
        self.Kn1n2_k = []
        for K in range(K1, K2):
            self.Kn1n2_k.append((K, n1, self.nocc2))
            n1 = 0
        if n2 > 0:
            self.Kn1n2_k.append((K2, n1, n2))
        self.fd.write('k-points: %s\n' % self.calc.wfs.kd.description)
        self.fd.write('Distributing %d x %d bands over %d process%s\n' %
                      (nk, self.nocc2,
                       world.size, ['es', ''][world.size == 1]))
            
    def get_k_point(self, K, n1, n2):
        wfs = self.calc.wfs
        
        U_cc, T, a_a, U_aii, shift_c, time_reversal = \
            self.construct_symmetry_operators(K)
        ik = wfs.kd.bz2ibz_k[K]
        kpt = wfs.kpt_u[ik]
        
        psit_nG = kpt.psit_nG
        ut_nR = wfs.gd.empty(n2 - n1, complex)
        for n in range(n1, n2):
            ut_nR[n - n1] = T(wfs.pd.ifft(psit_nG[n], ik))

        eps_n = kpt.eps_n[n1:n2]
        f_n = kpt.f_n[n1:n2] / kpt.weight
        
        P_ani = []
        for b, U_ii in zip(a_a, U_aii):
            P_ni = np.dot(kpt.P_ani[b][n1:n2], U_ii)
            if time_reversal:
                P_ni = P_ni.conj()
            P_ani.append(P_ni)
        
        return KPoint(K, n1, n2, ut_nR, eps_n, f_n, P_ani, shift_c)
    
    def calculate(self, q_c):
        # Start from scratch and do all empty bands:
        wfs = self.calc.wfs

        q_c = np.asarray(q_c, dtype=float)

        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, complex, qd)
        
        nG = pd.ngmax
        nw = len(self.omega_w)
        chi0_wGG = np.zeros((nw, nG, nG), complex)
        if not q_c.any():
            chi0_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
        else:
            chi0_wxvG = None
            
        Q_aGii = self.calculate_paw_corrections(pd)
        
        return self._calculate(pd, chi0_wGG, chi0_wxvG, Q_aGii,
                               self.nocc1, wfs.bd.nbands)

    def _calculate(self, pd, chi0_wGG, chi0_wxvG, Q_aGii, m1, m2):
        wfs = self.calc.wfs

        if self.eta == 0.0:
            update = self.update_hermitian
        elif self.hilbert:
            update = self.update_hilbert
        else:
            update = self.update
            
        q_c = pd.kd.bzk_kc[0]
        optical_limit = not q_c.any()
            
        for kpt1 in self.mykpts:
            K2 = wfs.kd.find_k_plus_q(q_c, [kpt1.K])[0]
            kpt2 = self.get_k_point(K2, m1, m2)
            Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                       kpt1.shift_c - kpt2.shift_c)
            for n in range(kpt1.n2 - kpt1.n1):
                eps = kpt1.eps_n[n]
                f = kpt1.f_n[n]
                utcc_R = kpt1.ut_nR[n].conj()
                C_aGi = [np.dot(Q_Gii, P_ni[n].conj())
                         for Q_Gii, P_ni in zip(Q_aGii, kpt1.P_ani)]
                for ma in range(0, m2 - m1, self.blocksize):
                    mb = min(ma + self.blocksize, m2 - m1)
                    n_mG = self.calculate_pair_densities(utcc_R, C_aGi,
                                                         kpt2, ma, mb, pd, Q_G)
                    deps_m = eps - kpt2.eps_n[ma:mb]
                    df_m = f - kpt2.f_n[ma:mb]
                    if optical_limit:
                        self.update_optical_limit(
                            n, kpt1, kpt2, deps_m, df_m, n_mG,
                            ma, mb, chi0_wxvG)
                    update(n_mG, deps_m, df_m, chi0_wGG)
                    
        self.world.sum(chi0_wGG)
        if optical_limit:
            self.world.sum(chi0_wxvG)
        
        if self.eta == 0.0:
            # Fill in upper triangle also:
            nG = pd.ngmax
            il = np.tril_indices(nG, -1)
            iu = il[::-1]
            for chi0_GG in chi0_wGG:
                chi0_GG[iu] = chi0_GG[il].conj()
                
        elif self.hilbert:
            for G in range(nG):
                chi0_wGG[:, :, G] = np.dot(A_ww, chi0_wGG[:, :, G])
        
        return pd, chi0_wGG, chi0_wxvG
        
    def update(self, n_mG, deps_m, df_m, chi0_wGG):
        for w, omega in enumerate(self.omega_w):
            x_m = df_m * (1.0 / (omega + deps_m + 1j * self.eta) -
                          1.0 / (omega - deps_m + 1j * self.eta))
            nx_mG = n_mG * x_m[:, np.newaxis]
            gemm(self.prefactor, n_mG.conj(), np.ascontiguousarray(nx_mG.T),
                 1.0, chi0_wGG[w])

    def update_hermitian(self, n_mG, deps_m, df_m, chi0_wGG):
        for w, omega in enumerate(self.omega_w):
            x_m = (-2 * df_m * deps_m / (omega.imag**2 + deps_m**2))**0.5
            nx_mG = n_mG.T.copy() * x_m
            rk(-self.prefactor, nx_mG, 1.0, chi0_wGG[w])

    def update_hilbert(self, n_mG, deps_m, df_m, chi0_wGG):
        domega = self.omega_w[1]
        for omega, df, n_G in zip(deps_m, df_m, n_mG):
            w = omega / domega
            iw = int(w)
            weights = df * np.array([[1 - w + iw], [w - iw]])
            x_2G = n_G * weights**0.5
            rk(self.prefactor, x_2G, 1.0, chi0_wGG[iw:iw + 2])

    def calculate_pair_densities(self, utcc_R, C_aGi, kpt2,
                                 ma, mb, pd, Q_G):
        dv = pd.gd.dv
        n_mG = pd.empty(mb - ma)
        for m in range(ma, mb):
            n_R = utcc_R * kpt2.ut_nR[m]
            pd.tmp_R[:] = n_R
            pd.fftplan.execute()
            n_mG[m - ma] = pd.tmp_Q.ravel()[Q_G] * dv
        
        # PAW corrections:
        for C_Gi, P_mi in zip(C_aGi, kpt2.P_ani):
            gemm(1.0, C_Gi, P_mi[ma:mb], 1.0, n_mG, 't')
            
        return n_mG
    
    def update_optical_limit(self, n, kpt1, kpt2, deps_m, df_m, n_mG,
                             ma, mb, chi0_wxvG):
        if self.ut_KnvR is None:
            self.ut_KnvR = self.calculate_derivatives()
            
        ut_vR = self.ut_KnvR[kpt1.K][n]
        
        atomdata_a = self.calc.wfs.setups
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]
        
        n0_mv = -self.calc.wfs.gd.integrate(ut_vR, kpt2.ut_nR[ma:mb]).T
        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            gemm(1.0, C_vi, P_mi[ma:mb], 1.0, n0_mv, 'c')

        n0_mv *= 1j / deps_m[:, np.newaxis]
        n_mG[:, 0] = n0_mv[:, 0]

        for w, omega in enumerate(self.omega_w):
            x_m = (self.prefactor *
                   df_m * (1.0 / (omega + deps_m + 1j * self.eta) -
                           1.0 / (omega - deps_m + 1j * self.eta)))
            chi0_wxvG[w, :, :, 0] += np.dot(x_m, n0_mv * n0_mv.conj())
            chi0_wxvG[w, 0, :, 1:] += np.dot(x_m * n0_mv.T, n_mG[:, 1:].conj())
            chi0_wxvG[w, 1, :, 1:] += np.dot(x_m * n0_mv.T.conj(), n_mG[:, 1:])
            
    def get_fft_indices(self, K, K2, q_c, pd, shift0_c):
        kd = self.calc.wfs.kd
        Q_G = pd.Q_qG[0]
        shift_c = (shift0_c +
                   (q_c - kd.bzk_kc[K2] + kd.bzk_kc[K]).round().astype(int))
        if shift_c.any():
            q_cG = np.unravel_index(Q_G, pd.gd.N_c)
            q_cG = [q_G + shift for q_G, shift in zip(q_cG, shift_c)]
            Q_G = np.ravel_multi_index(q_cG, pd.gd.N_c, 'wrap')
        return Q_G
        
    def construct_symmetry_operators(self, K):
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
            
        See the get_k_point() method for how tu use these tuples.
        """
        
        wfs = self.calc.wfs
        kd = wfs.kd

        s = kd.sym_k[K]
        U_cc = kd.symmetry.op_scc[s]
        time_reversal = kd.time_reversal_k[K]
        ik = kd.bz2ibz_k[K]
        k_c = kd.bzk_kc[K]
        ik_c = kd.ibzk_kc[ik]
        
        sign = 1 - 2 * time_reversal
        shift_c = np.dot(U_cc, ik_c) - k_c * sign
        assert np.allclose(shift_c.round(), shift_c)
        shift_c = shift_c.round().astype(int)
        
        if (U_cc == np.eye(3)).all():
            T = lambda f_R: f_R
        else:
            N_c = self.calc.wfs.gd.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            T = lambda f_R: f_R.ravel()[i].reshape(N_c)

        if time_reversal:
            T0 = T
            T = lambda f_R: T0(f_R).conj()
            shift_c *= -1
        
        a_a = []
        U_aii = []
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            b = kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        return U_cc, T, a_a, U_aii, shift_c, time_reversal

    def calculate_paw_corrections(self, pd):
        wfs = self.calc.wfs
        q_v = pd.K_qv[0]
        optical_limit = not q_v.any()

        G_Gv = pd.G_Qv[pd.Q_qG[0]] + q_v
        if optical_limit:
            G_Gv[0] = 1
            
        pos_av = np.dot(self.spos_ac, pd.gd.cell_cv)
        
        # Collect integrals for all species:
        Q_xGii = {}
        for id, atomdata in wfs.setups.setups.items():
            Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
            ni = atomdata.ni
            Q_xGii[id] = Q_Gii.reshape((-1, ni, ni))
            
        Q_aGii = []
        for a, atomdata in enumerate(wfs.setups):
            id = wfs.setups.id_a[a]
            Q_Gii = Q_xGii[id]
            x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
            Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)
                
        return Q_aGii

    def calculate_derivatives(self):
        if 0:
            from gpaw.fd_operators import Gradient
            g_v = [Gradient(self.calc.wfs.gd, v, 1.0, 4,complex).apply
                   for v in range(3)]
        wfs = self.calc.wfs
        ut_KnvR = {}
        for K, n1, n2 in self.Kn1n2_k:
            U_cc, T, a_a, U_aii, shift_c, time_reversal = \
                self.construct_symmetry_operators(K)
            A_cv = wfs.gd.cell_cv
            M_vv = np.dot(np.dot(A_cv.T, U_cc.T), np.linalg.inv(A_cv).T)
            ik = wfs.kd.bz2ibz_k[K]
            kpt = wfs.kpt_u[ik]
            psit_nG = kpt.psit_nG
            iG_Gv = 1j * wfs.pd.G_Qv[wfs.pd.Q_qG[ik]]
            ut_nvR = wfs.gd.zeros((n2 - n1, 3), complex)
            for n in range(n1, n2):
                for v in range(3):
                    if 1:
                        ut_R = T(wfs.pd.ifft(iG_Gv[:, v] * psit_nG[n], ik))
                        for v2 in range(3):
                            ut_nvR[n - n1, v2] += ut_R * M_vv[v,v2]
                    else:
                        ut_R = T(wfs.pd.ifft(psit_nG[n], ik))
                        g_v[v](ut_R, ut_nvR[n - n1, v],
                               np.ones((3, 2), complex))
            ut_KnvR[K] = ut_nvR
            
        return ut_KnvR


if np.__version__ < '1.6':
    old_unravel_index = np.unravel_index
    def new_unravel_index(indices, dims):
        if isinstance(indices, int):
            return old_unravel_index(indices, dims)
        return np.array([old_unravel_index(index, dims)
                         for index in indices]).T
    np.unravel_index = new_unravel_index
    def ravel_multi_index(i, d, mode):
        i = i % d[:, None]
        return np.dot([d[1] * d[2], d[2], 1], i)
    np.ravel_multi_index = ravel_multi_index
