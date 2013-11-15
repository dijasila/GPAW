"""Todo: Hilbert transform"""

import sys

import numpy as np
from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw.utilities.blas import gemm, rk
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor


class Chi0(PairDensity):
    def __init__(self, calc, omega_w, ecut=50, hilbert=False,
                 eta=0.2, blocksize=50, ftol=1e-6,
                 real_space_derivatives=False,
                 world=mpi.world, txt=sys.stdout):
        PairDensity.__init__(self, calc, ecut, blocksize, ftol,
                             real_space_derivatives, world, txt)
        
        eta /= Hartree
        
        self.omega_w = omega_w = np.asarray(omega_w) / Hartree
        self.hilbert = hilbert
        self.eta = eta
        
        if eta == 0.0:
            assert not hilbert
            assert not omega_w.real.any()
            
        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(self.nocc2)
        
        self.mykpts = [self.get_k_point(s, K, n1, n2)
                       for s, K, n1, n2 in self.mysKn1n2]
        
        self.prefactor = (2 / self.vol / calc.wfs.kd.nbzkpts /
                          self.calc.wfs.nspins)
        
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
            kpt2 = self.get_k_point(kpt1.s, K2, m1, m2)
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

    def update_optical_limit(self, n, kpt1, kpt2, deps_m, df_m, n_mG,
                             ma, mb, chi0_wxvG):
        if self.ut_sKnvR is None:
            self.ut_sKnvR = self.calculate_derivatives()
            
        ut_vR = self.ut_sKnvR[kpt1.s][kpt1.K][n]
        
        atomdata_a = self.calc.wfs.setups
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]
        
        n0_mv = -self.calc.wfs.gd.integrate(ut_vR, kpt2.ut_nR[ma:mb]).T
        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            gemm(1.0, C_vi, P_mi[ma:mb], 1.0, n0_mv, 'c')

        deps_m = deps_m.copy()
        deps_m[deps_m > -1e-3] = np.inf
        n0_mv *= 1j / deps_m[:, np.newaxis]
        n_mG[:, 0] = n0_mv[:, 0]

        for w, omega in enumerate(self.omega_w):
            x_m = (self.prefactor *
                   df_m * (1.0 / (omega + deps_m + 1j * self.eta) -
                           1.0 / (omega - deps_m + 1j * self.eta)))
            chi0_wxvG[w, :, :, 0] += np.dot(x_m, n0_mv * n0_mv.conj())
            chi0_wxvG[w, 0, :, 1:] += np.dot(x_m * n0_mv.T, n_mG[:, 1:].conj())
            chi0_wxvG[w, 1, :, 1:] += np.dot(x_m * n0_mv.T.conj(), n_mG[:, 1:])
