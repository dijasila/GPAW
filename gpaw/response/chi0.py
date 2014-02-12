"""Todo: Hilbert transform"""

import sys
from math import pi

import numpy as np
from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw.utilities.blas import gemm, rk, czher
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor


def frequency_grid(domega0, omegamax, alpha):
    wmax = int(omegamax / domega0 / (1 + alpha)) + 1
    w = np.arange(wmax)
    omega_w = w * domega0 / (1 - alpha * domega0 / omegamax * w)
    return omega_w
    

class Chi0(PairDensity):
    def __init__(self, calc,
                 frequencies=None, domega0=0.1, omegamax=None, alpha=3.0,
                 ecut=50, hilbert=False,
                 timeordered=False, eta=0.2, ftol=1e-6,
                 real_space_derivatives=False,
                 world=mpi.world, txt=sys.stdout):
        PairDensity.__init__(self, calc, ecut, ftol,
                             real_space_derivatives, world, txt)

        eta /= Hartree
        domega0 /= Hartree
        omegamax = (omegamax or ecut) / Hartree
        
        if frequencies is None:
            self.omega_w = frequency_grid(domega0, omegamax, alpha)
            self.domega0 = domega0
            self.omegamax = omegamax
            self.alpha = alpha
            print(wmax)
        else:
            self.omega_w = np.asarray(frequencies) / Hartree
        
        self.hilbert = hilbert
        self.timeordered = bool(timeordered)
        self.eta = eta

        if eta == 0.0:
            assert not hilbert
            assert not timeordered
            assert not self.omega_w.real.any()

        # Occupied states:
        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(self.nocc2)
        self.mykpts = [self.get_k_point(s, K, n1, n2)
                       for s, K, n1, n2 in self.mysKn1n2]

        wfs = self.calc.wfs
        self.prefactor = 2 / self.vol / wfs.kd.nbzkpts / wfs.nspins
        
    def calculate(self, q_c, spin='all'):
        wfs = self.calc.wfs

        if spin == 'all':
            spins = range(wfs.nspins)
        else:
            assert spin in range(wfs.nspins)
            spins = [spin]

        q_c = np.asarray(q_c, dtype=float)
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, complex, qd)

        nG = pd.ngmax
        nw = len(self.omega_w)
        chi0_wGG = np.zeros((nw, nG, nG), complex)

        if not q_c.any():
            chi0_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
            chi0_wvv = np.zeros((len(self.omega_w), 3, 3), complex)
        else:
            chi0_wxvG = None
            chi0_wvv = None

        Q_aGii = self.initialize_paw_corrections(pd)

        # Do all empty bands:
        m1 = self.nocc1
        m2 = wfs.bd.nbands
        return self._calculate(pd, chi0_wGG, chi0_wxvG, chi0_wvv, Q_aGii,
                               m1, m2, spins)

    def _calculate(self, pd, chi0_wGG, chi0_wxvG, chi0_wvv, Q_aGii,
                   m1, m2, spins):
        wfs = self.calc.wfs

        if self.eta == 0.0:
            update = self.update_hermitian
        elif self.hilbert:
            update = self.update_hilbert
        else:
            update = self.update

        q_c = pd.kd.bzk_kc[0]
        optical_limit = not q_c.any()
        
        # kpt1 occupied and kpt2 empty:
        for kpt1 in self.mykpts:
            if not kpt1.s in spins:
                continue
            K2 = wfs.kd.find_k_plus_q(q_c, [kpt1.K])[0]
            kpt2 = self.get_k_point(kpt1.s, K2, m1, m2)
            Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                       kpt1.shift_c - kpt2.shift_c)

            for n in range(kpt1.n2 - kpt1.n1):
                eps1 = kpt1.eps_n[n]
                f1 = kpt1.f_n[n]
                ut1cc_R = kpt1.ut_nR[n].conj()
                C1_aGi = [np.dot(Q_Gii, P1_ni[n].conj())
                          for Q_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
                n_mG = self.calculate_pair_densities(ut1cc_R, C1_aGi, kpt2,
                                                     pd, Q_G)
                deps_m = eps1 - kpt2.eps_n
                df_m = f1 - kpt2.f_n

                # Avoid double counting if occupied and empty states
                # overlap (metals)
                m0 = kpt1.n1 + n - kpt2.n1
                if m0 >= 0:
                    df_m[:m0] = 0.0
                    df_m[m0] *= 0.5
                    
                if optical_limit:
                    self.update_optical_limit(
                        n, kpt1, kpt2, deps_m, df_m, n_mG, chi0_wxvG, chi0_wvv)
                    #self.update_intraband(n, kpt1, kpt2, chi0_wvv)
                update(n_mG, deps_m, df_m, chi0_wGG)

        self.world.sum(chi0_wGG)
        if optical_limit:
            self.world.sum(chi0_wxvG)
            self.world.sum(chi0_wvv)

        if self.eta == 0.0:
            # Fill in upper triangle also:
            nG = pd.ngmax
            il = np.tril_indices(nG, -1)
            iu = il[::-1]
            for chi0_GG in chi0_wGG:
                chi0_GG[iu] = chi0_GG[il].conj()

        elif self.hilbert:
            ht = HilbertTransform(self.omega_w)
            ht(chi0_wGG)

        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    def update(self, n_mG, deps_m, df_m, chi0_wGG):
        if self.timeordered:
            deps1_m = deps_m + 1j * self.eta * np.sign(deps_m)
            deps2_m = deps1_m
        else:
            deps1_m = deps_m + 1j * self.eta
            deps2_m = deps_m - 1j * self.eta
            
        for w, omega in enumerate(self.omega_w):
            x_m = df_m * (1 / (omega + deps1_m) - 1 / (omega - deps2_m))
            nx_mG = n_mG * x_m[:, np.newaxis]
            gemm(self.prefactor, n_mG.conj(), np.ascontiguousarray(nx_mG.T),
                 1.0, chi0_wGG[w])

    def update_hermitian(self, n_mG, deps_m, df_m, chi0_wGG):
        for w, omega in enumerate(self.omega_w):
            x_m = (-2 * df_m * deps_m / (omega.imag**2 + deps_m**2))**0.5
            nx_mG = n_mG.conj() * x_m[:, np.newaxis]
            rk(-self.prefactor, nx_mG, 1.0, chi0_wGG[w], 'n')

    def update_hilbert(self, n_mG, deps_m, df_m, chi0_wGG):
        for deps, df, n_G in zip(deps_m, df_m, n_mG):
            o = abs(deps)
            w = int(o / self.domega0 / (1 + self.alpha * o / self.omegamax))
            if w + 2 > len(self.omega_w):
                break
            o1, o2 = self.omega_w[w:w + 2]
            assert o1 < o < o2, (o1,o,o2)
            p = self.prefactor * abs(df) / (o2 - o1)**2
            czher(p * (o2 - o), n_G, chi0_wGG[w])
            czher(p * (o - o1), n_G, chi0_wGG[w + 1])

    def update_optical_limit(self, n, kpt1, kpt2, deps_m, df_m, n_mG,
                             chi0_wxvG, chi0_wvv):
        n0_mv = PairDensity.update_optical_limit(self, n, kpt1, kpt2,
                                                 deps_m, df_m, n_mG)

        if self.timeordered:
            deps1_m = deps_m + 1j * self.eta * np.sign(deps_m)
            deps2_m = deps1_m
        else:
            deps1_m = deps_m + 1j * self.eta
            deps2_m = deps_m - 1j * self.eta
            
        for w, omega in enumerate(self.omega_w):
            x_m = self.prefactor * df_m * (1 / (omega + deps1_m) -
                                           1 / (omega - deps2_m))
            
            chi0_wvv[w] += np.dot(x_m * n0_mv.T, n0_mv.conj())
            chi0_wxvG[w, 0, :, 1:] += np.dot(x_m * n0_mv.T, n_mG[:, 1:].conj())
            chi0_wxvG[w, 1, :, 1:] += np.dot(x_m * n0_mv.T.conj(), n_mG[:, 1:])

    def update_intraband(self, n, kpt1, kpt2, chi0_wvv):
        width = self.calc.occupations.width
        dfde_n = - 1. / width * (kpt1.f_n[n] - kpt1.f_n[n]**2.0)

        if np.abs(dfde_n) > 1e-3:
            nabla0_mv = PairDensity.update_intraband(self, n, kpt1, kpt2)
            veln_v = - 1j * nabla0_mv[kpt1.n1 + n - kpt2.n1]            
            x_vv = -self.prefactor * dfde_n * np.outer(veln_v.conj(),veln_v)

            for w, omega in enumerate(self.omega_w):
                chi0_wvv[w, :, :] += (x_vv / ((omega + 1j * self.eta) *
                                              (omega - 1j * self.eta)))


class HilbertTransform:
    def __init__(self, omega_w, blocksize=500):
        """Analytic Hilbert transformation using linear interpolation."""
        self.omega_w = omega_w
        self.blocksize = blocksize

        nw = len(omega_w)
        self.H_ww = np.zeros((nw, nw))
        o_w = omega_w
        do_w = o_w[1:] - o_w[:-1]
        for w, o in enumerate(o_w):
            d_w = o_w - o
            x1_w = d_w[1:] / do_w
            x2_w = d_w[:-1] / do_w
            d_w[w] = 1.0
            y_w = np.log(abs(d_w[1:] / d_w[:-1]))
            self.H_ww[w, :-1] = x1_w * y_w
            self.H_ww[w, 1:] -= x2_w * y_w
            d_w = o_w + o
            x1_w = d_w[1:] / do_w
            x2_w = d_w[:-1] / do_w
            d_w[0] = 1.0
            y_w = np.log(abs(d_w[1:] / d_w[:-1]))
            self.H_ww[w, :-1] += x1_w * y_w
            self.H_ww[w, 1:] -= x2_w * y_w
    
    def __call__(self, A_wx):
        B_wx = A_wx.reshape((len(A_wx), -1))
        nx = B_wx.shape[1]
        for x in range(0, nx, self.blocksize):
            C_wx = B_wx[:, x:x + self.blocksize]
            C_wx[:] = pi * 1j * C_wx + np.dot(self.H_ww, C_wx)
        

if __name__ == '__main__':
    do = 0.05
    eta = 0.1
    omega_w = frequency_grid(do, 100.0, 3)
    ht = HilbertTransform(omega_w)
    print(len(omega_w))
    X_w = omega_w * 0j
    Xt_w = omega_w * 0j
    Xh_w = omega_w * 0j
    for o in np.linspace(2.5, 25.0, 1000):
        X_w -= (1 / (omega_w - o + 1j * eta) -
                1 / (omega_w + o + 1j * eta)) / o**2
        Xt_w -= (1 / (omega_w - o + 1j * eta) -
                 1 / (omega_w + o - 1j * eta)) / o**2
        w = int(o / do / (1 + 3 * o / 100))
        o1, o2 = omega_w[w:w + 2]
        assert o1 - 1e-12 <= o <= o2 + 1e-12, (o1,o,o2)
        p = 1 / (o2 - o1)**2 / o**2
        Xh_w[w] += p * (o2 - o)
        Xh_w[w + 1] += p * (o - o1)
        
    ht(Xh_w)
    import matplotlib.pyplot as plt
    plt.plot(omega_w, X_w.imag, label='ImX')
    plt.plot(omega_w, X_w.real, label='ReX')
    plt.plot(omega_w, Xt_w.imag, label='ImXt')
    plt.plot(omega_w, Xt_w.real, label='ReXt')
    plt.plot(omega_w, Xh_w.imag, label='ImXh')
    plt.plot(omega_w, Xh_w.real, label='ReXh')
    plt.legend()
    plt.show()
    