from __future__ import print_function

import sys
from time import ctime

import numpy as np
from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw import extra_parameters
from gpaw.utilities.timing import timer
from gpaw.utilities.memory import maxrss
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.utilities.blas import gemm, rk, czher
from gpaw.kpt_descriptor import KPointDescriptor


def frequency_grid(domega0, omegamax, alpha):
    print('Using nonlinear frequency grid from 0 to %d  eV'%(omegamax*Hartree), file=self.fd)
    wmax = int(omegamax / domega0 / (1 + alpha)) + 1
    w = np.arange(wmax)
    omega_w = w * domega0 / (1 - alpha * domega0 / omegamax * w)
    return omega_w
    

class Chi0(PairDensity):
    def __init__(self, calc,
                 frequencies=None, domega0=0.1, omegamax=None, alpha=3.0,
                 ecut=50, hilbert=False, nbands=None,
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
        else:
            self.omega_w = np.asarray(frequencies) / Hartree
            self.domega0 = self.omega_w[1]
            self.omegamax = -42.0 #np.max(self.omega_w)?
            self.alpha = 0.0
            
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
        self.mykpts = None

        self.nbands = nbands or self.calc.wfs.bd.nbands

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

        self.print_chi(pd)

        if extra_parameters.get('df_dry_run'):
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        nG = pd.ngmax
        nw = len(self.omega_w)
        chi0_wGG = np.zeros((nw, nG, nG), complex)

        if np.allclose(q_c, 0.0):
            chi0_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
            chi0_wvv = np.zeros((len(self.omega_w), 3, 3), complex)
        else:
            chi0_wxvG = None
            chi0_wvv = None
        print('    Initializing PAW Corrections', file=self.fd)
        Q_aGii = self.initialize_paw_corrections(pd)
        print('        Done.', file=self.fd)

        # Do all empty bands:
        m1 = self.nocc1
        m2 = self.nbands
        return self._calculate(pd, chi0_wGG, chi0_wxvG, chi0_wvv, Q_aGii,
                               m1, m2, spins)

    @timer('Calculate CHI_0')
    def _calculate(self, pd, chi0_wGG, chi0_wxvG, chi0_wvv, Q_aGii,
                   m1, m2, spins):
        wfs = self.calc.wfs

        if self.mykpts is None:
            self.mykpts = [self.get_k_point(s, K, n1, n2)
                           for s, K, n1, n2 in self.mysKn1n2]

        numberofkpts = len(self.mykpts)

        if self.eta == 0.0:
            update = self.update_hermitian
        elif self.hilbert:
            update = self.update_hilbert
        else:
            update = self.update

        q_c = pd.kd.bzk_kc[0]
        optical_limit = np.allclose(q_c, 0.0)

        print('\n    Starting summation', file=self.fd)
        # kpt1 occupied and kpt2 empty:
        for kn, kpt1 in enumerate(self.mykpts):
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
                update(n_mG, deps_m, df_m, chi0_wGG)
            
            if optical_limit:
                # Avoid that more ranks are summing up
                # the intraband contributions
                if kpt1.n1 == 0:
                    self.update_intraband(kpt2, chi0_wvv)
            
            if numberofkpts > 10 and kn % (numberofkpts // 10) == 0:
                print('    %s,' % ctime() +
                      ' local Kpoint no: %d / %d,' % (kn, numberofkpts) +
                      '\n        mem. used.: ' +
                      '%f M / cpu' % (maxrss() / 1024**2),
                      file=self.fd)

        print('    %s, Finished kpoint sum' % ctime(), file=self.fd)
        self.world.sum(chi0_wGG)
        if optical_limit:
            self.world.sum(chi0_wxvG)
            self.world.sum(chi0_wvv)

        if self.eta == 0.0 or self.hilbert:
            # Fill in upper/lower triangle also:
            nG = pd.ngmax
            il = np.tril_indices(nG, -1)
            iu = il[::-1]
            if self.hilbert:
                for chi0_GG in chi0_wGG:
                    chi0_GG[il] = chi0_GG[iu].conj()
            else:
                for chi0_GG in chi0_wGG:
                    chi0_GG[iu] = chi0_GG[il].conj()

        if self.hilbert:
            ht = HilbertTransform(self.omega_w, self.eta, self.timeordered)
            ht(chi0_wGG, chi0_wGG)
            print('Hilbert transform done', file=self.fd)

        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    @timer('CHI_0 update')
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

    @timer('CHI_0 hermetian update')
    def update_hermitian(self, n_mG, deps_m, df_m, chi0_wGG):
        for w, omega in enumerate(self.omega_w):
            x_m = (-2 * df_m * deps_m / (omega.imag**2 + deps_m**2))**0.5
            nx_mG = n_mG.conj() * x_m[:, np.newaxis]
            rk(-self.prefactor, nx_mG, 1.0, chi0_wGG[w], 'n')

    @timer('CHI_0 spectral function update')
    def update_hilbert(self, n_mG, deps_m, df_m, chi0_wGG):
        for deps, df, n_G in zip(deps_m, df_m, n_mG):
            o = abs(deps)
            w = int(o / self.domega0 / (1 + self.alpha * o / self.omegamax))
            if w + 2 > len(self.omega_w):
                break
            o1, o2 = self.omega_w[w:w + 2]
            assert o1 <= o <= o2, (o1, o, o2)

            p = self.prefactor * abs(df) / (o2 - o1)**2  # XXX abs()?
            czher(p * (o2 - o), n_G.conj(), chi0_wGG[w])
            czher(p * (o - o1), n_G.conj(), chi0_wGG[w + 1])

    @timer('CHI_0 optical limit update')
    def update_optical_limit(self, n, kpt1, kpt2, deps_m, df_m, n_mG,
                             chi0_wxvG, chi0_wvv):
        n0_mv = PairDensity.update_optical_limit(self, n, kpt1, kpt2,
                                                 deps_m, df_m, n_mG)

        if self.timeordered:
            # avoid getting a zero from np.sign():
            deps1_m = deps_m + 1j * self.eta * np.sign(deps_m + 1e-20)
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

    @timer('CHI_0 intraband update')
    def update_intraband(self, kpt, chi0_wvv):
        """Check whether there are any partly occupied bands."""
        width = self.calc.occupations.width
        if width == 0.0:
            return
            
        dfde_m = - 1. / width * (kpt.f_n - kpt.f_n**2.0)
        partocc_m = np.abs(dfde_m) > 1e-5
        if not partocc_m.any():
            return
        
        # Break bands into degenerate chunks
        deginds_cm = []  # indexing c as chunk number
        for m in range(kpt.n2 - kpt.n1):
            inds_m = np.nonzero(np.abs(kpt.eps_n[m] - kpt.eps_n) < 1e-5)[0]
            if m == np.min(inds_m) and partocc_m[m]:
                deginds_cm.append((inds_m))

        # Sum over the chunks of degenerate bands
        for inds_m in deginds_cm:
            deg = len(inds_m)
            vel_mmv = -1j * PairDensity.update_intraband(self, inds_m, kpt)
            vel_mv = np.zeros((deg, 3), dtype=complex)
            
            for iv in range(3):
                w, v = np.linalg.eig(vel_mmv[..., iv])
                vel_mv[:, iv] = w
                                
            for m in range(deg):
                velm_v = vel_mv[m]
                x_vv = (-self.prefactor * dfde_m[inds_m[m]] *
                        np.outer(velm_v.conj(), velm_v))

                for w, omega in enumerate(self.omega_w):
                    chi0_wvv[w, :, :] += x_vv / omega**2.0

    def print_chi(self, pd):
        calc = self.calc
        gd = calc.wfs.gd
        
        ns = calc.wfs.nspins
        nk = calc.wfs.kd.nbzkpts
        nb = self.nocc2

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
        else:
            world = self.world

        nw = len(self.omega_w)
        q_c = pd.kd.bzk_kc[0]
        nstat = (ns * nk * nb + world.size - 1) // world.size

        print('%s' % ctime(), file=self.fd)
        print('Called response.chi0.calculate with', file=self.fd)
        print('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]), file=self.fd)
        print('    [min(freq), max(freq)]: [%f, %f]'
              % (np.min(self.omega_w) * Hartree,
                 np.max(self.omega_w) * Hartree), file=self.fd)
        print('    Number of frequency points   : %d' % nw, file=self.fd)
        print('    Planewave cutoff: %f' % (self.ecut * Hartree), file=self.fd)
        print('    Number of spins: %d' % ns, file=self.fd)
        print('    Number of bands: %d' % self.nbands, file=self.fd)
        print('    Number of kpoints: %d' % nk, file=self.fd)
        print('    Number of planewaves: %d' % pd.ngmax, file=self.fd)
        print('    Broadening (eta): %f' % (self.eta * Hartree), file=self.fd)

        print('', file=self.fd)
        print('    Related to parallelization', file=self.fd)
        print('        world.size: %d' % world.size, file=self.fd)
        print('        Number of completely occupied states: %d'
              % self.nocc1, file=self.fd)
        print('        Number of partially occupied states: %d'
              % self.nocc2, file=self.fd)
        print('        Number of terms handled in chi-sum by each rank: %d'
              % nstat, file=self.fd)

        print('', file=self.fd)
        print('    Related to hilbert transform:', file=self.fd)
        print('        Use Hilbert Transform: %s' % self.hilbert, file=self.fd)
        print('        Calculate time-ordered Response Function: %s'
              % self.timeordered, file=self.fd)
        print('        domega0: %f' % (self.domega0 * Hartree), file=self.fd)
        print('        omegamax: %f' % (self.omegamax * Hartree), file=self.fd)
        print('        alpha: %f' % self.alpha, file=self.fd)

        print('', file=self.fd)
        print('    Memory estimate:', file=self.fd)
        print('        chi0_wGG: %f M / cpu'
              % (nw * pd.ngmax**2 * 32. / 1024**2), file=self.fd)
        if np.allclose(q_c, 0.0):
            print('        ut_sKnvR: %f M / cpu'
                  % (nstat * 3 * gd.N_c[0] * gd.N_c[1]
                     * gd.N_c[2] * 32. / 1024**2), file=self.fd)
        print('        Max mem sofar   : %f M / cpu'
              % (maxrss() / 1024**2), file=self.fd)

        print('', file=self.fd)


class HilbertTransform:
    def __init__(self, omega_w, eta, timeordered=False, blocksize=500):
        """Analytic Hilbert transformation using linear interpolation."""
        self.blocksize = blocksize

        if timeordered:
            self.H_ww = self.H(omega_w, -eta) + self.H(omega_w, -eta, -1)
        else:
            self.H_ww = self.H(omega_w, eta) + self.H(omega_w, -eta, -1)

    def H(self, o_w, eta, sign=1):
        nw = len(o_w)
        H_ww = np.zeros((nw, nw), complex)
        do_w = o_w[1:] - o_w[:-1]
        for w, o in enumerate(o_w):
            d_w = o_w - o * sign
            y_w = 1j * np.arctan(d_w / eta) + 0.5 * np.log(d_w**2 + eta**2)
            y_w = (y_w[1:] - y_w[:-1]) / do_w
            H_ww[w, :-1] = 1 - (d_w[1:] - 1j * eta) * y_w
            H_ww[w, 1:] -= 1 - (d_w[:-1] - 1j * eta) * y_w
        return H_ww
    
    def __call__(self, A_wx, out=None):
        if out is None:
            C_wx = np.empty_like(A_wx)
        else:
            C_wx = out
            
        B_wx = A_wx.reshape((len(A_wx), -1))
        D_wx = C_wx.reshape((len(A_wx), -1))
        
        nx = B_wx.shape[1]
        for x in range(0, nx, self.blocksize):
            b_wx = B_wx[:, x:x + self.blocksize]
            d_wx = D_wx[:, x:x + self.blocksize]
            d_wx[:] = np.dot(self.H_ww, b_wx)
        
        return C_wx
        

if __name__ == '__main__':
    do = 0.025
    eta = 0.1
    omega_w = frequency_grid(do, 10.0, 3)
    print(len(omega_w))
    X_w = omega_w * 0j
    Xt_w = omega_w * 0j
    Xh_w = omega_w * 0j
    for o in -np.linspace(2.5, 2.9, 10):
        X_w += (1 / (omega_w + o + 1j * eta) -
                1 / (omega_w - o + 1j * eta)) / o**2
        Xt_w += (1 / (omega_w + o - 1j * eta) -
                 1 / (omega_w - o + 1j * eta)) / o**2
        w = int(-o / do / (1 + 3 * -o / 10))
        o1, o2 = omega_w[w:w + 2]
        assert o1 - 1e-12 <= -o <= o2 + 1e-12, (o1, -o, o2)
        p = 1 / (o2 - o1)**2 / o**2
        Xh_w[w] += p * (o2 - -o)
        Xh_w[w + 1] += p * (-o - o1)
        
    ht = HilbertTransform(omega_w, eta, 1)
    ht(Xh_w, Xh_w)
    
    import matplotlib.pyplot as plt
    plt.plot(omega_w, X_w.imag, label='ImX')
    plt.plot(omega_w, X_w.real, label='ReX')
    plt.plot(omega_w, Xt_w.imag, label='ImXt')
    plt.plot(omega_w, Xt_w.real, label='ReXt')
    plt.plot(omega_w, Xh_w.imag, label='ImXh')
    plt.plot(omega_w, Xh_w.real, label='ReXh')
    plt.legend()
    plt.show()
