from __future__ import print_function, division

import sys
from time import ctime

import numpy as np
from ase.units import Hartree
from ase.utils.timing import timer, Timer

import gpaw.mpi as mpi
from gpaw import extra_parameters
from gpaw.blacs import (BlacsGrid, BlacsDescriptor, Redistributor,
                        DryRunBlacsGrid)
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.occupations import FermiDirac
from gpaw.response.pair import PairDensity
from gpaw.utilities.memory import maxrss
from gpaw.utilities.blas import gemm, rk, czher, mmm
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.response.pair import PWSymmetryAnalyzer
from functools import partial

def theta(self, x):
    """Heaviside step function."""
    return 0.5 * (np.sign(x) + 1)

class Integrator():
    def __init__(self, kd, n1, n2, m1, m2, ns, comm=mpi.world,
                 txt=sys.stdout, timer=None,  nblocks=1):
        """Baseclass for Brillouin zone integration and band summation.
        
        Simple class to calculate integrals over Brilloun zones
        and summation of bands.
        
        kd: KPointDescriptor
        n1: int
        n2: int
        m1: int
        m2: int
        comm: mpi.communicator
        nblocks: block parallelization
        """

        self.n1 = n1
        self.n2 = n2
        self.m1 = m1
        self.m2 = m2
        self.kd = kd
        self.ns = ns
        self.comm = comm
        self.nblocks = nblocks

        if nblocks == 1:
            self.blockcomm = self.comm.new_communicator([comm.rank])
            self.kncomm = comm
        else:
            assert comm.size % nblocks == 0, comm.size
            rank1 = comm.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.blockcomm = self.comm.new_communicator(range(rank1, rank2))
            ranks = range(comm.rank % nblocks, comm.size, nblocks)
            self.kncomm = self.comm.new_communicator(ranks)

        if comm.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        self.timer = timer or Timer()

    def distribute_integration(self):
        """Distribute spins, k-points and bands.

        The attribute self.mysKn1n2 will be set to a list of (s, K, n1, n2)
        tuples that this process handles.
        """

        kpts = range(self.kd.nbzkpts)
        band1 = self.n1
        band2 = self.n2
        nbands = band2 - band1
        size = self.kncomm.size
        rank = self.kncomm.rank
        ns = len(self.ns)
        nk = len(kpts)
        n = (ns * nk * nbands + size - 1) // size
        i1 = rank * n
        i2 = min(i1 + n, ns * nk * nbands)

        self.mysKn1n2 = []
        i = 0
        for s in range(ns):
            for K in kpts:
                n1 = min(max(0, i1 - i), nbands)
                n2 = min(max(0, i2 - i), nbands)
                if n1 != n2:
                    self.mysKn1n2.append((s, K, n1 + band1, n2 + band1))
                i += nbands

        print('BZ k-points:', self.kd.description, file=self.fd)
        print('Distributing spins, k-points and bands (%d x %d x %d)' %
              (ns, nk, nbands),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)
    
    def integrate(self, driver=None, *args, **kwargs):
        """Integration method wrapper."""

        if driver is None:
            return self.pointwise_integration(*args, **kwargs)
        else:
            raise NotImplementedError

    def pointwise_integration(self, func, out_x=None):
        """Simple integration by summation.
        
        Simple integration by iterating through kpoints and summing up bands::

             __
             \
        S =  /_  func(s, k_c, n, m)
            snmk

        where func is the function to be integrated, s is spin index,
        n and m are band indices, and k_c is a kpoint coordinate.

        f: unbound_method
        out_x: np.ndarray
            output array
        """
        
        # Summing up elements one spin and kpoint
        # but many bands at a time
        for s, K, n1, n2 in self.mysKn1n2:
            kc = self.kd.bzk_kc[K]
            func_nmx = func(s, k_c, range(n1, n2), range(self.m1, self.m2))
            
            if out_x is None:
                out_x = np.sum(func_nmx, axis=(0, 1))
            else:
                out_x += np.sum(func_nmx, axis=(0, 1))

        self.kncomm.sum(out_x)

        return out_x        


class BroadeningIntegrator(Integrator):
    def __init__(self, eta, *args, **kwargs):
        """Integrate brillouin zone using a broadening technique.

        The broadening technique consists of smearing out the
        delta functions appearing in many integrals by some factor
        eta. In this code we use Lorentzians."""

        Integrator.__init__(self, *args, **kwargs)
        self.eta = eta
        self.distribute_integration()

    def get_integration_function(self, kind=None):
        if kind is None:
            return self.pointwise_integration
        elif kind is 'response_function':
            return self.response_function_integration
        else:
            raise NotImplementedError

    def response_function_integration(self, func, omega_w, out_wxx=None,
                                      timeordered=False, hermitian=False,
                                      hilbert=True):
        """Integrate a response function over bands and kpoints.
        
        func: method
        omega_w: ndarray
        out: np.ndarray
        timeordered: Bool
        """
        if out_wxx is None:
            raise NotImplementedError

        if omega_w is None:
            raise NotImplementedError

        # Sum kpoints
        for s, K, n1, n2 in self.mysKn1n2:
            k_c = self.kd.bzk_kc[K]
            M_nmx, e_n, e_m, f_n, f_m = func(s, k_c, n1, n2,
                                             self.m1, self.m2)

            nx = M_nmx.shape[-1]
            M_mx =  M_nmx.reshape((-1, nx))
            de_m = (e_n[:, np.newaxis] - e_m).ravel()            
            df_m = (f_n[:, np.newaxis] - f_m).ravel()
            
            if hermitian:
                self.update_hermitian(M_nmx, de_m, df_m, out_wxx)
            elif hilbert:
                self.update_hilbert(M_nmx, de_m, df_m, out_wxx)
            else:
                self.update(M_mx, de_m, df_m, omega_w, out_wxx,
                            timeordered=timeordered)
        # Sum over 
        for out_xx in out_wxx:
            self.kncomm.sum(out_xx)

        if (hermitian or hilbert) and self.blockcomm.size == 1:
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

        if hilbert:
            with self.timer('Hilbert transform'):
                ht = HilbertTransform(omega_w, self.eta,
                                      timeordered)
                ht(chi0_wGG)
            print('Hilbert transform done', file=self.fd)

    @timer('CHI_0 update')
    def update(self, n_mG, deps_m, df_m, omega_w, chi0_wGG, timeordered=False):
        """Update chi."""

        if timeordered:
            deps1_m = deps_m + 1j * self.eta * np.sign(deps_m)
            deps2_m = deps1_m
        else:
            deps1_m = deps_m + 1j * self.eta
            deps2_m = deps_m - 1j * self.eta

        for omega, chi0_GG in zip(omega_w, chi0_wGG):
            x_m = df_m * (1 / (omega + deps1_m) - 1 / (omega - deps2_m))
            if self.blockcomm.size > 1:
                nx_mG = n_mG[:, self.Ga:self.Gb] * x_m[:, np.newaxis]
            else:
                nx_mG = n_mG * x_m[:, np.newaxis]
            gemm(1.0, n_mG.conj(), np.ascontiguousarray(nx_mG.T),
                 1.0, chi0_GG)

    @timer('CHI_0 hermetian update')
    def update_hermitian(self, n_mG, deps_m, df_m, chi0_wGG):
        """If eta=0 use hermitian update."""
        for w, omega in enumerate(omega_w):
            if self.blockcomm.size == 1:
                x_m = (-2 * df_m * deps_m / (omega.imag**2 + deps_m**2))**0.5
                nx_mG = n_mG.conj() * x_m[:, np.newaxis]
                rk(-1.0, nx_mG, 1.0, chi0_wGG[w], 'n')
            else:
                x_m = 2 * df_m * deps_m / (omega.imag**2 + deps_m**2)
                mynx_mG = n_mG[:, self.Ga:self.Gb] * x_m[:, np.newaxis]
                mmm(1.0, mynx_mG, 'c', n_mG, 'n', 1.0, chi0_wGG[w])

    @timer('CHI_0 spectral function update')
    def update_hilbert(self, n_mG, deps_m, df_m, chi0_wGG):
        """Update spectral function.

        Updates spectral function A_wGG and saves it to chi0_wGG for
        later hilbert-transform."""

        self.timer.start('prep')
        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        o_m = abs(deps_m)
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        o1_m = omega_w[w_m]
        o2_m = omega_w[w_m + 1]
        p_m = abs(df_m) / (o2_m - o1_m)**2  # XXX abs()?
        p1_m = p_m * (o2_m - o_m)
        p2_m = p_m * (o_m - o1_m)
        self.timer.stop('prep')

        if self.blockcomm.size > 1:
            for p1, p2, n_G, w in zip(p1_m, p2_m, n_mG, w_m):
                myn_G = n_G[self.Ga:self.Gb].reshape((-1, 1))
                gemm(p1, n_G.reshape((-1, 1)), myn_G, 1.0, chi0_wGG[w], 'c')
                gemm(p2, n_G.reshape((-1, 1)), myn_G, 1.0, chi0_wGG[w + 1],
                     'c')
            return

        for p1, p2, n_G, w in zip(p1_m, p2_m, n_mG, w_m):
            czher(p1, n_G.conj(), chi0_wGG[w])
            czher(p2, n_G.conj(), chi0_wGG[w + 1])

    @timer('CHI_0 optical limit update')
    def update_optical_limit(self, n0_mv, deps_m, df_m, n_mG,
                             chi0_wxvG, chi0_wvv):
        """Optical limit update of chi."""

        if self.hilbert:  # Do something special when hilbert transforming
            self.update_optical_limit_hilbert(n0_mv, deps_m, df_m, n_mG,
                                              chi0_wxvG, chi0_wvv)
            return

        if timeordered:
            # avoid getting a zero from np.sign():
            deps1_m = deps_m + 1j * self.eta * np.sign(deps_m + 1e-20)
            deps2_m = deps1_m
        else:
            deps1_m = deps_m + 1j * self.eta
            deps2_m = deps_m - 1j * self.eta

        for w, omega in enumerate(omega_w):
            x_m = df_m * (1 / (omega + deps1_m) -
                          1 / (omega - deps2_m))

            chi0_wvv[w] += np.dot(x_m * n0_mv.T, n0_mv.conj())
            chi0_wxvG[w, 0, :, 1:] += np.dot(x_m * n0_mv.T, n_mG[:, 1:].conj())
            chi0_wxvG[w, 1, :, 1:] += np.dot(x_m * n0_mv.T.conj(), n_mG[:, 1:])

    @timer('CHI_0 optical limit hilbert-update')
    def update_optical_limit_hilbert(self, n0_mv, deps_m, df_m, n_mG,
                                     chi0_wxvG, chi0_wvv):
        """Optical limit update of chi-head and -wings."""

        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        for deps, df, n0_v, n_G in zip(deps_m, df_m, n0_mv, n_mG):
            o = abs(deps)
            w = int(o / (self.domega0 + beta * o))
            if w + 2 > len(omega_w):
                break
            o1, o2 = omega_w[w:w + 2]
            assert o1 <= o <= o2, (o1, o, o2)

            p = abs(df) / (o2 - o1)**2  # XXX abs()?
            p1 = p * (o2 - o)
            p2 = p * (o - o1)
            x_vv = np.outer(n0_v, n0_v.conj())
            chi0_wvv[w] += p1 * x_vv
            chi0_wvv[w + 1] += p2 * x_vv
            x_vG = np.outer(n0_v, n_G[1:].conj())
            chi0_wxvG[w, 0, :, 1:] += p1 * x_vG
            chi0_wxvG[w + 1, 0, :, 1:] += p2 * x_vG
            chi0_wxvG[w, 1, :, 1:] += p1 * x_vG.conj()
            chi0_wxvG[w + 1, 1, :, 1:] += p2 * x_vG.conj()

    @timer('CHI_0 intraband update')
    def update_intraband(self, f_m, vel_mv, chi0_vv):
        """Add intraband contributions"""
        assert len(f_m) == len(vel_mv), print(len(f_m), len(vel_mv))

        width = self.calc.occupations.width
        if width == 0.0:
            return

        assert isinstance(self.calc.occupations, FermiDirac)
        dfde_m = - 1. / width * (f_m - f_m**2.0)
        partocc_m = np.abs(dfde_m) > 1e-5
        if not partocc_m.any():
            return

        for dfde, vel_v in zip(dfde_m, vel_mv):
            x_vv = (-dfde *
                    np.outer(vel_v, vel_v))
            chi0_vv += x_vv


class TetrahedronIntegrator(Integrator):
    """Integrate brillouin zone using tetrahedron integration.

    Tetrahedron integration uses linear interpolation of
    the eigenenergies and of the matrix elements 
    between the vertices of the tetrahedron."""
    def __init__(self, *args, **kwargs):
        Integrator.__init__(self, *args, **kwargs)
        self.indices_t = self.tetrahedralize()

    def tetrahedralize(self):
        """Determine kpoint indices for tetrahedrons.

        Assumes monkhorst pack grid! """

        bzk_kc = self.kd.bzk_kc
        N_c = self.kd.N_c
        k_kkkc = np.reshape(bzk_kc, (N_c[0], N_c[1], N_c[2], 3))

        # Storing indices in this
        indices_t = []
        # Iterating through submesh unit cell
        for i in range(N_c[0] - 1):
            for j in range(N_c[1] - 1):
                for l in range(N_c[2] - 1):
                    K_k = [np.ravel_multi_index((i + t, j + u, l + v), N_c)
                               for v in range(2) for u in range(2)
                               for t in range(2)]

                    # There are six tetrahedrons in each cell
                    indices_t.append([K_k[0], K_k[1], K_k[2], K_k[5]].sort())
                    indices_t.append([K_k[0], K_k[4], K_k[2], K_k[5]].sort())
                    indices_t.append([K_k[2], K_k[4], K_k[5], K_k[6]].sort())
                    indices_t.append([K_k[1], K_k[2], K_k[3], K_k[5]].sort())
                    indices_t.append([K_k[2], K_k[3], K_k[5], K_k[7]].sort())
                    indices_t.append([K_k[2], K_k[5], K_k[6], K_k[7]].sort())
                    
        return indices_t

    def integrate(self, func):
        bzk_kc = self.kd.bzk_kc
        oldindices_k = []
        funcvals_k = []
        for indices_k in indices_t:
            # Calculate new function values
            for i, index in enumerate(indices_k):
                if index in oldindices_k:
                    continue
                k_c = bzk_kc[index]
                funcvals_k[i] = func(k_c)

            self.integrate_single_tetrahedron(indices_k, funcvals_k)
            
    def integrate_single_tetrahedron(self):
        raise NotImplementedError

    def integrate_spectral_function(self, func, omega_w):
        bzk_kc = self.kd.bzk_kc
        funcvals_k = {}
        for indices_k in self.indices_t:
            # Calculate new function values
            for index in indices_k:
                if index not in funcvals_k:
                    k_c = bzk_kc[index]
                    funcvals_k[index] = func(k_c)

            # Function vales at tetrahedron vertices
            tetfuncvals_k = {}
            for i, index in enumerate(indices_k):
                tetfuncvals_k[i] = funcvals[index]

            self.charlesworth_yeung_integrate(tetfuncvals_k, omega_w)

    def charlesworth_yeung_integrate(self, funcvals_k, omega):
        """Integrate a single tetrahedron using the CY-formula.

        Analytical tetrahedron integration using average matrix elements.
        
        [1] Charlesworth and Yeung, Phys. Rev. B. 53, 5, Feb. 1996.
        [2] Charlesworth and Yeung, Comp. phys. Comm. 88, 186-194.

        Use the average matrix elements and apply analytical
        formula from the authors. Valid for T != 0. Get contribution
        to tetrahedron for a single tetrahdron.
        
        funcvals_k: dict
            Dictionary containing the values of
            a function evaluated at the vertices of
            the tetrahedron.
        omega_w: list or np.ndarray
            Frequency hbar * omega.
        """

        e_k = [funcvals_k[i][0] for i in range(4)]
        ekq_k = [funcvals_k[i][1] for i in range(4)]
        M_kGG = [funcvals_k[i][2] for i in range(4)]

        Mavg_GG = M_kGG[0]
        for i in range(1, 4):
            Mavg_GG += M_kGG[i]
        
        Mavg_GG /= 4.
        
        # Quantities needed for later
        D_k = e_k - ekq_k + omega
        a = D_k[0]
        b_k = D_k[1:] - D_k[0]
        c = e_k[0]
        d_k = e_k[1:] - e_k[0]

        # Levi-civita
        e_kkk = np.zeros((3, 3, 3))
        e_kkk[0, 1, 2] = e_kkk[1, 2, 0] = e_kkk[2, 0, 1] = 1
        e_kkk[0, 2, 1] = e_kkk[2, 1, 0] = e_kkk[1, 0, 2] = -1

        mu_k = (np.sum(np.tensordot(e_kkk, b_k, axes=[1, 0]), axis=1)
                / np.sum(np.tensordot(e_kkk, d_k, axes=[1, 0]), axis=1))

        zeta = 1. / ((b_k[0] - b_k[1]) * (b_k[1] - b_k[2]) * (b_k[2] - b_k[0]))
        xi = 1. / ((d_k[0] - d_k[1]) * (d_k[1] - d_k[2]) * (d_k[2] - d_k[0]))
        nu = 1. / ((b_k[0] / d_k[0] - b_k[1] - d_k[1]) *
                   (b_k[1] / d_k[1] - b_k[2] - d_k[2]) *
                   (b_k[2] / d_k[2] - b_k[0] - d_k[0]))

        sum1 = 0
        sum2 = 0
        sum3 = 0        
        for t in range(3):
            for u in range(3):
                for v in range(3):
                    sum1 += (e_kkk[t, u, v] * (b_k[u] / b_k[t])
                             * (a ** 2 * self.theta(c) * np.sign(a)
                                - (a + b_k[t])**2 * self.theta(c + d_k[t])
                                * np.sign(a + b_k[t])))        

                    sum2 += (e_kkk[t, u ,v] * mu_k[v] * mu_k[t] / mu_k[u] /
                             (b_k[t] - mu_k[u] * d_k[t]) *
                             ((d_k[u] - d_k[v]) / (mu_k[u] - mu_k[v])) * self.theta(c + d_k[t]) *
                             (a + b_k[t] - mu_k[u] * (c + d_k[t]))**2 *
                             np.sign(a + b_k[t] - mu_k[u] * (c + d_k[t])))
        
                    sum3 += (e_kkk[t, u, v] * d_k[t] / b_k[t] * b_k[u] / d_k[u] *
                             (self.theta(c) - self.theta(c + d_k[t]))
                             * (a - c * b_k[t] / d_k[t])**2
                             * np.sign(a - c * b_k[t] / d_k[t]))

        I1 = (xi / np.prod(mu_k) * sum1 
              - xi * sum2 - nu / np.prod(d_k) * sum3) / 4
                
        return I1


class HilbertTransform:
    def __init__(self, omega_w, eta, timeordered=False, gw=False,
                 blocksize=500):
        """Analytic Hilbert transformation using linear interpolation.

        Hilbert transform::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        With timeordered=True, you get::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' - i eta   w + w' + i eta
          0

        With gw=True, you get::

           oo
          /           1                1
          |dw' (-------------- + --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        """

        self.blocksize = blocksize

        if timeordered:
            self.H_ww = self.H(omega_w, -eta) + self.H(omega_w, -eta, -1)
        elif gw:
            self.H_ww = self.H(omega_w, eta) - self.H(omega_w, -eta, -1)
        else:
            self.H_ww = self.H(omega_w, eta) + self.H(omega_w, -eta, -1)

    def H(self, o_w, eta, sign=1):
        """Calculate transformation matrix.

        With s=sign (+1 or -1)::

                        oo
                       /       dw'
          X (w, eta) = | ---------------- S(w').
           s           / s w - w' + i eta
                       0

        Returns H_ij so that X_i = np.dot(H_ij, S_j), where::

            X_i = X (omega_w[i]) and S_j = S(omega_w[j])
                   s
        """

        nw = len(o_w)
        H_ij = np.zeros((nw, nw), complex)
        do_j = o_w[1:] - o_w[:-1]
        for i, o in enumerate(o_w):
            d_j = o_w - o * sign
            y_j = 1j * np.arctan(d_j / eta) + 0.5 * np.log(d_j**2 + eta**2)
            y_j = (y_j[1:] - y_j[:-1]) / do_j
            H_ij[i, :-1] = 1 - (d_j[1:] - 1j * eta) * y_j
            H_ij[i, 1:] -= 1 - (d_j[:-1] - 1j * eta) * y_j
        return H_ij

    def __call__(self, S_wx):
        """Inplace transform"""
        B_wx = S_wx.reshape((len(S_wx), -1))
        nw, nx = B_wx.shape
        tmp_wx = np.zeros((nw, min(nx, self.blocksize)), complex)
        for x in range(0, nx, self.blocksize):
            b_wx = B_wx[:, x:x + self.blocksize]
            c_wx = tmp_wx[:, :b_wx.shape[1]]
            gemm(1.0, b_wx, self.H_ww, 0.0, c_wx)
            b_wx[:] = c_wx


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
