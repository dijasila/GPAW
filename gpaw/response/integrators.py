from __future__ import print_function, division

import sys
from time import ctime
import itertools

import numpy as np
from scipy.spatial import Delaunay

from ase.units import Hartree
from ase.utils import devnull
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
from gpaw.utilities.progressbar import ProgressBar
from functools import partial

# Levi-civita
e_kkk = np.zeros((3, 3, 3))
e_kkk[0, 1, 2] = e_kkk[1, 2, 0] = e_kkk[2, 0, 1] = 1
e_kkk[0, 2, 1] = e_kkk[2, 1, 0] = e_kkk[1, 0, 2] = -1

def theta(x):
    """Heaviside step function."""
    return 0.5 * (np.sign(x) + 1)

class Integrator():
    def __init__(self, gd, comm=mpi.world,
                 txt=sys.stdout, timer=None,  nblocks=1):
        """Baseclass for Brillouin zone integration and band summation.
        
        Simple class to calculate integrals over Brilloun zones
        and summation of bands.
        
        comm: mpi.communicator
        nblocks: block parallelization
        """

        self.gd = gd
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
        self.A_cv = self.gd.cell_cv
        self.iA_cv = self.gd.icell_cv
        self.vol = abs(np.linalg.det(self.A_cv))

    def distribute_domain(self, domain_dl):
        """Distribute integration domain. """
        domainsize = [len(domain_l) for domain_l in domain_dl]
        nterms = np.prod(domainsize)
        size = self.kncomm.size
        rank = self.kncomm.rank

        n = (nterms + size - 1) // size
        i1 = rank * n
        i2 = min(i1 + n, nterms)
        mydomain = []
        for i in range(i1, i2):
            unravelled_d = np.unravel_index(i, domainsize)
            arguments = []
            for domain_l, index in zip(domain_dl, unravelled_d):
                arguments.append(domain_l[index])
            mydomain.append(tuple(arguments))

        print('Distributing domain %s' % (domainsize, ),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)

        return mydomain

    def distribute_integral(self, spins, kpts, n1, n2):
        """Distribute spins, k-points and bands.

        The attribute self.mysKn1n2 will be set to a list of (s, K, n1, n2)
        tuples that this process handles.
        """

        band1 = n1
        band2 = n2
        nbands = band2 - band1
        size = self.kncomm.size
        rank = self.kncomm.rank
        ns = len(spins)
        nk = len(kpts)
        n = (ns * nk * nbands + size - 1) // size
        i1 = rank * n
        i2 = min(i1 + n, ns * nk * nbands)

        mysKn1n2 = []
        i = 0
        for s in range(ns):
            for K in kpts:
                n1tmp = min(max(0, i1 - i), nbands)
                n2tmp = min(max(0, i2 - i), nbands)
                if n1tmp != n2tmp:
                    mysKn1n2.append((s, K, n1tmp + band1,
                                     n2tmp + band1))
                i += nbands

        print('Distributing spins, k-points and bands (%d x %d x %d)' %
              (ns, nk, nbands),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)

        return mysKn1n2
    
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
        self.mysKn1n2 = self.distribute_integral()
        self.prefactor = 2 / self.vol / self.kd.nbzkpts / len(self.ns)

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

    def tesselate(self, vertices):
        """Get tesselation descriptor."""

        td = Delaunay(vertices)

        return td

    def get_simplex_volume(self, td, S):
        """Get volume of simplex S"""
        
        K_k = td.simplices[S]
        k_kc = td.points[K_k]
        k_kv = np.dot(k_kc, self.iA_cv) * 2 * np.pi
        vol = np.abs(np.linalg.det(k_kv[1:] - k_kv[0])) / 6.

        return vol

    def integrate(self, kind, *args, **kwargs):
        if kind is 'response_function':
            return self.response_function_integration(*args, **kwargs)
        else:
            raise NotImplementedError

    @timer('Response function integration')
    def response_function_integration(self, domain, functions, wd,
                                      kwargs=None, out_wxx=None):
        """Integrate response function.
        
        Assume that the integral has the
        form of a response function. For the linear tetrahedron
        method it is possible calculate frequency dependent weights
        and do a point summation using these weights."""

        # Input domain
        td = domain[0]
        args = domain[1:]
        get_matrix_element, get_eigenvalues = functions

        # The kwargs contain any constant
        # arguments provided by the user
        if kwargs is not None:
            get_matrix_element = partial(get_matrix_element, **kwargs[0])
            get_eigenvalues = partial(get_eigenvalues, **kwargs[1])
        
        # Relevant quantities
        bzk_kc = td.points
        nk = len(bzk_kc)
        nw = len(wd)

        # Distribute integral summation
        myterms = self.distribute_domain([range(nk)] + list(args))

        # Calculate integrations weight
        pb = ProgressBar(self.fd)

        # Treat each terms by itself
        for _, arguments in pb.enumerate(myterms):
            # Assuming calculation of weights is cheap
            K = arguments[0]

            with self.timer('Calculate weights'):
                I_SMw = []
                for S in range(td.nsimplex):
                    K_k = td.simplices[S]
                    if K not in K_k:
                        continue

                    vol = self.get_simplex_volume(td, S)
                    deps_kM = []

                    with self.timer('get_eigenvalues'):
                        for Ks in K_k:
                            k_c = bzk_kc[Ks]
                            deps_M = get_eigenvalues(k_c, *arguments[1:])
                            deps_kM.append(deps_M)

                    Itmp_KMw = self.calculate_integration_weights(wd,
                                                                  deps_kM,
                                                                  vol)
                    for ik, I_Mw in enumerate(Itmp_KMw):
                        if K_k[ik] == K:
                            I_SMw.append(I_Mw)


            with self.timer('Integrate'):
                # Integrate values
                k_c = bzk_kc[K]
                n_MG = get_matrix_element(k_c, *arguments[1:])

                for I_Mw in I_SMw:
                    for n_G, I_w in zip(n_MG, I_Mw):
                        i0 = I_w[0]
                        weight_w = I_w[1]
                        for iw, weight in enumerate(weight_w):
                            czher(weight, n_G.conj(), out_wxx[i0 + iw])

        self.kncomm.sum(out_wxx)

        prefactor = 1 / (2 * np.pi)**3
        out_wxx *= prefactor

        self.timer.write()

    @timer('Single simplex weight')
    def calculate_integration_weights(self, wd, deps_kM, vol):
        """Calculate the integration weights."""
        nM = len(deps_kM[0])
        nw = len(wd)
        omega_w = wd.get_data()

        f = lambda o, de1, de2: (o - de2) / (de1 - de2)

        I_KMw = [[] for j in range(4)]
        for M in range(nM):
            de_k = np.array([deps_kM[ik][M] for ik in range(4)])
            permute = np.argsort(de_k)
            de_k = de_k[permute]

            # Frequency ranges from i0 to i3
            i0 = wd.get_closest_index(de_k[0])
            i3 = wd.get_closest_index(de_k[3]) + 1

            I_wk = np.zeros((i3 - i0, 4), float)

            for iw in range(i0, i3):
                omega = omega_w[iw]

                if de_k[0] < omega <= de_k[1]:
                    case = 0
                elif de_k[1] < omega <= de_k[2]:
                    case = 1
                elif de_k[2] < omega < de_k[3]:
                    case = 2
                else:
                    continue

                gi, I_k = self.get_kpoint_weight(omega, de_k, case)
                I_wk[iw - i0, permute] = vol * gi * I_k

            with self.timer('append'):
                for K in range(4):
                    I_KMw[K].append((i0, I_wk[:, K]))

        return I_KMw

    @timer('get_kpoint_weight')
    def get_kpoint_weight(self, omega, de_k, case):
        I_k = np.empty(4, float)
        f_kk = np.empty((4, 4), float)

        if case == 0:
            f_kk[1:, 0] = (omega - de_k[0]) / (de_k[1:] - de_k[0])
            f_kk[0, 1:] = 1 - f_kk[1:, 0]
            ni = f_kk[1, 0] * f_kk[2, 0] * f_kk[3, 0]
            gi = 3 * ni / (omega - de_k[0])
            I_k[0] = 1. / 3 * (f_kk[0, 1] + f_kk[0, 2] + f_kk[0, 3])
            I_k[1:] = 1. / 3 * f_kk[1:, 0]
        elif case == 1:
            f_kk[2:, :2] = ((omega - de_k[:2][np.newaxis]) /
                            (de_k[2:][:, np.newaxis]
                             - de_k[:2][np.newaxis]))
            f_kk[:2, 2:] = 1 - f_kk[2:, :2].T
            delta = de_k[3] - de_k[0]
            gi = 3. / delta * (f_kk[1, 2] * f_kk[2, 0] +
                               f_kk[2, 1] * f_kk[1, 3])
            I_k[:] = (f_kk[([0, 1, 2, 3], [3, 2, 1, 0])] / 3. +
                      f_kk[([0, 1, 2, 3], [2, 3, 0, 1])] *
                      f_kk[([2, 1, 2, 1], [0, 3, 0, 3])] *
                      f_kk[([1, 2, 1, 2], [2, 1, 2, 1])] *
                      (gi * delta))
        elif case == 2:
            f_kk[:3, 3] = (omega - de_k[3]) / (de_k[:3] - de_k[3])
            f_kk[3, :3] = 1 - f_kk[:3, 3]
            ni = (1 - f_kk[0, 3] * f_kk[1, 3] * f_kk[2, 3])
            gi = 3. * (1 - ni) / (de_k[3] - omega)
            I_k[:3] = f_kk[:3, 3]
            I_k[3] = (f_kk[3, 0] + f_kk[3, 1] + f_kk[3,2]) / 3.

        return gi, I_k


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
