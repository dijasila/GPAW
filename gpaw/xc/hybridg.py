# Copyright (C) 2010  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange with k-point sampling."""

from time import time
from math import pi, sqrt

import numpy as np
from ase.utils import prnt
from ase.units import Hartree

import gpaw.fftw as fftw
from gpaw.xc.hybrid import HybridXCBase
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC
from gpaw.utilities import pack, unpack2, packed_index, logfile
        

class HybridXC(HybridXCBase):
    orbital_dependent = True

    def __init__(self, name, hybrid=None, xc=None,
                 alpha=None, skip_gamma=False, finegrid=False,
                 method='standard',
                 bandstructure=False,
                 logfilename='-', bands=None,
                 fcut=1e-10):
        """Mix standard functionals with exact exchange.

        name: str
            Name of functional: EXX, PBE0, B3LYP.
        hybrid: float
            Fraction of exact exchange.
        xc: str or XCFunctional object
            Standard DFT functional with scaled down exchange.
        method: str
            Use 'standard' standard formula and 'acdf for
            adiabatic-connection dissipation fluctuation formula.
        finegrid: boolean
            Use fine grid for energy functional evaluations?
        alpha: float
            XXX describe
        skip_gamma: bool
            Skip k2-k1=0 interactions.
        bandstructure: bool
            Calculate bandstructure instead of just the total energy.
        bands: list of int
            List of bands to calculate bandstructure for.  Default is
            all bands.
        fcut: float
            Threshold for empty band.
        """

        self.alpha = alpha
        self.fcut = fcut

        self.finegrid = finegrid
        self.skip_gamma = skip_gamma
        self.method = method
        self.bandstructure = bandstructure
        self.bands = bands

        self.fd = logfilename
        self.write_timing_information = True

        HybridXCBase.__init__(self, name, hybrid, xc)

        # EXX energies:
        self.exx = None  # total
        self.evv = None  # valence-valence (pseudo part)
        self.evvacdf = None  # valence-valence (pseudo part)
        self.devv = None  # valence-valence (PAW correction)
        self.evc = None  # valence-core
        self.ecc = None  # core-core

        self.exx_skn = None  # bandstructure

    def log(self, *args, **kwargs):
        prnt(file=self.fd, *args, **kwargs)
        self.fd.flush()

    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg,
                         dndr_sLg=None, rnablaY_Lv=None,
                         tau_sg=None, dedtau_sg=None):
        return self.xc.calculate_radial(rgd, n_sLg, Y_L, v_sg,
                                        dndr_sLg, rnablaY_Lv)
    
    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        return self.xc.calculate_paw_correction(setup, D_sp, dEdD_sp,
                                 addcoredensity, a)
    
    def initialize(self, dens, ham, wfs, occupations):
        self.xc.initialize(dens, ham, wfs, occupations)

        self.dens = dens
        self.wfs = wfs

        # Make a k-point descriptor that is not distributed
        # (self.kd.comm is serial_comm):
        self.kd = wfs.kd.copy()

        self.fd = logfile(self.fd, wfs.world.rank)

        vol = wfs.gd.dv * wfs.gd.N_c.prod()
        
        if self.alpha is None:
            self.alpha = 6 * vol**(2 / 3.0) / pi**2
        
        # Make q-point descriptor:
        N_c = self.kd.N_c
        i_qc = np.indices(N_c * 2 - 1).transpose((1, 2, 3, 0)).reshape((-1, 3))
        bzq_qc = (i_qc - N_c + 1.0) / N_c
        self.qd = KPointDescriptor(bzq_qc)

        self.pd = PWDescriptor(wfs.pd.ecut, wfs.gd,
                               dtype=wfs.pd.dtype, kd=self.kd)
        self.pd2 = PWDescriptor(dens.pd2.ecut, dens.gd,
                                dtype=wfs.dtype, kd=self.qd)
        if self.finegrid:
            self.pd3 = PWDescriptor(dens.pd3.ecut, dens.finegd,
                                    dtype=wfs.dtype, kd=self.qd)
        else:
            self.pd3 = self.pd2

        # Calculate 1/|G+q|^2 with special treatment of |G+q|=0:
        G2_qG = self.pd3.G2_qG
        q0 = ((N_c * 2 - 1).prod() - 1) // 2
        assert not bzq_qc[q0].any()
        G2_qG[q0][0] = 117.0
        self.iG2_qG = [1.0 / G2_G for G2_G in G2_qG]
        G2_qG[q0][0] = 0.0
        self.iG2_qG[q0][0] = 0.0

        self.gamma = (vol / (2 * pi)**2 * sqrt(pi / self.alpha) *
                      self.kd.nbzkpts)

        for q in range(self.qd.nbzkpts):
            if abs(bzq_qc[q] + 1e-7).max() < 0.5:
                self.gamma -= np.dot(np.exp(-self.alpha * G2_qG[q]),
                                     self.iG2_qG[q])

        self.iG2_qG[q0][0] = self.gamma
        
        # Compensation charges:
        self.ghat = PWLFC([setup.ghat_l for setup in wfs.setups], self.pd3)
        
        self.log('Value of alpha parameter: %.3f Bohr^2' % self.alpha)
        self.log('Value of gamma parameter: %.3f Bohr^2' % self.gamma)
        self.log('Cutoff energies:')
        self.log('    Wave functions:       %10.3f eV' %
                 (self.pd.ecut * Hartree))
        self.log('    Density:              %10.3f eV' %
                 (self.pd2.ecut * Hartree))
        self.log('    Compensation charges: %10.3f eV' %
                 (self.pd3.ecut * Hartree))
        self.log('%d x %d x %d k-points' % tuple(self.kd.N_c))

        if fftw.FFTPlan is fftw.NumpyFFTPlan:
            self.log('Not using FFTW!')
            
        if self.bandstructure:
            self.log('Calculating eigenvalue shifts.')

    def set_positions(self, spos_ac):
        self.ghat.set_positions(spos_ac)
        self.spos_ac = spos_ac

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)

         # Add EXX contribution:
        return exc + self.exx * self.hybrid

    def calculate_exx(self):
        """Non-selfconsistent calculation."""

        kd = self.kd
        K = kd.nibzkpts
        W = max(1, self.wfs.world.size // self.wfs.nspins)
        parallel = (W > 1)

        self.log("%d CPU's used for %d IBZ k-points" % (W, K))
        self.log('Spins:', self.wfs.nspins)

        # Find number of occupied bands:
        self.nocc_sk = np.zeros((self.wfs.nspins, kd.nibzkpts), int)
        for kpt in self.wfs.kpt_u:
            for n, f in enumerate(kpt.f_n):
                if abs(f) < self.fcut:
                    self.nocc_sk[kpt.s, kpt.k] = n
                    break
            else:
                self.nocc_sk[kpt.s, kpt.k] = self.wfs.bd.nbands
        self.wfs.kd.comm.sum(self.nocc_sk)

        noccmin = self.nocc_sk.min()
        noccmax = self.nocc_sk.max()
        self.log('Number of occupied bands (min, max): %d, %d' %
                 (noccmin, noccmax))

        self.log('Number of valence electrons:', self.wfs.setups.nvalence)

        if self.bandstructure:
            # allocate array for eigenvalue shifts:
            self.exx_skn = np.zeros((self.wfs.nspins, K, self.wfs.bd.nbands))

            if self.bands is None:
                noccmax = self.wfs.bd.nbands
            else:
                noccmax = max(max(self.bands) + 1, noccmax)

        B = noccmax
        E = B - self.wfs.setups.nvalence / 2.0  # empty bands
        self.npairs_estimate = (K * kd.nbzkpts - 0.5 * K**2) * (B**2 - E**2)
        self.log('Approximate number of pairs:', self.npairs_estimate)
        
        self.npairs = 0
        self.evv = 0.0
        self.evvacdf = 0.0
        for s in range(self.wfs.nspins):
            kpt1_q = [KPoint(self.wfs, noccmax).initialize(kpt)
                      for kpt in self.wfs.kpt_u if kpt.s == s]
            kpt2_q = kpt1_q[:]

            if len(kpt1_q) == 0:
                # No s-spins on this CPU:
                continue

            # Send and receive ranks:
            srank = self.wfs.kd.get_rank_and_index(s,
                                                   (kpt1_q[0].k - 1) % K)[0]
            rrank = self.wfs.kd.get_rank_and_index(s,
                                                   (kpt1_q[-1].k + 1) % K)[0]

            # Shift k-points K - 1 times:
            for i in range(K):
                if i < K - 1:
                    if parallel:
                        kpt = kpt2_q[-1].next(self.wfs)
                        kpt.start_receiving(rrank)
                        kpt2_q[0].start_sending(srank)
                    else:
                        kpt = kpt2_q[0]

                for kpt1, kpt2 in zip(kpt1_q, kpt2_q):
                    # Loop over all k-points that k2 can be mapped to:
                    for k, ik in enumerate(kd.bz2ibz_k):
                        if ik == kpt2.k:
                            self.apply(kpt1, kpt2, k)

                if i < K - 1:
                    if parallel:
                        kpt.wait()
                        kpt2_q[0].wait()
                    kpt2_q.pop(0)
                    kpt2_q.append(kpt)

        self.evv = self.wfs.world.sum(self.evv)
        self.evvacdf = self.wfs.world.sum(self.evvacdf)
        self.calculate_exx_paw_correction()
        
        if self.method == 'standard':
            self.exx = self.evv + self.devv + self.evc + self.ecc
        elif self.method == 'acdf':
            self.exx = self.evvacdf + self.devv + self.evc + self.ecc
        else:
            1 / 0

        self.log('Exact exchange energy:')
        for txt, e in [
            ('core-core', self.ecc),
            ('valence-core', self.evc),
            ('valence-valence (pseudo, acdf)', self.evvacdf),
            ('valence-valence (pseudo, standard)', self.evv),
            ('valence-valence (correction)', self.devv),
            ('total (%s)' % self.method, self.exx)]:
            self.log('    %-36s %14.6f eV' % (txt + ':', e * Hartree))

        self.log('Number of pairs:', self.npairs)

    def apply(self, kpt1, kpt2, k):
        k1_c = self.kd.ibzk_kc[kpt1.k]
        k20_c = self.kd.ibzk_kc[kpt2.k]
        k2_c = self.kd.bzk_kc[k]
        q_c = k2_c - k1_c
        q = abs(self.qd.bzk_kc - q_c).sum(1).argmin()

        same = abs(k1_c - k2_c).max() < 1e-9

        if self.skip_gamma and same:
            return

        if k2_c.any():
            eik2r_R = self.wfs.gd.plane_wave(k2_c)
            eik20r_R = self.wfs.gd.plane_wave(k20_c)
        else:
            eik2r_R = 1.0
            eik20r_R = 1.0

        iG2_G = self.iG2_qG[q]
        f_IG = self.ghat.expand(q)
        if self.finegrid:
            G3_G = self.pd2.map(self.pd3, q)
        else:
            G3_G = None

        w1 = self.kd.weight_k[kpt1.k]
        w2 = self.kd.weight_k[kpt2.k]

        nocc1 = self.nocc_sk[kpt1.s, kpt1.k]
        nocc2 = self.nocc_sk[kpt2.s, kpt2.k]

        # Is k2 in the 1. BZ?
        is_ibz2 = abs(k2_c - k20_c).max() < 1e-9

        for n2 in range(self.wfs.bd.nbands):
            f2 = kpt2.f_n[n2]
            eps2 = kpt2.eps_n[n2]

            # Find range of n1's (from n1a to n1b-1):
            if is_ibz2:
                # We get this combination twice, so let only do half:
                if kpt1.k >= kpt2.k:
                    n1a = n2
                else:
                    n1a = n2 + 1
            else:
                n1a = 0

            n1b = self.wfs.bd.nbands

            if self.bandstructure:
                if n2 >= nocc2:
                    n1b = min(n1b, nocc1)
            else:
                if n2 >= nocc2:
                    break
                n1b = min(n1b, nocc1)

            if self.bands is not None:
                assert self.bandstructure
                n1_n = []
                for n1 in range(n1a, n1b):
                    if (n1 in self.bands and n2 < nocc2 or
                        is_ibz2 and n2 in self.bands and n1 < nocc1):
                        n1_n.append(n1)
                n1_n = np.array(n1_n)
            else:
                n1_n = np.arange(n1a, n1b)

            if len(n1_n) == 0:
                continue

            e_n = self.calculate_interaction(n1_n, n2, kpt1, kpt2, q, k,
                                             eik20r_R, eik2r_R, G3_G, f_IG,
                                             iG2_G, is_ibz2)

            e_n *= 1.0 / self.kd.nbzkpts / self.wfs.nspins

            if same:
                e_n[n1_n == n2] *= 0.5

            f1_n = kpt1.f_n[n1_n]
            eps1_n = kpt1.eps_n[n1_n]
            s_n = np.sign(eps2 - eps1_n)

            evv = (f1_n * f2 * e_n).sum()
            evvacdf = 0.5 * (f1_n * (1 - s_n) * e_n +
                             f2 * (1 + s_n) * e_n).sum()
            self.evv += evv * w1
            self.evvacdf += evvacdf * w1
            if is_ibz2:
                self.evv += evv * w2
                self.evvacdf += evvacdf * w2
            
            if self.bandstructure:
                x = self.wfs.nspins
                self.exx_skn[kpt1.s, kpt1.k, n1_n] += x * f2 * e_n
                if is_ibz2:
                    self.exx_skn[kpt2.s, kpt2.k, n2] += x * np.dot(f1_n, e_n)

    def calculate_interaction(self, n1_n, n2, kpt1, kpt2, q, k,
                              eik20r_R, eik2r_R, G3_G, f_IG, iG2_G, is_ibz2):
        """Calculate Coulomb interactions.

        For all n1 in the n1_n list, calculate interaction with n2."""

        t0 = time()

        # number of plane waves:
        ng1 = self.wfs.ng_k[kpt1.k]
        ng2 = self.wfs.ng_k[kpt2.k]

        # Transform to real space and apply symmetry operation:
        if is_ibz2:
            u2_R = self.pd.ifft(kpt2.psit_nG[n2, : ng2], kpt2.k)
        else:
            psit2_R = self.pd.ifft(kpt2.psit_nG[n2, : ng2], kpt2.k) * eik20r_R
            u2_R = self.kd.transform_wave_function(psit2_R, k) / eik2r_R

        # Calculate pair densities:
        nt_nG = self.pd3.zeros(len(n1_n), q=q)
        for n1, nt_G in zip(n1_n, nt_nG):
            u1_R = self.pd.ifft(kpt1.psit_nG[n1, :ng1], kpt1.k)
            nt_R = u1_R.conj() * u2_R

            if self.finegrid:
                nt_G[G3_G] = self.pd2.interpolate(nt_R, self.pd3, q)[1] * 8
            else:
                nt_G[:] = self.pd2.fft(nt_R, q)
        
        s = self.kd.sym_k[k]
        time_reversal = self.kd.time_reversal_k[k]
        k2_c = self.kd.ibzk_kc[kpt2.k]

        Q_anL = {}  # coefficients for shape functions
        for a, P1_ni in kpt1.P_ani.items():
            P1_ni = P1_ni[n1_n]

            if is_ibz2:
                P2_i = kpt2.P_ani[a][n2]
            else:
                b = self.kd.symmetry.a_sa[s, a]
                S_c = (np.dot(self.spos_ac[a], self.kd.symmetry.op_scc[s]) -
                       self.spos_ac[b])
                assert abs(S_c.round() - S_c).max() < 1e-5
                if self.ghat.dtype == complex:
                    x = np.exp(2j * pi * np.dot(k2_c, S_c))
                else:
                    x = 1.0
                P2_i = np.dot(self.wfs.setups[a].R_sii[s],
                              kpt2.P_ani[b][n2]) * x
                if time_reversal:
                    P2_i = P2_i.conj()

            D_np = []
            for P1_i in P1_ni:
                D_ii = np.outer(P1_i.conj(), P2_i)
                D_np.append(pack(D_ii))
            Q_anL[a] = np.dot(D_np, self.wfs.setups[a].Delta_pL)

        # Add compensation charges:
        self.ghat.add(nt_nG, Q_anL, q, f_IG)

        # Calculate energies:
        e_n = np.empty(len(n1_n))
        for n, nt_G in enumerate(nt_nG):
            e_n[n] = -4 * pi * np.real(self.pd3.integrate(nt_G, nt_G * iG2_G))
            self.npairs += 1

        if self.write_timing_information:
            t = (time() - t0) / len(n1_n)
            self.log('Time for first pair-density: %10.3f seconds' % t)
            self.log('Estimated total time:        %10.3f seconds' %
                     (t * self.npairs_estimate / self.wfs.world.size))
            self.write_timing_information = False

        return e_n

    def calculate_exx_paw_correction(self):
        self.devv = 0.0
        self.evc = 0.0
        self.ecc = 0.0
                         
        deg = 2 // self.wfs.nspins  # spin degeneracy
        for a, D_sp in self.dens.D_asp.items():
            setup = self.wfs.setups[a]
            for D_p in D_sp:
                D_ii = unpack2(D_p)
                ni = len(D_ii)

                for i1 in range(ni):
                    for i2 in range(ni):
                        A = 0.0
                        for i3 in range(ni):
                            p13 = packed_index(i1, i3, ni)
                            for i4 in range(ni):
                                p24 = packed_index(i2, i4, ni)
                                A += setup.M_pp[p13, p24] * D_ii[i3, i4]
                        self.devv -= D_ii[i1, i2] * A / deg

                self.evc -= np.dot(D_p, setup.X_p)
            self.ecc += setup.ExxC
        
        if not self.bandstructure:
            return

        for kpt in self.wfs.kpt_u:
            for a, D_sp in self.dens.D_asp.items():
                setup = self.wfs.setups[a]
                for D_p in D_sp:
                    D_ii = unpack2(D_p)
                    ni = len(D_ii)
                    P_ni = kpt.P_ani[a]
                    for i1 in range(ni):
                        for i2 in range(ni):
                            A = 0.0
                            for i3 in range(ni):
                                p13 = packed_index(i1, i3, ni)
                                for i4 in range(ni):
                                    p24 = packed_index(i2, i4, ni)
                                    A += setup.M_pp[p13, p24] * D_ii[i3, i4]
                            self.exx_skn[kpt.s, kpt.k] -= \
                                (A * P_ni[:, i1].conj() * P_ni[:, i2]).real
                            p12 = packed_index(i1, i2, ni)
                            self.exx_skn[kpt.s, kpt.k] -= \
                                (P_ni[:, i1].conj() * setup.X_p[p12] *
                                 P_ni[:, i2]).real / self.wfs.nspins

        self.exx_skn *= self.hybrid
        self.wfs.world.sum(self.exx_skn)


class KPoint:
    def __init__(self, wfs, nbands):
        """Helper class for parallelizing over k-points.

        Placeholder for wave functions, occupation numbers, eigenvalues,
        projections, spin index and global k-point index."""
        
        self.kd = wfs.kd
        self.ng_k = wfs.ng_k

        # Array large enough to hold wave functions from all k-points:
        self.psit_nG = wfs.pd.empty(nbands)

        self.requests = []

    def initialize(self, kpt):
        ng = self.ng_k[kpt.k]
        nbands = len(self.psit_nG)
        self.psit_nG[:, :ng] = kpt.psit_nG[:nbands]
        self.f_n = kpt.f_n / kpt.weight  # will be in the range [0,1]
        self.eps_n = kpt.eps_n
        self.P_ani = kpt.P_ani
        self.k = kpt.k
        self.s = kpt.s

        return self

    def next(self, wfs):
        """Create empty object.

        Data will be received from another process."""
        
        nbands = len(self.psit_nG)
        kpt = KPoint(wfs, nbands)

        # Allocate arrays for receiving:
        kpt.f_n = wfs.bd.empty()
        kpt.eps_n = wfs.bd.empty()

        # Total number of projector functions:
        I = sum([P_ni.shape[1] for P_ni in self.P_ani.values()])
        
        kpt.P_In = np.empty((I, wfs.bd.nbands), wfs.dtype)

        kpt.P_ani = {}
        I1 = 0
        assert self.P_ani.keys() == range(len(self.P_ani))  # ???
        for a, P_ni in self.P_ani.items():
            I2 = I1 + P_ni.shape[1]
            kpt.P_ani[a] = kpt.P_In[I1:I2].T
            I1 = I2

        kpt.k = (self.k + 1) % self.kd.nibzkpts
        kpt.s = self.s
        
        return kpt
        
    def start_sending(self, rank):
        assert self.P_ani.keys() == range(len(self.P_ani))  # ???
        P_In = np.concatenate([P_ni.T for P_ni in self.P_ani.values()])
        self.requests += [
            self.kd.comm.send(self.psit_nG, rank, block=False, tag=1),
            self.kd.comm.send(self.f_n, rank, block=False, tag=2),
            self.kd.comm.send(self.eps_n, rank, block=False, tag=3),
            self.kd.comm.send(P_In, rank, block=False, tag=4)]
        
    def start_receiving(self, rank):
        self.requests += [
            self.kd.comm.receive(self.psit_nG, rank, block=False, tag=1),
            self.kd.comm.receive(self.f_n, rank, block=False, tag=2),
            self.kd.comm.receive(self.eps_n, rank, block=False, tag=3),
            self.kd.comm.receive(self.P_In, rank, block=False, tag=4)]
    
    def wait(self):
        self.kd.comm.waitall(self.requests)
        self.requests = []
