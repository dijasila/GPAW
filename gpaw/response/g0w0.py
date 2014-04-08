from __future__ import division, print_function

from math import pi

import numpy as np
from ase.units import Hartree
from ase.utils import opencew
from ase.dft.kpoints import monkhorst_pack

import gpaw.mpi as mpi
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
from gpaw.utilities.timing import timer
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.chi0 import Chi0, HilbertTransform
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb


class G0W0(PairDensity):
    def __init__(self, calc, filename='gw',
                 kpts=None, bands=None, nbands=None, ppa=False, hilbert=True,
                 wstc=False, fast=True,
                 ecut=150.0, eta=0.1, E0=1.0 * Hartree,
                 domega0=0.025, alpha=3.0,
                 world=mpi.world):
    
        PairDensity.__init__(self, calc, ecut, world=world,
                             txt=filename + '.txt')
        
        self.filename = filename
        
        ecut /= Hartree
        
        self.ppa = ppa
        self.fast = fast
        self.hilbert = hilbert
        self.wstc = wstc
        self.eta = eta / Hartree
        self.E0 = E0 / Hartree
        self.domega0 = domega0 / Hartree
        self.alpha = alpha

        print('  ___  _ _ _ ', file=self.fd)
        print(' |   || | | |', file=self.fd)
        print(' | | || | | |', file=self.fd)
        print(' |__ ||_____|', file=self.fd)
        print(' |___|       ', file=self.fd)
        print(file=self.fd)

        if kpts is None:
            kpts = range(self.calc.wfs.kd.nibzkpts)
        
        if bands is None:
            bands = [0, self.nocc2]
            
        self.kpts = kpts
        self.bands = bands

        b1, b2 = bands
        self.shape = shape = (self.calc.wfs.nspins, len(kpts), b2 - b1)
        self.eps_sin = np.empty(shape)     # KS-eigenvalues
        self.f_sin = np.empty(shape)       # occupation numbers
        self.sigma_sin = np.zeros(shape)   # self-energies
        self.dsigma_sin = np.zeros(shape)  # derivatives of self-energies
        self.vxc_sin = None                # KS XC-contributions
        self.exx_sin = None                # exact exchange contributions
        self.Z_sin = None                  # renormalization factors

        self.nbands = nbands
        
        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(nbands)
        
        # Find q-vectors and weights in the IBZ:
        kd = self.calc.wfs.kd
        assert -1 not in kd.bz2bz_ks
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.calc.atoms, self.calc.wfs.setups,
                             usesymm=self.calc.input_parameters.usesymm,
                             N_c=self.calc.wfs.gd.N_c)
        
    @timer('G0W0')
    def calculate(self):
        kd = self.calc.wfs.kd

        self.calculate_ks_xc_contribution()
        self.calculate_exact_exchange()
        self.calculate_screened_potential()
        
        mykpts = [self.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]

        for s in range(self.calc.wfs.nspins):
            for i, k1 in enumerate(self.kpts):
                K1 = kd.ibz2bz_k[k1]
                kpt1 = self.get_k_point(s, K1, *self.bands)
                self.eps_sin[s, i] = kpt1.eps_n
                self.f_sin[s, i] = kpt1.f_n
                for kpt2 in mykpts:
                    if kpt2.s == s:
                        self.calculate_q(i, kpt1, kpt2)

        self.world.sum(self.sigma_sin)
        self.world.sum(self.dsigma_sin)
        
        self.Z_sin = 1 / (1 - self.dsigma_sin)
        self.qp_sin = self.eps_sin + self.Z_sin * (self.sigma_sin +
                                                   self.exx_sin -
                                                   self.vxc_sin)
        
        description = ['f:     Occupation numbers',
                       'eps:   KS-eigenvalues [eV]',
                       'vxc:   KS vxc [eV]',
                       'exx:   Exact exchange [eV]',
                       'sigma: Self-energies [eV]',
                       'Z:     Renormalization factors',
                       'qp:    QP-energies [eV]']

        print('\nResults:', file=self.fd)
        for line in description:
            print(line, file=self.fd)
            
        results = {'f': self.f_sin,
                   'eps': self.eps_sin * Hartree,
                   'vxc': self.vxc_sin * Hartree,
                   'exx': self.exx_sin * Hartree,
                   'sigma': self.sigma_sin * Hartree,
                   'Z': self.Z_sin,
                   'qp': self.qp_sin * Hartree}

        b1, b2 = self.bands
        names = [line.split(':', 1)[0] for line in description]
        for s in range(self.calc.wfs.nspins):
            for i, ik in enumerate(self.kpts):
                print('\nk-point ' +
                      '{0} ({1}): ({2:.3f}, {3:.3f}, {4:.3f})'.format(
                          i, ik, *kd.ibzk_kc[ik]), file=self.fd)
                print('band' +
                      ''.join('{0:>8}'.format(name) for name in names),
                      file=self.fd)
                for n in range(b2 - b1):
                    print('{0:4}'.format(n + b1) +
                          ''.join('{0:8.3f}'.format(results[name][s, i, n])
                                  for name in names),
                          file=self.fd)

        self.timer.write(self.fd)
        
        return results
        
    def calculate_q(self, i, kpt1, kpt2):
        wfs = self.calc.wfs
        qd = self.qd
        
        q_c = wfs.kd.bzk_kc[kpt2.K] - wfs.kd.bzk_kc[kpt1.K]
        
        # Find index of q-vector:
        d_Qc = ((qd.bzk_kc - q_c) * qd.N_c).round() % qd.N_c
        Q = abs(d_Qc).sum(axis=1).argmin()

        # Find symmetry related q-vector in IBZ:
        s = qd.sym_k[Q]
        U_cc = qd.symmetry.op_scc[s]
        time_reversal = qd.time_reversal_k[Q]
        iq = qd.bz2ibz_k[Q]
        iq_c = qd.ibzk_kc[iq]
        sign = 1 - 2 * time_reversal
        suiq_c = sign * np.dot(U_cc, iq_c)
        shift_c = suiq_c - q_c

        assert np.allclose(shift_c.round(), shift_c)
        shift_c = shift_c.round().astype(int)
        
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, wfs.gd, complex, qd)
        N0_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                    kpt1.shift_c - kpt2.shift_c)

        N_c = pd.gd.N_c

        # Read W and transform from IBZ to BZ:
        fd = open('{0}.w.q{1}.npy'.format(self.filename, iq))
        assert (iq_c == np.load(fd)).all()
        N_G = np.load(fd)
        nG = len(N_G)
        
        n_cG = np.unravel_index(N_G, N_c)
        N3_G = np.ravel_multi_index(
            sign * np.dot(U_cc, n_cG) +
            (shift_c + kpt1.shift_c - kpt2.shift_c)[:, None],
            N_c, 'wrap')
        G_N = np.empty(N_c.prod(), int)
        G_N[:] = -1
        G_N[N3_G] = np.arange(nG)
        G_G = G_N[N0_G]
        assert (G_G >= 0).all()

        W = self.read_screened_potential(fd, G_G)
        fd.close()

        Q_aGii = self.initialize_paw_corrections(pd)
        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            eps1 = kpt1.eps_n[n]
            C1_aGi = [np.dot(Q_Gii, P1_ni[n].conj())
                     for Q_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
            n_mG = self.calculate_pair_densities(ut1cc_R, C1_aGi, kpt2,
                                                 pd, N0_G)
            f_m = kpt2.f_n
            deps_m = eps1 - kpt2.eps_n
            sigma, dsigma = self.calculate_sigma(fd, n_mG, deps_m, f_m, W)
            self.sigma_sin[kpt1.s, i, n] += sigma
            self.dsigma_sin[kpt1.s, i, n] += dsigma
            
    def read_screened_potential(self, fd, G_G):
        def T(W_GG):
            W_GG[:] = W_GG.take(G_G, 0).take(G_G, 1)
            return W_GG
            
        if self.ppa:
            with self.timer('Read W'):
                W_GG = np.load(fd)
                omegat_GG = np.load(fd)
            with self.timer('Symmetry transform of W'):
                return [T(W_GG), T(omegat_GG)]
            
        if self.fast:
            with self.timer('Read W'):
                W_swGG = [np.load(fd), np.load(fd)]
            with self.timer('Symmetry transform of W'):
                for W_wGG in W_swGG:
                    for W_GG in W_wGG:
                        T(W_GG)
            return W_swGG
        
        with self.timer('Read W'):
            W_wGG = [np.load(fd) for o in self.omega_w]
        with self.timer('Symmetry transform of W'):
            for W_GG in W_wGG:
                T(W_GG)
        return W_wGG
        
    @timer('Sigma')
    def calculate_sigma(self, fd, n_mG, deps_m, f_m, W_wGG):
        if self.ppa:
            return self.calculate_sigma_ppa(fd, n_mG, deps_m, f_m, *W_wGG)
        elif self.fast:
            return self.calculate_sigma2(fd, n_mG, deps_m, f_m, W_wGG)
            
        sigma = 0.0
        dsigma = 0.0
        
        x = self.domega0 / (self.qd.nbzkpts * 2 * pi * self.vol)
        assert self.alpha == 0
        for W_GG, omegap in zip(W_wGG, self.omega_w):
            x1_m = 1 / (deps_m + omegap - 2j * self.eta * (f_m - 0.5))
            x2_m = 1 / (deps_m - omegap - 2j * self.eta * (f_m - 0.5))
            x_m = x1_m + x2_m
            dx_m = x1_m**2 + x2_m**2
            nW_mG = np.dot(n_mG, W_GG)
            sigma += x * np.vdot(n_mG * x_m[:, np.newaxis], nW_mG).imag
            dsigma -= x * np.vdot(n_mG * dx_m[:, np.newaxis], nW_mG).imag

        return sigma, dsigma

    @timer('Sigma2')
    def calculate_sigma2(self, fd, n_mG, deps_m, f_m, C_swGG):
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)
        
        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2
        
        w_m = (o_m / self.domega0 /
               (1 + self.alpha * o_m / self.omegamax)).astype(int)
        o1_m = self.omega_w[w_m]
        o2_m = self.omega_w[w_m + 1]
        
        x = 1.0 / (self.qd.nbzkpts * 2 * pi * self.vol)
        sigma = 0.0
        dsigma = 0.0
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):
            C1_GG = C_swGG[s][w]
            C2_GG = C_swGG[s][w + 1]
            p = x * sgn
            sigma1 = p * np.dot(np.dot(n_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(n_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)
            
        return sigma, dsigma

    @timer('PPA-Sigma')
    def calculate_sigma_ppa(self, fd, n_mG, deps_m, f_m, W_GG, omegat_GG):
        deps_mGG = deps_m[:, np.newaxis, np.newaxis]
        sign_mGG = 2 * f_m[:, np.newaxis, np.newaxis] - 1
        x1_mGG = 1 / (deps_mGG + omegat_GG - 1j * self.eta)
        x2_mGG = 1 / (deps_mGG - omegat_GG + 1j * self.eta)
        x3_mGG = 1 / (deps_mGG + omegat_GG - 1j * self.eta * sign_mGG)
        x4_mGG = 1 / (deps_mGG - omegat_GG - 1j * self.eta * sign_mGG)
        x_mGG = W_GG * (sign_mGG * (x1_mGG - x2_mGG) + x3_mGG + x4_mGG)
        dx_mGG = W_GG * (sign_mGG * (x1_mGG**2 - x2_mGG**2) +
                         x3_mGG**2 + x4_mGG**2)

        sigma = 0.0
        dsigma = 0.0
        for m in range(np.shape(n_mG)[0]):
            nW_mG = np.dot(n_mG[m], x_mGG[m])
            sigma += np.vdot(n_mG[m], nW_mG).real
            nW_mG = np.dot(n_mG[m], dx_mGG[m])
            dsigma -= np.vdot(n_mG[m], nW_mG).real
        
        x = 1 / (self.qd.nbzkpts * 2 * pi * self.vol)
        return x * sigma, x * dsigma

    @timer('W')
    def calculate_screened_potential(self):
        chi0 = None
        
        if self.ppa:
            print('Using Godby-Needs plasmon-pole approximation:',
                  file=self.fd)
            print('    Fitting energy: i*E0, E0 = %.3f Hartee' % self.E0,
                  file=self.fd)

            # use small imaginary frequency to avoid dividing by zero:
            frequencies = [1e-10j, 1j * self.E0 * Hartree]
            
            parameters = {'eta': 0,
                          'hilbert': False,
                          'timeordered': False,
                          'frequencies': frequencies}
        else:
            parameters = {'eta': self.eta * Hartree,
                          'hilbert': self.hilbert,
                          'timeordered': True,
                          'domega0': self.domega0 * Hartree,
                          'alpha': self.alpha}
            
        chi0 = Chi0(self.calc,
                    nbands=self.nbands,
                    ecut=self.ecut * Hartree,
                    intraband=False,
                    real_space_derivatives=False,
                    txt=self.filename + '.w.txt',
                    timer=self.timer,
                    **parameters)
        
        self.omega_w = chi0.omega_w
        self.omegamax = chi0.omegamax
        
        first_time = True
        
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            fd = opencew('%s.w.q%d.npy' % (self.filename, iq))
            if fd is None:
                continue
                
            if first_time:
                first_time = False
                print('Calulating screened Coulomb potential', file=self.fd)
            
                if self.wstc:
                    wstc = WignerSeitzTruncatedCoulomb(
                        self.calc.wfs.gd.cell_cv,
                        self.calc.wfs.kd.N_c,
                        chi0.fd)
                
                if self.fast:
                    htp = HilbertTransform(self.omega_w, self.eta, gw=True)
                    htm = HilbertTransform(self.omega_w, -self.eta, gw=True)
            
            pd, chi0_wGG = chi0.calculate(q_c)[:2]

            if self.wstc:
                iG_G = (wstc.get_potential(pd) / (4 * pi))**0.5
                if np.allclose(q_c, 0):
                    chi0_wGG[:, 0] = 0.0
                    chi0_wGG[:, :, 0] = 0.0
                    G0inv = 0.0
                    G20inv = 0.0
            else:
                if np.allclose(q_c, 0):
                    dq3 = (2 * pi)**3 / (self.qd.nbzkpts * self.vol)
                    qc = (dq3 / 4 / pi * 3)**(1 / 3)
                    G0inv = 2 * pi * qc**2 / dq3
                    G20inv = 4 * pi * qc / dq3
                    G_G = pd.G2_qG[0]**0.5
                    G_G[0] = 1
                    iG_G = 1 / G_G
                else:
                    iG_G = pd.G2_qG[0]**-0.5
                
            delta_GG = np.eye(len(iG_G))
            
            np.save(fd, q_c)
            np.save(fd, pd.Q_qG[0])

            if self.ppa:
                einv_wGG = []
                for chi0_GG in chi0_wGG:
                    e_GG = (delta_GG -
                            4 * pi * chi0_GG * iG_G * iG_G[:, np.newaxis])
                    einv_wGG.append(np.linalg.inv(e_GG) - delta_GG)

                if self.wstc and np.allclose(q_c, 0):
                    einv_wGG[0][0] = 42
                    einv_wGG[0][:, 0] = 42
                omegat_GG = self.E0 * np.sqrt(einv_wGG[1] /
                                              (einv_wGG[0] - einv_wGG[1]))
                R_GG = -0.5 * omegat_GG * einv_wGG[0]
                W_GG = 4 * pi**2 * R_GG * iG_G * iG_G[:, np.newaxis]
                if np.allclose(q_c, 0):
                    W_GG[0, 0] *= G20inv
                    W_GG[1:, 0] *= G0inv
                    W_GG[0, 1:] *= G0inv

                np.save(fd, W_GG)
                np.save(fd, omegat_GG)
            else:
                for chi0_GG in chi0_wGG:
                    e_GG = (delta_GG -
                            4 * pi * chi0_GG * iG_G * iG_G[:, np.newaxis])
                    W_GG = chi0_GG
                    W_GG[:] = 4 * pi * (np.linalg.inv(e_GG) -
                                        delta_GG) * iG_G * iG_G[:, np.newaxis]
                    if np.allclose(q_c, 0):
                        W_GG[0, 0] *= G20inv
                        W_GG[1:, 0] *= G0inv
                        W_GG[0, 1:] *= G0inv
                        
                if self.fast:
                    Wp_wGG = chi0_wGG.copy()
                    Wm_wGG = chi0_wGG
                    with self.timer('Hilbert transform'):
                        htp(Wp_wGG)
                        htm(Wm_wGG)
                    for W_wGG in [Wp_wGG, Wm_wGG]:
                        np.save(fd, W_wGG)
                else:
                    np.save(fd, chi0_wGG)
                            
            fd.close()

    @timer('Kohn-Sham XC-contribution')
    def calculate_ks_xc_contribution(self):
        name = self.filename + '.vxc.npy'
        fd = opencew(name)
        if fd is None:
            print('Reading Kohn-Sham XC contribution from file:', name,
                  file=self.fd)
            with open(name) as fd:
                self.vxc_sin = np.load(fd)
            assert self.vxc_sin.shape == self.shape, self.vxc_sin.shape
            return
            
        print('Calculating Kohn-Sham XC contribution', file=self.fd)
        vxc_skn = vxc(self.calc, self.calc.hamiltonian.xc) / Hartree
        n1, n2 = self.bands
        self.vxc_sin = vxc_skn[:, self.kpts, n1:n2]
        np.save(fd, self.vxc_sin)
        
    @timer('EXX')
    def calculate_exact_exchange(self):
        name = self.filename + '.exx.npy'
        fd = opencew(name)
        if fd is None:
            print('Reading EXX contribution from file:', name, file=self.fd)
            with open(name) as fd:
                self.exx_sin = np.load(fd)
            assert self.exx_sin.shape == self.shape, self.exx_sin.shape
            return
            
        print('Calculating EXX contribution', file=self.fd)
        exx = EXX(self.calc, kpts=self.kpts, bands=self.bands,
                  txt=self.filename + '.exx.txt', timer=self.timer)
        exx.calculate()
        self.exx_sin = exx.get_eigenvalue_contributions() / Hartree
        np.save(fd, self.exx_sin)
