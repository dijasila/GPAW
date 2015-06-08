# This makes sure that division works as in Python3
from __future__ import division, print_function

import sys
from math import pi
import numpy as np
import pickle
from ase.dft.kpoints import monkhorst_pack
from gpaw.kpt_descriptor import to1bz, KPointDescriptor
from gpaw.response.pair import PairDensity, PWSymmetryAnalyzer
from gpaw.response.chi0 import Chi0, HilbertTransform
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.response.df import DielectricFunction
from gpaw.wavefunctions.pw import count_reciprocal_vectors
import gpaw.mpi as mpi
from ase.utils import devnull, opencew
from ase.utils.timing import timer
from ase.units import Hartree

class SelfEnergy:

    def __init__(self):
        pass

class GWSelfEnergy(SelfEnergy):

    def __init__(self, calc, kpts=None, bandrange=None,
                 filename=None, txt=sys.stdout, savew=False,
                 nbands=None, ecut=150., nblocks=1, hilbert=True, eta=0.1,
                 domega0=0.025, omega2=10., omegamax=None,
                 qptint=None, truncation='wigner-seitz',
                 world=mpi.world):

        # Create output buffer
        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        SelfEnergy.__init__(self)

        self.ecut = ecut / Hartree
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.omegamax = omegamax / Hartree if not omegamax is None else None
        self.eta = eta / Hartree
        
        self.calc = calc

        if kpts is None:
            kpts = range(len(calc.get_ibz_k_points()))
        self.kpts = kpts

        vol = abs(np.linalg.det(self.calc.wfs.gd.cell_cv))
        if nbands is None:
            nbands = int(vol * self.ecut**1.5 * 2**0.5 / 3 / pi**2)
        self.nbands = nbands

        if bandrange is None:
            bandrange = (0, calc.get_number_of_bands())
        
        self.bandrange = bandrange

        self.filename = filename
        self.savew = savew

        self.truncation = truncation

        kd = calc.wfs.kd

        self.world = world
        self.pairDensity = PairDensity(calc, ecut=ecut, nblocks=nblocks,
                                       world=mpi.world)
        self.timer = self.pairDensity.timer
        self.blockcomm = self.pairDensity.blockcomm

        #self.dielectric_function = DielectricFunction(calc, ecut=ecut,
        #    domega0=self.domega0 * Hartree, omega2=self.omega2 * Hartree,
        #    txt=devnull, truncation=truncation)
        #self.omega_w = self.dielectric_function.chi0.omega_w
        #self.omegamax = self.dielectric_function.chi0.omegamax
        #self.Ga = self.dielectric_function.chi0.Ga
        #self.Gb = self.dielectric_function.chi0.Gb
        
        self.wstc = None
        

        self.mysKn1n2 = None
        b1, b2 = self.bandrange

        self.shape = (1, len(kpts), b2 - b1)
        
        self.pairDensity.distribute_k_points_and_bands(b1, b2,
                                                       kd.ibz2bz_k[self.kpts])
        self.mysKn1n2 = self.pairDensity.mysKn1n2
        
        # Find q-vectors and weights in the IBZ:
        assert -1 not in kd.bz2bz_ks
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.calc.atoms, kd.symmetry)

        if qptint is None:
            cell_cv = calc.wfs.gd.cell_cv
            weights_q = 1.0 / len(bzq_qc) * np.ones(len(bzq_qc))
            self.qpt_integration = QuadQPointIntegration(bzq_qc,
                                                        cell_cv,
                                                        weights_q)
        else:
            self.qpt_integration = qptint
        

    def calculate(self, readw=True):

        # My part of the states we want to calculate QP-energies for:
        mykpts = [self.pairDensity.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]

        kd = self.calc.wfs.kd
        #print('bzk_kc=%s' % kd.bzk_kc)
        #print('ibzk_kc=%s' % kd.ibzk_kc)
        #print('sym_k=%s' % kd.sym_k)
        kplusqdone_u = [set() for kpt in mykpts]
        
        nbzq = len(self.qpt_integration.qpts_qc)
        self.sigma_qsin = np.zeros((nbzq, ) + self.shape)
        self.dsigma_qsin = np.zeros((nbzq, ) + self.shape)
        
        for Q1, Q2, pd0, W0, Q0_aGii in self.do_qpt_loop(readw=readw):
            ibzq = self.qd.bz2ibz_k[Q1]
            q_c = self.qd.ibzk_kc[ibzq]
            
            s = self.qd.sym_k[Q2]
            U_cc = self.qd.symmetry.op_scc[s]
            timerev = self.qd.time_reversal_k[Q2]
            sign = 1 - 2 * timerev

            Q_c = self.qd.bzk_kc[Q2]
            diff_c = sign * np.dot(U_cc, q_c) - Q_c
            assert np.allclose(diff_c.round(), diff_c), ("Difference should only " +
                                                   "be equal to a reciprocal" +
                                                   " lattice vector")
            diff_c = diff_c.round().astype(int)

            # Find all BZ q-points included in the integration that are
            # equivalent to Q_c, i.e. differ by a whole reciprocal lattice
            # vector. The contribution to the self-energy for these points
            # is the same as for Q_c
            bz1q_qc = to1bz(self.qpt_integration.qpts_qc, kd.symmetry.cell_cv)
            bzqs = []
            for bzq, bzq_c in enumerate(bz1q_qc):
                dq_c = bzq_c - Q_c
                if np.allclose(dq_c.round(), dq_c):
                    bzqs.append(bzq)

            G_Gv = pd0.get_reciprocal_vectors()
            pos_av = np.dot(self.pairDensity.spos_ac, pd0.gd.cell_cv)
            M_vv = np.dot(pd0.gd.cell_cv.T,
                          np.dot(U_cc.T, np.linalg.inv(pd0.gd.cell_cv).T))
            # Transform PAW corrections from IBZ to full BZ
            Q_aGii = []
            for a, Q_Gii in enumerate(Q0_aGii):
                x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] - sign *
                                                np.dot(M_vv, pos_av[a]))))
                U_ii = self.calc.wfs.setups[a].R_sii[s]
                Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, None]),
                               U_ii.T).transpose(1, 0, 2)
                Q_aGii.append(Q_Gii)

            for u1, kpt1 in enumerate(mykpts):
                k1 = kd.bz2ibz_k[kpt1.K]
                spin = kpt1.s
                i = self.kpts.index(k1)
                
                K2 = kd.find_k_plus_q(Q_c, [kpt1.K])[0] # K2 will be in 1st BZ
                # This k+q or symmetry related points have not been
                # calculated yet.
                kpt2 = self.pairDensity.get_k_point(0, K2, 0,
                                                    self.nbands,
                                                    block=True)

                N_c = pd0.gd.N_c
                i_cG = sign * np.dot(U_cc, np.unravel_index(pd0.Q_qG[0], N_c))

                k1_c = kd.bzk_kc[kpt1.K]
                k2_c = kd.bzk_kc[K2]
                # This is the q that connects K1 and K2 in the 1st BZ
                q1_c = kd.bzk_kc[K2] - kd.bzk_kc[kpt1.K]

                # G-vector that connects the full Q_c with q1_c
                shift1_c = q1_c - sign * np.dot(U_cc, q_c)
                assert np.allclose(shift1_c.round(), shift1_c)
                shift1_c = shift1_c.round().astype(int)
                shift_c = kpt1.shift_c - kpt2.shift_c - shift1_c
                I_G = np.ravel_multi_index(i_cG + shift_c[:, None], N_c, 'wrap')

                for n in range(kpt1.n2 - kpt1.n1):
                    ut1cc_R = kpt1.ut_nR[n].conj()
                    C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                              for Qa_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
                    n_mG = self.pairDensity \
                        .calculate_pair_densities(ut1cc_R, C1_aGi,
                                                  kpt2, pd0, I_G)
                    
                    if sign == 1:
                        n_mG = n_mG.conj()

                    if np.allclose(q1_c, 0) and not self.wstc:
                        # If we're at the Gamma point the G=0 component of the
                        # pair density is a delta in the band index
                        n_mG[:, 0] = 0
                        m = n + kpt1.n1 - kpt2.n1
                        if 0 <= m < len(n_mG):
                            n_mG[m, 0] = 1.0
                        # Why is this necessary?

                    f_m = kpt2.f_n
                    deps_m = kpt1.eps_n[n] - kpt2.eps_n
                    nn = kpt1.n1 + n - self.bandrange[0]

                    sigma, dsigma = self.calculate_sigma(n_mG, deps_m, f_m, W0)
                    
                    for bzq in bzqs:
                        self.sigma_qsin[bzq, spin, i, nn] = sigma
                        self.dsigma_qsin[bzq, spin, i, nn] = dsigma

        self.world.sum(self.sigma_qsin)
        self.world.sum(self.dsigma_qsin)
        
        self.sigma_sin, self.dsigma_sin = \
          self.qpt_integration.integrate(self.sigma_qsin, self.dsigma_qsin)

    def do_qpt_loop(self, readw=True):
        """Do the loop over q-points in the q-point integration"""

        use_wstc = False
        if not self.wstc is None:
            use_wstc = True

        parameters = {'eta': self.eta * Hartree,
                      'hilbert': True,
                      'timeordered': True,
                      'domega0': self.domega0 * Hartree,
                      'omega2': self.omega2 * Hartree,
                      'omegamax': self.omegamax * Hartree}
        
        chi0 = Chi0(self.calc,
                    nbands=self.nbands,
                    ecut=self.ecut * Hartree,
                    intraband=False,
                    real_space_derivatives=False,
                    #txt=self.filename + '.w.txt',
                    txt=devnull,
                    timer=self.timer,
                    keep_occupied_states=True,
                    nblocks=self.blockcomm.size,
                    no_optical_limit=self.wstc,
                    **parameters)

        if self.truncation == 'wigner-seitz':
            self.wstc = WignerSeitzTruncatedCoulomb(
                self.calc.wfs.gd.cell_cv,
                self.calc.wfs.kd.N_c,
                chi0.fd)

        self.omega_w = chi0.omega_w
        self.omegamax = chi0.omegamax

        self.htp = HilbertTransform(self.omega_w, self.eta, gw=True)
        self.htm = HilbertTransform(self.omega_w, -self.eta, gw=True)
        
        # Find maximum size of chi-0 matrices:
        gd = self.calc.wfs.gd
        nGmax = max(count_reciprocal_vectors(self.ecut, gd, q_c)
                    for q_c in self.qd.ibzk_kc)
        nw = len(self.omega_w)
        
        size = self.blockcomm.size
        mynGmax = (nGmax + size - 1) // size
        mynw = (nw + size - 1) // size
        
        # Allocate memory in the beginning and use for all q:
        A1_x = np.empty(nw * mynGmax * nGmax, complex)
        A2_x = np.empty(max(mynw * nGmax, nw * mynGmax) * nGmax, complex)
        

        # Find IBZ q-points included in the integration
        qd = self.qd
        bz1q_qc = to1bz(self.qpt_integration.qpts_qc, qd.symmetry.cell_cv)
        ibzqs = []
        for bzq_c in bz1q_qc:
            ibzq, iop, timerev, diff_c = qd.find_ibzkpt(qd.symmetry.op_scc,
                                                        qd.ibzk_kc,
                                                        bzq_c)
            if not ibzq in ibzqs:
                ibzqs.append(ibzq)

        # Loop over IBZ q-points
        for ibzq in ibzqs:
            q_c = qd.ibzk_kc[ibzq]
            print('q0_c=%s' % q_c, file=self.fd)

            # We now calculate the dielectric function
            #pd, df_wGG = self.dielectric_function.get_dielectric_matrix(q_c=q_c,
            #                symmetric=True)

            if (self.savew or readw) and self.filename:
                wfilename = self.filename + '.w.q%d.pckl' % ibzq
                fd = opencew(wfilename)
            if readw and fd is None:
                # Read screened potential from file
                print('Reading W from file: %s' % wfilename, file=self.fd)
                with open(wfilename) as fd:
                    pd, W = pickle.load(fd)
                # Initialize PAW corrections:
                Q_aGii = self.pairDensity.initialize_paw_corrections(pd)
            else:
                # First time calculation
                W, pd, Q_aGii = self.calculate_w(chi0, q_c, A1_x, A2_x)
                if self.savew:
                    pickle.dump((pd, W), fd, pickle.HIGHEST_PROTOCOL)

            nG = pd.ngmax
            mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
            self.Ga = self.blockcomm.rank * mynG
            self.Gb = min(self.Ga + mynG, nG)
            assert mynG * (self.blockcomm.size - 1) < nG
            # Get the PAW corrections to the pair density
            #Q_aGii = self.pairDensity.initialize_paw_corrections(pd)

            # Loop over all k-points in the BZ and find those that are related
            # to the current IBZ k-point by symmetry
            #Q1 = qd.ibz2bz_k[iq]

            Q1 = self.qd.ibz2bz_k[ibzq]
            done = set()
            for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                if Q2 >= 0 and Q2 not in done:
                    yield Q1, Q2, pd, W, Q_aGii
                    done.add(Q2)

    def calculate_w(self, chi0, q_c, A1_x, A2_x):
        """Calculates the screened potential for a specified q-point."""
        pd, chi0_wGG = chi0.calculate(q_c, A_x=A1_x)[:2]
        Q_aGii = chi0.Q_aGii
        self.Ga = chi0.Ga
        self.Gb = chi0.Gb
        
        if self.blockcomm.size > 1:
            A1_x = chi0_wGG.ravel()
            chi0_wGG = chi0.redistribute(chi0_wGG, A2_x)
        
        if self.wstc:
            iG_G = (self.wstc.get_potential(pd) / (4 * pi))**0.5
            if np.allclose(q_c, 0):
                #chi0_wGG[:, 0] = 0.0
                #chi0_wGG[:, :, 0] = 0.0
                G0inv = 0.0
                G20inv = 0.0
            else:
                G0inv = None
                G20inv = None
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
                G0inv = None
                G20inv = None

        delta_GG = np.eye(len(iG_G))

        #print('iG_G=%s' % iG_G[0:5])
        
        self.timer.start('Dyson eq.')
        # Calculate W and store it in chi0_wGG ndarray:
        for w, chi0_GG in enumerate(chi0_wGG):
            e_GG = (delta_GG -
                    4 * pi * chi0_GG * iG_G * iG_G[:, np.newaxis])
            
            W_GG = chi0_GG
            W_GG[:] = 4 * pi * (np.linalg.inv(e_GG) -
                                delta_GG) * iG_G * iG_G[:, np.newaxis]
            if np.allclose(q_c, 0):
                W_GG[0, 0] *= G20inv
                W_GG[1:, 0] *= G0inv
                W_GG[0, 1:] *= G0inv
        
        if self.blockcomm.size > 1:
            Wm_wGG = chi0.redistribute(chi0_wGG, A1_x)
        else:
            Wm_wGG = chi0_wGG
        
        Wp_wGG = A2_x[:Wm_wGG.size].reshape(Wm_wGG.shape)
        Wp_wGG[:] = Wm_wGG
        
        with self.timer('Hilbert transform'):
            self.htp(Wp_wGG)
            self.htm(Wm_wGG)
        self.timer.stop('Dyson eq.')
        
        return [Wp_wGG, Wm_wGG], pd, Q_aGii
    
    def calculate_sigma(self, n_mG, deps_m, f_m, W_swGG):
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)
        
        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2
        
        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        m_inb = np.where(w_m < len(self.omega_w))[0]
        o1_m = np.empty(len(o_m))
        o2_m = np.empty(len(o_m))
        o1_m[m_inb] = self.omega_w[w_m[m_inb]]
        o2_m[m_inb] = self.omega_w[w_m[m_inb] + 1]
        
        sigma = 0.0
        dsigma = 0.0
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):
            if w >= len(self.omega_w):
                continue
            C1_GG = W_swGG[s][w]
            C2_GG = W_swGG[s][w + 1]
            p = 1.0 * sgn
            myn_G = n_G[self.Ga:self.Gb]
            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)
            
        return sigma, dsigma

class FrequencyIntegration:

    def __init__(self):
        self.initialize()

    def initialize(self, df):
        self.df = df

    def calculate_sigma(self, deps_m, f_m, df_wGG, n_mG):
        pass

class RealFreqIntegration(FrequencyIntegration):

    def __init__(self, domega0, omega2):
        self.domega0 = domega0
        self.omega2 = omega2

    def initialize(self, df):

        self.omega_w = df.omega_w
        self.initialize(self, df)

    def calculate_sigma(self, deps_m, f_m, df_wGG, n_mG):
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)
        
        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2
        
        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        o1_m = self.omega_w[w_m]
        o2_m = self.omega_w[w_m + 1]
        
        sigma = 0.0
        dsigma = 0.0
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):
            C1_GG = C_swGG[s][w]
            C2_GG = C_swGG[s][w + 1]
            p = x * sgn
            myn_G = n_G[self.Ga:self.Gb]
            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)
            
        return sigma, dsigma

class QPointIntegration:

    def __init__(self, qpts_qc, cell_cv):
        self.qpts_qc = qpts_qc
        self.cell_cv = cell_cv
        self.sigma_sin = None
        self.dsigma_sin = None

    def add_terms(self, bzqs, sigma_sin, dsigma_sin):
        pass

    def integrate(self, ):
        pass
    

class QuadQPointIntegration(QPointIntegration):
    def __init__(self, qpts_qc, cell_cv, weight_q):
        self.weight_q = weight_q
        
        QPointIntegration.__init__(self, qpts_qc, cell_cv)

    def integrate(self, sigma_qsin, dsigma_qsin):
        vol = abs(np.linalg.det(self.cell_cv))
        dq = 1.0 / (2 * pi * vol)
        sigma_sin = dq * np.dot(self.weight_q,
                                  np.transpose(sigma_qsin, [1, 2, 0, 3]))
        dsigma_sin = dq * np.dot(self.weight_q,
                                   np.transpose(dsigma_qsin, [1, 2, 0, 3]))

        return sigma_sin, dsigma_sin

class TriangleQPointIntegration(QPointIntegration):
    def __init__(self, qpts_qc, cell_cv, simplices):
        self.simplices = simplices
        
        QPointIntegration.__init__(self, qpts_qc, cell_cv)

    def integrate(self, sigma_qsin, dsigma_qsin):
        icell_cv = np.linalg.inv(self.cell_cv).T
        h = icell_cv[2, 2]
        qpts_qv = np.dot(self.qpts_qc, icell_cv)

        sigma_sin = np.zeros(sigma_qsin.shape[1:])
        dsigma_sin = np.zeros(dsigma_qsin.shape[1:])
        for sim in self.simplices:
            q1_v = qpts_qv[sim[0]]
            q2_v = qpts_qv[sim[1]]
            q3_v = qpts_qv[sim[2]]
            J = np.array([[1, 1, 1],
                          [q1_v[0], q2_v[0], q3_v[0]],
                          [q1_v[1], q2_v[1], q3_v[1]]])
            V = 0.5 * h * abs(np.linalg.det(J)) / (2 * pi)
            sigma_sin += V / 3 * np.sum(sigma_qsin[sim], axis=0)
            dsigma_sin += V / 3 * np.sum(dsigma_qsin[sim], axis=0)

        return sigma_sin, dsigma_sin


