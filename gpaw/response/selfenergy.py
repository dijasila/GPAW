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

        self.freqint = RealFreqIntegration(self.calc,
                                           ecut=self.ecut * Hartree,
                                           nbands=self.nbands,
                                           domega0=self.domega0 * Hartree,
                                           omega2=self.omega2 * Hartree,
                                           txt=self.fd,
                                           timer=self.timer)

        self.vc = WignerSeitzTruncatedCoulomb(self.calc.wfs.gd.cell_cv,
                                              self.calc.wfs.kd.N_c,
                                              self.fd)

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
        
        for Q1, Q2, W0_wGG, S0_wvG, L0_wvv, pd0, Q0_aGii in \
          self.do_qpt_loop(readw=readw):
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

                    sigma, dsigma, S_G0, dS_G0, S_0G, dS_0G, S_00, dS_00 = \
                      self.freqint.calculate_integration(
                          n_mG, deps_m, f_m, W0_wGG, S0_wvG, L0_wvv)
                    
                    for bzq in bzqs:
                        self.sigma_qsin[bzq, spin, i, nn] = sigma
                        self.dsigma_qsin[bzq, spin, i, nn] = dsigma

        self.world.sum(self.sigma_qsin)
        self.world.sum(self.dsigma_qsin)
        
        self.sigma_sin, self.dsigma_sin = \
          self.qpt_integration.integrate(self.sigma_qsin, self.dsigma_qsin)

    def do_qpt_loop(self, readw=True):
        """Do the loop over q-points in the q-point integration"""

        # Find maximum size of chi-0 matrices:
        gd = self.calc.wfs.gd
        nGmax = max(count_reciprocal_vectors(self.ecut, gd, q_c)
                    for q_c in self.qd.ibzk_kc)
        nw = self.freqint.wsize
        
        size = self.blockcomm.size
        mynGmax = (nGmax + size - 1) // size
        #mynw = (nw + size - 1) // size
        
        # Allocate memory in the beginning and use for all q:
        A_x = np.empty(nw * mynGmax * nGmax, complex)

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

            W_wGG, S_wvG, L_wvv, vc_G0, vc_00, pd, Q_aGii = \
              self.calculate_w(q_c, A_x)

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
                    yield Q1, Q2, W_wGG, S_wvG, L_wvv, pd, Q_aGii
                    done.add(Q2)


    def calculate_w(self, q_c, A_x):

        # Get Coulomb potential
        #vc_G = self.vc.get_potential(pd)
        vc_G0 = None
        vc_00 = None
        #vc_G, vc_G0, vc_00 = self.vc.get_potential(pd)
        
        W_wGG, S_wvG, L_wvv, pd, Q_aGii = \
          self.freqint.calculate_w(q_c, self.vc, readw=False, A_x=A_x)
        
        if Q_aGii is None:
            Q_aGii = self.pairDensity.initialize_paw_corrections(pd)

        return W_wGG, S_wvG, L_wvv, vc_G0, vc_00, pd, Q_aGii
        #return [Wp_wGG, Wm_wGG], vc_G0, vc_00, pd, Q_aGii
    
    

class FrequencyIntegration:

    def __init__(self):
        self.initialize()

    def initialize(self, df):
        self.df = df

    def calculate_integration(self, n_mG, deps_m, f_m, W_swGG, S_wvG, L_wvv):
        pass

class RealFreqIntegration(FrequencyIntegration):

    def __init__(self, calc, filename=None, ecut=150., nbands=None,
                 domega0=0.025, omega2=10.,
                 timer=None, txt=sys.stdout, nblocks=1):
        self.timer = timer
        self.fd = txt
        
        self.calc = calc
        self.filename = filename

        self.ecut = ecut / Hartree
        self.nbands = nbands
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.eta = 0.1 / Hartree

        self.anistropy_correction = False
        self.savew = False
        self.wstc = WignerSeitzTruncatedCoulomb(self.calc.wfs.gd.cell_cv,
                                                self.calc.wfs.kd.N_c,
                                                self.fd)

        self.nblocks = nblocks
        
        parameters = {'eta': self.eta * Hartree,
                      'hilbert': True,
                      'timeordered': True,
                      'domega0': self.domega0 * Hartree,
                      'omega2': self.omega2 * Hartree}
        
        self.chi0 = Chi0(self.calc,
                         nbands=self.nbands,
                         ecut=self.ecut * Hartree,
                         intraband=False,
                         real_space_derivatives=False,
                         #txt=self.filename + '.w.txt',
                         txt=devnull,
                         timer=self.timer,
                         keep_occupied_states=True,
                         nblocks=self.nblocks,
                         no_optical_limit=True,
                         **parameters)

        self.omega_w = self.chi0.omega_w
        self.wsize = 2 * len(self.omega_w)

        self.htp = HilbertTransform(self.omega_w, self.eta, gw=True)
        self.htm = HilbertTransform(self.omega_w, -self.eta, gw=True)

    def calculate_w(self, q_c, vc, readw=False, A_x=None):
        """Calculates the inverse dielectric matrix for a specified q-point."""
        # Divide memory into two slots so we can redistribute via moving values
        # from one slot to the other
        nx = len(A_x)
        A1_x = A_x[:nx // 2]
        A2_x = A_x[nx // 2:]
        
        Q_aGii = None
        if readw and self.filename:
            chi_filename = (self.filename + '.chi0.%+d%+d%+d.pckl' %
                            tuple((q_c * self.calc.wfs.kd.N_c).round()))
            fd = opencew(chi_filename)
        if readw and fd is None:
            # Read chi0 from file and save it in second half of A1_x
            print('Reading chi0 from file: %s' % chi_filename, file=self.fd)
            pd, chi0_wGG, chi0_wxvGm, chi0_wvv = self.read_chi(chi_filename,
                                                               A1_x)
        else:
            # Calculate chi0 and save it in second half of A1_x
            pd, chi0_wGG, chi0_wxvG, chi0_wvv = \
              self.chi0.calculate(q_c, A_x=A1_x)
            Q_aGii = self.chi0.Q_aGii
            if self.savew:
                chi_filename = (self.filename + '.chi0.%+d%+d%+d.pckl' %
                                tuple((q_c * self.calc.wfs.kd.N_c).round()))
                self.write_chi(chi_filename, pd, chi0_wGG, chi0_wxvG, chi0_wvv)
        
        world = self.chi0.world
        blockcomm = self.chi0.blockcomm
        nblocks = blockcomm.size
        
        nw = len(self.chi0.omega_w)
        nG = pd.ngmax
        mynw = (nw + nblocks - 1) // nblocks
        mynG = (nG + nblocks - 1) // nblocks

        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)
        Ga = min(world.rank * mynG, nG)
        Gb = min(Ga + mynG, nG)
        sizew = (wb - wa) * nG**2
        sizeG = nw * mynG * nG
        
        if self.nblocks > 1:
            # Redistribute chi0 over frequencies and save the new array in A2_x
            chi0_wGG = self.chi0.redistribute(chi0_wGG, A2_x)
            # chi0_wGG now has shape (wb - wa, nG, nG)

        # Get the Coulomb kernel for the dielectric function calculation
        if self.wstc:
            iG_G = (self.wstc.get_potential(pd) / (4 * pi))**0.5
        else:
            if np.allclose(q_c, 0):
                G_G = pd.G2_qG[0]**0.5
                G_G[0] = 1
                iG_G = 1 / G_G
            else:
                iG_G = pd.G2_qG[0]**-0.5

        # Get the Coulomb kernel for the screened potential calculation
        vc_G = (vc.get_potential(pd) / (4 * pi))**0.5
        vc_G0 = np.zeros(len(vc_G))
        vc_00 = 0

        # These are entities related to the q->0 value
        S_wvG = None
        L_wvv = None
        if np.allclose(q_c, 0) and self.anistropy_correction:
            S_wvG = np.zeros((nw, 3, nG))
            L_wvv = np.zeros((nw, 3, 3))

        delta_GG = np.eye(len(iG_G))
        for w, chi0_GG in enumerate(chi0_wGG):
            # First we calculate the inverse dielectric function
            idf_GG = chi0_GG
            idf_GG[:] = np.linalg.inv(delta_GG -
                4 * pi * chi0_GG * iG_G * iG_G[:, np.newaxis])

            # Calculate the q->0 entities
            if (np.allclose(q_c, 0) and chi0_wxvG is not None and
                self.anistropy_correction):
                U_vG = chi0_wxvG[w, 0]
                S_vG = np.dot(U_vG[:, 1:], idf_GG[:, 1:].T)
                L_vv = chi0_wvv[w] - np.dot(U_vG.conj()[:, 1:], S_vG[:, 1:].T)
                S_wvG[w] = S_vG
                L_wvv[w] = L_vv

            # Calculate the screened potential
            W_GG = idf_GG
            W_GG -= delta_GG
            if np.allclose(q_c, 0):
                W_GG[1:, 1:] = 4 * pi * W_GG[1:, 1:] * vc_G[1:] * \
                  vc_G[1:, np.newaxis]
                #W_GG[1:, 0] = 4 * pi * (idf_GG[0, 1:] -
                #                        delta_GG[0, 1:]) * vc_G0[1:]
                #W_GG[1:, 0] = 4 * pi * (idf_GG[0, 1:] -
                #                        delta_GG[0, 1:]) * vc_G0[1:]
                #W_GG[0, 0] = 4 * pi * (idf_GG[0, 0] - 1.) * vc_00
            else:
                W_GG[:] = 4 * pi * W_GG * vc_G * vc_G[:, np.newaxis]
        
        W_wGG = chi0_wGG # rename

        # Since we are doing a Hilbert transform we get two editions of the
        # W_wGG matrix corresponding to +/- contributions. We are thus doubling
        # the number of frequencies and storing the +/- part in each of the two
        # halves.
        newshape = (2*nw, mynG, nG)
        size = np.prod(newshape)

        Wpm_wGG = A_x[:size].reshape(newshape)

        if self.nblocks > 1:
            # Now redistribute back on G rows and save in second half of A1_x
            # which is not used any more (was only used when reading/
            # calculating chi0 in the beginning
            Wpm_wGG[:nw] = self.chi0.redistribute(W_wGG, A1_x)
        else:
            Wpm_wGG[:nw] = W_wGG

        Wpm_wGG[nw:] = W_wGG

        Spm_wvG = None
        Lpm_wvv = None
        if np.allclose(q_c, 0) and self.anistropy_correction:
            Spm_wvG = np.empty((2*nw, 3, nG))
            Spm_wvG[:nw] = S_wvG
            Spm_wvG[nw:] = S_wvG
            Lpm_wvv = np.empty((2*nw, 3, 3))
            Lpm_wvv[:nw] = L_wvv
            Lpm_wvv[nw:] = L_wvv
        
        with self.timer('Hilbert transform'):
            self.htp(Wpm_wGG[:nw])
            self.htm(Wpm_wGG[nw:])
            if np.allclose(q_c, 0) and self.anistropy_correction:
                self.htp(Spm_wvG[:nw])
                self.htm(Spm_wvG[nw:])
                self.htp(Lpm_wvv[:nw])
                self.htm(Lpm_wvv[nw:])
        
        return Wpm_wGG, Spm_wvG, Lpm_wvv, pd, Q_aGii

    def calculate_integration(self, n_mG, deps_m, f_m, W_wGG, S_wvG, L_wvv,
                              gamma=False):
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)
        
        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2

        world = self.chi0.world
        comm = self.chi0.blockcomm
        nw = len(self.omega_w)
        nG = n_mG.shape[1]
        mynG = (nG + comm.size - 1) // comm.size
        Ga = min(world.rank * mynG, nG)
        Gb = min(Ga + mynG, nG)
        
        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        m_inb = np.where(w_m < len(self.omega_w) - 1)[0]
        o1_m = np.empty(len(o_m))
        o2_m = np.empty(len(o_m))
        o1_m[m_inb] = self.omega_w[w_m[m_inb]]
        o2_m[m_inb] = self.omega_w[w_m[m_inb] + 1]

        # G1 != 0, G2 != 0 part
        sigma = 0.0
        dsigma = 0.0

        if gamma:
            # G1 != 0, G2 == 0 part for q=Gamma
            S_G0 = np.zeros(len(nG))
            dS_G0 = np.zeros(len(nG))
            # G1 == 0, G2 != 0 part for q=Gamma
            S_0G = np.zeros(len(nG))
            dS_0G = np.zeros(len(nG))
            # G1 == 0, G2 == 0 part for q=Gamma
            S_00 = 0.0
            dS_00 = 0.0
        else:
            S_G0 = None
            dS_G0 = None
            S_0G = None
            dS_0G = None
            S_00 = None
            dS_00 = None
        
        
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):
            if w >= len(self.omega_w) - 1:
                continue
            C1_GG = W_wGG[s*nw + w]
            C2_GG = W_wGG[s*nw + w + 1]
            p = 1.0 * sgn
            myn_G = n_G[Ga:Gb]
            if not gamma:
                sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
                sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
                sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
                dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)
            else:
                G0 = 0
                if Ga == 0:
                    G0 = 1

                sigma1 = p * np.dot(np.dot(myn_G[G0:], C1_GG[G0:, 1:]),
                                    n_G[1:].conj())
                sigma2 = p * np.dot(np.dot(myn_G[G0:], C2_GG[G0:, 1:]),
                                    n_G[1:].conj())
                sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
                dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

                S1_G0 = p * mynG[G0:] * C1_GG[G0:, 0] * n_G[0].conj()
                S2_G0 = p * mynG[G0:] * C2_GG[G0:, 0] * n_G[0].conj()
                S_G0 += ((o - o1) * S2_G0 + (o2 - o) * S1_G0) / (o2 - o1)
                dS_G0 += sgn * (S2_G0 - S1_G0) / (o2 - o1)

                if Ga == 0:
                    S1_0G = p * mynG[0] * C1_GG[0, 1:] * n_G[1:].conj()
                    S2_0G = p * mynG[0] * C2_GG[0, 1:] * n_G[1:].conj()
                    S_0G += ((o - o1) * S2_0G + (o2 - o) * S1_0G) / (o2 - o1)
                    dS_0G += sgn * (S2_0G - S1_0G) / (o2 - o1)

                    S1_00 = p * mynG[0] * C1_GG[0, 0] * n_G[0].conj()
                    S2_00 = p * mynG[0] * C2_GG[0, 0] * n_G[0].conj()
                    S_00 += ((o - o1) * S2_00 + (o2 - o) * S1_00) / (o2 - o1)
                    dS_00 += sgn * (S2_00 - S1_00) / (o2 - o1)
            
        return sigma, dsigma, S_G0, dS_G0, S_0G, dS_0G, S_00, dS_00

    def write_chi(self, name, pd, chi0_wGG):
        nw = len(self.omega_w)
        nG = pd.ngmax
        mynG = chi0_wGG.shape[2]
        world = self.world

        if world.rank == 0:
            fd = open(name, 'wb')
            pickle.dump((self.omega_w, pd), fd, pickle.HIGHEST_PROTOCOL)
            for iG in range(mynG):
                pickle.dump((iG, chi_wGG[:, :, iG]), fd,
                            pickle.HIGHEST_PROTOCOL)

            tmp_wGG = np.empty((nw, nG, mynG), complex)
            Ga = mynG
            for rank in range(1, world.size):
                Gb = min(Ga + mynG, nG)
                world.recieve(tmp_wGG[:Gb - Ga], rank)
                for iG in range(Gb - Gb):
                    globG = rank * mynG + iG
                    pickle.dump((globG, tmp_wGG[:, :, iG]), fd,
                                pickle.HIGHEST_PROTOCOL)
                Ga = Gb
            fd.close()
        else:
            world.send(chi0_wGG, 0)

    def read_chi(self, name, A1_x):
        """Read chi0_wGG from a file."""
        world = self.world
        fd = open(name)
        omega_w, pd = pickle.load(fd)

        nw = len(omega_w)
        nG = pd.ngmax

        mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
        assert mynG * (self.blockcomm.size - 1) < nG

        myGa = self.blockcomm.rank * mynG
        myGb = min(self.Ga + mynG, nG)

        if A1_x is not None:
            nx = nw * (myGa - myGb) * nG
            chi0_wGG = A_x[:nx].reshape((nw, myGb - myGa, nG))
            chi0_wGG[:] = 0.0
        else:
            chi0_wGG = np.zeros((nw, myGb - myGa, nG), complex)

        if world.rank == 0:
            for iG in range(mynG):
                row, chi0_wGG[:, iG, :] = pickle.load(fd)
                assert n == iG, 'Row order not OK'
            tmp_wGG = np.empty((nw, mynG, nG), complex)
            Ga = mynG
            for rank in range(1, world.size):
                Gb = min(Ga + mynG, nG)
                for iG in range(Gb - Ga):
                    row, tmp_wGG[:, iG, :] = pickle.load(fd)
                    assert row == Ga + iG, 'Row order not OK'
                world.send(tmp_wGG[:, 0:Gb, :], rank)
                Ga = Gb
            else:
                world.recieve(chi0_wGG, 0)


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


