# This makes sure that division works as in Python3
from __future__ import division, print_function

import sys
import functools
from math import pi
import numpy as np
import pickle
from ase.dft.kpoints import monkhorst_pack
from gpaw.kpt_descriptor import to1bz, KPointDescriptor
from gpaw.response.pair import PairDensity, PWSymmetryAnalyzer
from gpaw.response.chi0 import Chi0, HilbertTransform
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.response.df import DielectricFunction
from gpaw.wavefunctions.pw import PWDescriptor, count_reciprocal_vectors
import gpaw.mpi as mpi
from ase.utils import devnull, opencew
from ase.utils.timing import timer, Timer
from ase.units import Hartree


class SelfEnergy:

    def __init__(self):
        pass

class GWSelfEnergy(SelfEnergy):

    def __init__(self, calc, kpts=None, bandrange=None,
                 filename=None, txt=sys.stdout, savechi0=False, scratch='./',
                 nbands=None, ecut=150., nblocks=1, hilbert=True, eta=0.1,
                 domega0=0.025, omega2=10., omegamax=None,
                 qptint=None, truncation='3D',
                 world=mpi.world, timer=None):

        # Create output buffer
        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w', 1)
        self.fd = txt

        SelfEnergy.__init__(self)

        if isinstance(ecut, (int, float)):
            pct = 0.8
            necuts = 5
            self.ecut = ecut / Hartree
            self.ecut_i = self.ecut * (1 + (1. / pct - 1) * np.arange(necuts) /
                                       (necuts - 1))**(-2 / 3)
        else:
            self.ecut = max(ecut) / Hartree
            self.ecut_i = np.sort(ecut)[::-1] / Hartree
        
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.omegamax = omegamax / Hartree if not omegamax is None else None
        self.eta = eta / Hartree
        
        self.calc = calc

        if kpts is None:
            kpts = range(len(calc.get_ibz_k_points()))
        self.kpts = kpts

        vol = abs(np.linalg.det(self.calc.wfs.gd.cell_cv))
        self.vol = vol
        if nbands is None:
            nbands = min(calc.get_number_of_bands(),
                         int(vol * self.ecut**1.5 * 2**0.5 / 3 / pi**2))
        self.nbands = nbands

        if bandrange is None:
            bandrange = (0, calc.get_number_of_bands())
        
        self.bandrange = bandrange

        self.filename = filename
        self.savechi0 = savechi0

        self.truncation = truncation

        kd = calc.wfs.kd

        self.timer = timer or Timer()
        
        self.world = world
        self.pairDensity = PairDensity(calc, ecut=self.ecut, nblocks=nblocks,
                                       world=mpi.world, txt=self.fd,
                                       timer=self.timer)
        self.blockcomm = self.pairDensity.blockcomm
        self.nblocks = nblocks
        
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
            weights_q = 1. / len(bzq_qc) * np.ones(len(bzq_qc))
            self.qpt_integration = QuadQPointIntegration(self.qd,
                                                         cell_cv,
                                                         bzq_qc,
                                                         weights_q)
        else:
            self.qpt_integration = qptint

        self.freqint = RealFreqIntegration(self.calc,
                                           filename=self.filename,
                                           savechi0=self.savechi0,
                                           ecut=self.ecut * Hartree,
                                           nbands=self.nbands,
                                           domega0=self.domega0 * Hartree,
                                           omega2=self.omega2 * Hartree,
                                           nblocks=self.nblocks,
                                           txt=self.fd,
                                           timer=self.timer)
        self.freqint.initialize(self)

        if truncation is None or truncation == '3D':
            self.vc = CoulombKernel3D()
        elif truncation == '2D':
            self.vc = CoulombKernel2D(cell_cv=calc.wfs.gd.cell_cv)
        elif truncation == 'wigner-seitz':
            self.vc = WignerSeitzTruncatedCoulomb(self.calc.wfs.gd.cell_cv,
                                                  self.calc.wfs.kd.N_c,
                                                  self.fd)
        
        self.qpt_integration.set_potential(self.vc)

        self.progressEvent = Event()

    def addEventHandler(self, event, handler):
        if event == 'progress' and handler not in self.progressEvent:
            self.progressEvent.append(handler)

    def calculate(self, readw=True):
        p = functools.partial(print, file=self.fd)
        p('Calculating the correlation self-energy')
        if len(self.ecut_i) > 1:
            p('    Using PW cut-offs: ' +
              ', '.join(['{0:.0f} eV'.format(ecut * Hartree)
                         for ecut in self.ecut_i]))
            p('    Extrapolating to infinite cut-off')
        else:
            p('    Using PW cut-off:   {0:.0f} eV'.format(self.ecut_i[0] * Hartree))

        nbzq = len(self.qpt_integration.qpts_qc)
        self.sigma_iskn = np.zeros((len(self.ecut_i), ) + self.shape)
        self.dsigma_iskn = np.zeros((len(self.ecut_i), ) + self.shape)

        self.qpt_integration.reset(self.shape)

        self.progressEvent(0.0)

        for i, ecut in enumerate(self.ecut_i):
            sigma_skn, dsigma_skn = self._calculate(ecut, readw)
            self.sigma_iskn[i, :] = sigma_skn
            self.dsigma_iskn[i, :] = dsigma_skn
        
        self.sigerr_skn = np.zeros(self.shape)
        if len(self.ecut_i) > 1:
            self.sigma_skn = np.zeros(self.shape)
            self.dsigma_skn = np.zeros(self.shape)
            invN_i = self.ecut_i**(-3. / 2)
            for s in range(self.shape[0]):
                for k in range(self.shape[1]):
                    for n in range(self.shape[2]):
                        psig = np.polyfit(invN_i,
                                          self.sigma_iskn[:, s, k, n], 1)
                        self.sigma_skn[s, k, n] = psig[1]
                        sigslopes = (np.diff(self.sigma_iskn[:, s, k, n]) /
                                     np.diff(invN_i))
                        imin = np.argmin(sigslopes)
                        imax = np.argmax(sigslopes)
                        sigmax = (self.sigma_iskn[imin, s, k, n] -
                                  sigslopes[imin] * invN_i[imin])
                        sigmin = (self.sigma_iskn[imax, s, k, n] -
                                  sigslopes[imax] * invN_i[imax])
                        assert (psig[1] < sigmax and psig[1] > sigmin)
                        sigerr = sigmax - sigmin
                        self.sigerr_skn[s, k, n] = sigerr
                        
                        pdsig = np.polyfit(invN_i,
                                           self.dsigma_iskn[:, s, k, n], 1)
                        self.dsigma_skn[s, k, n] = pdsig[1]
        else:
            self.sigma_skn = self.sigma_iskn[0]
            self.dsigma_skn = self.dsigma_iskn[0]

    def _calculate(self, ecut, readw=True):

        # My part of the states we want to calculate QP-energies for:
        mykpts = [self.pairDensity.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]

        kd = self.calc.wfs.kd
        
        kplusqdone_u = [set() for kpt in mykpts]
        
        prefactor = 1 / (2 * pi)**4

        self.qpt_integration.reset(self.shape)
        
        for Q1, Q2, W0_wGG, pd0, Q0_aGii in \
          self.do_qpt_loop(ecut, readw=readw):
            ibzq = self.qd.bz2ibz_k[Q1]
            q_c = self.qd.ibzk_kc[ibzq]

            world = self.world
            bsize = self.nblocks
            nG = W0_wGG.shape[2]
            mynG = (nG + bsize - 1) // bsize
            Ga = min(world.rank * mynG, nG)
            Gb = min(Ga + mynG, nG)
            
            s = self.qd.sym_k[Q2]
            U_cc = self.qd.symmetry.op_scc[s]
            timerev = self.qd.time_reversal_k[Q2]
            sign = 1 - 2 * timerev

            Q_c = self.qd.bzk_kc[Q2]
            diff_c = sign * np.dot(U_cc, q_c) - Q_c
            assert np.allclose(diff_c.round(), diff_c), \
                ("Difference should only be equal to a reciprocal "
                 "lattice vector")
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
                kpt2 = self.pairDensity.get_k_point(spin, K2, 0,
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

                    sigma, dsigma = self.freqint.calculate_integration(n_mG,
                                        deps_m, f_m, W0_wGG)

                    for bzq in bzqs:
                        self.qpt_integration.add_term(bzq, spin, i, nn,
                                                      prefactor * sigma,
                                                      prefactor * dsigma)
                    
                    """
                    for bzq in bzqs:
                        self.sigma_qsin[bzq, spin, i, nn] = prefactor * sigma
                        self.dsigma_qsin[bzq, spin, i, nn] = prefactor * dsigma
                    """

        #self.world.sum(self.sigma_qsin)
        #self.world.sum(self.dsigma_qsin)
        
        sigma_skn, dsigma_skn = self.qpt_integration.integrate() # (self.sigma_qsin, self.dsigma_qsin)
        self.world.sum(sigma_skn)
        self.world.sum(dsigma_skn)
        return sigma_skn, dsigma_skn

    def do_qpt_loop(self, ecut, readw=True):
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
        for nq, ibzq in enumerate(ibzqs):
            q_c = qd.ibzk_kc[ibzq]

            W_wGG, pd, Q_aGii = \
              self.calculate_idf(q_c, ecut, readw=readw, A_x=A_x)

            nG = pd.ngmax
            mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
            self.Ga = self.blockcomm.rank * mynG
            self.Gb = min(self.Ga + mynG, nG)
            assert mynG * (self.blockcomm.size - 1) < nG

            #W_wGG = self.qpt_integration.calculate_w(pd, idf_wGG,
            #                                         S_wvG, L_wvv,
            #                                         (self.Ga, self.Gb))

            #print('Wp_wGG:')
            #print(W_wGG[0, 0:3, 0:3])
            # Get the PAW corrections to the pair density
            #Q_aGii = self.pairDensity.initialize_paw_corrections(pd)

            # Loop over all k-points in the BZ and find those that are related
            # to the current IBZ k-point by symmetry
            #Q1 = qd.ibz2bz_k[iq]

            Q1 = self.qd.ibz2bz_k[ibzq]
            done = set()
            for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                if Q2 >= 0 and Q2 not in done:
                    yield Q1, Q2, W_wGG, pd, Q_aGii
                    done.add(Q2)
            
            self.progressEvent(1.0 * (nq + 1) / len(ibzqs))

    @timer('Screened potential')
    def calculate_idf(self, q_c, ecut, readw=True, A_x=None):
        
        W_wGG, pd, Q_aGii = \
          self.freqint.calculate_idf(q_c, self.vc, ecut, readw=readw, A_x=A_x)
        
        if Q_aGii is None:
            Q_aGii = self.pairDensity.initialize_paw_corrections(pd)

        return W_wGG, pd, Q_aGii

    def calculate_w(self, pd, idf_wGG, S_wvG, L_wvv):

        return self.qpt_integration.calculate_w(pd, idf_wGG, S_wvG, L_wvv)

class FrequencyIntegration:

    def __init__(self, txt=sys.stdout):
        self.fd = txt

    def initialize(self, selfenergy):
        self.selfenergy = selfenergy

    def calculate_integration(self, n_mG, deps_m, f_m, W_swGG, S_wvG, L_wvv):
        pass

class RealFreqIntegration(FrequencyIntegration):

    def __init__(self, calc, filename=None, savechi0=False,
                 ecut=150., nbands=None,
                 domega0=0.025, omega2=10.,
                 timer=None, txt=sys.stdout, nblocks=1):
        FrequencyIntegration.__init__(self, txt)
        
        self.timer = timer
        
        self.calc = calc
        self.filename = filename

        self.ecut = ecut / Hartree
        self.nbands = nbands
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.eta = 0.1 / Hartree

        self.anistropy_correction = False
        self.savechi0 = savechi0
        #self.wstc = WignerSeitzTruncatedCoulomb(self.calc.wfs.gd.cell_cv,
        #                                        self.calc.wfs.kd.N_c,
        #                                        self.fd)

        self.nblocks = nblocks
        
        parameters = {'eta': self.eta * Hartree,
                      'hilbert': True,
                      'timeordered': True,
                      'domega0': self.domega0 * Hartree,
                      'omega2': self.omega2 * Hartree}
        
        """
        self.chi0 = Chi0(self.calc,
                         nbands=self.nbands,
                         ecut=self.ecut * Hartree,
                         intraband=True,
                         real_space_derivatives=False,
                         txt=self.filename + '.chi0.txt',
                         timer=self.timer,
                         keep_occupied_states=True,
                         nblocks=self.nblocks,
                         no_optical_limit=False,
                         **parameters)
        """
        if self.savechi0:
            filename = self.filename + '.chi0'
        else:
            filename = None
        self.df = DielectricFunction(self.calc,
                                     nbands=self.nbands,
                                     ecut=self.ecut * Hartree,
                                     intraband=True,
                                     nblocks=self.nblocks,
                                     name=filename,
                                     txt=self.filename + '.chi0.txt',
                                     **parameters)
                                     

        self.omega_w = self.df.chi0.omega_w
        self.wsize = 2 * len(self.omega_w)

        self.htp = HilbertTransform(self.omega_w, self.eta, gw=True)
        self.htm = HilbertTransform(self.omega_w, -self.eta, gw=True)

    @timer('Inverse dielectric function')
    def calculate_idf(self, q_c, vc, ecut, readw=False, A_x=None):
        #print('qp: calculate_idf, q_c=%s' % q_c.round(3))
        """Calculates the inverse dielectric matrix for a specified q-point."""
        # Divide memory into two slots so we can redistribute via moving values
        # from one slot to the other
        nx = len(A_x)
        if A_x is not None:
            A1_x = A_x[:nx // 2]
            A2_x = A_x[nx // 2:]
        else:
            A1_x = None
            A2_x = None
        
        Q_aGii = None
        """
        if readw and self.filename:
            chi_filename = (self.filename + '.chi0.%+d%+d%+d.pckl' %
                            tuple((q_c * self.calc.wfs.kd.N_c).round()))
            fd = opencew(chi_filename)
        if readw and fd is None:
            # Read chi0 from file and save it in second half of A_x
            print('Reading chi0 from file: %s' % chi_filename, file=self.fd)
            pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.read_chi(chi_filename,
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

        if self.nblocks > 1:
            # Redistribute chi0 over frequencies and save the new array in A2_x
            chi0_wGG = self.chi0.redistribute(chi0_wGG, A2_x)
            # chi0_wGG now has shape (wb - wa, nG, nG)
        """
        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.df.calculate_chi0(q_c,
                                                                   A1_x=A1_x,
                                                                   A2_x=A2_x)

        if pd.ecut > ecut:
            bigpd = pd
            pd = PWDescriptor(ecut, bigpd.gd, dtype=bigpd.dtype,
                              kd=bigpd.kd)
            G2G = pd.map(bigpd, q=0)
            chi0_wGG = chi0_wGG.take(G2G, axis=1).take(G2G, axis=2)

            if chi0_wxvG is not None:
                chi0_wxvG = chi0_wxvG.take(G2G, axis=3)

            if Q_aGii is not None:
                for a, Q_Gii in enumerate(Q_aGii):
                    Q_aGii[a] = Q_Gii.take(G2G, axis=0)

        world = self.df.chi0.world
        blockcomm = self.df.chi0.blockcomm
        nblocks = blockcomm.size
        
        nw = len(self.df.chi0.omega_w)
        nG = chi0_wGG.shape[2]
        mynw = (nw + nblocks - 1) // nblocks
        mynG = (nG + nblocks - 1) // nblocks

        if nblocks > 1:
            wa = min(blockcomm.rank * mynw, nw)
            wb = min(wa + mynw, nw)
            Ga = min(blockcomm.rank * mynG, nG)
            Gb = min(Ga + mynG, nG)
            sizew = (wb - wa) * nG**2
            sizeG = nw * (Gb - Ga) * nG
        else:
            wa = 0
            wb = nw
            Ga = 0
            Gb = nG
        #print('nw=%s, nG=%s, mynw=%s, mynG=%s, Ga=%s, Gb=%s' % (nw, nG, mynw,
        #                                                        mynG, Ga, Gb))
        
        # Get the Coulomb kernel for the screened potential calculation
        vc_G = vc.get_potential(pd=pd)**0.5

        # These are entities related to the q->0 value
        S_wvG = None
        L_wvv = None
        if np.allclose(q_c, 0):
            vc_G0, vc_00 = vc.get_gamma_limits(pd)
            S_wvG = np.zeros((wb - wa, 3, nG - 1), complex)
            L_wvv = np.zeros((wb - wa, 3, 3), complex)

        delta_GG = np.eye(len(vc_G))
        for w, chi0_GG in enumerate(chi0_wGG[:wb - wa]):
            # First we calculate the inverse dielectric function
            idf_GG = chi0_GG

            # Calculate the q->0 entities
            if np.allclose(q_c, 0):
                idf_GG[0, :] = 0.0
                idf_GG[:, 0] = 0.0
                idf_GG[0, 0] = 1.0
                idf_GG[1:, 1:] = np.linalg.inv(delta_GG[1:, 1:] -
                                        chi0_GG[1:, 1:] * vc_G[np.newaxis, 1:] *
                                        vc_G[1:, np.newaxis])
                B_GG = idf_GG[1:, 1:]
                U_vG = -vc_G0[np.newaxis, :] * chi0_wxvG[wa + w, 0, :, 1:]
                F_vv = -vc_00 * chi0_wvv[wa + w]
                S_vG = np.dot(U_vG, B_GG.T)
                L_vv = F_vv - np.dot(U_vG.conj(), S_vG.T)
                S_wvG[w] = S_vG
                L_wvv[w] = L_vv
            else:
                idf_GG[:] = np.linalg.inv(delta_GG - chi0_GG * 
                                          vc_G[np.newaxis, :] *
                                          vc_G[:, np.newaxis])
            idf_GG -= delta_GG
        
        idf_wGG = chi0_wGG # rename

        W_wGG = idf_wGG
        W_wGG[:wb - wa] = self.selfenergy.calculate_w(pd, idf_wGG[:wb - wa],
                                                      S_wvG, L_wvv)

        # Since we are doing a Hilbert transform we get two editions of the
        # W_wGG matrix corresponding to +/- contributions. We are thus doubling
        # the number of frequencies and storing the +/- part in each of the two
        # halves.
        newshape = (2*nw, Gb - Ga, nG)
        size = np.prod(newshape)

        nblocks1 = 3 - world.size
        mynG1 = (nG + nblocks1 - 1) // nblocks1
        mynw1 = (nw + nblocks1 - 1) // nblocks1
        
        Wpm_wGG = A_x[:size].reshape(newshape)
        
        if self.nblocks > 1:
            # Now redistribute back on G rows and save in second half of A1_x
            # which is not used any more (was only used when reading/
            # calculating chi0 in the beginning
            Wpm_wGG[:nw] = self.df.chi0.redistribute(W_wGG, A1_x)
        else:
            Wpm_wGG[:nw] = W_wGG
        
        
        Wpm_wGG[nw:] = Wpm_wGG[0:nw]
        """
        mystr = 'rank=%d, q_c=%s\n' % (world.rank, q_c)
        for n in range(nblocks1):
            G1 = n * mynG1
            mystr += 'G1=%d, before hilbert: %s\n' % (Ga + G1, str(Wpm_wGG[0, n * mynG1, 0:3]))
        """
        with self.timer('Hilbert transform'):
            self.htp(Wpm_wGG[:nw])
            self.htm(Wpm_wGG[nw:])
        
        """
        for n in range(nblocks1):
            G1 = n * mynG1
            mystr += 'G1=%d, after hilbert: %s\n' % (Ga + G1, str(Wpm_wGG[0, n * mynG1, 0:3]))
        print(mystr)
        """
        return Wpm_wGG, pd, Q_aGii

    @timer('Frequency integration')
    def calculate_integration(self, n_mG, deps_m, f_m, W_wGG, gamma=False):
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)
        
        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2

        world = self.df.chi0.world
        comm = self.df.chi0.blockcomm
        nw = len(self.omega_w)
        nG = n_mG.shape[1]
        mynG = (nG + comm.size - 1) // comm.size
        Ga = min(comm.rank * mynG, nG)
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
        
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):
            if w >= len(self.omega_w) - 1:
                continue
            
            C1_GG = W_wGG[s*nw + w]
            C2_GG = W_wGG[s*nw + w + 1]
            p = 1.0 * sgn
            myn_G = n_G[Ga:Gb]
            
            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)
            
        return sigma, dsigma

    def write_chi(self, name, pd, chi0_wGG, chi0_wxvG=None, chi0_wvv=None):
        nw = len(self.omega_w)
        nG = pd.ngmax
        mynG = chi0_wGG.shape[1]
        world = self.chi0.world

        if world.rank == 0:
            fd = open(name, 'wb')
            pickle.dump((self.omega_w, pd, chi0_wxvG, chi0_wvv),
                        fd, pickle.HIGHEST_PROTOCOL)
            for iG in range(mynG):
                pickle.dump((iG, chi0_wGG[:, iG, :]), fd,
                            pickle.HIGHEST_PROTOCOL)

            tmp_wGG = np.empty((nw, mynG, nG), complex)
            Ga = mynG
            for rank in range(1, world.size):
                Gb = min(Ga + mynG, nG)
                world.receive(tmp_wGG, rank)
                for iG in range(Gb - Ga):
                    globG = rank * mynG + iG
                    pickle.dump((globG, tmp_wGG[:, iG, :]), fd,
                                pickle.HIGHEST_PROTOCOL)
                Ga = Gb
            fd.close()
        else:
            world.send(chi0_wGG, 0)

    def read_chi(self, name, A1_x=None):
        """Read chi0_wGG from a file."""
        world = self.chi0.world
        fd = open(name)
        omega_w, pd, chi0_wxvG, chi0_wvv = pickle.load(fd)

        nw = len(omega_w)
        nG = pd.ngmax

        mynG = (nG + self.chi0.blockcomm.size - 1) // self.chi0.blockcomm.size
        assert mynG * (self.chi0.blockcomm.size - 1) < nG

        myGa = self.chi0.blockcomm.rank * mynG
        myGb = min(myGa + mynG, nG)
        
        if A1_x is not None:
            nx = nw * mynG * nG
            chi0_wGG = A1_x[:nx].reshape((nw, mynG, nG))
            chi0_wGG[:] = 0.0
        else:
            chi0_wGG = np.zeros((nw, mynG, nG), complex)

        if world.rank == 0:
            for iG in range(mynG):
                row, chi0_wGG[:, iG, :] = pickle.load(fd)
                assert row == iG, 'Row order not OK'
            tmp_wGG = np.empty((nw, mynG, nG), complex)
            Ga = mynG
            for rank in range(1, world.size):
                Gb = min(Ga + mynG, nG)
                for iG in range(Gb - Ga):
                    row, tmp_wGG[:, iG, :] = pickle.load(fd)
                    assert row == Ga + iG, 'Row order not OK'
                print('Sending to rank=%d, Ga=%d, Gb=%d, tmp_wGG.shape=%s' %
                          (rank, Ga, Gb, str(tmp_wGG[:, :, :].shape)))
                world.send(tmp_wGG, rank)
                Ga = Gb
        else:
            world.receive(chi0_wGG, 0)
        
        return pd, chi0_wGG[:, 0:myGb - myGa], chi0_wxvG, chi0_wvv


class QPointIntegration:

    def __init__(self, qd, cell_cv, qpts_qc, txt=sys.stdout):
        self.qd = qd
        self.qpts_qc = qpts_qc
        self.cell_cv = cell_cv
        self.vc = None

        self.fd = txt

    def set_potential(self, vc):
        self.vc = vc

    def add_term(bzq, spin, i, nn, sigma, dsigma):
        pass

    def calculate_w(self, pd, idf_wGG, S_wvG, L_wvv, GaGb=None):
        """This function calculates the screened potential W integrated over
        a region around each q-point."""
        
        if GaGb is None:
            Ga = 0
            Gb = idf_wGG.shape[1]
        else:
            Ga = GaGb[0]
            Gb = GaGb[1]
        
        nG = idf_wGG.shape[2]
        mynG = idf_wGG.shape[1]
        W_wGG = idf_wGG # Rename

        q_c = pd.kd.bzk_kc[0]
        
        vol = abs(np.linalg.det(self.cell_cv))
        rvol = (2 * pi)**3 / vol

        if isinstance(self.vc, WignerSeitzTruncatedCoulomb):
            vc_G = self.vc.get_potential(pd)**0.5
            W_wGG[:] = (vc_G[np.newaxis, Ga:Gb, np.newaxis] * idf_wGG *
                        vc_G[np.newaxis, np.newaxis, :])

        elif isinstance(self.vc, CoulombKernel3D):
            dq = rvol / self.qd.nbzkpts
            
            if np.allclose(q_c, 0):
                # For now, suppose 3D
                qdensity = 0.025
                rcell_cv = 2 * pi * np.linalg.inv(self.cell_cv).T
                N_c = self.qd.N_c
                
                npts_c = np.ceil(np.sum(rcell_cv**2, axis=1)**0.5 /
                                 N_c / qdensity).astype(int)
                print('Calculating Gamma-point integral on a %dx%dx%d grid' %
                      tuple(npts_c), file=self.fd)
                qpts_qc = ((np.indices(npts_c).transpose((1, 2, 3, 0)) \
                            .reshape((-1, 3)) + 0.5) / npts_c - 0.5) / N_c
                qgamma = np.argmin(np.sum(qpts_qc**2, axis=1))
                #qpts_qc[qgamma] += np.array([1e-16, 0, 0])
                dq0 = dq / len(qpts_qc)
                
                qpts_qv = np.dot(qpts_qc, rcell_cv)
                
                G_Gv = pd.get_reciprocal_vectors()
                delta_GG = np.eye(len(G_Gv))

                W_wGG[:, :, 0] = 0.0
                W_wGG[:, 0, :] = 0.0

                # I think we have to do this as a dump loop to avoid memory
                # problems
                for q, q_v in enumerate(qpts_qv):
                    if np.allclose(q_v, 0):
                        continue
                    
                    if q % 100 == 0:
                        print('q=%d/%d' % (q + 1, len(qpts_qv)))
                    vq_G = self.vc.get_potential(Gq_Gv=G_Gv + q_v)**0.5
                    
                    self.add_anisotropy_correction3D(W_wGG, dq0 * vq_G,
                                                     q_v, S_wvG, L_wvv)
                
                print('done')
                G_Gv[0] += np.array([1e-16, 0, 0])
                v0_G = self.vc.get_potential(Gq_Gv=G_Gv)**0.5
                v0_G[0] = 1.0
                dirx_v = np.array([1, 0, 0])
                diry_v = np.array([0, 1, 0])
                dirz_v = np.array([0, 0, 1])

                rq = (3. / (4 * pi) * dq0)**(1. / 3)
                va = (4*pi)**2 * rq
                vb = 2. / 5 * (2 * pi)**(3 / 2) * rq**(5 / 2)
                vc = dq0
                self.add_anisotropy_correction3D(W_wGG, 1. / 3 * v0_G, dirx_v,
                                                 S_wvG, L_wvv, va, vb, vc)
                self.add_anisotropy_correction3D(W_wGG, 1. / 3 * v0_G, diry_v,
                                                 S_wvG, L_wvv, va, vb, vc)
                self.add_anisotropy_correction3D(W_wGG, 1. / 3 * v0_G, dirz_v,
                                                 S_wvG, L_wvv, va, vb, vc)
                
            else:
                vc_G = self.vc.get_potential(pd=pd)**0.5
                W_wGG[:] = dq * (vc_G[np.newaxis, Ga:Gb, np.newaxis] * idf_wGG *
                                 vc_G[np.newaxis, :, np.newaxis])
            
        elif isinstance(self.vc, CoulombKernel2D):
            L = abs(self.cell_cv[2, 2])
            dq = rvol * L / (2 * pi) / self.qd.nbzkpts
            
            if np.allclose(q_c, 0):
                # For now, suppose 3D
                qdensity = 0.05
                rcell_cv = 2 * pi * np.linalg.inv(self.cell_cv).T
                N_c = self.qd.N_c
                
                npts_c = np.ceil(np.sum(rcell_cv**2, axis=1)**0.5 /
                                 N_c / qdensity).astype(int)
                npts_c[2] = 1
                
                qpts_qc = ((np.indices(npts_c).transpose((1, 2, 3, 0)) \
                            .reshape((-1, 3)) + 0.5) / npts_c - 0.5) / N_c
                qgamma = np.argmin(np.sum(qpts_qc**2, axis=1))
                #qpts_qc[qgamma] += np.array([1e-16, 0, 0])
                dq0 = dq / len(qpts_qc)
                
                qpts_qv = np.dot(qpts_qc, rcell_cv)
                
                G_Gv = pd.get_reciprocal_vectors()
                v_G = self.vc.get_potential(Gq_Gv=G_Gv[1:])**0.5

                W_wGG[:, :, 0] = 0.0
                W_wGG[:, 0, :] = 0.0
                W_wGG[:, 1:, 1:] *= (v_G[np.newaxis, :, np.newaxis] *
                                     v_G[np.newaxis, np.newaxis, :])

                if self.anistropy_correction:
                    # I think we have to do this as a dump loop to avoid memory
                    # problems
                    for q, q_v in enumerate(qpts_qv):
                        if np.allclose(q_v, 0):
                            continue
                    
                        vq_G = self.vc.get_potential(Gq_Gv=G_Gv + q_v)**0.5

                        dn = 1. / len(qpts_qv)
                        self.add_anisotropy_correction2D(W_wGG, dn * vq_G, q_v,
                                                         S_wvG, L_wvv)
                    
                    rq = (dq0 / pi)**0.5
                    W_wGG[:, 0, 0] += (-2 * pi**2 * L * (rq**2 / 2) *
                                       (L_wvv[:, 0, 0] + L_wvv[:, 1, 1])) / dq0
                
                    W_wGG[:, 1:, 1:] += (pi * rq**3 / 3 / dq0 *
                                         (S_wvG[:, 0, :, np.newaxis] *
                                          S_wvG[:, 0, np.newaxis, :].conj() +
                                          S_wvG[:, 1, :, np.newaxis] *
                                          S_wvG[:, 1, np.newaxis, :].conj()))
            
            else:
                vc_G = self.vc.get_potential(pd=pd)**0.5
                W_wGG[:] = 1.0 * (vc_G[np.newaxis, Ga:Gb, np.newaxis] * idf_wGG *
                                 vc_G[np.newaxis, :, np.newaxis])

        return W_wGG

    def add_anisotropy_correction3D(self, W_wGG, v_G, q_v, S_wvG, L_wvv,
                                    a=1.0, b=1.0, c=1.0):
        qdir_v = q_v / np.linalg.norm(q_v)
        
        qLq_w = np.dot(np.dot(np.eye(3)[np.newaxis, :, :] +
                              L_wvv, qdir_v), qdir_v)
        qS_wG = np.dot(S_wvG.transpose((0, 2, 1)), qdir_v)
        
        W_wGG[:, 0, 0] += a * v_G[0] * (1.0 / qLq_w - 1)
        W_wGG[:, 1:, 0] += b * (-v_G[np.newaxis, 1:] * v_G[0] *
                                qS_wG / qLq_w[:, np.newaxis])
        W_wGG[:, 0, 1:] += b * (-v_G[np.newaxis, 1:] * v_G[0] *
                                qS_wG.conj() / qLq_w[:, np.newaxis])
        W_wGG[:, 1:, 1:] += c * (v_G[np.newaxis, 1:, np.newaxis] *
                                 v_G[np.newaxis, np.newaxis, 1:] *
                                 (qS_wG[:, :, np.newaxis] *
                                  qS_wG.conj()[:, np.newaxis, :] /
                                  qLq_w[:, np.newaxis, np.newaxis]))

    def add_anisotropy_correction2D(self, W_wGG, v_G, q_v, S_wvG, L_wvv,
                                    a=1.0, b=1.0, c=1.0):
        qnorm = np.linalg.norm(q_v)
        qdir_v = q_v / qnorm
        
        qLq_w = np.dot(np.dot(np.eye(3)[np.newaxis, :, :] +
                              qnorm * L_wvv, qdir_v), qdir_v)
        qS_wG = qnorm**0.5 * np.dot(S_wvG.transpose((0, 2, 1)), qdir_v)
        
        W_wGG[:, 0, 0] += a * v_G[0] * (1.0 / qLq_w - 1)
        W_wGG[:, 1:, 0] += b * (-v_G[np.newaxis, 1:] * v_G[0] *
                                qS_wG / qLq_w[:, np.newaxis])
        W_wGG[:, 0, 1:] += b * (-v_G[np.newaxis, 1:] * v_G[0] *
                                qS_wG.conj() / qLq_w[:, np.newaxis])
        W_wGG[:, 1:, 1:] += c * (v_G[np.newaxis, 1:, np.newaxis] *
                                 v_G[np.newaxis, np.newaxis, 1:] *
                                 (qS_wG[:, :, np.newaxis] *
                                  qS_wG.conj()[:, np.newaxis, :] /
                                  qLq_w[:, np.newaxis, np.newaxis]))

    def integrate(self, sigma_qsin, dsigma_qsin):
        pass
    

class QuadQPointIntegration(QPointIntegration):
    def __init__(self, qd, cell_cv, qpts_qc=None, weight_q=None,
                 txt=sys.stdout, anisotropic=True):
        if qpts_qc is None:
            qpts_qc = qd.bzk_kc
        
        if weight_q is None:
            weight_q = 1.0 * np.ones(len(qpts_qc)) / len(qpts_qc)

        assert abs(np.sum(weight_q) - 1.0) < 1e-9, "Weights should sum to 1"
        
        self.weight_q = weight_q
        self.anistropy_correction = anisotropic
        
        QPointIntegration.__init__(self, qd, cell_cv, qpts_qc, txt)

    def reset(self, shape):
        self.sigma_skn = np.zeros(shape)
        self.dsigma_skn = np.zeros(shape)

    def add_term(self, bzq, spin, k, n, sigma, dsigma):
        vol = abs(np.linalg.det(self.cell_cv))
        dq = (2 * pi)**3 / vol
        
        self.sigma_skn[spin, k, n] += dq * self.weight_q[bzq] * sigma
        self.dsigma_skn[spin, k, n] += dq * self.weight_q[bzq] * dsigma

    def integrate(self):
        
        """
        sigma_sin = dq * np.dot(self.weight_q,
                                np.transpose(sigma_qsin, [1, 2, 0, 3]))
        dsigma_sin = dq * np.dot(self.weight_q,
                                 np.transpose(dsigma_qsin, [1, 2, 0, 3]))
        """
        return self.sigma_skn, self.dsigma_skn

class TriangleQPointIntegration(QPointIntegration):
    def __init__(self, qd, cell_cv, qpts_qc, simplices):
        self.simplices = simplices
        
        QPointIntegration.__init__(self, qd, cell_cv, qpts_qc)

    def integrate(self, sigma_qsin, dsigma_qsin):
        rcell_cv = 2 * pi * np.linalg.inv(self.cell_cv).T
        h = abs(self.cell_cv[2, 2])
        qpts_qv = np.dot(self.qpts_qc, rcell_cv)

        Vtot = 0

        sigma_sin = np.zeros(sigma_qsin.shape[1:])
        dsigma_sin = np.zeros(dsigma_qsin.shape[1:])
        for sim in self.simplices:
            q1_v = qpts_qv[sim[0]]
            q2_v = qpts_qv[sim[1]]
            q3_v = qpts_qv[sim[2]]
            J = np.array([[1, 1, 1],
                          [q1_v[0], q2_v[0], q3_v[0]],
                          [q1_v[1], q2_v[1], q3_v[1]]])
            V = 0.5 * abs(np.linalg.det(J)) * 2 * pi / h
            Vtot += V
            sigma_sin += V / 3 * np.sum(sigma_qsin[sim], axis=0)
            dsigma_sin += V / 3 * np.sum(dsigma_qsin[sim], axis=0)

        return sigma_sin, dsigma_sin

class BaseKernel:
    def __init__(self):
        pass

    def get_potential(pd, kpts_kc=None):
        pass

    def get_gamma_limits(self, pd):
        pass

class CoulombKernel3D:
    
    def get_potential(self, Gq_Gv=None, pd=None):
        if Gq_Gv is None:
            Gq_Gv = pd.get_reciprocal_vectors()
            Gq_Gv[0] += np.array([1e-16, 0, 0])
        K_G = 4 * pi / np.sum(Gq_Gv**2, axis=1)
        return K_G

    def get_gamma_limits(self, pd):
        if np.allclose(pd.kd.bzk_kc[0], 0):
            K0_0 = 4 * pi
            K0_G = 4 * pi / pd.G2_qG[0][1:]
            return K0_G, K0_0
        else:
            return None, None

class CoulombKernel2D:

    def __init__(self, cell_cv, R=None):
        self.L = abs(cell_cv[2, 2])
        if R is None:
            R = self.L / 2
        self.R = R
    
    def get_potential(self, Gq_Gv=None, pd=None):
        if Gq_Gv is None:
            Gq_Gv = pd.get_reciprocal_vectors()
            Gq_Gv[0] += np.array([1e-16, 0, 0])
        Gq2_G = np.sum(Gq_Gv**2, axis=1)
        Gqpar_G = np.sum(Gq_Gv[:, 0:2]**2, axis=1)**0.5
        R = self.R
        if abs(self.R - self.L / 2) < 1e-9:
            K_G = (4 * pi / Gq2_G * (1 - np.exp(-Gqpar_G * R) *
                                     np.cos(np.abs(Gq_Gv[:, 2]) * R)))
        else:
            K_G = (4 * pi / Gq2_G * (1 + np.exp(-Gqpar_G * R) *
                                     (Gq_Gv[:, 2] / Gqpar_G *
                                      np.sin(Gq_Gv[:, 2] * R) -
                                     np.cos(np.abs(Gq_Gv[:, 2]) * R))))

        return K_G

    def get_gamma_limits(self, pd):
        if np.allclose(pd.kd.bzk_kc[0], 0):
            K0_0 = 4 * pi * self.R
            K0_G = self.get_potential(
                Gq_Gv=pd.get_reciprocal_vectors(add_q=False)[1:])
            return K0_G, K0_0
        else:
            return None, None

class Event(list):
    """Event subscription.

    A list of callable objects. Calling an instance of this will cause a
    call to each item in the list in ascending order by index.

    Example Usage:
    >>> def f(x):
    ...     print 'f(%s)' % x
    >>> def g(x):
    ...     print 'g(%s)' % x
    >>> e = Event()
    >>> e()
    >>> e.append(f)
    >>> e(123)
    f(123)
    >>> e.remove(f)
    >>> e()
    >>> e += (f, g)
    >>> e(10)
    f(10)
    g(10)
    >>> del e[0]
    >>> e(2)
    g(2)

    """
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)
