# This makes sure that division works as in Python3
from __future__ import division, print_function

import sys
import os
from tempfile import TemporaryFile
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

    def __init__(self, calc, kpts=None, bandrange=None, filename=None, 
                 txt=sys.stdout, temp=None, savechi0=False, savepair=False,
                 nbands=None, ecut=150., nblocks=1, hilbert=True,      
                 domega0=0.025, omega2=10., omegamax=None,  eta=0.1, 
                 qptint=None, truncation='3D', world=mpi.world, timer=None):
        
        # Create output buffer
        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w', 1)
        self.fd = txt

        SelfEnergy.__init__(self)

        if isinstance(ecut, (int, float)):
            pct = 0.8
            necuts = 3
            self.ecut = ecut / Hartree
            self.ecut_i = self.ecut * (1 + (1. / pct - 1) * np.arange(necuts) /
                                       (necuts - 1))**(-2 / 3)
        else:
            self.ecut = max(ecut) / Hartree
            self.ecut_i = np.sort(ecut)[::-1] / Hartree
        
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
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

        # Set omegamax to maximum shift plus some extra
        e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)[0:nbands]
                           for k in range(len(calc.get_ibz_k_points()))]
                          for s in range(calc.get_number_of_spins())]) / Hartree
        omax = np.amax(e_skn) - np.amin(e_skn)
        if omegamax is None:
            self.omegamax = omax # + 10.0 / Hartree
        else:
            self.omegamax = omegamax

        if bandrange is None:
            bandrange = (0, calc.get_number_of_bands())
        
        self.bandrange = bandrange

        self.filename = filename

        self.savechi0 = savechi0
        self.savepair = savepair
        self.use_temp = bool(temp)
        self.temp_dir = './'
        if self.use_temp and isinstance(temp, str):
            self.temp_dir = temp
        self.saved_pair_density_files = dict()

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
                                                         weights_q,
                                                         txt=self.fd)
        else:
            self.qpt_integration = qptint
            self.qpt_integration.fd = self.fd

        self.freqint = RealFreqIntegration(self.calc,
                                           filename=self.filename,
                                           savechi0=self.savechi0,
                                           use_temp=self.use_temp,
                                           temp_dir=self.temp_dir,
                                           ecut=self.ecut * Hartree,
                                           nbands=self.nbands,
                                           domega0=self.domega0 * Hartree,
                                           omega2=self.omega2 * Hartree,
                                           omegamax=self.omegamax * Hartree,
                                           eta = self.eta * Hartree,
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
        self.qpt_integration.fd = self.fd

        self.progressEvent = Event()

        self.estimate_pair_density_space()

        self.sigma_iskn = None
        self.dsigma_iskn = None
        self.sigma_skn = None
        self.dsigma_skn = None
        self.sigerr_skn = None

        self.complete = False
        self.nq = 0
        if self.load_state_file():
            if self.complete:
                print('Self-energy loaded from file', file=self.fd)

    def addEventHandler(self, event, handler):
        if event == 'progress' and handler not in self.progressEvent:
            self.progressEvent.append(handler)

    def save_state_file(self, q=0):
        data = {'kpts': self.kpts,
                'bandrange': self.bandrange,
                'nbands': self.nbands,
                'ecut_i': self.ecut_i,
                'freqint': (type(self.freqint).__module__ +
                            type(self.freqint).__name__),
                'qptint': (type(self.qpt_integration).__module__ +
                           type(self.qpt_integration).__name__),
                'qpts_qc': self.qpt_integration.qpts_qc,
                'last_q': self.nq,
                'complete': self.complete,
                'qptstate': self.qpt_integration.get_state(self.world),
                'sigma_iskn': self.sigma_iskn,
                'dsigma_iskn': self.dsigma_iskn,
                'sigma_skn': self.sigma_skn,
                'dsigma_skn': self.dsigma_skn,
                'sigerr_skn': self.sigerr_skn}
        if self.world.rank == 0:
            with open(self.filename + '.sigma.pckl', 'wb') as fd:
                pickle.dump(data, fd)

    def load_state_file(self):
        try:
            data = pickle.load(open(self.filename + '.sigma.pckl'))
        except IOError:
            return False
        else:
            if (data['kpts'] == self.kpts and
                data['bandrange'] == self.bandrange and
                data['nbands'] == self.nbands and
                data['ecut_i'] == self.ecut_i,
                data['freqint'] == (type(self.freqint).__module__ +
                                    type(self.freqint).__name__) and
                data['qptint'] == (type(self.qpt_integration).__module__ +
                                   type(self.qpt_integration).__name__) and
                np.allclose(data['qpts_qc'], self.qpt_integration.qpts_qc)):
                self.nq = data['last_q']
                self.qpt_integration.load_state(data['qptstate'], self.world)
                self.complete = data['complete']
                self.sigma_iskn = data['sigma_iskn']
                self.dsigma_iskn = data['dsigma_iskn']
                self.sigma_skn = data['sigma_skn']
                self.dsigma_skn = data['dsigma_skn']
                self.sigerr_skn = data['sigerr_skn']
                return True
            else:
                return False

    def calculate(self, readw=True):
        p = functools.partial(print, file=self.fd)
        p('Calculating the correlation self-energy')
        p('selfenergy2')
        if len(self.ecut_i) > 1:
            p('    Using PW cut-offs: ' +
              ', '.join(['{0:.0f} eV'.format(ecut * Hartree)
                         for ecut in self.ecut_i]))
            p('    Extrapolating to infinite cut-off')
        else:
            p('    Using PW cut-off:   {0:.0f} eV'.format(self.ecut_i[0] * Hartree))

        nbzq = len(self.qpt_integration.qpts_qc)
        self.sigma_iskn = np.zeros((len(self.ecut_i), ) + self.shape, complex)
        self.dsigma_iskn = np.zeros((len(self.ecut_i), ) + self.shape, complex)

        if self.complete:
            # Start a new calculation
            self.complete = False
            self.nq = 0

        self.progressEvent(0.0)

        #for i, ecut in enumerate(self.ecut_i):
        try:
            sigma_iskn, dsigma_iskn = self._calculate(readw)
        except:
            self.clear_temp_files()
            self.freqint.clear_temp_files()
            raise
        else:
            self.complete = True
        
        self.sigma_iskn = sigma_iskn
        self.dsigma_iskn = dsigma_iskn
        
        self.sigerr_skn = np.zeros(self.shape, complex)
        if len(self.ecut_i) > 1:
            # Do linear fit of selfenergy vs. inverse of number of plane waves
            # to extrapolate to infinite number of plane waves
            self.sigma_skn = np.zeros(self.shape, complex)
            self.dsigma_skn = np.zeros(self.shape, complex)
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
                        assert (psig[1].real < sigmax.real and
                                psig[1].real > sigmin.real)
                        sigerr = sigmax - sigmin
                        self.sigerr_skn[s, k, n] = sigerr
                        
                        pdsig = np.polyfit(invN_i,
                                           self.dsigma_iskn[:, s, k, n], 1)
                        self.dsigma_skn[s, k, n] = pdsig[1]
        else:
            self.sigma_skn = self.sigma_iskn[0]
            self.dsigma_skn = self.dsigma_iskn[0]

        self.save_state_file()
        
        self.timer.write(self.fd)

    def _calculate(self, readw=True):
        # My part of the states we want to calculate QP-energies for:
        mykpts = [self.pairDensity.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]

        kd = self.calc.wfs.kd
        
        kplusqdone_u = [set() for kpt in mykpts]
        
        prefactor = 1 / (2 * pi)**4

        if self.complete or self.nq == 0:
            self.qpt_integration.reset((len(self.ecut_i), ) + self.shape)
        
        for i, Q1, Q2, W0_wGG, pd0, pdi0, Q0_aGii in self.do_qpt_loop(readw=readw):
            ibzq = self.qd.bz2ibz_k[Q1]
            q_c = self.qd.ibzk_kc[ibzq]

            #print('i=%d, W0_GG=' % i, file=self.fd)
            #for iG in range(4):
            #    print(W0_wGG[0, iG, 0:4].round(4), file=self.fd)

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

            G_Gv = pdi0.get_reciprocal_vectors()
            pos_av = np.dot(self.pairDensity.spos_ac, pdi0.gd.cell_cv)
            M_vv = np.dot(pdi0.gd.cell_cv.T,
                          np.dot(U_cc.T, np.linalg.inv(pdi0.gd.cell_cv).T))
            # Transform PAW corrections from IBZ to full BZ
            Q_aGii = []
            for a, Q_Gii in enumerate(Q0_aGii):
                x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] - sign *
                                                np.dot(M_vv, pos_av[a]))))
                U_ii = self.calc.wfs.setups[a].R_sii[s]
                Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, None]),
                               U_ii.T).transpose(1, 0, 2)
                Q_aGii.append(Q_Gii)

            G2G = pdi0.map(pd0, q=0)

            for u1, kpt1 in enumerate(mykpts):
                k1 = kd.bz2ibz_k[kpt1.K]
                spin = kpt1.s
                ik = self.kpts.index(k1)
                
                K2 = kd.find_k_plus_q(Q_c, [kpt1.K])[0] # K2 will be in 1st BZ
                # This k+q or symmetry related points have not been
                # calculated yet.
                kpt2 = self.pairDensity.get_k_point(spin, K2, 0,
                                                    self.nbands,
                                                    block=True)

                N_c = pdi0.gd.N_c
                i_cG = sign * np.dot(U_cc, np.unravel_index(pdi0.Q_qG[0], N_c))

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
                    n_mG = None
                    myid = 'k1=%dk2=%dn=%d' % (kpt1.K, kpt2.K, n)
                    if i > 0 and self.use_temp and self.savepair:
                        n_mG = np.empty((kpt2.n2 - kpt2.n1, nG), complex)
                        if self.blockcomm.rank == 0:
                            tmpfile = self.saved_pair_density_files[myid]
                            tmpfile.seek(0)
                            n_mG[:] = np.load(tmpfile).take(G2G, axis=1)
                        self.blockcomm.broadcast(n_mG, 0)
                            
                    if n_mG is None:
                        C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                                  for Qa_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
                        ut1cc_R = kpt1.ut_nR[n].conj()
                        n_mG = self.pairDensity.calculate_pair_densities(
                            ut1cc_R, C1_aGi, kpt2, pdi0, I_G)
                        if self.use_temp and self.savepair and self.blockcomm.rank == 0:
                            tmpfile = TemporaryFile(dir=self.temp_dir)
                            np.save(tmpfile, n_mG)
                            self.saved_pair_density_files[myid] = tmpfile
                    
                    #n_mG = self.pairDensity \
                    #    .calculate_pair_densities(ut1cc_R, C1_aGi,
                    #                              kpt2, pd0, I_G)
                    #n_mG = self.get_pair_densities(kpt1, kpt2, n, pd0,
                    #                               C1_aGi, I_G)
                    
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
                        self.qpt_integration.add_term(bzq, i, spin, ik, nn,
                                                      prefactor * sigma,
                                                      prefactor * dsigma)

        #self.world.sum(self.sigma_qsin)
        #self.world.sum(self.dsigma_qsin)
        
        sigma_iskn, dsigma_iskn = self.qpt_integration.integrate() # (self.sigma_qsin, self.dsigma_qsin)

        self.world.sum(sigma_iskn)
        self.world.sum(dsigma_iskn)
        
        return sigma_iskn, dsigma_iskn

    def get_pair_densities(self, kpt1, kpt2, n1, pd, C1_aGi, I_G):
        tmpfile = (self.temp_dir + self.filename +
                   '.nk%dq%dn%d.pckl' % (kpt1.K, kpt2.K, n1))
        if self.use_temp:
            if tmpfile in self.saved_pair_density_files:
                try:
                    fd = open(tmpfile, 'rb')
                    pd0, n0_mG = pickle.load(fd)
                    fd.close()
                    if pd0.ecut > pd.ecut:
                        G2G = pd.map(pd0, q=0)
                        return n0_mG.take(G2G, axis=1)
                    else:
                        return n0_mG
                except IOError as err:
                    print('read error: %s' % err)
        
        ut1cc_R = kpt1.ut_nR[n1].conj()
        n_mG = self.pairDensity.calculate_pair_densities(ut1cc_R, C1_aGi,
                                                         kpt2, pd, I_G)
        if self.use_temp and self.blockcomm.rank == 0:
            try:
                fd = open(tmpfile, 'wb')
                pickle.dump((pd, n_mG), fd, pickle.HIGHEST_PROTOCOL)
                fd.close()
                self.saved_pair_density_files.append(tmpfile)
            except IOError as err:
                print('write err: %s' % err)
        
        return n_mG

    def clear_temp_files(self):
        for fileid, tmpfile in self.saved_pair_density_files.iteritems():
            tmpfile.close()
        self.saved_pair_density_files = dict()

    def do_qpt_loop(self, readw=True):
        """Do the loop over q-points in the q-point integration"""
        # Find maximum size of chi-0 matrices:
        gd = self.calc.wfs.gd
        nGmax = max(count_reciprocal_vectors(self.ecut, gd, q_c)
                    for q_c in self.qd.ibzk_kc)
        nw = self.freqint.wsize
        
        size = self.blockcomm.size
        mynwmax = (nw + size - 1) // size
        mynGmax = (nGmax + size - 1) // size
        #mynw = (nw + size - 1) // size
        
        # Allocate memory in the beginning and use for all q:
        #maxsize = max(nw * mynGmax * nGmax, mynwmax * nGmax**2)
        maxsize = self.freqint.get_max_size(self.qd)
        A_x = np.empty(maxsize, complex)

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
        nibzqs = len(ibzqs)
        for nq in range(self.nq, nibzqs):
            self.nq = nq
            self.save_state_file()
            ibzq = ibzqs[nq]
            q_c = qd.ibzk_kc[ibzq]

            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            print('Calculating contribution from IBZ q-pointq #%d/%d, q_c=%s' %
                  (nq + 1, len(ibzqs), qcstr), file=self.fd)
            for i, ecut in enumerate(self.ecut_i):

                W_wGG, pd, pdi, Q_aGii = self.calculate_idf(q_c, ecut,
                                                            readw=readw,
                                                            A_x=A_x)
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
                Q2s = set()
                for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                    if Q2 >= 0 and Q2 not in Q2s:
                        Q2s.add(Q2)
                
                for nq2, Q2 in enumerate(Q2s):
                    yield i, Q1, Q2, W_wGG, pd, pdi, Q_aGii
                    q2p = (nq2 + 1.) / len(Q2s)
                    ecutp = (i + q2p) / len(self.ecut_i)
                    q1p = (nq + ecutp) / len(ibzqs)
                    self.progressEvent(q1p)
                print('ecut=%.0f done' % (ecut * Hartree), file=self.fd)
            
            self.clear_temp_files()
            self.freqint.clear_temp_files()

    @timer('Screened potential')
    def calculate_idf(self, q_c, ecut, readw=True, A_x=None):
        W_wGG, pd, pdi, Q_aGii = \
          self.freqint.calculate_idf(q_c, self.vc, ecut, readw=readw, A_x=A_x)
        
        if Q_aGii is None:
            Q_aGii = self.pairDensity.initialize_paw_corrections(pdi)

        return W_wGG, pd, pdi, Q_aGii

    def calculate_w(self, pd, idf_wGG, S_wvG, L_wvv):

        return self.qpt_integration.calculate_w(pd, idf_wGG, S_wvG, L_wvv)

    def estimate_pair_density_space(self):
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
        
        
        nbzq_q = np.zeros(len(ibzqs))
        for i, ibzq in enumerate(ibzqs):
            Q1 = self.qd.ibz2bz_k[ibzq]
            done = set()
            for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                if Q2 >= 0 and Q2 not in done:
                    nbzq_q[i] += 1
                    done.add(Q2)
        maxnbzq = np.amax(nbzq_q)
        knsize = self.pairDensity.kncomm.size
        nu = np.prod(self.shape)
        mynu = (nu + knsize - 1) // knsize
        maxsize = mynu * maxnbzq * self.nbands * 16.0 / 1024**2
        print('max nbzq=%d, mynu=%d, maxsize=%.2f MB' % (maxnbzq, mynu, maxsize),
              file=self.fd)

class FrequencyIntegration:

    def __init__(self, txt=sys.stdout):
        self.fd = txt

    def initialize(self, selfenergy):
        self.selfenergy = selfenergy

    def calculate_integration(self, n_mG, deps_m, f_m, W_swGG, S_wvG, L_wvv):
        pass

class RealFreqIntegration(FrequencyIntegration):

    def __init__(self, calc, filename=None, savechi0=False,
                 use_temp=True, temp_dir='./',
                 ecut=150., nbands=None,
                 domega0=0.025, omega2=10., omegamax=None,
                 eta = 0.1, timer=None, txt=sys.stdout, nblocks=1):
        FrequencyIntegration.__init__(self, txt)
        
        self.timer = timer
        
        self.calc = calc

        self.ecut = ecut / Hartree
        self.nbands = nbands
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.eta = eta / Hartree

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
                      'omega2': self.omega2 * Hartree,
                      'omegamax': omegamax}
        
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
        self.fd = txt
        self.filename = filename
        self.use_temp = use_temp
        self.temp_dir = temp_dir
        
        self.df = DielectricFunction(self.calc,
                                     nbands=self.nbands,
                                     ecut=self.ecut * Hartree,
                                     intraband=True,
                                     nblocks=self.nblocks,
                                     name=None,
                                     txt=self.filename + '.chi0.txt',
                                     **parameters)
                                     

        self.omega_w = self.df.chi0.omega_w
        self.wsize = 2 * len(self.omega_w)

        self.htp = HilbertTransform(self.omega_w, self.eta, gw=True)
        self.htm = HilbertTransform(self.omega_w, -self.eta, gw=True)

        self.temp_files = []

    def get_max_size(self, qd):
        # Find maximum size of chi-0 matrices:
        gd = self.calc.wfs.gd
        nGmax = max(count_reciprocal_vectors(self.ecut, gd, q_c)
                    for q_c in qd.ibzk_kc)
        nw = len(self.omega_w)

        chi0 = self.df.chi0
        size = chi0.blockcomm.size
        mynwmax = (nw + size - 1) // size
        mynGmax = (nGmax + size - 1) // size
        #mynw = (nw + size - 1) // size
        
        # Allocate memory in the beginning and use for all q:
        maxsize = max(2 * nw * mynGmax * nGmax, 2 * mynwmax * nGmax**2)

        return maxsize

    @timer('Inverse dielectric function')
    def calculate_idf(self, q_c, vc, ecut, readw=False, A_x=None):
        """Calculates the inverse dielectric matrix for a specified q-point."""
        # Divide memory into two slots so we can redistribute via moving values
        # from one slot to the other
        if A_x is not None:
            nx = len(A_x)
            A1_x = A_x[:nx // 2]
            A2_x = A_x[nx // 2:]
        else:
            A1_x = None
            A2_x = None
        
        """
        if self.use_temp and self.tmpfile is not None:
            pd = self.pd
            Q_aGii = self.Q_aGii
            chi0_wGG, chi0_wxvG, chi0_wvv = self.read_chi0(self.tmpfile,
                                                           A1_x, A2_x)
        else:
            pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.chi0.calculate(q_c,
                                                                    A_x=A1_x)
            Q_aGii = self.chi0.Q_aGii
            chi0_wGG = self.chi0.redistribute(chi0_wGG, A2_x)
            if self.use_temp:
                self.tmpfile = TemporaryFile(dir=self.tmp_dir)
                self.write_chi0(self.tmpfile, pd, chi0_wGG, chi0_wxvG, chi0_wvv)
                self.pd = pd
                self.Q_aGii = Q_aGii
        """
        
        kd = self.df.chi0.calc.wfs.kd
        filename = (self.temp_dir + self.filename + '.chi0' +
                    '%+d%+d%+d.pckl' % tuple((q_c * kd.N_c).round()))
        if (readw or self.use_temp) and os.path.isfile(filename):
            pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.df.read(filename,
                                                             A1_x, A2_x)
            Q_aGii = None
        else:
            pd, chi0_wGG, chi0_wxvG, chi0_wvv = self.df.calculate_chi0(
                q_c, A1_x=A2_x, A2_x=A1_x)
            Q_aGii = self.df.chi0.Q_aGii
            if self.savechi0 or self.use_temp:
                self.df.write(filename, pd, chi0_wGG, chi0_wxvG, chi0_wvv,
                              A1_x)
                self.temp_files.append(filename)

        if pd.ecut > ecut:
            pdi = PWDescriptor(ecut, pd.gd, dtype=pd.dtype,
                               kd=pd.kd)
            G2G = pdi.map(pd, q=0)
            
            chi0_wGG = chi0_wGG.take(G2G, axis=1).take(G2G, axis=2)

            if chi0_wxvG is not None:
                chi0_wxvG = chi0_wxvG.take(G2G, axis=3)

            if Q_aGii is not None:
                for a, Q_Gii in enumerate(Q_aGii):
                    Q_aGii[a] = Q_Gii.take(G2G, axis=0)
        else:
            pdi = pd


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
        vc_G = vc.get_potential(pd=pdi)**0.5

        # These are entities related to the q->0 value
        S_wvG = None
        L_wvv = None
        if np.allclose(q_c, 0):
            vc_G0, vc_00 = vc.get_gamma_limits(pdi)
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
                u_vG = vc_G0[np.newaxis, :]**0.5 * \
                       chi0_wxvG[wa + w, 0, :, 1:]
                U_vv = -chi0_wvv[wa + w]
                a_vG = -np.dot(u_vG, B_GG.T)
                A_vv = U_vv - np.dot(u_vG.conj(), a_vG.T)
                S_wvG[w] = a_vG
                L_wvv[w] = A_vv
            else:
                idf_GG[:] = np.linalg.inv(delta_GG - chi0_GG * 
                                          vc_G[np.newaxis, :] *
                                          vc_G[:, np.newaxis])
            idf_GG -= delta_GG
        
        idf_wGG = chi0_wGG # rename

        W_wGG = idf_wGG
        W_wGG[:wb - wa] = self.selfenergy.calculate_w(pdi, idf_wGG[:wb - wa],
                                                      S_wvG, L_wvv)

        # Since we are doing a Hilbert transform we get two editions of the
        # W_wGG matrix corresponding to +/- contributions. We are thus doubling
        # the number of frequencies and storing the +/- part in each of the two
        # halves.
        newshape = (2*nw, Gb - Ga, nG)
        size = np.prod(newshape)
        
        Wpm_wGG = A_x[:size].reshape(newshape)
        
        if self.nblocks > 1:
            # Now redistribute back on G rows and save in second half of A1_x
            # which is not used any more (was only used when reading/
            # calculating chi0 in the beginning
            W_wGG = self.df.chi0.redistribute(W_wGG, A2_x)
            Wpm_wGG[:nw] = W_wGG
            #Wpm_wGG[:nw] = self.df.chi0.redistribute(W_wGG, A1_x)
        else:
            Wpm_wGG[:nw] = W_wGG
        
        
        Wpm_wGG[nw:] = Wpm_wGG[0:nw]
        
        with self.timer('Hilbert transform'):
            self.htp(Wpm_wGG[:nw])
            self.htm(Wpm_wGG[nw:])

        #print('rank=%d, Wpm_wGG.shape=%s' % (world.rank, str(Wpm_wGG.shape)))
        
        return Wpm_wGG, pd, pdi, Q_aGii

    def clear_temp_files(self):
        if not self.savechi0:
            world = self.df.chi0.world
            if world.rank == 0:
                while len(self.temp_files) > 0:
                    filename = self.temp_files.pop()
                    os.remove(filename)

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

    def get_state(self):
        pass

    def load_state(self, state):
        pass

    def calculate_w(self, pd, idf_wGG, S_wvG, L_wvv, GaGb=None):
        pass

    def integrate(self, sigma_qsin, dsigma_qsin):
        pass
    

class QuadQPointIntegration(QPointIntegration):
    def __init__(self, qd, cell_cv, qpts_qc=None, weight_q=None,
                 txt=sys.stdout, anisotropic=True, x0density=0.01,
                 only_head=False):
        if qpts_qc is None:
            qpts_qc = qd.bzk_kc
        
        if weight_q is None:
            weight_q = 1.0 * np.ones(len(qpts_qc)) / len(qpts_qc)

        assert abs(np.sum(weight_q) - 1.0) < 1e-9, "Weights should sum to 1"
        
        self.weight_q = weight_q
        self.anisotropy_correction = anisotropic
        self.only_head = only_head
        self.x0density = x0density
        
        QPointIntegration.__init__(self, qd, cell_cv, qpts_qc, txt)

    def print_info():
        print('BZ Integration using numerical quadrature', file=self.fd)
        print('  Use anisotropy correction: %s' % self.anisotropy_correction,
              file=self.fd)
        print('  Include only anistropy corrections to HEAD: %s'
              % self.only_head, file=self.fd)
        print('  q-point density for numerical Gamma-point integration: ' +
              '%s Angstrom^(-1)' % self.x0density, file=self.fd)

    def reset(self, shape):
        self.sigma_iskn = np.zeros(shape, complex)
        self.dsigma_iskn = np.zeros(shape, complex)

    def add_term(self, bzq, i, spin, k, n, sigma, dsigma):
        vol = abs(np.linalg.det(self.cell_cv))
        dq = (2 * pi)**3 / vol
        
        self.sigma_iskn[i, spin, k, n] += dq * self.weight_q[bzq] * sigma
        self.dsigma_iskn[i, spin, k, n] += dq * self.weight_q[bzq] * dsigma

    def integrate(self):
        
        """
        sigma_sin = dq * np.dot(self.weight_q,
                                np.transpose(sigma_qsin, [1, 2, 0, 3]))
        dsigma_sin = dq * np.dot(self.weight_q,
                                 np.transpose(dsigma_qsin, [1, 2, 0, 3]))
        """
        return self.sigma_iskn, self.dsigma_iskn

    def get_state(self, world):
        sigma_iskn = self.sigma_iskn.copy()
        dsigma_iskn = self.dsigma_iskn.copy()
        world.sum(sigma_iskn, 0)
        world.sum(dsigma_iskn, 0)
        return {'sigma_iskn': sigma_iskn,
                'dsigma_iskn': dsigma_iskn}

    def load_state(self, state, world):
        if world.rank == 0:
            self.sigma_iskn = state['sigma_iskn'].astype(complex)
            self.dsigma_iskn = state['dsigma_iskn'].astype(complex)
        else:
            myshape = state['sigma_iskn'].shape
            self.sigma_iskn = np.zeros(myshape, complex)
            self.dsigma_iskn = np.zeros(myshape, complex)
    
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
        
        if np.allclose(q_c, 0):
            if isinstance(self.vc, WignerSeitzTruncatedCoulomb):
                vc_G = self.vc.get_potential(pd=pd)**0.5
                W_wGG[:] = (vc_G[np.newaxis, Ga:Gb, np.newaxis] * idf_wGG *
                            vc_G[np.newaxis, np.newaxis, :])
            elif isinstance(self.vc, CoulombKernel3D):
                pass
            elif isinstance(self.vc, CoulombKernel2D):
                self.calculate_W_q0_2D(pd, idf_wGG, L_wvv, S_wvG)
        else:
            vc_G = self.vc.get_potential(pd=pd)**0.5
            W_wGG[:] = (vc_G[np.newaxis, :, np.newaxis] * idf_wGG *
                        vc_G[np.newaxis, np.newaxis, :])
        
        return W_wGG

    def calculate_W_q0_2D(self, pd, invdf_wGG, A_wvv, a_wvG):
        L = self.cell_cv[2, 2]
        # First get potential
        G_Gv = pd.get_reciprocal_vectors()[1:]
        G_Gv += np.array([1e-9, 1e-9, 0])
        G2_G = np.sum(G_Gv**2, axis=1)
        Gpar_G = np.sum(G_Gv[:, 0:2]**2, axis=1)**0.5
        v_G = (4 * pi / G2_G * (1 - np.exp(-0.5 * L * Gpar_G) * \
                                np.cos(0.5 * L * G_Gv[:, 2])))**0.5
        W_wGG = invdf_wGG # Rename
        nG = W_wGG.shape[1]

        # Generate numerical q-point grid
        rcell_cv = 2 * pi * np.linalg.inv(self.cell_cv).T
        rvol = abs(np.linalg.det(rcell_cv))
        N_c = self.qd.N_c
        
        q0weight = 1
        qf = q0weight
        q0cell_cv = np.array([qf, qf, 1])**0.5 * rcell_cv / N_c
        q0vol = abs(np.linalg.det(q0cell_cv))

        x0density = self.x0density
        q0density = 2. / L * x0density
        npts_c = np.ceil(np.sum(q0cell_cv**2, axis=1)**0.5 /
                         q0density).astype(int)
        npts_c[2] = 1
        npts_c += (npts_c + 1) % 2
        print('qf=%.3f, q0density=%s, q0 volume=%.5f ~ %.2f %%' %
              (qf, q0density, q0vol, q0vol / rvol * 100.),
              file=self.fd)
        print('Evaluating Gamma point contribution to W on a ' +
              '%dx%dx%d grid' % tuple(npts_c), file=self.fd)
        #npts_c = np.array([1, 1, 1])
                
        qpts_qc = ((np.indices(npts_c).transpose((1, 2, 3, 0)) \
                    .reshape((-1, 3)) + 0.5) / npts_c - 0.5) #/ N_c
        qgamma = np.argmin(np.sum(qpts_qc**2, axis=1))
        #qpts_qc[qgamma] += np.array([1e-16, 0, 0])
                    
        qpts_qv = np.dot(qpts_qc, q0cell_cv)
        qpts_q = np.sum(qpts_qv**2, axis=1)**0.5
        qpts_q[qgamma] = 1e-14
        qdir_qv = qpts_qv / qpts_q[:, np.newaxis]
        qdir_qvv = qdir_qv[:, :, np.newaxis] * qdir_qv[:, np.newaxis, :]
        nq = len(qpts_qc)
        q0area = q0vol / q0cell_cv[2, 2]
        dq0 = q0area / nq
        dq0rad = (dq0 / pi)**0.5

        exp_q = 4 * pi * (1 - np.exp(-qpts_q * L / 2))
        dv_G = ((pi * L * G2_G * np.exp(-0.5 * L * Gpar_G) * \
                 np.cos(0.5 * L * G_Gv[:, 2]) -
                 4 * pi * Gpar_G * (1 - np.exp(-0.5 * Gpar_G) * \
                                    np.cos(0.5 * L * G_Gv[:, 2]))) / \
                (G2_G**1.5 * Gpar_G * (4 * pi * (1 - np.exp(-0.5 * L * Gpar_G) * \
                                                 np.cos(0.5 * L * G_Gv[:, 2])))**0.5))
        dv_Gv = dv_G[:, np.newaxis] * G_Gv
        
        # Calculate q=0 value
        nw = W_wGG.shape[0]
        nG = W_wGG.shape[1]
        for w in range(nw):

            W_wGG[w, :, 0] = 0.0
            W_wGG[w, 0, :] = 0.0
            W_wGG[w, 1:, 1:] = (v_G[:, None] * v_G[None, :] * invdf_wGG[w, 1:, 1:])
            

            if not self.anisotropy_correction:
                # Skip q=0 corrections
                continue
            
            A_q = np.sum(qdir_qv * np.dot(qdir_qv, A_wvv[w]), axis=1)
            frac_q = 1. / (1 + exp_q * A_q)

            # HEAD:
            w00_q = -(exp_q / qpts_q)**2 * A_q * frac_q
            w00_q[qgamma] = 0.0
            W_wGG[w, 0, 0] = w00_q.sum() / nq
            a0 = 2 * pi * (A_wvv[w, 0, 0] + A_wvv[w, 1, 1]) + 1
            W_wGG[w, 0, 0] += -(a0 * dq0rad - np.log(a0 * dq0rad + 1)) / a0**2 / dq0

            # WINGS:
            u_q = -exp_q / qpts_q * frac_q
            #u_q[qgamma] = 0.0
            W_wGG[w, 1:, 0] = 1. / nq * np.dot(
                np.sum(qdir_qv * u_q[:, np.newaxis], axis=0),
                a_wvG[w] * v_G[np.newaxis, :])
            
            W_wGG[w, 0, 1:] = W_wGG[w, 1:, 0].conj()
            
            # BODY:
            # Constant corrections:
            W_wGG[w, 1:, 1:] += 1. / nq * v_G[:, None] * v_G[None, :] * \
                np.tensordot(a_wvG[w], np.dot(a_wvG[w].T.conj(),
                np.sum(-qdir_qvv * exp_q[:, None, None] * frac_q[:, None, None],
                axis=0)), axes=(0, 1))
            u_vvv = np.tensordot(u_q[:, None] * qpts_qv, qdir_qvv, axes=(0, 0))
            # Gradient corrections:
            W_wGG[w, 1:, 1:] += 1. / nq * np.sum(
                dv_Gv[:, :, None] * np.tensordot(
                    a_wvG[w], np.tensordot(u_vvv, a_wvG[w].conj() * v_G[None, :],
                                           axes=(2, 0)), axes=(0, 1)), axis=1)
            

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

    def add_anisotropy_correction2D_lin(self, W_wGG, v_G, q_v, S_wvG, L_wvv,
                                    a=1.0, b=1.0, c=1.0):
        qnorm = np.linalg.norm(q_v)
        qdir_v = q_v / qnorm
        
        qLq_w = np.dot(np.dot(np.eye(3)[np.newaxis, :, :] +
                              qnorm * L_wvv, qdir_v), qdir_v)
        qS_wG = qnorm**0.5 * np.dot(S_wvG.transpose((0, 2, 1)), qdir_v)
        
        W_wGG[:, 0, 0] += a * v_G[0]**2 * (1.0 / qLq_w - 1)
        W_wGG[:, 1:, 0] += b * (-v_G[np.newaxis, 1:] * v_G[0] *
                                qS_wG / qLq_w[:, np.newaxis])
        W_wGG[:, 0, 1:] += b * (-v_G[np.newaxis, 1:] * v_G[0] *
                                qS_wG.conj() / qLq_w[:, np.newaxis])
        W_wGG[:, 1:, 1:] += c * (v_G[np.newaxis, 1:, np.newaxis] *
                                 v_G[np.newaxis, np.newaxis, 1:] *
                                 (qS_wG[:, :, np.newaxis] *
                                  qS_wG.conj()[:, np.newaxis, :] /
                                  qLq_w[:, np.newaxis, np.newaxis]))

    def add_anisotropy_correction2D(self, W_wGG, v_G, q_v, S_wvG, L_wvv, L,
                                    a=1.0, b=1.0, c=1.0):
        qnorm = np.linalg.norm(q_v)
        qdir_v = q_v / qnorm
        
        A_w = np.dot(np.dot(L_wvv, qdir_v), qdir_v)
        a_wG = v_G[np.newaxis, 1:] * np.dot(qdir_v, S_wvG)
        exp = 4 * pi * (1 - np.exp(-0.5 * L * qnorm))

        W_wGG[:, 0, 0] += -(exp / qnorm)**2 / (1 + A_w * exp) * a
        
        W_wGG[:, 1:, 0] += -exp / qnorm * a_wG / \
                           (1 + A_w[:, np.newaxis] * exp) * b
        W_wGG[:, 0, 1:] += -exp / qnorm * a_wG.conj() / \
                           (1 + A_w[:, np.newaxis] * exp) * b
        
        W_wGG[:, 1:, 1:] += -exp * a_wG[:, :, np.newaxis] * \
                            a_wG[:, np.newaxis, :] / \
                            (1 + A_w[:, np.newaxis, np.newaxis] * exp) * c
        

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
