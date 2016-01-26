from __future__ import division, print_function

import sys
import os
import functools
from math import pi
import pickle

import numpy as np

from ase.utils import opencew, devnull
from ase.utils.timing import timer, Timer
from ase.units import Hartree
from ase.parallel import paropen
from ase.dft.kpoints import monkhorst_pack

import gpaw
from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw.kpt_descriptor import to1bz, KPointDescriptor
from gpaw.response.chi0 import HilbertTransform
from gpaw.response.pair import PairDensity, PWSymmetryAnalyzer
from gpaw.response.selfenergy import RealFreqIntegration, QuadQPointIntegration

from gpaw.wavefunctions.pw import PWDescriptor, count_reciprocal_vectors
from gpaw.utilities.progressbar import ProgressBar
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
import gpaw.io.tar as io
from ase.units import Bohr


class GWQEHCorrection:
    """ Class for calculating quasiparticle energies of van der Waals
    heterostructures using the GW approximation for the self-energy. 
    The quasiparticle energy correction due to increased screening from
    surrounding layers is obtained from the QEH model.
    """
    def __init__(self, gwfile,
                 structure=None, d=None, layer=0, qqeh=None, wqeh=None,
                 dW_qw=None, d0=None, filename=None,
                 txt=sys.stdout, calc=None, kpts=[0], bandrange=None,
                 world=mpi.world, qptint=None, domega0=None, omega2=None): 
        
        self.d0 = d0 / Bohr
        self.gwfile = gwfile

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w', 1)
        self.fd = txt

        self.timer = Timer()
        self.filename = filename
        self.world = world
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree

        #  Initialize variables
        self.qp_skn = None
        self.Z_skn = None
        self.f_skn = None

        with self.timer('Read ground state'):
            if isinstance(calc, str):
                if not calc.split('.')[-1] == 'gpw':
                    calc = calc + '.gpw'
                    self.calc_file = calc
                self.reader = io.Reader(calc, comm=mpi.serial_comm)
                print('Reading ground state calculation from file: %s' % calc,
                      file=self.fd)
                calc = GPAW(calc, txt=None, communicator=mpi.serial_comm)
                            #  ,read_projections=False)
            else:
                self.reader = None
                assert calc.wfs.world.size == 1

        assert calc.wfs.kd.symmetry.symmorphic, \
          'Can only handle symmorhpic symmetries at the moment'
        self.calc = calc

        if kpts is None:
            kpts = range(len(calc.get_ibz_k_points()))
        self.kpts = kpts

        if bandrange is None:
            bandrange = (0, calc.get_number_of_bands())
        self.bandrange = bandrange

        self.nspins = calc.get_number_of_spins()

        self.shape = (self.nspins, len(kpts), bandrange[1] - bandrange[0])
        # Just put fake ecut in order to use pairdensity object
        self.ecut = 1. / Hartree 
        self.ecutnb = 150 / Hartree

        vol = abs(np.linalg.det(self.calc.wfs.gd.cell_cv))
        self.vol = vol
        self.nbands = min(calc.get_number_of_bands(),
                          int(vol * (self.ecutnb)**1.5 * 2**0.5 / 3 / pi**2))

        # Get data from ground state
        self.ibzk_kc = calc.get_ibz_k_points()
        na, nb = self.bandrange
        kd = calc.wfs.kd
        self.pairDensity = PairDensity(calc, ecut=self.ecut*Hartree, 
                                       txt=self.fd, timer=self.timer)

        b1, b2 = self.bandrange

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
                                           ecut=self.ecut * Hartree,
                                           nbands=self.nbands,
                                           txt=self.fd,
                                           timer=self.timer,
                                           domega0=self.domega0 * Hartree,
                                           omega2=self.omega2 * Hartree
                                           )
        self.freqint.initialize(self)

        self.sigma_skn = None
        self.dsigma_skn = None
        self.qp_skn = None
        self.Qp_skn = None

        if dW_qw is None:
            try: 
                self.qqeh, self.wqeh, dW_qw = pickle.load(
                    open(filename + '_dW_qw.pckl', 'r'))
            except:
                dW_qw = self.calculate_W_QEH(structure, d, 
                                                              layer)
        else:
            self.qqeh = qqeh
            self.wqeh = None  # wqeh

        self.dW_qw = self.get_W_on_grid(dW_qw)

        self.complete = False
        self.nq = 0
        if self.load_state_file():
            if self.complete:
                print('Self-energy loaded from file', file=self.fd)
        
    def calculate_QEH(self):
        print('Calculating QEH self-energy contribution', file=self.fd)

        self.qpt_integration.sigma_iskn = np.zeros((1, ) + self.shape, 
                                                   complex)
        self.qpt_integration.dsigma_iskn = np.zeros((1, ) + self.shape, 
                                                    complex)

        mykpts = [self.pairDensity.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]
        
        kd = self.calc.wfs.kd
        
        kplusqdone_u = [set() for kpt in mykpts]

        prefactor = 1 / (2 * pi)**4

        # Find IBZ q-points included in the integration
        qd = self.qd
        bz1q_qc = to1bz(self.qpt_integration.qpts_qc, qd.symmetry.cell_cv)
        ibzqs = []
        for bzq_c in bz1q_qc:
            ibzq, iop, timerev, diff_c = qd.find_ibzkpt(qd.symmetry.op_scc,
                                                        qd.ibzk_kc,
                                                        bzq_c)
            if ibzq not in ibzqs:
                ibzqs.append(ibzq)

        # Loop over IBZ q-points
        nibzqs = len(ibzqs)
        for nq in range(self.nq, nibzqs):
            self.nq = nq
            self.save_state_file()
            
            ibzq = ibzqs[nq]
            q_c = self.qd.ibzk_kc[ibzq]

            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            print('Calculating contribution from IBZ q-pointq #%d/%d, q_c=%s' %
                  (nq + 1, len(ibzqs), qcstr), file=self.fd)
            
            rcell_cv = 2 * pi * np.linalg.inv(self.calc.wfs.gd.cell_cv).T
            q_abs = np.linalg.norm(np.dot(q_c, rcell_cv))
            dW_w = self.dW_qw[nq]
            dW_w = dW_w[:, np.newaxis, np.newaxis]
            L = abs(self.calc.wfs.gd.cell_cv[2, 2])
            d0 = self.d0

            Lcorr = 1
            dW_w *= L 

            nw = dW_w.shape[0]

            assert nw == len(self.freqint.omega_w), \
                ('Frequency grids doesnt match!')

            Wpm_w = np.zeros([2 * nw, 1, 1], dtype=complex)
            Wpm_w[:nw] = dW_w
            Wpm_w[nw:] = Wpm_w[0:nw]
            
            with self.timer('Hilbert transform'):
                self.freqint.htp(Wpm_w[:nw])
                self.freqint.htm(Wpm_w[nw:])
            # print('Hilbert transform done')
            
            qd = KPointDescriptor([q_c])
            pd0 = PWDescriptor(self.ecut, self.calc.wfs.gd, complex, qd)

            # modify pd0 by hand - only G=0 component is needed
            pd0.G_Qv = np.array([1e-17, 1e-17, 1e-17])[np.newaxis, :] 
            pd0.Q_qG = [np.array([0], dtype='int32')]
            pd0.ngmax = 1
            G_Gv = pd0.get_reciprocal_vectors()
            Q_aGii = self.pairDensity.initialize_paw_corrections(pd0)

            # Loop over all k-points in the BZ and find those that are related
            # to the current IBZ k-point by symmetry
            # Q1 = qd.ibz2bz_k[iq]

            Q1 = self.qd.ibz2bz_k[ibzq]

            Q2s = set()
            for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                if Q2 >= 0 and Q2 not in Q2s:
                    Q2s.add(Q2)
            
            for Q2 in Q2s:
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
                bz1q_qc = to1bz(self.qpt_integration.qpts_qc, 
                                kd.symmetry.cell_cv)
                bzqs = []
                for bzq, bzq_c in enumerate(bz1q_qc):
                    dq_c = bzq_c - Q_c
                    if np.allclose(dq_c.round(), dq_c):
                        bzqs.append(bzq)
                
                pos_av = np.dot(self.pairDensity.spos_ac, pd0.gd.cell_cv)
                M_vv = np.dot(pd0.gd.cell_cv.T,
                              np.dot(U_cc.T, np.linalg.inv(pd0.gd.cell_cv).T))
               
                for u1, kpt1 in enumerate(mykpts):
                    k1 = kd.bz2ibz_k[kpt1.K]
                    spin = kpt1.s
                    ik = self.kpts.index(k1)
                    # K2 will be in 1st BZ
                    K2 = kd.find_k_plus_q(Q_c, [kpt1.K])[0] 

                    # This k+q or symmetry related points have not been
                    # calculated yet.
                    kpt2 = self.pairDensity.get_k_point(spin, K2, 0,
                                                        self.nbands,
                                                        block=True)
                    
                    N_c = pd0.gd.N_c
                    i_cG = sign * np.dot(U_cc, 
                                         np.unravel_index(pd0.Q_qG[0], N_c))

                    k1_c = kd.bzk_kc[kpt1.K]
                    k2_c = kd.bzk_kc[K2]
                    # This is the q that connects K1 and K2 in the 1st BZ
                    q1_c = kd.bzk_kc[K2] - kd.bzk_kc[kpt1.K]

                    # G-vector that connects the full Q_c with q1_c
                    shift1_c = q1_c - sign * np.dot(U_cc, q_c)
                    assert np.allclose(shift1_c.round(), shift1_c)
                    shift1_c = shift1_c.round().astype(int)
                    shift_c = kpt1.shift_c - kpt2.shift_c - shift1_c
                    I_G = np.ravel_multi_index(i_cG + shift_c[:, None], 
                                               N_c, 'wrap') 

                    for n in range(kpt1.n2 - kpt1.n1):
                        n_mG = None
                        myid = 'k1=%dk2=%dn=%d' % (kpt1.K, kpt2.K, n)
                        if n_mG is None:
                            C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                                      for Qa_Gii, P1_ni in zip(Q_aGii, 
                                                               kpt1.P_ani)]
                            ut1cc_R = kpt1.ut_nR[n].conj()
                            n_mG = self.pairDensity.calculate_pair_densities(
                                ut1cc_R, C1_aGi, kpt2, pd0, I_G)

                        if sign == 1:
                            n_mG = n_mG.conj()

                        if np.allclose(q1_c, 0):
                            """ If we're at the Gamma point the G=0 
                            component of the pair density is a delta in 
                            the band index"""
                            n_mG[:, 0] = 0
                            m = n + kpt1.n1 - kpt2.n1
                            if 0 <= m < len(n_mG):
                                n_mG[m, 0] = 1.0
                            # Why is this necessary?

                        f_m = kpt2.f_n
                        deps_m = kpt1.eps_n[n] - kpt2.eps_n
                        nn = kpt1.n1 + n - self.bandrange[0]
                        n_mG *= Lcorr

                        sigma, dsigma = \
                            self.freqint.calculate_integration(n_mG[:],
                                                               deps_m, 
                                                               f_m, Wpm_w)

                        for bzq in bzqs:
                            self.qpt_integration.add_term(bzq, 0, spin, ik, nn,
                                                          prefactor * sigma,
                                                          prefactor * dsigma)

        sigma_iskn, dsigma_iskn = self.qpt_integration.integrate()
        
        self.world.sum(sigma_iskn)
        self.world.sum(dsigma_iskn)

        self.sigma_skn = sigma_iskn[0].real
        self.dsigma_skn = dsigma_iskn[0].real

        self.complete = True
        self.save_state_file()
    
        return self.sigma_skn, self.dsigma_skn

    def calculate_qp_correction(self):
        if self.filename:
            pckl = self.filename + '.sigma.pckl'
        else:
            pckl = 'sigma_qeh.pckl'

        if self.complete:
            print('Self-energy loaded from file', file=self.fd)
        else:
            self.calculate_QEH()  

        # Need GW result for renormalization factor
        gwdata = pickle.load(open(self.gwfile))   
        self.dsigmagw_skn = gwdata['dsigma_skn']
        self.sigmagw_skn = gwdata['sigma_skn']
        self.qpgw_skn = gwdata['qp_skn']
        nk = self.qpgw_skn.shape[1]
        if not self.sigma_skn.shape[1] == nk:
            self.sigma_skn = np.repeat(self.sigma_skn[:,:1,:], nk, axis=1)
            self.dsigma_skn = np.repeat(self.dsigma_skn[:,:1,:], nk, axis=1)
        self.Z_skn = 1. / (1 - self.dsigma_skn - self.dsigmagw_skn)

        self.qp_skn = self.Z_skn * self.sigma_skn
        
        return self.qp_skn * Hartree
    
    def calculate_qp_energies(self):
        # calculate 
        qp_skn = self.calculate_qp_correction() / Hartree
        self.Qp_skn = self.qpgw_skn + qp_skn
        self.save_state_file()
        return self.Qp_skn * Hartree

    def save_state_file(self, q=0):
        data = {'kpts': self.kpts,
                'bandrange': self.bandrange,
                'nbands': self.nbands,
                'freqint': (type(self.freqint).__module__ +
                            type(self.freqint).__name__),
                'qptint': (type(self.qpt_integration).__module__ +
                           type(self.qpt_integration).__name__),
                'qpts_qc': self.qpt_integration.qpts_qc,
                'last_q': self.nq,
                'complete': self.complete,
                'qptstate': self.qpt_integration.get_state(self.world),
                'sigma_skn': self.sigma_skn,
                'dsigma_skn': self.dsigma_skn,
                'qp_skn' : self.qp_skn,
                'Qp_skn' : self.Qp_skn}
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
                data['freqint'] == (type(self.freqint).__module__ +
                                    type(self.freqint).__name__) and
                data['qptint'] == (type(self.qpt_integration).__module__ +
                                   type(self.qpt_integration).__name__) and
                np.allclose(data['qpts_qc'], self.qpt_integration.qpts_qc)):
                self.nq = data['last_q']
                self.complete = data['complete']
                self.qpt_integration.load_state(data['qptstate'], self.world)
                self.complete = data['complete']
                self.sigma_skn = data['sigma_skn']
                self.dsigma_skn = data['dsigma_skn']
                return True
            else:
                return False    

    def get_W_on_grid(self, dW_qw):
        """This function transforms the screened potential W(q,w) to the 
        (q,w)-grid of the GW calculation. Also, W is integrated over
        a region around each q=0."""
    
        qd = self.qd
        bz1q_qc = to1bz(self.qpt_integration.qpts_qc, qd.symmetry.cell_cv)
        ibzqs = []
        for bzq_c in bz1q_qc:
            ibzq, iop, timerev, diff_c = qd.find_ibzkpt(qd.symmetry.op_scc,
                                                        qd.ibzk_kc,
                                                        bzq_c)
            if ibzq not in ibzqs:
                ibzqs.append(ibzq)

        q_cs = self.qd.ibzk_kc[ibzqs]

        rcell_cv = 2 * pi * np.linalg.inv(self.calc.wfs.gd.cell_cv).T
        q_vs = np.dot(q_cs, rcell_cv)
        q_grid = (q_vs**2).sum(axis=1)**0.5
        w_grid = self.freqint.omega_w

        wqeh = self.wqeh  # w_grid.copy() # self.qeh
        qqeh = self.qqeh
        sort = np.argsort(qqeh)
        qqeh = qqeh[sort]
        dW_qw = dW_qw[sort] 

        from scipy.interpolate import RectBivariateSpline
        yr = RectBivariateSpline(qqeh, wqeh, dW_qw.real, s=0)
        yi = RectBivariateSpline(qqeh, wqeh, dW_qw.imag, s=0)
        
        sort = np.argsort(q_grid)
        isort = np.argsort(sort)
        dWgw_qw = yr(q_grid[sort], w_grid) + 1j * yi(q_grid[sort], w_grid)
        dW_qw = yr(qqeh, w_grid) + 1j * yi(qqeh, w_grid)

        q_cut = q_grid[sort][1] / 2.
        q0 = np.array([q for q in qqeh if q <= q_cut])
        if len(q0) > 1:
            vol = np.pi * (q0[-1] + q0[1] / 2.)**2 
            if np.isclose(q0[0], 0):
                weight0 = np.pi * (q0[1] / 2.)**2 / vol
                c = (1 - weight0) / np.sum(q0)
                weights = c * q0        
                weights[0] = weight0
            else:
                c = 1 / np.sum(q0)
                weights = c * q0        

            dWgw_qw[0] = (np.repeat(weights[:, np.newaxis], len(w_grid), 
                                    axis=1) * dW_qw[:len(q0)]).sum(axis=0)

        dWgw_qw = dWgw_qw[isort]

        return dWgw_qw

    def calculate_W_QEH(self, structure, d, layer=0):
        from gpaw.response.qeh import Heterostructure, expand_layers
        structure = expand_layers(structure)
        self.w_grid = self.freqint.omega_w
        wmax = self.w_grid[-1]
        # qmax = (self.q_grid).max()

        # Single layer
        s = (np.insert(d, 0, d[0]) +
             np.append(d, d[-1])) / 2.
        d0 = s[layer]
        HS0 = Heterostructure(structure=[structure[layer]], 
                              d=[],
                              d0=d0,
                              wmax=wmax * Hartree,
                              # qmax=qmax / Bohr
                              ) 

        W0_qw = HS0.get_screened_potential()
        
        # Full heterostructure
        HS = Heterostructure(structure=structure, d=d,
                             wmax=wmax * Hartree,
                             #qmax=qmax / Bohr
                             ) 
        
        W_qw = HS.get_screened_potential(layer=layer)
    
        # Difference in screened potential:
        dW_qw = W_qw - W0_qw
        self.wqeh = HS.frequencies
        self.qqeh = HS.q_abs

        if self.world.rank == 0:
            pickle.dump((self.qqeh, self.wqeh, dW_qw), 
                        open(self.filename + '_dW_qw.pckl', 'w'))

        return dW_qw
