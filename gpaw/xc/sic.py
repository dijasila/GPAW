"""Perdew-Zunger SIC to DFT functionals (currently only LDA)

Self-consistent minimization of self-interaction corrected
LDA functionals (Perdew-Zunger)
"""

from math import pi, cos, sin, log10, exp, atan2

import numpy as np
from ase.units import Bohr, Hartree

import gpaw.mpi as mpi
from gpaw.utilities import pack, unpack
from gpaw.utilities.blas import axpy, gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.xc import XC
from gpaw.xc.functional import XCFunctional
from gpaw.utilities.timing import Timer
from gpaw.poisson import PoissonSolver
from gpaw.atom.generator import Generator, parameters
from gpaw.lfc import LFC
from gpaw.hgh import NullXCCorrection
from gpaw import extra_parameters
import _gpaw


class SIC(XCFunctional):
    def __init__(self, xc='LDA', finegrid=True, dtype=float,
                 coulomb_factor=1.0, xc_factor=1.0,
                 uominres=1E-1, uomaxres=1E-12, uorelres=1.0E-2,
                 test=0, txt=None, rattle=-0.1, maxuoiter=30):
        """Self-Interaction Corrected (SIC) Functionals.

        xc: str
            Name of LDA functional which acts as
            a starting point for the construction of
            the self-interaction corrected functional

        finegrid: boolean
            Use fine grid for energy functional evaluations?

        coulomb_factor:
            Scaling factor for Hartree-functional

        xc_factor:
            Scaling factor for xc-functional

        uominres:
            Minimum residual before unitary optimization starts

        uomaxres:
            Target accuracy for unitary optimization
            (absolute variance)

        uorelres:
            Target accuracy for unitary optimization
            (rel. to basis residual)

        maxuoiter:
            Maximum number of unitary optimization steps

        test:
            debug level

        txt:
            log file for unitary optimization

        rattle:
            perturbation to the initial states

        """

        # parameters
        self.coulomb_factor = coulomb_factor    
        self.xc_factor = xc_factor    

        self.finegrid = finegrid
        self.rattle = rattle    
        self.uominres = uominres
        self.uomaxres = uomaxres
        self.uorelres = uorelres
        self.maxuoiter = maxuoiter 

        self.dtype = dtype

        self.adderror = False      # add unit-opt. residual to basis-residual
        self.maxlsiter = 1         # maximum number of line-search steps
        self.maxcgiter = 2         # maximum number of CG-iterations
        self.lsinterp = True       # interpolate for minimum during line search

        # initialization         
        self.basiserror = 1E+20

        if isinstance(xc, str):
            xc = XC(xc)
        self.xc = xc
        XCFunctional.__init__(self, xc.name + '-SIC')
        self.hybrid = 0.0

    def initialize(self, density, hamiltonian, wfs):
        assert wfs.gamma
        self.xc.initialize(density, hamiltonian, wfs)
        self.timer = wfs.timer
        self.eigensolver = wfs.eigensolver
        self.kpt_comm = wfs.kpt_comm

        self.spin_s = {}
        for kpt in wfs.kpt_u:
            self.spin_s[kpt.s] = SICSpin(kpt, float, 1, self.xc,
                                         self.xc_factor, self.coulomb_factor,
                                         density, hamiltonian, wfs)

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)

        self.esic = 0.0
        self.ekin = 0.0
        for spin in self.spin_s.values():
            if spin.kpt.psit_nG is not None:
                desic, dekin = spin.calculate()
                self.esic += desic
                self.ekin += dekin
        self.esic = self.kpt_comm.sum(self.esic)
        self.ekin = self.kpt_comm.sum(self.ekin)
        return exc + self.esic

    def add_correction(self, kpt, psit_xG, Htpsit_xG, approximate, n_x):
        self.spin_s[kpt.s].add_correction(psit_xG, Htpsit_xG, approximate, n_x)

    def rotate(self, kpt, U_nn):
        self.spin_s[kpt.s].rotate(U_nn)


class SICSpin:
    def __init__(self, kpt, dtype, nocc, xc, xc_factor, coulomb_factor,
                 density, hamiltonian, wfs):
        self.kpt = kpt
        self.dtype = dtype
        self.nocc = nocc
        self.xc = xc
        self.xc_factor = xc_factor
        self.coulomb_factor = coulomb_factor

        self.gd = wfs.gd
        self.finegd = density.finegd
        self.interpolator = density.interpolator
        self.restrictor = hamiltonian.restrictor
        self.poissonsolver = hamiltonian.poisson
        self.nspins = wfs.nspins
        self.timer = wfs.timer

        self.W_mn = None
        self.vt_mG = None
        self.exc_m = None
        self.ecoulomb_m = None

    def initialize(self):
        assert self.gd.orthogonal
        Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
        for v in range(3):
            Z_mmv[:, :, v] = self.gd.wannier_matrix(self.kpt.psit_nG,
                                                    self.kpt.psit_nG, v, 1,
                                                    self.nocc)
        self.gd.comm.sum(Z_mmv)
        W_nm = np.identity(self.nocc)
        localization = 0.0
        for iter in range(30):
            loc = _gpaw.localize(Z_mmv, W_nm)
            print iter, loc
            if loc - localization < 1e-4:
                break
            localization = loc

        print W_nm
        self.W_mn = W_nm.T.copy()

        spos_mc = -np.angle(Z_mmv.diagonal()).T / (2 * pi)
        print np.dot(spos_mc % 1.0, self.gd.cell_cv)

    def calculate_sic_matrixelements(self):
        Htphit_mG = self.vt_mG * self.phit_mG
        V_mm = np.zeros((self.nocc, self.nocc), dtype=self.dtype)
        gemm(self.gd.dv, self.phit_mG, Htphit_mG, 0.0, V_mm, 't')
        
        self.gd.comm.sum(V_mm)

        # Symmetrization of V and kappa-matrix:
        K_mm = 0.5 * (V_mm - V_mm.T.conj())
        V_mm = 0.5 * (V_mm + V_mm.T.conj())

        self.ekin = -np.trace(V_mm) * (3 - self.nspins) / self.gd.comm.size

        return V_mm, K_mm, np.vdot(K_mm, K_mm).real

    def update_optimal_states(self):
        self.phit_mG = self.gd.zeros(self.nocc)
        gemm(1.0, self.kpt.psit_nG[:self.nocc], self.W_mn, 0.0, self.phit_mG)
        self.P_ami = {}
        for a, P_ni in self.kpt.P_ani.items():
            self.P_ami[a] = np.dot(self.W_mn, P_ni[:self.nocc])

    def update_potentials(self):
        self.timer.start('ODD-potentials')
        nt_G = self.gd.empty()
        nt_sg = self.finegd.empty(2)
        nt_sg[1] = 0.0
        vt_sg = self.finegd.empty(2)

        if self.vt_mG is None:
            self.vt_mG = self.gd.zeros(self.nocc)
            self.exc_m = np.empty(self.nocc)
            self.ecoulomb_m = np.empty(self.nocc)
            
        for m, phit_G in enumerate(self.phit_mG):
            nt_G[:] = phit_G**2
            Nt = self.gd.integrate(nt_G)
            self.interpolator.apply(nt_G, nt_sg[0])
            Ntfine = self.finegd.integrate(nt_sg[0])
            nt_sg[0]  *= Nt / Ntfine
            self.timer.start('XC')
            vt_sg[:] = 0.0
            exc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
            self.exc_m[m] = -self.xc_factor * exc
            vt_sg[0] *= -self.xc_factor
            self.timer.stop('XC')

            self.timer.start('Hartree')
            vt_sg[1] = 0.0
            self.poissonsolver.solve(vt_sg[1], nt_sg[0], zero_initial_phi=True)
            ecoulomb = 0.5 * self.finegd.integrate(nt_sg[0] * vt_sg[1])
            self.ecoulomb_m[m] = -self.coulomb_factor * ecoulomb
            vt_sg[0] -= self.coulomb_factor * vt_sg[1]
            self.timer.stop('Hartree')

            self.restrictor.apply(vt_sg[0], self.vt_mG[m])
        self.timer.stop('ODD-potentials')
        self.esic = (self.exc_m.sum() +
                     self.ecoulomb_m.sum()) * (3 - self.nspins)

    def add_correction(self, psit_xG, Htpsit_xG, approximate, n_x):
        if self.W_mn is None:
            return

        if approximate:
            assert len(n_x) == 1
            n = n_x[0]
            if n < self.nocc:
                Htpsit_xG += np.dot(self.vt_mG.T,
                                    self.W_mn[:, n]**2).T * psit_xG
        else:
            Htpsit_xG[:self.nocc] += np.dot((self.vt_mG * self.phit_mG).T,
                                            self.W_mn).T

    def rotate(self, U_nn):
        if self.W_mn is not None:
            self.W_mn = np.dot(self.W_mn, U_nn[:self.nocc, :self.nocc].T)
            self.phit_mG = None

    def calculate(self):
        if self.W_mn is None:
            self.initialize()
        self.unitary_optimization()
        return self.esic, self.ekin

    def unitary_optimization(self, maxiter=30):
        ESI_init = 0.0
        ESI      = 0.0
        dE       = 1e-16  
        # compensate the change in the basis functions during subspace
        # diagonalization and update the energy optimal states

        #U_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)
        #O_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)

        optstep  = 0.0
        Gold     = 0.0
        cgiter   = 0
        #
        epsstep  = 0.001  # 0.005
        dltstep  = 0.1    # 0.1
        prec     = 1E-6
        #
        #
        # get the initial ODD potentials/energy/matrixelements
        self.update_optimal_states()
        self.update_potentials()
        ESI = self.esic
        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
        ESI_init = ESI

        if norm < 1E-10 or self.nocc <= 1:
            return
        #
        # optimize the unitary transformation
        # --------------------------------------------------------------
        #
        # allocate arrays for the search direction,
        # i.e., the (conjugate) gradient
        D_old_mm = np.zeros_like(self.W_mn)
        #
        for iter in range(maxiter):
            #
            # copy the initial unitary transformation and orbital
            # dependent energies
            W_old_mn    = self.W_mn.copy()
            ESI_old  = ESI
            #
            # setup the steepest-descent/conjugate gradient
            # D_nn:  search direction
            # K_nn:  inverse gradient
            # G0  :  <K,D> (projected length of D along K)
            if (Gold!=0.0):
                #
                # conjugate gradient
                G0        = np.sum(K_mm*K_mm.conj()).real
                beta      = G0/Gold
                Gold      = G0
                D_mm      = K_mm + beta*D_old_mm
                G0        = np.sum(K_mm*D_mm.conj()).real
            else:
                #
                # steepest-descent
                G0        = np.sum(K_mm*K_mm.conj()).real
                Gold      = G0
                D_mm      = K_mm
            #
            updated  = False
            minimum  = False
            failed   = True
            E0       = ESI
            #
            # try to estimate optimal step-length from change in length
            # of the gradient (force-only)
            # ----------------------------------------------------------
            if (epsstep!=0.0):
                #
                # infinitesimal steepest descent
                step = max(min(epsstep/np.sqrt(abs(G0)),1.0),1E-3)
                while (True):
                    U_mm = matrix_exponential(D_mm, step)
                    self.W_mn = np.dot(U_mm, W_old_mn)
                    self.update_optimal_states()
                    E1 = self.update_potentials()
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                    #
                    # projected length of the gradient at the new position
                    G1 = np.sum(K_mm*D_mm.conj()).real
                    #
                    if (abs(E1-E0)<prec):
                        #
                        eps_works = True
                        Eeps      = E1
                    elif (E1<E0):
                        #
                        # trial step reduced energy
                        eps_works = True
                        Eeps      = E1
                    else:
                        #
                        # scale down trial step
                        eps_works = False
                        optstep   = 0.0
                        break
                        #step = 0.5*step
                        #if step<1.0:
                        #    eps_works=False
                        #    break
                        #print 'scaling down steplength', step
                        #continue
                    #
                    # compute the optimal step size
                    optstep = step/(1.0-G1/G0)
                    #
                    if (eps_works):
                        break
                    #
                #print 'trial step: ',step,optstep,E1-E0,G0,G1
                #
                # decide on the method for stepping
                if (optstep > 0.0):
                    #
                    # convex region -> force only estimate for minimum
                    U_mm = matrix_exponential(D_mm,optstep)
                    self.W_mn = np.dot(U_mm,W_old_mn)
                    self.update_optimal_states()
                    E1=self.update_potentials()
                    if (abs(E1-E0)<prec):
                        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                        ESI       = E1
                        optstep   = optstep
                        lsiter    = 0
                        maxlsiter = -1
                        updated   = True
                        minimum   = True
                        failed    = False
                        lsmethod  = 'CV-N'
                    if (E1<E0):
                        V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                        ESI       = E1
                        optstep   = optstep
                        lsiter    = 0
                        maxlsiter = -1
                        updated   = True
                        minimum   = True
                        failed    = False
                        lsmethod  = 'CV-S'
                    else:
                        #self.K_unn[q] = K_nn
                        ESI       = E0
                        step      = optstep
                        optstep   = 0.0
                        lsiter    = 0
                        maxlsiter = self.maxlsiter
                        updated   = False
                        minimum   = False
                        failed    = True
                        lsmethod  = 'CV-F-CC'
                else:
                    #self.K_unn[q] = K_nn
                    ESI       = E0
                    step      = optstep
                    optstep   = 0.0
                    lsiter    = 0
                    maxlsiter = self.maxlsiter
                    updated   = False
                    minimum   = False
                    failed    = True
                    lsmethod  = 'CC'
                #
            if (optstep==0.0):
                #
                # we are in the concave region or force-only estimate failed,
                # just follow the (conjugate) gradient
                step = dltstep * abs(step)
                #print step
                U_mm = matrix_exponential(D_mm,step)
                self.W_mn = np.dot(U_mm,W_old_mn)
                self.update_optimal_states()
                E1 = self.update_potentials()
                #
                #
                if (abs(E1-E0)<prec):
                    ESI       = E1
                    optstep   = 0.0
                    updated   = False
                    minimum   = True
                    failed    = True
                    lsmethod  = lsmethod+'-DLT-N'
                    maxlsiter = -1
                elif (E1<E0):
                    ESI       = E1
                    optstep   = step
                    updated   = True
                    minimum   = False
                    failed    = False
                    lsmethod  = lsmethod+'-DLT'
                    maxlsiter = self.maxlsiter
                elif (eps_works):
                    ESI       = Eeps
                    E1        = Eeps
                    step      = epsstep
                    updated   = False
                    minimum   = False
                    failed    = False
                    lsmethod  = lsmethod+'-EPS'
                    maxlsiter = self.maxlsiter
                else:
                    optstep   = 0.0
                    updated   = False
                    minimum   = False
                    failed    = True
                    lsmethod  = lsmethod+'-EPS-failed'
                    maxlsiter = -1
                #
                G       = (E1-E0)/step
                step0   = 0.0
                step1   = step
                step2   = 2*step
                #
                for lsiter in range(maxlsiter):
                    #
                    # energy at the new position
                    U_mm = matrix_exponential(D_mm,step2)
                    self.W_mn = np.dot(U_mm,W_old_mn)
                    self.update_optimal_states()
                    E2=self.update_potentials()
                    G  = (E2-E1)/(step2-step1)
                    #
                    #print lsiter,E2,G,step2,step
                    #
                    if (G>0.0):
                        if self.lsinterp:
                            a= E0/((step2-step0)*(step1-step0)) \
                             + E2/((step2-step1)*(step2-step0)) \
                             - E1/((step2-step1)*(step1-step0))
                            b=(E2-E0)/(step2-step0)-a*(step2+step0)
                            if (a<step**2):
                                optstep = 0.5*(step0+step2)
                            else:
                                optstep =-0.5*b/a
                            updated  = False
                            minimum  = True
                            break
                        else:
                            optstep  = step1
                            updated  = False
                            minimum  = True
                            break
                    #
                    elif (G<0.0):
                        optstep = step2
                        step0   = step1
                        step1   = step2
                        step2   = step2 + step
                        E0      = E1
                        E1      = E2
                        ESI     = E2
                        updated = True
                        minimum = False
            #
            if (cgiter!=0):
                lsmethod = lsmethod + ' CG'
            #
            if (cgiter>=self.maxcgiter or not minimum):
                Gold        = 0.0
                cgiter      = 0
            else:
                cgiter      = cgiter + 1
                D_old[:,:]  = D_nn[:,:]
            #
            # update the energy and matrixelements of V and Kappa
            # and accumulate total residual of unitary optimization
            if (not updated):
                if (optstep==0.0):
                    self.W_mn[:] = W_old_mn
                    self.update_optimal_states()
                    ESI=self.update_potentials()
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()
                else:
                    U_mm = matrix_exponential(D_mm,optstep)
                    self.W_mn = np.dot(U_mm,W_old_mn)
                    self.update_optimal_states()
                    ESI=self.update_potentials()
                    V_mm, K_mm, norm = self.calculate_sic_matrixelements()

            if (lsiter==maxlsiter-1):
                optstep = step1
                V_mm, K_mm, norm = self.calculate_sic_matrixelements()
            #
            E0=ESI
            #
            # orthonormalize the energy optimal orbitals
            self.W_mn = ortho(self.W_mn)
            K  =max(norm, 1.0e-16)
            if K<basiserror*self.uorelres or K<self.uomaxres:
                break
