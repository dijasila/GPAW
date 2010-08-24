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
                 coufac=1.0, excfac=1.0,
                 uominres=1E-1, uomaxres=1E-12, uorelres=1.0E-2,
                 test=0, txt=None, rattle=-0.1, maxuoiter=30):
        """Self-Interaction Corrected (SIC) Functionals.

        xc: str
            Name of LDA functional which acts as
            a starting point for the construction of
            the self-interaction corrected functional

        finegrid: boolean
            Use fine grid for energy functional evaluations?

        coufac:
            Scaling factor for Hartree-functional

        excfac:
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
        self.coufac = coufac    
        self.excfac = excfac    
        self.finegrid = finegrid
        self.rattle = rattle    
        self.uominres = uominres
        self.uomaxres = uomaxres
        self.uorelres = uorelres
        self.maxuoiter = maxuoiter 
        self.dtype = dtype

        self.adderror = False      # add unit-opt. residual to basis-residual
        self.virt_SIC = False      # evaluate SIC for virtual orbitals
        self.new_coul = True       # use the ODD-coulomb solver 
        self.maxlsiter = 1         # maximum number of line-search steps
        self.maxcgiter = 2         # maximum number of CG-iterations
        self.lsinterp = True       # interpolate for minimum during line search
        self.act_SIC = True        # self-consistent SIC
        self.opt_SIC = True        # unitary optimization

        # debugging parameters
        self.debug = test          # debug level

        # initialization         
        self.init_SIC = True     # SIC functional has to be initialized?
        self.init_cou = True     # coulomb solver has to be initialized?
        self.active_SIC = False    # SIC is activated
        self.ESI = 0.0      # orbital dependent energy (SIC)
        self.RSI = 0.0      # residual of unitary optimization
        self.Sha = 0.0      
        self.Sxc = 0.0
        self.Stot = 0.0
        self.basiserror = 1E+20

        if isinstance(xc, str):
            xc = XC(xc)
        self.xc = xc
        XCFunctional.__init__(self, xc.name + '-SIC')
        self.hybrid = 0.0
        #self.type = xc.type

        # turn off unitary optimization if ODD functional is zero
        if self.coufac == 0.0 and self.excfac == 0.0:
            self.opt_SIC = False

    def initialize(self, density, hamiltonian, wfs):
        assert wfs.gamma
        self.xc.initialize(density, hamiltonian, wfs)
        self.timer = wfs.timer
        self.eigensolver = wfs.eigensolver
        self.spin_u = [SICSpin(kpt, 1, wfs.nbands, wfs.gd, float)
                       for kpt in wfs.kpt_u]

    def get_non_local_kinetic_corrections(self):
        if (self.act_SIC):
            Ecorr = 0.0
            for u in self.myblocks:
                q = self.myblocks.index(u)
                Ecorr = Ecorr + self.Ecorr_q[q]
            return self.wfs.kpt_comm.sum(Ecorr)
        else:
            return 0.0

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)
        if self.eigensolver.error < 0.1:
            for spin in self.spin_u:
                spin.unitary_optimization()
        return exc

class SICSpin:
    def __init__(self, kpt, nocc, nbands, gd, dtype):
        self.kpt = kpt
        self.nocc = nocc
        self.gd = gd
        self.dtype = dtype
        self.W_nm = None

    def initialize(self):
        assert self.gd.orthogonal
        Z_mmv = np.empty((self.nocc, self.nocc, 3), complex)
        for v in range(3):
            Z_mmv[:, :, v] = self.gd.wannier_matrix(self.kpt.psit_nG,
                                                    self.kpt.psit_nG, v, 1,
                                                    self.nocc)
        self.W_nm = np.identity(self.nocc)
        for iter in range(3 * self.nocc):
            _gpaw.localize(Z_mmv, self.W_nm)
        print self.W_nm

    def calculate_sic_matrixelements(self, gd, phit_mG, v_mG):
        M = self.nocc
        Htphit_mG = v_mG * phit_mG
        V_mm = np.zeros((M, M), dtype=phit_mG.dtype)
        gemm(gd.dv, phit_mG, Htphit_mG, 0.0, V_mm, 't')
        gd.comm.sum(V_mm)

        # Symmetrization of V and kappa-matrix:
        K_mm = 0.5 * (V_mm - V_mm.T.conj())
        V_mm = 0.5 * (V_mm + V_mm.T.conj())
        return V_mm, K_mm, np.vdot(K_mm, K_mm).real

    def update_optimal_states(self):
        self.phit_mG = self.gd.zeros(self.nocc)
        gemm(1.0, self.kpt.psit_nG, self.W_mm, 0.0, self.phit_mG)

    def unitary_optimization(self, maxiter=30):
        ESI_init = 0.0
        ESI      = 0.0
        dE       = 1e-16  
        # compensate the change in the basis functions during subspace
        # diagonalization and update the energy optimal states
        if self.W_nm is None:
            self.initialize()

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
        ESI=self.update_potentials([u])
        self.calculate_sic_matrixelements([u])
        ESI_init = ESI
        #
        # decide if unitary optimization is necessary
        # ------------------------------------------------------------------
        if self.normK_q[q] < 1E-10 or self.subiter != 0:
            #
            # no unitary optimization necessary
            # reason: optimal states already sufficiently converged
            # --------------------------------------------------------------
            dE2      = 1E-16
            dE       = 1E-16
            K        = 1E-16
            self.RSI_q[q] = 1E-16
            optstep  = 0.0
            lsiter   = 0
            failed   = False
            lsmethod = 'skipped'
            #
        elif self.npart_q[q] == 0 or self.npart_q[q] == 1:
            #
            # no unitary optimization necessary
            # reason: no or only one particle in current (spin) block
            # --------------------------------------------------------------
            dE2      = 1E-16
            dE       = 1E-16
            K        = 1E-16
            self.RSI_q[q] = 1E-16
            optstep  = 0.0
            lsiter   = 0
            failed   = False
            lsmethod = 'skipped'
            #
        elif not self.opt_SIC:
            #
            # no unitary optimization necessary
            # reason: deactivated unitary optimization
            # --------------------------------------------------------------
            dE2      = 1E-16
            dE       = 1E-16                
            K        = 1E-16
            self.RSI_q[q] = 1E-16
            optstep  = 0.0
            lsiter   = 0
            failed   = False
            lsmethod = 'skipped'
        else:
            #
            # optimize the unitary transformation
            # --------------------------------------------------------------
            #
            # allocate arrays for the search direction,
            # i.e., the (conjugate) gradient
            D_nn  = np.zeros_like(self.W_unn[q])
            D_old = np.zeros_like(self.W_unn[q])
            W_old = np.zeros_like(self.W_unn[q])
            K_nn  = np.zeros_like(self.W_unn[q])
            #
            for iter in range(maxiter):
                #
                # copy the initial unitary transformation and orbital
                # dependent energies
                W_old    = self.W_unn[q]
                K_nn     = self.K_unn[q]
                ESI_old  = ESI
                #
                # setup the steepest-descent/conjugate gradient
                # D_nn:  search direction
                # K_nn:  inverse gradient
                # G0  :  <K,D> (projected length of D along K)
                if (Gold!=0.0):
                    #
                    # conjugate gradient
                    G0        = np.sum(K_nn*K_nn.conj()).real
                    beta      = G0/Gold
                    Gold      = G0
                    D_nn      = K_nn + beta*D_old
                    G0        = np.sum(K_nn*D_nn.conj()).real
                else:
                    #
                    # steepest-descent
                    G0        = np.sum(K_nn*K_nn.conj()).real
                    Gold      = G0
                    D_nn      = K_nn
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
                        U_nn = matrix_exponential(D_nn, step)
                        self.W_unn[q] = np.dot(U_nn, W_old)
                        self.update_optimal_states([u],rotate_only=True)
                        E1 = self.update_potentials([u])
                        self.calculate_sic_matrixelements([u])
                        #
                        # projected length of the gradient at the new position
                        G1 = np.sum(self.K_unn[q]*D_nn.conj()).real
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
                        U_nn = matrix_exponential(D_nn,optstep)
                        self.W_unn[q] = np.dot(U_nn,W_old)
                        self.update_optimal_states([u],rotate_only=True)
                        E1=self.update_potentials([u])
                        if (abs(E1-E0)<prec):
                            self.calculate_sic_matrixelements([u])
                            ESI       = E1
                            optstep   = optstep
                            lsiter    = 0
                            maxlsiter = -1
                            updated   = True
                            minimum   = True
                            failed    = False
                            lsmethod  = 'CV-N'
                        if (E1<E0):
                            self.calculate_sic_matrixelements([u])
                            ESI       = E1
                            optstep   = optstep
                            lsiter    = 0
                            maxlsiter = -1
                            updated   = True
                            minimum   = True
                            failed    = False
                            lsmethod  = 'CV-S'
                        else:
                            self.K_unn[q] = K_nn
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
                        self.K_unn[q] = K_nn
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
                    U_nn = matrix_exponential(D_nn,step)
                    self.W_unn[q] = np.dot(U_nn,W_old)
                    self.update_optimal_states([u],rotate_only=True)
                    E1 = self.update_potentials([u])
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
                        U_nn = matrix_exponential(D_nn,step2)
                        self.W_unn[q] = np.dot(U_nn,W_old)
                        self.update_optimal_states([u],rotate_only=True)
                        E2=self.update_potentials([u])
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
                        self.W_unn[q,:] = W_old[:]
                        self.update_optimal_states([u],rotate_only=True)
                        ESI=self.update_potentials([u])
                        self.calculate_sic_matrixelements([u])
                    else:
                        U_nn = matrix_exponential(D_nn,optstep)
                        self.W_unn[q] = np.dot(U_nn,W_old)
                        self.update_optimal_states([u],rotate_only=True)
                        ESI=self.update_potentials([u])
                        self.calculate_sic_matrixelements([u])

                if (lsiter==maxlsiter-1):
                    optstep = step1
                    self.calculate_sic_matrixelements([u])
                #
                E0=ESI
                #
                # orthonormalize the energy optimal orbitals
                self.W_unn[q] = ortho(self.W_unn[q])
                self.RSI_q[q] = self.normK_q[q]
                #
                # logging
                dE2=max(abs(ESI-ESI_old),1.0E-16)
                dE =max(abs(ESI-ESI_init),1.0E-16)
                K  =max(self.RSI_q[q],1.0E-16)
                #
                # logging
                if self.finegd.comm.rank == 0 and self.logging:
                    log.write((" %3i  %10.5f %10.5f %5.1f %5.1f %5.1f %10.3f %3i %s\n" %
                               (iter+1,ESI*self.units,
                                (ESI-ESI_init)*self.units,
                                log10(dE),log10(dE2),log10(K),
                                optstep,lsiter+1,lsmethod)))
                    log.flush()
                #
                # iteration converged
                if K<basiserror*self.uorelres or K<self.uomaxres:
                    localerror = localerror + K
                    break
            #
        if self.finegd.comm.rank == 0 and self.logging:
            log.write("\n")
            log.flush()
        #
        #
        for u in self.myblocks:
            q=self.myblocks.index(u)
            self.setup_unified[q] = True
        #
        #print self.subiter
        if self.subiter==self.nsubiter:
            self.subiter=0
        else:
            self.subiter+=1
        
        

    def add_non_local_terms(self, psit_nG, Htpsit_nG, u):
        #
        # skip if SIC is not initialized or if feedback is
        # temporary or permanently disabled
        if self.init_SIC or not self.act_SIC or not self.active_SIC:
            return
        #
        q      = self.myblocks.index(u)
        f      = self.f_un[u]/(3-self.nspins)
        eps_n  = self.eps_un[u]
        nbands = psit_nG.shape[0]
        #
        #if (not self.unified_type==0):
            
        #
        # start the timer
        self.timer.start('ODD - basis action')
        #
        if (nbands==self.nbands and self.setup_unified[q]):
            #
            #q_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            #
            # get the unitary transformation from
            # energy optimal states |phi_k> to canonic states |k>
            W_nn  =  self.W_unn[q].T.conj().copy()
            #
            # action of the unitary invariant hamiltonian on the canonic
            # states (stored on psit_nG) is stored on Htpsit_nG
            #
            # compute matrixelements H^0_{ij} in the basis of the canonic
            # states psit_nG
            gemm(self.gd.dv,psit_nG,Htpsit_nG,0.0,self.H0_unn[q],'t')
            self.gd.comm.sum(self.H0_unn[q])
            #
            # add matrix-elements of the ODD-potential
            #
            # transform V_ij from energy optimal basis to the canonic
            # basis
            V_nn = np.dot(np.dot(W_nn,self.V_unn[q]),W_nn.T.conj())
            V_nn += self.H0_unn[q]
            #
            # separate occupied subspace from unoccupied subspace
            V_nn *= np.outer(f,f) + np.outer(1-f,1-f)
            #
            # diagonalize hamiltonian matrix 
            diagonalize(V_nn,eps_n)
            W_nn = np.dot(V_nn,W_nn.copy())
            #
            # store V_ij (basis of new canonic states)
            self.H0_unn[q]  =  np.dot(np.dot(W_nn,self.V_unn[q]),W_nn.T.conj())
            self.H0_unn[q] *=  np.outer(f,f)
            self.Ecorr_q[q] = -np.sum(np.diag(self.H0_unn[q]))*(3-self.nspins)
            #
            # new canonic states are now defined by |k> \mapsto V|k>
            #
            # action of ODD potential V_i|phi_i>
            self.v_unG[q,:] *= self.phit_unG[q,:]
            #
            # action of the canonic ODD potential
            gemm(1.0,self.v_unG[q],W_nn,0.0,self.Htphit_unG[q])
            #
            # setup new canonic states |k>
            gemm(1.0,  psit_nG,V_nn,0.0,self.phit_unG[q])
            #
            for i in range(nbands):
                self.phit_unG[q,i,:]   *= f[i]
                self.Htphit_unG[q,i,:] *= f[i]
            
            #print self.H0_unn[q]
            #
            q_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            H_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            #
            gemm(self.gd.dv,self.phit_unG[q]  ,psit_nG,0.0,q_nn,'t')
            gemm(self.gd.dv,self.Htphit_unG[q],psit_nG,0.0,H_nn,'t')
            self.gd.comm.sum(q_nn)
            self.gd.comm.sum(H_nn)
            #
            V_nn  = H_nn - np.dot(q_nn,self.H0_unn[q])
            #
            gemm(+1.0,self.phit_unG[q]  ,V_nn, 1.0,Htpsit_nG)
            gemm(+1.0,self.Htphit_unG[q],q_nn, 1.0,Htpsit_nG)
            #
            self.setup_unified[q]=False
            #
        else:
            #
            q_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            H_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            #
            gemm(self.gd.dv,self.phit_unG[q]  ,psit_nG,0.0,q_nn,'t')
            gemm(self.gd.dv,self.Htphit_unG[q],psit_nG,0.0,H_nn,'t')
            self.gd.comm.sum(q_nn)
            self.gd.comm.sum(H_nn)
            #
            V_nn  = H_nn - np.dot(q_nn,self.H0_unn[q])            
            #
            gemm(+1.0,self.phit_unG[q]  ,V_nn, 1.0,Htpsit_nG)
            gemm(+1.0,self.Htphit_unG[q],q_nn, 1.0,Htpsit_nG)
            #
        self.timer.stop('ODD - basis action')

    def get_sic_energy(self):
        #
        return self.Stot*Hartree

    def update_potentials(self):
        nt_G = np.gd.empty()
        nt_sg = np.finegd.zeros(2)
        nt_g = nt_sg[0]
        vt_sg = np.finegd.zeros(2)
        vt_g = vt_sg[0]
        for m, phit_G in enumerate(self.phit_mG):
            nt_G[:] = phit_G**2
            Nt = self.gd.integrate(nt_G)
            self.interpolator.apply(nt_G, nt_g)
            Ntfine = self.finegd.integrate(nt_g)
            nt_g  *= Nt / Ntfine
            self.timer.start('SIC-XC')
            vt_sg[0] = 0.0
            exc = self.xc.calculate(self.finegd, nt_sg, vt_sg)
            exc_m[m] = -self.excfac * exc
            vt_g *= -self.excfac
            self.timer.stop('SIC-XC')

            self.timer.start('Hartree')
            self.interpolator.apply(self.vt_mG[m], vt_sg[1])
                        self.solve_poisson_charged(v_cou_g,nt_g,self.pos_un[:,u,n],
                                                   self.phi_gauss, self.rho_gauss,
                                                   zero_initial_phi=self.init_cou)
                        #
                    #
                    #
                    # compose the energy density and add potential
                    # contributions to orbital dependent potential.
                    e_g         = nt_g * v_cou_g
                    Sha_un[u,n] = -0.5*self.coufac*self.finegd.integrate(e_g)
                    v_g[:]     -= self.coufac * occ * v_cou_g[:]
                    #Sha_un[u,n] = -0.5 * self.coufac * \
                    #               np.sum(e_g.ravel()) * self.finegd.dv
                    #
                    # add PAW corrections to the self-Hartree-Energy
                    ##Sha_un[u, n] += self.gd.comm.sum(self.paw_sic_hartree_energy(D_ap, dH_ap))
                    #Sha_un[u, n] += self.paw_sic_hartree_energy(D_ap, dH_ap)
                #
                self.timer.stop('Hartree')
                #
                # apply the localization mask to the potentials
                # and set potential to zero for metallic and unoccupied states
                # -------------------------------------------------------------
                if (self.periodic):
                    v_g[:] = v_g[:]*self.mask[:]
                #
                # restrict to the coarse-grid
                # -------------------------------------------------------------
                if self.finegrid:
                    hamiltonian.restrictor.apply(v_g    , v_unG[q,n])
                    hamiltonian.restrictor.apply(v_cou_g, v_cou_unG[q,n])
                else:
                    v_unG[q,n,:]     = v_g[:]
                    v_cou_unG[q,n,:] = v_cou_g[:]
                #
                #
                # accumulate total SIC-energies and number of occupied states
                # in block
                # -------------------------------------------------------------
                #nocc_u[q]    = nocc_u[q]   + occ
                self.npart_q[q] = self.npart_q[q] + occ
                #
            #self.npart_q[q] = int(nocc_u[q]+0.5)
            self.npart_q[q] = int(self.npart_q[q] + 0.5)
        #
        Stot = 0.0
        for u in myblocks:
            Stot_un[u,:] = Sxc_un[u,:] + Sha_un[u,:]
            Stot = Stot + np.sum(Stot_un[u,:]*self.f_un[u,:])
        self.timer.stop('ODD - potentials')
        return Stot

    def paw_sic_hartree_energy(self, D_ap, dH_ap):
        """Calculates the PAW corrections for the SIC Hartree energy.

        returns the PAW correction to the SIC energy and adds corrections
        to the derivatives dH_ap."""

        setups      = self.wfs.setups
        dE = 0.0
        for a, D_p in D_ap.items():
            M_pp = setups[a].M_pp
            dE += np.dot(D_p, np.dot(M_pp, D_p))
            dH_ap[a] -= 2 * self.coufac * np.dot(M_pp, D_p)

        return -self.coufac * dE



    def update_unitary_transformation(self,blocks=[]):
        #
        test    = self.debug
        nbands  = self.nbands
        #
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        # compensate for the changes to the orbitals due to
        # last subspace diagonalization
        for u in myblocks:
            #
            # get the local index of the block u (which is
            q    = self.myblocks.index(u)
            f    = self.wfs.kpt_u[q].f_n
            mask = np.outer(f,f) 
            #
            # account for changes to the canonic states
            # during diagonalization of the unitary invariant hamiltonian
            W_nn = self.wfs.kpt_u[q].W_nn
            #W_nn *= mask
            #for n in range(nbands):
            #    if (mask[n,n]==0.0):
            #        W_nn[n,n]=1.0
            #W_nn=ortho(W_nn)
            #
            if (1==0):
                if self.optcmplx:
                    self.Tmp2_nn = self.W_unn[q].real
                    gemm(1.0,W_nn,self.Tmp2_nn,0.0,self.Tmp_nn)
                    self.W_unn[q]=self.Tmp_nn
                    self.Tmp2_nn = self.W_unn[q].imag
                    print self.Tmp2_nn
                    gemm(1.0,W_nn,self.Tmp2_nn,0.0,self.Tmp_nn)
                    self.W_unn[q] += 1j*self.Tmp_nn
                else:
                    gemm(1.0,W_nn,self.W_unn[q],0.0,self.O_unn[q])
                    self.W_unn[q]=self.O_unn[q]
                    #self.W_unn[q]=np.dot(self.W_unn[q],W_nn.T)
            else:
                self.W_unn[q] = np.dot(self.W_unn[q],W_nn)
            #
            # adjust transformation for the occupied states
            self.W_unn[q] = self.W_unn[q]*mask
            for n in range(nbands):
                if (mask[n,n]==0.0):
                    self.W_unn[q,n,n]=1.0
            #
            # reset to unit matrix
            self.wfs.kpt_u[q].W_nn=np.eye(self.nbands)
            #
            # orthonormalize W
            self.W_unn[q]=ortho(self.W_unn[q])
            #            
                       

    def solve_poisson_charged(self,phi,rho,pos,phi0,rho0,
                              zero_initial_phi=False):
        #
        #
        #
        # monopole moment
        q1    = self.finegd.integrate(rho)/np.sqrt(4 * np.pi)
        q0    = self.finegd.integrate(rho0)/np.sqrt(4 * np.pi)
        q     = q1/q0
        #
        self.rho_n     = rho - q * rho0
        #
        if zero_initial_phi==True:
            phi[:]     = 0.0
        else:
            axpy(-q, phi0, phi) # phi -= q * self.phi_gauss
        #
        # Determine potential from neutral density using standard solver
        niter = self.psolver.solve_neutral(phi, self.rho_n)
        #
        # correct error introduced by removing monopole
        axpy(+q, phi0, phi)      # phi += q * self.phi_gauss
        #
        return niter

