"""Module defining  ``Eigensolver`` classes."""

import numpy as np

from gpaw.utilities.blas import axpy, dotc
from gpaw.utilities.mblas import multi_axpy, multi_scal, multi_dotc
from gpaw.eigensolvers.eigensolver import Eigensolver



class RMM_DIIS(Eigensolver):
    """RMM-DIIS eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated

    Solution steps are:

    * Subspace diagonalization
    * Calculation of residuals
    * Improvement of wave functions:  psi' = psi + lambda PR + lambda PR'
    * Orthonormalization"""

    def __init__(self, keep_htpsit=True, blocksize=10, cuda=False):
        Eigensolver.__init__(self, keep_htpsit, blocksize, cuda)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        psit_nG, R_nG = self.subspace_diagonalize(hamiltonian, wfs, kpt)

        self.timer.start('RMM-DIIS')
        if self.keep_htpsit:
            self.calculate_residuals(kpt, wfs, hamiltonian, psit_nG,
                                     kpt.P_ani, kpt.eps_n, R_nG)
            
        def integrate(a_xG, b_xG):
            if self.cuda:
                return multi_dotc(a_xG, b_xG).real * wfs.gd.dv
            else:
                return[np.real(wfs.integrate(a_G, b_G, global_integral=False))
                              for a_G, b_G in zip(a_xG, b_xG)]


        comm = wfs.gd.comm
        B = self.blocksize
        error = 0
        dR_xG = wfs.empty(B, q=kpt.q)
        #dR_nG = wfs.empty(wfs.bd.mynbands, q=kpt.q)
        #dR_nG = kpt.psit_nG
        #if self.cuda:
        #    dR_nG = self.operator.work1_xG_gpu
        #else:
        #    dR_nG = self.operator.work1_xG
                    
        P_axi = wfs.pt.dict(B)

        weight=np.empty(wfs.bd.mynbands,float)
        if kpt.f_n is None:
            weight[:] = kpt.weight
        else:
            weight[:] = kpt.f_n
            
        if self.nbands_converge != 'occupied':
            for n in range(0, wfs.bd.mynbands):
                if wfs.bd.global_index(n) < self.nbands_converge:
                    weight[n] = kpt.weight
                else:
                    weight[n] = 0.0

        if self.keep_htpsit:
            error = sum(weight * integrate(R_nG, R_nG))

        for n1 in range(0, wfs.bd.mynbands, B):
            n2 = n1 + B
            if n2 > wfs.bd.mynbands:
                n2 = wfs.bd.mynbands
                B = n2 - n1
                P_axi = dict((a, P_xi[:B]) for a, P_xi in P_axi.items())
                dR_xG = dR_xG[:B]            
                
            n_x = range(n1, n2)
            psit_xG = psit_nG[n1:n2]
            #dR_xG = dR_nG[n1:n2]            
            if self.keep_htpsit:
                R_xG = R_nG[n1:n2]
            else:
                R_xG = wfs.empty(B, q=kpt.q)
                wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_xG, R_xG)
                wfs.pt.integrate(psit_xG, P_axi, kpt.q)
                self.calculate_residuals(kpt, wfs, hamiltonian, psit_xG,
                                         P_axi, kpt.eps_n[n_x], R_xG, n_x)
                
                error += sum(weight[n1:n2] * integrate(R_xG, R_xG))
            # Precondition the residual:
            self.timer.start('precondition')
            ekin_x = self.preconditioner.calculate_kinetic_energy(
                psit_xG, kpt)
            dpsit_xG = self.preconditioner(R_xG, kpt, ekin_x)
            self.timer.stop('precondition')

            # Calculate the residual of dpsit_G, dR_G = (H - e S) dpsit_G:
            wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, dpsit_xG, dR_xG)
            self.timer.start('projections')
            wfs.pt.integrate(dpsit_xG, P_axi, kpt.q)
            self.timer.stop('projections')
            self.calculate_residuals(kpt, wfs, hamiltonian, dpsit_xG,
                                     P_axi, kpt.eps_n[n_x], dR_xG, n_x,
                                     calculate_change=True)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR_x=np.array(integrate(R_xG, dR_xG))
            dRdR_x=np.array(integrate(dR_xG, dR_xG))
            comm.sum(RdR_x)
            comm.sum(dRdR_x)
            lam_x = -RdR_x / dRdR_x
            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            multi_scal(2.0 * lam_x, R_xG)
            multi_axpy(lam_x**2, dR_xG, R_xG)
            self.timer.start('precondition')
            psit_G = psit_nG[n1:n2]
            psit_G += self.preconditioner(R_xG, kpt, ekin_x)
            self.timer.stop('precondition')        

        self.timer.stop('RMM-DIIS')
        error = comm.sum(error)
        return error, psit_nG
