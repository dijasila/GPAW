"""Module defining  ``Eigensolver`` classes."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.utilities import unpack
from gpaw.mpi import run


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

    def __init__(self, keep_htpsit=True):
        Eigensolver.__init__(self, keep_htpsit)

    def initialize(self, wfs):
        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.subspace_diagonalize(hamiltonian, wfs, kpt)

        self.timer.start('RMM-DIIS')
        if self.keep_htpsit:
            R_nG = self.Htpsit_nG
            wfs.calculate_residuals(hamiltonian, kpt, kpt.eps_n, kpt.psit_nG,
                                    R_nG, kpt.P_ani,
                                    apply_hamiltonian=False)

        B = 1
        dR_bG = wfs.gd.empty(B)
        error = 0.0
        assert B == 1
        for n1 in range(0, wfs.bd.mynbands, B):
            n2 = min(n1 + B, wfs.bd.mynbands)
            if self.keep_htpsit:
                R_bG = R_nG[n1:n2]
            else:
                R_bG = wfs.gd.empty(B)
                wfs.calculate_residuals(hamiltonian, kpt,
                                        kpt.eps_n[n1:n2], kpt.psit_nG[n1:n2],
                                        R_bG)

            for n in range(n1, n2):
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if wfs.bd.global_index(n) < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * np.vdot(R_bG[n - n1], R_bG[n - n1]).real

            # Precondition the residual:
            self.timer.start('precondition')
            pR_bG = self.preconditioner(R_bG, kpt)
            self.timer.stop('precondition')

            # Calculate the residual of pR_G, dR_G = (H - e S) pR_G:
            wfs.calculate_residuals(hamiltonian, kpt, kpt.eps_n[n1:n2],
                                    pR_bG, dR_bG, approximate=True,
                                    n_x=range(n1, n2))
            
            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR = self.gd.comm.sum(np.vdot(R_bG, dR_bG).real)
            dRdR = self.gd.comm.sum(np.vdot(dR_bG, dR_bG).real)

            lam = -RdR / dRdR
            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            R_bG *= 2.0 * lam
            axpy(lam**2, dR_bG, R_bG)  # R_G += lam**2 * dR_G
            self.timer.start('precondition')
            kpt.psit_nG[n1:n2] += self.preconditioner(R_bG, kpt)
            self.timer.stop('precondition')
            
        self.timer.stop('RMM-DIIS')
        error = self.gd.comm.sum(error)
        return error
    
