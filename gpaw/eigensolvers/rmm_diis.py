"""Module defining  ``Eigensolver`` classes."""

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.mpi import run
from gpaw import sl_inverse_cholesky


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

    def __init__(self, keep_hpsit=True, nblocks=1):
        Eigensolver.__init__(self, keep_hpsit, nblocks)

    def initialize(self, paw):
        Eigensolver.initialize(self, paw)
        self.overlap = paw.overlap

    def iterate_one_k_point(self, hamiltonian, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.subspace_diagonalize(hamiltonian, kpt)

        self.timer.start('Residuals')

        if self.keep_htpsit:
            R_nG = self.Htpsit_nG

            self.timer.start('Residuals: axpy psit_G')
            for R_G, eps, psit_G in zip(R_nG, kpt.eps_n, kpt.psit_nG):
                # R_G -= eps * psit_G
                axpy(-eps, psit_G, R_G)
            self.timer.stop('Residuals: axpy psit_G')

            self.timer.start('Residuals: run')
            run([nucleus.adjust_residual(R_nG, kpt.eps_n, kpt.s, kpt.u, kpt.k)
                 for nucleus in hamiltonian.pt_nuclei])
            self.timer.stop('Residuals: run')

        self.timer.stop('Residuals')

        self.timer.start('RMM-DIIS')
        vt_G = hamiltonian.vt_sG[kpt.s]
        dR_G = self.big_work_arrays['work_nG'][0]
        error = 0.0
        n0 = self.band_comm.rank * self.nmybands
        for n in range(self.nmybands):
            if self.keep_htpsit:
                R_G = R_nG[n]
            else:
                R_G = self.big_work_arrays['work_nG'][1]
                psit_G = kpt.psit_nG[n]
                self.timer.start('RMM-DIIS: hamiltonian.apply')
                hamiltonian.apply(psit_G, R_G, kpt,
                                  local_part_only=True,
                                  calculate_projections=False)
                self.timer.stop('RMM-DIIS: hamiltonian.apply')
                self.timer.start('RMM-DIIS: axpy pR_G')
                axpy(-kpt.eps_n[n], psit_G, R_G)
                self.timer.stop('RMM-DIIS: axpy pR_G')
                self.timer.start('RMM-DIIS: adjust_residual2')
                run([nucleus.adjust_residual2(psit_G, R_G, kpt.eps_n[n],
                                             kpt.s, kpt.u, kpt.k, n)
                     for nucleus in hamiltonian.pt_nuclei])
                self.timer.stop('RMM-DIIS: adjust_residual2')

            weight = kpt.f_n[n]
            if self.nbands_converge != 'occupied':
                if n0 + n < self.nbands_converge:
                    weight = kpt.weight
                else:
                    weight = 0.0
            error += weight * np.vdot(R_G, R_G).real

            # Precondition the residual:
            self.timer.start('RMM-DIIS: kpt.psit_nG')
            pR_G = self.preconditioner(R_G, kpt.phase_cd, kpt.psit_nG[n],
                                       kpt.k_c)
            self.timer.stop('RMM-DIIS: kpt.psit_nG')

            self.timer.start('RMM-DIIS: hamiltonian.apply')
            # Calculate the residual of pR_G, dR_G = (H - e S) pR_G:
            hamiltonian.apply(pR_G, dR_G, kpt, local_part_only=True,
                              calculate_projections=False)
            self.timer.stop('RMM-DIIS: hamiltonian.apply')
            self.timer.start('RMM-DIIS: axpy pR_G')
            axpy(-kpt.eps_n[n], pR_G, dR_G)  # dR_G -= kpt.eps_n[n] * pR_G
            self.timer.stop('RMM-DIIS: axpy pR_G')

            self.timer.start('RMM-DIIS: adjust_residual2')
            run([nucleus.adjust_residual2(pR_G, dR_G, kpt.eps_n[n],
                                          kpt.u, kpt.s, kpt.k, n)
                 for nucleus in hamiltonian.pt_nuclei])
            self.timer.stop('RMM-DIIS: adjust_residual2')

            hamiltonian.xc.xcfunc.adjust_non_local_residual(
                pR_G, dR_G, kpt.eps_n[n], kpt.u, kpt.s, kpt.k, n)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            self.timer.start('RMM-DIIS: self.comm.sum RdR')
            RdR = self.comm.sum(np.vdot(R_G, dR_G).real)
            self.timer.stop('RMM-DIIS: self.comm.sum RdR')
            self.timer.start('RMM-DIIS: self.comm.sum dRdR')
            dRdR = self.comm.sum(np.vdot(dR_G, dR_G).real)
            self.timer.stop('RMM-DIIS: self.comm.sum dRdR')

            lam = -RdR / dRdR

            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            R_G *= 2.0 * lam
            self.timer.start('RMM-DIIS: axpy dR_G')
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            self.timer.stop('RMM-DIIS: axpy dR_G')
            self.timer.start('RMM-DIIS: kpt.psit_nG')
            kpt.psit_nG[n] += self.preconditioner(R_G, kpt.phase_cd,
                                                  kpt.psit_nG[n], kpt.k_c)
            self.timer.stop('RMM-DIIS: kpt.psit_nG')

        self.timer.stop('RMM-DIIS')

        # Orthonormalize the wave functions
        self.overlap.orthonormalize(kpt)

        error = self.comm.sum(error)
        return error
