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

    def __init__(self, keep_hpsit=True, nblocks=1):
        Eigensolver.__init__(self, keep_hpsit, nblocks)

    def initialize(self, paw):
        Eigensolver.initialize(self, paw)
        self.overlap = paw.overlap

    def calculate_residual(self, wfs, hamiltonian, kpt, eps, R_G, psit_G):
        wfs.kin.apply(psit_G, R_G, kpt.phase_cd)
        hamiltonian.apply_local_potential(psit_G, R_G, kpt.s)
        P_ai = dict([(a, np.zeros(wfs.setups[a].ni, wfs.dtype))
                      for a in kpt.P_ani])
        wfs.pt.integrate(psit_G, P_ai, kpt.q)
        axpy(-eps, psit_G, R_G)
        c_ai = {}
        for a, P_i in P_ai.items():
            dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
            dO_ii = hamiltonian.setups[a].O_ii
            c_i = np.dot(P_i, dH_ii - eps * dO_ii)
            c_ai[a] = c_i
        wfs.pt.add(R_G, c_ai, kpt.q)
        
    def calculate_residuals(self, wfs, hamiltonian, kpt, R_nG, psit_nG):
        wfs.kin.apply(psit_nG, R_nG, kpt.phase_cd)
        hamiltonian.apply_local_potential(psit_nG, R_nG, kpt.s)
        P_ani = dict([(a, np.zeros_like(P_ni))
                      for a, P_ni in kpt.P_ani.items()])
        wfs.pt.integrate(psit_nG, P_ani, kpt.q)
        self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG, psit_nG, P_ani)
        
    def calculate_residuals2(self, wfs, hamiltonian, kpt, R_nG,
                             eps_n=None, psit_nG=None, P_ani=None):
        if psit_nG is None:
            psit_nG = kpt.psit_nG
        if P_ani is None:
            P_ani = kpt.P_ani
        for R_G, eps, psit_G in zip(R_nG, kpt.eps_n, psit_nG):
            axpy(-eps, psit_G, R_G)
        c_ani = {}
        for a, P_ni in P_ani.items():
            dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
            dO_ii = hamiltonian.setups[a].O_ii
            c_ni = (np.dot(P_ni, dH_ii) -
                    np.dot(P_ni * kpt.eps_n[:, np.newaxis], dO_ii))
            c_ani[a] = c_ni
        wfs.pt.add(R_nG, c_ani, kpt.q)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.subspace_diagonalize(hamiltonian, wfs, kpt)

        self.timer.start('Residuals')
        if self.keep_htpsit:
            R_nG = self.Htpsit_nG
            self.calculate_residuals2(wfs, hamiltonian, kpt, R_nG)
        self.timer.stop('Residuals')

        self.timer.start('RMM-DIIS')
        vt_G = hamiltonian.vt_sG[kpt.s]
        dR_G = self.big_work_arrays['work_nG'][0]
        error = 0.0
        n0 = self.band_comm.rank * self.mynbands
        for n in range(self.mynbands):
            if self.keep_htpsit:
                R_G = R_nG[n]
            else:
                R_G = self.big_work_arrays['work_nG'][1]
                psit_G = kpt.psit_nG[n]
                hamiltonian.apply(psit_G, R_G, kpt,
                                  local_part_only=True,
                                  calculate_projections=False)
                axpy(-kpt.eps_n[n], psit_G, R_G)
                run([nucleus.adjust_residual2(psit_G, R_G, kpt.eps_n[n],
                                             kpt.s, kpt.u, kpt.k, n)
                     for nucleus in hamiltonian.pt_nuclei])

            if kpt.f_n is None:
                weight = kpt.weight
            else:
                weight = kpt.f_n[n]
            if self.nbands_converge != 'occupied':
                if n0 + n < self.nbands_converge:
                    weight = kpt.weight
                else:
                    weight = 0.0
            error += weight * np.vdot(R_G, R_G).real

            # Precondition the residual:
            pR_G = self.preconditioner(R_G, kpt.phase_cd, kpt.psit_nG[n])

            # Calculate the residual of pR_G, dR_G = (H - e S) pR_G:
            self.calculate_residual(wfs, hamiltonian, kpt, kpt.eps_n[n],
                                    dR_G, pR_G)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR = self.comm.sum(np.vdot(R_G, dR_G).real)
            dRdR = self.comm.sum(np.vdot(dR_G, dR_G).real)

            lam = -RdR / dRdR

            # Calculate new psi'_G = psi_G + lam pR_G + lam pR'_G
            #                      = psi_G + p(2 lam R_G + lam**2 dR_G)
            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            kpt.psit_nG[n] += self.preconditioner(R_G, kpt.phase_cd,
                                                  kpt.psit_nG[n])

        self.timer.stop('RMM-DIIS')

        error = self.comm.sum(error)
        return error
