"""Module defining  ``Eigensolver`` classes."""

import numpy as npy

from gpaw.utilities.blas import axpy, rk, gemm
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.utilities import elementwise_multiply_add, utilities_vdot, utilities_vdot_self
from gpaw.utilities import unpack
from gpaw.utilities.complex import cc, real
from gpaw.eigensolvers.eigensolver import Eigensolver
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

    def __init__(self, rotate=True):
        Eigensolver.__init__(self)
        self.rotate = rotate
        self.nbands = None

    def initialize(self, paw):
        Eigensolver.initialize(self, paw)

        self.S_nn = npy.empty((self.nbands, self.nbands), self.dtype)
        self.S_nn[:] = 0.0  # rk fails the first time without this!
        
    def iterate_one_k_point(self, hamiltonian, kpt):      
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.diagonalize(hamiltonian, kpt, self.rotate)

        self.timer.start('Residuals')
        R_nG = self.Htpsit_nG

        if self.rotate:
            for R_G, eps, psit_G in zip(R_nG, kpt.eps_n, kpt.psit_nG):
                # R_G -= eps * psit_G
                axpy(-eps, psit_G, R_G)
                
            run([nucleus.adjust_residual(R_nG, kpt.eps_n, kpt.s, kpt.u, kpt.k)
                 for nucleus in hamiltonian.pt_nuclei])
        else:
            H_nn = self.H_nn
            # Filling up the upper triangle:
            for n in range(self.nbands - 1):
                H_nn[n, n:] = H_nn[n:, n]

            kpt.eps_n = npy.diagonal(H_nn)

            gemm(-1.0, kpt.psit_nG, H_nn, 1.0, R_nG)

            for nucleus in hamiltonian.pt_nuclei:
                if nucleus.in_this_domain:
                    H_ii = unpack(nucleus.H_sp[kpt.s])
                    P_ni = nucleus.P_uni[kpt.u]
                    coefs_ni =  (npy.dot(P_ni, H_ii) -
                                 npy.dot(npy.dot(H_nn, P_ni),
                                         nucleus.setup.O_ii))

                    nucleus.pt_i.add(R_nG, coefs_ni, kpt.k, communicate=True)
                else:
                    nucleus.pt_i.add(R_nG, None, kpt.k, communicate=True)

        self.timer.stop('Residuals')

        self.timer.start('RMM-DIIS')
        vt_G = hamiltonian.vt_sG[kpt.s]
        dR_G = self.work[0]
        error = 0.0
        for n in range(kpt.nbands):
            R_G = R_nG[n]

            weight = kpt.f_n[n]
            if self.nbands_converge != 'occupied':
                weight = kpt.weight * float(n < self.nbands_converge)
            error += weight * real(npy.vdot(R_G, R_G))

            pR_G = self.preconditioner(R_G, kpt.phase_cd, kpt.psit_nG[n],
                                  kpt.k_c)

            hamiltonian.kin.apply(pR_G, dR_G, kpt.phase_cd)
                
            if (dR_G.dtype.char == float):
                elementwise_multiply_add(vt_G, pR_G, dR_G)
            else:
                dR_G += vt_G * pR_G
            
            axpy(-kpt.eps_n[n], pR_G, dR_G)  # dR_G -= kpt.eps_n[n] * pR_G

            run([nucleus.adjust_residual2(pR_G, dR_G, kpt.eps_n[n],
                                          kpt.u, kpt.s, kpt.k, n)
                 for nucleus in hamiltonian.pt_nuclei])

            hamiltonian.xc.xcfunc.adjust_non_local_residual(
                pR_G, dR_G, kpt.eps_n[n], kpt.u, kpt.s, kpt.k, n)
            
            if (dR_G.dtype.char == float):
                RdR = self.comm.sum(utilities_vdot(R_G, dR_G))
                dRdR = self.comm.sum(utilities_vdot_self(dR_G))
            else:
                RdR = self.comm.sum(real(npy.vdot(R_G, dR_G)))
                dRdR = self.comm.sum(real(npy.vdot(dR_G, dR_G)))
            lam = -RdR / dRdR

            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            kpt.psit_nG[n] += self.preconditioner(R_G, kpt.phase_cd,
                                                 kpt.psit_nG[n], kpt.k_c)
            
        self.timer.stop('RMM-DIIS')

        self.timer.start('Orthogonalize')
        run([nucleus.calculate_projections(kpt)
             for nucleus in hamiltonian.pt_nuclei])

        S_nn = self.S_nn

        # Fill in the lower triangle:
        rk(self.gd.dv, kpt.psit_nG, 0.0, S_nn)

        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            S_nn += npy.dot(P_ni, cc(npy.inner(nucleus.setup.O_ii, P_ni)))

        self.comm.sum(S_nn, kpt.root)

        if self.comm.rank == kpt.root:
            info = inverse_cholesky(S_nn)
            if info != 0:
                raise RuntimeError('Orthogonalization failed!')

        self.comm.broadcast(S_nn, kpt.root)
        
        gemm(1.0, kpt.psit_nG, S_nn, 0.0, self.work)
        kpt.psit_nG, self.work = self.work, kpt.psit_nG  # swap

        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            gemm(1.0, P_ni.copy(), S_nn, 0.0, P_ni)
        self.timer.stop('Orthogonalize')
     
        error = self.comm.sum(error)
        return error
    
