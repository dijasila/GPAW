"""Module defining  ``Eigensolver`` classes."""

import numpy as np

from gpaw.utilities.blas import axpy, gemm, rk, r2k
from gpaw.utilities.lapack import general_diagonalize
from gpaw.eigensolvers.eigensolver import Eigensolver


class RMM_DIIS_new(Eigensolver):
    """RMM-DIIS eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated

    Solution steps are:

    * Subspace diagonalization
    * Calculation of residuals
    * Improvement of wave functions:  psi' = psi + lambda PR + lambda PR'
    * Orthonormalization"""

    def __init__(self, keep_htpsit=True, blocksize=10, niter=4, rtol=1e-6,
                 limit_lambda=False):
        Eigensolver.__init__(self, keep_htpsit, blocksize)
        self.niter = niter
        self.rtol = rtol
        self.limit_lambda = limit_lambda

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        psit_nG, R_nG = self.subspace_diagonalize(hamiltonian, wfs, kpt)

        self.timer.start('RMM-DIIS')
        self.timer.start('Calculate residuals')
        if self.keep_htpsit:
            self.calculate_residuals(kpt, wfs, hamiltonian, psit_nG,
                                     kpt.P_ani, kpt.eps_n, R_nG)
        self.timer.stop('Calculate residuals')

        def integrate(a_G, b_G):
            return np.real(wfs.integrate(a_G, b_G, global_integral=False))

        comm = wfs.gd.comm
        B = self.blocksize
        dR_xG = wfs.empty(B, q=kpt.q)
        P_axi = wfs.pt.dict(B)
        errors = np.zeros(B)
        
        # Arrays needed for DIIS step
        if self.niter > 1:
            psit_diis_nxG = wfs.empty(B * self.niter, q=kpt.q)
            R_diis_nxG = wfs.empty(B * self.niter, q=kpt.q)
            P_diis_anxi = wfs.pt.dict(B * self.niter)
            eig_n = np.zeros(self.niter)  # eigenvalues for diagonalization
                                          # not needed in any step

        error = 0.0
        for n1 in range(0, wfs.bd.mynbands, B):
            n2 = n1 + B
            if n2 > wfs.bd.mynbands:
                n2 = wfs.bd.mynbands
                B = n2 - n1
                P_axi = dict((a, P_xi[:B]) for a, P_xi in P_axi.items())
                dR_xG = dR_xG[:B]
                
            n_x = range(n1, n2)
            psit_xG = psit_nG[n1:n2]

            self.timer.start('Calculate residuals')
            if self.keep_htpsit:
                R_xG = R_nG[n1:n2]
            else:
                R_xG = wfs.empty(B, q=kpt.q)
                wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_xG, R_xG)
                wfs.pt.integrate(psit_xG, P_axi, kpt.q)
                self.calculate_residuals(kpt, wfs, hamiltonian, psit_xG,
                                         P_axi, kpt.eps_n[n_x], R_xG, n_x)
            self.timer.stop('Calculate residuals')

            error_block = 0.0
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
                error_block += weight * integrate(R_xG[n - n1], 
                                                  R_xG[n - n1])
            comm.sum(error_block)
            error += error_block

            # Insert first vectors and residuals for DIIS step
            if self.niter > 1:
                # Save the previous vectors contiguously for each band
                # in the block
                psit_diis_nxG[:B * self.niter:self.niter] = psit_xG
                R_diis_nxG[:B * self.niter:self.niter] = R_xG

            # Precondition the residual:
            self.timer.start('precondition')
            # ekin_x = self.preconditioner.calculate_kinetic_energy(
            #     R_xG, kpt)
            ekin_x = self.preconditioner.calculate_kinetic_energy(
                psit_xG, kpt)
            dpsit_xG = self.preconditioner(R_xG, kpt, ekin_x)
            self.timer.stop('precondition')

            # Calculate the residual of dpsit_G, dR_G = (H - e S) dpsit_G:
            # self.timer.start('Apply Hamiltonian')
            wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, dpsit_xG, dR_xG)
            # self.timer.stop('Apply Hamiltonian')
            self.timer.start('projections')
            wfs.pt.integrate(dpsit_xG, P_axi, kpt.q)
            self.timer.stop('projections')
            self.timer.start('Calculate residuals')
            self.calculate_residuals(kpt, wfs, hamiltonian, dpsit_xG,
                                     P_axi, kpt.eps_n[n_x], dR_xG, n_x,
                                     calculate_change=True)
            self.timer.stop('Calculate residuals')

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            self.timer.start('Find lambda')
            RdR_x = np.array([integrate(dR_G, R_G)
                              for R_G, dR_G in zip(R_xG, dR_xG)])
            dRdR_x = np.array([integrate(dR_G, dR_G) for dR_G in dR_xG])
            comm.sum(RdR_x)
            comm.sum(dRdR_x)
            lam_x = -RdR_x / dRdR_x
            # Limit abs(lam) to [0.15, 1.0]
            if self.limit_lambda:
                lam_x = np.where(np.abs(lam_x) < 0.1, 0.1 * np.sign(lam_x), lam_x)
                lam_x = np.where(np.abs(lam_x) > 1.0, 1.0 * np.sign(lam_x), lam_x)
            self.timer.stop('Find lambda')
            # New trial wavefunction and residual          
            self.timer.start('Update psi')
            for lam, psit_G, dpsit_G, R_G, dR_G in zip(lam_x, psit_xG, 
                                                       dpsit_xG, R_xG, dR_xG):
                axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
                axpy(lam, dR_G, R_G)  # R_G += lam** dR_G
            self.timer.stop('Update psi')

            self.timer.start('DIIS step')
            # DIIS step
            for nit in range(1, self.niter):
                # Do not perform DIIS if error is small
                if abs(error_block / B) < self.rtol:
                    break 
                
                # Update the subspace
                psit_diis_nxG[nit:B * self.niter:self.niter] = psit_xG
                R_diis_nxG[nit:B * self.niter:self.niter] = R_xG

                # XXX Only integrals of nit old psits would be needed
                wfs.pt.integrate(psit_diis_nxG, P_diis_anxi, kpt.q)
                for ib in range(B):
                    istart = ib * self.niter
                    iend = istart + nit + 1

                    # Residual matrix
                    R_nn = np.zeros((nit+1, nit+1), wfs.dtype)
                    rk(wfs.gd.dv, R_diis_nxG[istart:iend], 0.0, R_nn)
                    comm.sum(R_nn)

                    # Overlap matrix
                    S_nn = np.zeros((nit + 1, nit + 1), wfs.dtype)
                    rk(wfs.gd.dv, psit_diis_nxG[istart:iend], 0.0, S_nn)
                    for a, P_nxi in P_diis_anxi.items():
                        dO_ii = wfs.setups[a].dO_ii
                        gemm(1.0, P_nxi[istart:iend], 
                             np.dot(P_nxi[istart:iend], dO_ii), 1.0, S_nn, 'c')
                    comm.sum(S_nn)

                    general_diagonalize(R_nn, eig_n[:nit+1], S_nn)
                    alpha_i = R_nn[0, :]

                    # Obtain new trial wave function from previous ones
                    psit_diis_nxG[istart + nit] *= alpha_i[nit]
                    R_diis_nxG[istart + nit] *= alpha_i[nit]
                    for i in range(nit):
                        axpy(alpha_i[i], psit_diis_nxG[istart + i], 
                             psit_diis_nxG[istart + nit])
                        axpy(alpha_i[i], R_diis_nxG[istart + i], 
                             R_diis_nxG[istart + nit])

                psit_xG[:] = psit_diis_nxG[nit:B * self.niter:self.niter]
                R_xG[:] = R_diis_nxG[nit:B * self.niter:self.niter]
                self.timer.start('precondition')
                # ekin_x = self.preconditioner.calculate_kinetic_energy(
                #     R_xG, kpt)
                dpsit_xG = self.preconditioner(R_xG, kpt, ekin_x)
                self.timer.stop('precondition')

                for psit_G, lam, dpsit_G in zip(psit_xG, lam_x, dpsit_xG):
                    axpy(lam, dpsit_G, psit_G)

                # Calculate the new residuals
                self.timer.start('Calculate residuals')
                wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_xG, R_xG)
                wfs.pt.integrate(psit_xG, P_axi, kpt.q)
                self.calculate_residuals(kpt, wfs, hamiltonian, psit_xG,
                                         P_axi, kpt.eps_n[n_x], R_xG, n_x,
                                         calculate_change=True)
                self.timer.stop('Calculate residuals')
                
                error_block = 0.0
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
                    error_block += weight * integrate(R_xG[n - n1], 
                                                      R_xG[n - n1])
                comm.sum(error_block)

            self.timer.stop('DIIS step')                
            # Final trial step
            self.timer.start('precondition')
            # ekin_x = self.preconditioner.calculate_kinetic_energy(
            #     R_xG, kpt)
            dpsit_xG = self.preconditioner(R_xG, kpt, ekin_x)
            self.timer.stop('precondition')
            self.timer.start('Update psi')
            for lam, psit_G, dpsit_G in zip(lam_x, psit_xG, dpsit_xG):
                axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
            self.timer.stop('Update psi')
        self.timer.stop('RMM-DIIS')
        return error, psit_nG

    def __repr__(self):
        repr_string = 'RMM-DIIS eigensolver\n'
        repr_string += '       Block size: %d\n' % self.blocksize
        repr_string += '       DIIS iterations: %d\n' % self.niter
        repr_string += '       Threshold for DIIS: %5.1e\n' % self.rtol
        repr_string += '       Limit lambda: %s' % self.limit_lambda
        return repr_string
