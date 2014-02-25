import numpy as np

from gpaw.utilities import unpack
from gpaw.utilities.blas import gemm
from gpaw.hs_operators import reshape
from gpaw.utilities.lapack import general_diagonalize
from gpaw.eigensolvers.eigensolver import Eigensolver


class Davidson(Eigensolver):
    """Simple Davidson eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated.

    Solution steps are:

    * Subspace diagonalization
    * Calculate all residuals
    * Add preconditioned residuals to the subspace and diagonalize
    """

    def __init__(self, niter=2, preconditioner=1):
        Eigensolver.__init__(self)
        self.niter = niter
        self.preconditioner = preconditioner  # 1 or 2 (old/new)
        self.orthonormalization_required = False

    def initialize(self, wfs):
        if wfs.bd.comm.size > 1:
            raise ValueError('CG eigensolver does not support band '
                             'parallelization.  This calculation parallelizes '
                             'over %d band groups.' % wfs.bd.comm.size)
        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap
        # Allocate arrays
        self.H_nn = np.zeros((self.nbands, self.nbands), self.dtype)
        self.S_nn = np.zeros((self.nbands, self.nbands), self.dtype)
        self.H_2n2n = np.empty((2 * self.nbands, 2 * self.nbands), self.dtype)
        self.S_2n2n = np.empty((2 * self.nbands, 2 * self.nbands), self.dtype)
        self.eps_2n = np.empty(2 * self.nbands)

    def estimate_memory(self, mem, wfs):
        Eigensolver.estimate_memory(self, mem, wfs)
        nbands = wfs.bd.nbands
        mem.subnode('H_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('H_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('eps_2n', 2 * nbands * mem.floatsize)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do Davidson iterations for the kpoint"""
        niter = self.niter
        nbands = self.nbands

        gd = wfs.matrixoperator.gd

        psit_nG, Htpsit_nG = self.subspace_diagonalize(hamiltonian, wfs, kpt)
        # Note that psit_nG is now in self.operator.work1_nG and
        # Htpsit_nG is in kpt.psit_nG!

        H_2n2n = self.H_2n2n
        S_2n2n = self.S_2n2n
        eps_2n = self.eps_2n
        psit2_nG = reshape(self.Htpsit_nG, psit_nG.shape)

        self.timer.start('Davidson')
        R_nG = Htpsit_nG
        self.calculate_residuals(kpt, wfs, hamiltonian, psit_nG,
                                 kpt.P_ani, kpt.eps_n, R_nG)

        def integrate(a_G, b_G):
            return np.real(wfs.integrate(a_G, b_G, global_integral=False))

        for nit in range(niter):
            H_2n2n[:] = 0.0
            S_2n2n[:] = 0.0

            error = 0.0
            for n in range(nbands):
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if n < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * integrate(R_nG[n], R_nG[n])

                if self.preconditioner == 1:
                    p_1G = R_nG[n:n + 1]
                else:
                    p_1G = psit_nG[n:n + 1]
                    
                ekin_1 = self.preconditioner.calculate_kinetic_energy(p_1G,
                                                                      kpt)
                psit2_nG[n] = self.preconditioner(R_nG[n:n + 1], kpt, ekin_1)

                H_2n2n[n, n] = kpt.eps_n[n]
                S_2n2n[n, n] = 1.0

            # Calculate projections
            P2_ani = wfs.pt.dict(nbands)
            wfs.pt.integrate(psit2_nG, P2_ani, kpt.q)
            
            # Hamiltonian matrix
            # <psi2 | H | psi>
            wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit2_nG, Htpsit_nG)
            gd.integrate(psit_nG, Htpsit_nG, global_integral=False,
                          _transposed_result=self.H_nn)
            # gemm(1.0, psit_nG, Htpsit_nG, 0.0, self.H_nn, 'c')

            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                self.H_nn += np.dot(P2_ni, np.dot(dH_ii, P_ni.T.conj()))

            gd.comm.sum(self.H_nn, 0)
            H_2n2n[nbands:, :nbands] = self.H_nn

            # <psi2 | H | psi2>
            gd.integrate(psit2_nG, Htpsit_nG, global_integral=False,
                          _transposed_result=self.H_nn)
            # r2k(0.5 * gd.dv, psit2_nG, Htpsit_nG, 0.0, self.H_nn)
            for a, P2_ni in P2_ani.items():
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                self.H_nn += np.dot(P2_ni, np.dot(dH_ii, P2_ni.T.conj()))

            gd.comm.sum(self.H_nn, 0)
            H_2n2n[nbands:, nbands:] = self.H_nn

            # Overlap matrix
            # <psi2 | S | psi>
            gd.integrate(psit_nG, psit2_nG, global_integral=False,
                          _transposed_result=self.S_nn)
            # gemm(1.0, psit_nG, psit2_nG, 0.0, self.S_nn, 'c')
        
            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                dO_ii = wfs.setups[a].dO_ii
                self.S_nn += np.dot(P2_ni, np.inner(dO_ii, P_ni.conj()))

            gd.comm.sum(self.S_nn, 0)
            S_2n2n[nbands:, :nbands] = self.S_nn

            # <psi2 | S | psi2>
            gd.integrate(psit2_nG, psit2_nG, global_integral=False,
                          _transposed_result=self.S_nn)
            # rk(gd.dv, psit2_nG, 0.0, self.S_nn)
            for a, P2_ni in P2_ani.items():
                dO_ii = wfs.setups[a].dO_ii
                self.S_nn += np.dot(P2_ni, np.dot(dO_ii, P2_ni.T.conj()))

            gd.comm.sum(self.S_nn, 0)
            S_2n2n[nbands:, nbands:] = self.S_nn

            if gd.comm.rank == 0:
                general_diagonalize(H_2n2n, eps_2n, S_2n2n)

            gd.comm.broadcast(H_2n2n, 0)
            gd.comm.broadcast(eps_2n, 0)

            kpt.eps_n[:] = eps_2n[:nbands]

            # Rotate psit_nG
            gd.gemm(1.0, psit_nG, H_2n2n[:nbands, :nbands],
                    0.0, Htpsit_nG)
            gd.gemm(1.0, psit2_nG, H_2n2n[:nbands, nbands:],
                    1.0, Htpsit_nG)
            psit_nG, Htpsit_nG = Htpsit_nG, psit_nG

            # Rotate P_uni:
            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                gemm(1.0, P_ni.copy(), H_2n2n[:nbands, :nbands],
                     0.0, P_ni)
                gemm(1.0, P2_ni, H_2n2n[:nbands, nbands:], 1.0, P_ni)

            if nit < niter - 1:
                wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_nG,
                                             Htpsit_nG)
                R_nG = Htpsit_nG
                self.calculate_residuals(kpt, wfs, hamiltonian, psit_nG,
                                         kpt.P_ani, kpt.eps_n, R_nG)

        self.timer.stop('Davidson')
        error = gd.comm.sum(error)
        return error, psit_nG
