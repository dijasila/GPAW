from functools import partial

import numpy as np

from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.matrix import PAWMatrix
from gpaw.utilities import unpack
from gpaw.utilities.lapack import general_diagonalize


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

    def __init__(self, niter=1, smin=None, normalize=True):
        Eigensolver.__init__(self)
        self.niter = niter
        self.smin = smin
        self.normalize = normalize

        if smin is not None:
            raise NotImplementedError(
                'See https://trac.fysik.dtu.dk/projects/gpaw/ticket/248')

        self.orthonormalization_required = False

    def __repr__(self):
        return 'Davidson(niter=%d, smin=%r, normalize=%r)' % (
            self.niter, self.smin, self.normalize)

    def todict(self):
        return {'name': 'dav', 'niter': self.niter}

    def initialize(self, wfs):

        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap

        # Allocate arrays
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

    def iterate_one_k_point(self, ham, wfs, kpt):
        """Do Davidson iterations for the kpoint"""
        B = self.nbands
        mynbands = self.mynbands
        #gd = wfs.matrixoperator.gd
        bd = wfs.bd

        def integrate(a_G):
            return np.real(wfs.integrate(a_G, a_G, global_integral=False))

        self.subspace_diagonalize(ham, wfs, kpt)

        psit_n = kpt.psit_n
        psit2_n = psit_n.new(buf=wfs.work_array_nG)
        P_nI = kpt.P_nI
        P2_nI = P_nI.new()
        dMP_nI = P_nI.new()
        M_nn = wfs.M_nn

        H_2n2n = self.H_2n2n
        S_2n2n = self.S_2n2n
        eps_2n = self.eps_2n

        self.timer.start('Davidson')

        if self.keep_htpsit:
            R_n = psit_n.new(buf=self.Htpsit_nG)
        else:
            1 / 0
            # R_nG = wfs.empty(mynbands, q=kpt.q)
            # psit2_nG = wfs.empty(mynbands, q=kpt.q)
            # wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG, R_nG)
            # wfs.pt.integrate(psit_nG, kpt.P_ani, kpt.q)

        self.calculate_residuals(kpt, wfs, ham, psit_n.A,
                                 kpt.P_ani, kpt.eps_n, R_n.A)

        for nit in range(self.niter):
            H_2n2n[:] = 0.0
            S_2n2n[:] = 0.0

            norm_n = np.zeros(mynbands)
            error = 0.0
            for n in range(mynbands):
                if kpt.f_n is None:
                    weight = kpt.weight
                else:
                    weight = kpt.f_n[n]
                if self.nbands_converge != 'occupied':
                    if n < self.nbands_converge:
                        weight = kpt.weight
                    else:
                        weight = 0.0
                error += weight * integrate(R_n[n])

                ekin = self.preconditioner.calculate_kinetic_energy(
                    psit_n[n:n + 1], kpt)
                psit2_n[n] = self.preconditioner(R_n[n:n + 1], kpt, ekin)

                if self.normalize:
                    norm_n[n] = integrate(psit2_n[n])

                N = bd.global_index(n)
                H_2n2n[N, N] = kpt.eps_n[n]
                S_2n2n[N, N] = 1.0

            #bd.comm.sum(H_2n2n)
            #bd.comm.sum(S_2n2n)

            if self.normalize:
                #gd.comm.sum(norm_n)
                for norm, psit2_G in zip(norm_n, psit2_n.A):
                    psit2_G *= norm**-0.5

            Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)
            dH_II = PAWMatrix(unpack(ham.dH_asp[a][kpt.s]) for a in kpt.P_ani)
            dS_II = PAWMatrix(wfs.setups[a].dO_ii for a in kpt.P_ani)

            def mat(a_n, b_n, Pa_nI, Pb_nI, dM_II, C_nn, hermitian=False):
                """Fill C_nn with <a|b> matrix elements."""
                a_n.matrix_elements(b_n, M_nn, hermitian)
                dMP_nI[:] = Pb_nI * dM_II
                C_nn += Pa_nI.C * dMP_nI.T
                C_nn[:] = M_nn

            # Calculate projections
            psit2_n.project(wfs.pt, P2_nI)

            psit2_n.apply(Ht, R_n)

            with self.timer('calc. matrices'):
                # <psi2 | H | psi>
                mat(R_n, psit_n, P2_nI, dH_II, P_nI, H21_nn)
                # <psi2 | S | psi>
                mat(psit2_n, psit_n, P2_nI, dS_II, P_nI, S21_nn)
                # <psi2 | H | psi2>
                mat(R_n, psit2_n, P2_nI, dH_II, P2_nI, H22_nn)
                # <psi2 | S | psi2>
                mat(psit2_n, psit2_n, P2_nI, dS_II, P2_nI, S22_nn)

            with self.timer('diagonalize'):
                #if gd.comm.rank == 0 and bd.comm.rank == 0:
                general_diagonalize(H_2n2n, eps_2n, S_2n2n)

            #gd.comm.broadcast(H_2n2n, 0)
            #gd.comm.broadcast(eps_2n, 0)
            #bd.comm.broadcast(H_2n2n, 0)
            #bd.comm.broadcast(eps_2n, 0)

            kpt.eps_n[:] = eps_2n[bd.get_slice()]

            with self.timer('rotate_psi'):
                M_nn[:] = H_2n2n[:nbands, :nbands]
                R_n[:] = M_nn * psit_n
                dMP_nI[:] = M_nn * P_nI
                M_nn[:] = H_2n2n[:nbands, :nbands]
                R_n += M_nn * psit2_n
                dMP_nI += M_nn * P2_nI
                psit_n[:] = R_n
                dMP_nI.extract_to(kpt.P_ani)

            if nit < self.niter - 1:
                psit_n.apply(Ht, R_n)
                self.calculate_residuals(kpt, wfs, ham, psit_n.A,
                                         kpt.P_ani, kpt.eps_n, R_n.A)

        self.timer.stop('Davidson')
        #error = gd.comm.sum(error)
        return error
