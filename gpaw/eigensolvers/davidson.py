from functools import partial

from ase.utils.timing import timer
import numpy as np

from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.matrix import matrix_matrix_multiply as mmm


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

    @timer('Davidson')
    def iterate_one_k_point(self, ham, wfs, kpt):
        """Do Davidson iterations for the kpoint"""
        bd = wfs.bd
        B = bd.nbands

        H_NN = self.H_2n2n
        S_NN = self.S_2n2n
        eps_N = self.eps_2n

        def integrate(a_G):
            return np.real(wfs.integrate(a_G, a_G, global_integral=False))

        self.subspace_diagonalize(ham, wfs, kpt)

        psit = kpt.psit
        psit2 = psit.new(buf=wfs.work_array)
        P = kpt.P
        P2 = P.new()
        dMP = P.new()
        M = wfs.work_matrix_nn
        dS = wfs.setups.dS
        comm = wfs.gd.comm

        def matrix_elements(a, b, Pa, dM, Pb, C_nn, symmetric=False,
                            tmp=M):
            """Fill C_nn with <a|b> matrix elements."""
            a.matrix_elements(b, out=tmp, symmetric=symmetric)
            dM.apply(Pb, out=dMP)
            mmm(1.0, Pa, 'H', dMP, 'N', 1.0, tmp)
            comm.sum(tmp.array, 0)
            if comm.rank == 0:
                C_nn[:] = tmp.array

        if self.keep_htpsit:
            R = psit.new(buf=self.Htpsit_nG)
        else:
            1 / 0
            # R_nG = wfs.empty(mynbands, q=kpt.q)
            # psit2_nG = wfs.empty(mynbands, q=kpt.q)
            # wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG, R_nG)
            # wfs.pt.integrate(psit_nG, kpt.P_ani, kpt.q)

        self.calculate_residuals(kpt, wfs, ham, psit, P, kpt.eps_n, R, P2)

        weights = self.weights(kpt)

        Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

        for nit in range(self.niter):
            if nit == self.niter - 1:
                error = np.dot(weights, [integrate(R_G) for R_G in R.array])

            for psit_G, R_G, psit2_G in zip(psit.array,
                                            R.array,
                                            psit2.array):
                pre = self.preconditioner
                ekin = pre.calculate_kinetic_energy(psit_G, kpt)
                psit2_G[:] = pre(R_G, kpt, ekin)

            if self.normalize:
                norms = np.array([integrate(psit2_G)
                                  for psit2_G in psit2.array])
                comm.sum(norms)
                for norm, psit2_G in zip(norms, psit2.array):
                    psit2_G *= norm**-0.5

            # Calculate projections
            wfs.pt.matrix_elements(psit2, out=P2)

            psit2.apply(Ht, out=R)

            with self.timer('calc. matrices'):
                H_NN[:B, :B] = np.diag(kpt.eps_n)
                S_NN[:B, :B] = np.eye(B)
                me = matrix_elements

                # <psi2 | H | psi>
                me(R, psit, P2, ham.dH, P, H_NN[B:, :B])
                # <psi2 | S | psi>
                me(psit2, psit, P2, dS, P, S_NN[B:, :B])
                # <psi2 | H | psi2>
                me(R, psit2, P2, ham.dH, P2, H_NN[B:, B:], True)
                # <psi2 | S | psi2>
                me(psit2, psit2, P2, dS, P2, S_NN[B:, B:])

            with self.timer('diagonalize'):
                if comm.rank == 0 and bd.comm.rank == 0:
                    H_NN[:B, B:] = 0.0
                    S_NN[:B, B:] = 0.0
                    from scipy.linalg import eigh
                    eps_N, H_NN[:] = eigh(H_NN, S_NN,
                                          lower=True,
                                          check_finite=not False)
                # H_NN[:B, B:] = H_NN[B:, :B].conj().T
                # S_NN[:B, B:] = S_NN[B:, :B].conj().T
                # general_diagonalize(H_NN, eps_N, S_NN)
                # H_NN = H_NN.T.copy()

            comm.broadcast(H_NN, 0)
            comm.broadcast(eps_N, 0)
            bd.comm.broadcast(H_NN, 0)
            bd.comm.broadcast(eps_N, 0)

            kpt.eps_n[:] = eps_N[bd.get_slice()]

            with self.timer('rotate_psi'):
                M.array[:] = H_NN[:B, :B]
                mmm(1.0, M, 'T', psit, 'N', 0.0, R)
                mmm(1.0, P, 'N', M, 'N', 0.0, dMP)
                M.array[:] = H_NN[B:, :B]
                mmm(1.0, M, 'T', psit2, 'N', 1.0, R)
                mmm(1.0, P2, 'N', M, 'N', 1.0, dMP)
                psit[:] = R
                P, dMP = dMP, P
                kpt.P = P

            if nit < self.niter - 1:
                psit.apply(Ht, out=R)
                self.calculate_residuals(kpt, wfs, ham, psit,
                                         P, kpt.eps_n, R, P2)

        error = wfs.gd.comm.sum(error)
        return error
