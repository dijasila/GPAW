from functools import partial

from ase.utils.timing import timer
import numpy as np
import scipy.linalg as linalg

from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.matrix import AtomBlockMatrix


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

        psit_n = kpt.psit_n
        psit2_n = psit_n.new(buf=wfs.work_array)
        P_In = kpt.P_In
        P2_In = P_In.new()
        dMP_In = P_In.new()
        M_nn = wfs.work_matrix_nn
        dS_II = AtomBlockMatrix(wfs.setups[a].dO_ii
                                for a in kpt.P_In.my_atom_indices)

        if self.keep_htpsit:
            R_n = psit_n.new(buf=self.Htpsit_nG)
        else:
            1 / 0
            # R_nG = wfs.empty(mynbands, q=kpt.q)
            # psit2_nG = wfs.empty(mynbands, q=kpt.q)
            # wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG, R_nG)
            # wfs.pt.integrate(psit_nG, kpt.P_ani, kpt.q)

        self.calculate_residuals(kpt, wfs, ham, psit_n,
                                 P_In, kpt.eps_n, dS_II, R_n, P2_In)

        weights = self.weights(kpt)

        for nit in range(self.niter):
            if nit == self.niter - 1:
                error = np.dot(weights, [integrate(R_G) for R_G in R_n.array])

            for psit_G, R_G, psit2_G in zip(psit_n.array,
                                            R_n.array,
                                            psit2_n.array):
                P = self.preconditioner
                ekin = P.calculate_kinetic_energy(psit_G, kpt)
                psit2_G[:] = P(R_G, kpt, ekin)

            if self.normalize:
                norms = [integrate(psit2_G) for psit2_G in psit2_n.array]
                #gd.comm.sum(norm_n)
                for norm, psit2_G in zip(norms, psit2_n.array):
                    psit2_G *= norm**-0.5

            Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

            def mat(a_n, b_n, Pa_In, dM_II, Pb_In, C_nn,
                    hermitian=False, M_nn=M_nn):
                """Fill C_nn with <a|b> matrix elements."""
                a_n.matrix_elements(b_n, out=M_nn, hermitian=hermitian)
                dMP_In[:] = dM_II * Pb_In
                M_nn += Pa_In.H * dMP_In
                C_nn[:] = M_nn.array

            # Calculate projections
            wfs.pt_I.matrix_elements(psit2_n, out=P2_In)

            psit2_n.apply(Ht, out=R_n)

            with self.timer('calc. matrices'):
                H_NN[:B, :B] = np.diag(kpt.eps_n)
                S_NN[:B, :B] = np.eye(B)

                # <psi2 | H | psi>
                mat(R_n, psit_n, P2_In, ham.dH_II, P_In, H_NN[:B, B:])
                # <psi2 | S | psi>
                mat(psit2_n, psit_n, P2_In, dS_II, P_In, S_NN[:B, B:])
                # <psi2 | H | psi2>
                mat(R_n, psit2_n, P2_In, ham.dH_II, P2_In, H_NN[B:, B:], True)
                # <psi2 | S | psi2>
                mat(psit2_n, psit2_n, P2_In, dS_II, P2_In, S_NN[B:, B:])

            with self.timer('diagonalize'):
                #if gd.comm.rank == 0 and bd.comm.rank == 0:
                #H_NN[B:, :B] = 0.0
                #S_NN[B:, :B] = 0.0
                from gpaw.utilities.lapack import general_diagonalize

                #eps_N, H_NN[:] = linalg.eigh(H_NN, S_NN,
                #                             lower=False,
                #                             check_finite=not False)
                H_NN[B:, :B] = H_NN[:B, B:]
                S_NN[B:, :B] = S_NN[:B, B:]
                general_diagonalize(H_NN, eps_N, S_NN)
                H_NN = H_NN.T.copy()

            #gd.comm.broadcast(H_2n2n, 0)
            #gd.comm.broadcast(eps_2n, 0)
            #bd.comm.broadcast(H_2n2n, 0)
            #bd.comm.broadcast(eps_2n, 0)

            kpt.eps_n[:] = eps_N[bd.get_slice()]

            with self.timer('rotate_psi'):
                M_nn.array[:] = H_NN[:B, :B]
                R_n[:] = M_nn.T * psit_n
                dMP_In[:] = P_In * M_nn
                M_nn.array[:] = H_NN[B:, :B]
                R_n += M_nn.T * psit2_n
                dMP_In += P2_In * M_nn
                psit_n[:] = R_n
                kpt.P_In, dMP_In = dMP_In, kpt.P_In
                P_In = kpt.P_In

            if nit < self.niter - 1:
                psit_n.apply(Ht, out=R_n)
                self.calculate_residuals(kpt, wfs, ham, psit_n,
                                         P_In, kpt.eps_n, dS_II, R_n,
                                         P2_In)

        #error = gd.comm.sum(error)
        return error
