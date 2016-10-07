import numpy as np

from gpaw.utilities.lapack import general_diagonalize
from gpaw.utilities import unpack
from gpaw.eigensolvers.eigensolver import Eigensolver, reshape


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
        niter = self.niter
        nbands = self.nbands
        mynbands = self.mynbands

        gd = wfs.matrixoperator.gd
        bd = self.operator.bd

        self.subspace_diagonalize(ham, wfs, kpt)

        psit_n = kpt.psit_n
        psit2_n = psit_n.new(buf=wfs.work_array_nG)
        P_nI = kpt.P_nI
        dHP_nI = P_nI.new()
        H_nn = wfs.M_nn

        H_2n2n = self.H_2n2n
        S_2n2n = self.S_2n2n
        eps_2n = self.eps_2n

        self.timer.start('Davidson')

        if self.keep_htpsit:
            R_n = psit_n.new(buf=self.Htpsit_nG)
        else:
            pass
            # R_nG = wfs.empty(mynbands, q=kpt.q)
            # psit2_nG = wfs.empty(mynbands, q=kpt.q)
            # wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG, R_nG)
            # wfs.pt.integrate(psit_nG, kpt.P_ani, kpt.q)

        self.calculate_residuals(kpt, wfs, ham, psit_n.data,
                                 kpt.P_ani, kpt.eps_n, R_n.data)

        def integrate(a_G):
            return np.real(wfs.integrate(a_G, a_G, global_integral=False))

        # Note on band parallelization
        # The "large" H_2n2n and S_2n2n matrices are at the moment
        # global and replicated over band communicator, and the
        # general diagonalization is performed in serial i.e. without
        # scalapack

        for nit in range(niter):
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
                psit2_n.data[n] = self.preconditioner(R_n[n:n + 1], kpt, ekin)

                if self.normalize:
                    norm_n[n] = integrate(psit2_n[n])

                N = bd.global_index(n)
                H_2n2n[N, N] = kpt.eps_n[n]
                S_2n2n[N, N] = 1.0

            bd.comm.sum(H_2n2n)
            bd.comm.sum(S_2n2n)

            if self.normalize:
                gd.comm.sum(norm_n)
                for norm, psit2_G in zip(norm_n, psit2_n.data):
                    psit2_G *= norm**-0.5

            self.timer.start('calc. matrices')

            Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

            dH_II = P_nI.paw_matrix(unpack(ham.dH_asp[a][kpt.s])
                                    for a in kpt.P_ani)
            dS_II = P_nI.paw_matrix(wfs.setups[a].dO_ii for a in kpt.P_ani)

            # Hamiltonian matrix
            # <psi2 | H | psi>
            psit2_n.apply(Ht, R_n)
            psit_n.matrix_elements(R_n, H_nn)
            dHP_nI[:] = P_nI * dH_II
            H_nn += P_nI.C * dHP_nI.T
            H_2n2n[nbands:, :nbands] = H_nn

            # Overlap matrix
            # <psi2 | S | psi>
            psit_n.matrix_elements(psit2_n, H_nn)
            dHP_nI[:] = P_nI * dS_II
            H_nn += P_nI.C * dHP_nI.T
            S_2n2n[nbands:, :nbands] = H_nn

            # Calculate projections
            P2_ani = wfs.pt.dict(mynbands)
            wfs.pt.integrate(psit2_n.data, P2_ani, kpt.q)

            # <psi2 | H | psi2>
            psit2_n.matrix_elements(R_n, H_nn)
            dHP_nI[:] = P_nI * dH_II
            H_nn += P_nI.C * dHP_nI.T
            H_2n2n[nbands:, :nbands] = H_nn

            # <psi2 | S | psi2>
            psit2_n.matrix_elements(R_n, H_nn)
            dHP_nI[:] = P_nI * dH_II
            H_nn += P_nI.C * dHP_nI.T
            S_2n2n[nbands:, nbands:] = H_nn

            self.timer.stop('calc. matrices')

            self.timer.start('diagonalize')
            if gd.comm.rank == 0 and bd.comm.rank == 0:
                general_diagonalize(H_2n2n, eps_2n, S_2n2n)

            gd.comm.broadcast(H_2n2n, 0)
            gd.comm.broadcast(eps_2n, 0)
            bd.comm.broadcast(H_2n2n, 0)
            bd.comm.broadcast(eps_2n, 0)

            kpt.eps_n[:] = eps_2n[self.operator.bd.get_slice()]

            self.timer.stop('diagonalize')

            self.timer.start('rotate_psi')

            psit_nG = self.operator.matrix_multiply(H_2n2n[:nbands, :nbands],
                                                    psit_nG, kpt.P_ani,
                                                    out_nG=R_nG)

            tmp_nG = self.operator.matrix_multiply(H_2n2n[:nbands, nbands:],
                                                   psit2_nG, P2_ani)

            if bd.comm.size > 1:
                psit_nG += tmp_nG
            else:
                tmp_nG += psit_nG
                psit_nG, R_nG = tmp_nG, psit_nG

            for a, P_ni in kpt.P_ani.items():
                P2_ni = P2_ani[a]
                P_ni += P2_ni

            self.timer.stop('rotate_psi')

            if nit < niter - 1:
                wfs.apply_pseudo_hamiltonian(kpt, ham, psit_nG,
                                             R_nG)
                self.calculate_residuals(kpt, wfs, ham, psit_nG,
                                         kpt.P_ani, kpt.eps_n, R_nG)

        self.timer.stop('Davidson')
        error = gd.comm.sum(error)
        kpt.psit_nG[:] = psit_nG
        return error
