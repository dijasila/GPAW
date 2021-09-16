from functools import partial

import numpy as np
from ase.utils.timing import timer
from gpaw import debug
from gpaw.eigensolvers.diagonalizerbackend import (ScalapackDiagonalizer,
                                                   ScipyDiagonalizer)
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.hybrids import HybridXC
from gpaw.utilities import unpack


class DummyArray:
    def __getitem__(self, x):
        return np.empty((0, 0))


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

    def __init__(
            self, niter=2):
        Eigensolver.__init__(self)
        self.niter = niter
        self.diagonalizer_backend = None

        self.orthonormalization_required = False
        self.H_NN = DummyArray()
        self.S_NN = DummyArray()
        self.eps_N = DummyArray()

    def __repr__(self):
        return 'Davidson(niter=%d)' % (
            self.niter)

    def todict(self):
        return {'name': 'dav', 'niter': self.niter}

    def initialize(self, wfs):
        Eigensolver.initialize(self, wfs)
        slcomm, nrows, ncols, slsize = wfs.scalapack_parameters

        if wfs.gd.comm.rank == 0 and wfs.bd.comm.rank == 0:
            # Allocate arrays
            B = self.nbands
            self.H_NN = np.zeros((2 * B, 2 * B), self.dtype)
            self.S_NN = np.zeros((2 * B, 2 * B), self.dtype)
            self.eps_N = np.zeros(2 * B)

        if slsize is not None:
            self.diagonalizer_backend = ScalapackDiagonalizer(
                arraysize=self.nbands * 2,
                grid_nrows=nrows,
                grid_ncols=ncols,
                scalapack_communicator=slcomm,
                dtype=self.dtype,
                blocksize=slsize)
        else:
            self.diagonalizer_backend = ScipyDiagonalizer()

    def estimate_memory(self, mem, wfs):
        Eigensolver.estimate_memory(self, mem, wfs)
        nbands = wfs.bd.nbands
        mem.subnode('H_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_nn', nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('H_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('S_2n2n', 4 * nbands * nbands * mem.itemsize[wfs.dtype])
        mem.subnode('eps_2n', 2 * nbands * mem.floatsize)

    @timer('Davidson')
    def iterate_one_k_point(self, ham, wfs, kpt, weights):
        """Do Davidson iterations for the kpoint"""
        if isinstance(ham.xc, HybridXC):
            self.niter = 1

        bd = wfs.bd
        B = bd.nbands

        H_NN = self.H_NN
        S_NN = self.S_NN
        eps_N = self.eps_N

        def integrate(a_G):
            if wfs.collinear:
                return np.real(wfs.integrate(a_G, a_G, global_integral=False))
            return sum(
                np.real(wfs.integrate(b_G, b_G, global_integral=False))
                for b_G in a_G)

        self.subspace_diagonalize(ham, wfs, kpt)

        psit = kpt.psit.wave_functions
        psit2 = psit.new(data=wfs.work_array)

        proj = kpt.projections
        proj2 = proj.new()
        proj3 = proj.new()

        M = wfs.work_matrix_nn

        comm = wfs.gd.comm

        is_gridband_master: bool = (
            comm.rank == 0) and (bd.comm.rank == 0)

        if bd.comm.size > 1:
            M0 = M.new(dist=(bd.comm, 1, 1))
        else:
            M0 = M

        if comm.rank == 0:
            e_N = bd.collect(kpt.eps_n)
            if e_N is not None:
                eps_N[:B] = e_N

        if self.keep_htpsit:
            residual = psit.new(data=self.Htpsit_nG)
        else:
            residual = psit.new()

        def Ht(x):
            wfs.apply_pseudo_hamiltonian(kpt, ham, x.data, residual.data)
            return residual

        def dH(proj, out):
            for a, I1, I2 in proj.layout.myindices:
                dh = unpack(ham.dH_asp[a][kpt.s])
                out.data[I1:I2] = dh @ proj.data[I1:I2]
            return out

        def dS(proj, out):
            for a, I1, I2 in proj.layout.myindices:
                ds = wfs.setups[a].dO_ii
                out.data[I1:I2] = ds @ proj.data[I1:I2]
            return out

        def me(a, b, **kwargs):
            return a.matrix_elements(b, domain_sum=False, out=M, **kwargs)

        if not self.keep_htpsit:
            Ht(psit, proj3)

        self.calculate_residuals(kpt, wfs, ham, dH, dS,
                                 psit, proj, kpt.eps_n, residual,
                                 proj2, proj3)

        precond = self.preconditioner

        for nit in range(self.niter):
            if nit == self.niter - 1:
                error = np.dot(weights, [integrate(residual_G)
                                         for residual_G in residual.data])

            for psit_G, residual_G, psit2_G in zip(psit.data, residual.data,
                                                   psit2.data):
                ekin = precond.calculate_kinetic_energy(psit_G, kpt)
                precond(residual_G, kpt, ekin, out=psit2_G)

            # Calculate projections
            kpt.projectors.integrate(psit2, out=proj2)

            def copy(M, C_nn):
                comm.sum(M.data, 0)
                if comm.rank == 0:
                    M.redist(M0)
                    if bd.comm.rank == 0:
                        C_nn[:] = M0.data

            # <psi2 | H | psi2>
            me(psit2, psit2, function=Ht)
            me(proj2, proj2,
               function=partial(dH, out=proj3),
               add_to_out=True)
            copy(M, H_NN[B:, B:])

            # <psi2 | H | psi>
            me(residual, psit)
            proj3.matrix.multiply(proj, opa='C', beta=1.0, out=M)
            copy(M, H_NN[B:, :B])

            # <psi2 | S | psi2>
            me(psit2, psit2)
            me(proj2, proj2,
               function=partial(dS, out=proj3),
               add_to_out=True)
            copy(M, S_NN[B:, B:])

            # <psi2 | S | psi>
            me(psit2, psit)
            proj3.matrix.multiply(proj, opa='C', beta=1.0, out=M)
            copy(M, S_NN[B:, :B])

            if is_gridband_master:
                H_NN[:B, :B] = np.diag(eps_N[:B])
                S_NN[:B, :B] = np.eye(B)

            if is_gridband_master and debug:
                H_NN[np.triu_indices(2 * B, 1)] = 42.0
                S_NN[np.triu_indices(2 * B, 1)] = 42.0

            self.diagonalizer_backend.diagonalize(
                H_NN, S_NN, eps_N,
                is_master=is_gridband_master,
                debug=debug)

            if comm.rank == 0:
                bd.distribute(eps_N[:B], kpt.eps_n)
            comm.broadcast(kpt.eps_n, 0)

            if comm.rank == 0:
                if bd.comm.rank == 0:
                    M0.data[:] = H_NN[:B, :B].T
                M0.redist(M)
            comm.broadcast(M.data, 0)

            M.multiply(psit, out=residual)
            proj.matrix.multiply(M, opb='T', out=proj3)

            if comm.rank == 0:
                if bd.comm.rank == 0:
                    M0.data[:] = H_NN[B:, :B].T
                M0.redist(M)
            comm.broadcast(M.data, 0)

            M.multiply(psit2, beta=1.0, out=residual)
            proj2.matrix.multiply(M, opb='T', beta=1.0, out=proj3)
            psit.data[:] = residual.data
            proj, proj3 = proj3, proj
            kpt.projections = proj

            if nit < self.niter - 1:
                Ht(psit)
                self.calculate_residuals(
                    kpt, wfs,
                    ham, dH, dS,
                    psit, proj, kpt.eps_n, residual,
                    proj2, proj3)

        error = wfs.gd.comm.sum(error)
        return error
