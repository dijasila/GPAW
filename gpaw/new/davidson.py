from functools import partial

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix

        for R_G, eps, psit_G in zip(R.data, eps_n, psit.data):
            axpy(-eps, psit_G, R_G)

        dH(projections, out=tmp1)
        tmp2.data[:] = projections.data * eps_n
        dS(tmp2, out=tmp2)
        tmp1.data -= tmp2.data

        ham.xc.add_correction(kpt, psit.data, R.data,
                              projections,
                              tmp1,
                              n_x,
                              calculate_change)
        kpt.projectors.add_to(R, tmp1)

class Davidson:
    def __init__(self,
                 nbands, band_comm, basis,
                 niter=2, scalapack_parameters=None):
        B = nbands
        domain_comm = basis.comm
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H = Matrix(2 * B, 2 * B, basis.dtype)
            self.S = Matrix(2 * B, 2 * B, basis.dtype)

        self.work_array1 = basis.empty(B, band_comm)
        self.work_array2 = basis.empty(B, band_comm)

    def iterate(self, ibzwfs, hamiltonian):
        for wfs in ibzwfs:
            self.iterate1(wfs, Ht, dH, dS)

    def iterate1(self, wfs, Ht, dH, dS):
        H = self.H
        S = self.S
        M = H.new()

        B = len(H)  # number of bands

        eigs = np.empty(2 * B)

        psit = wfs.wave_functions
        psit2 = psit.new(data=self.work_array1)
        psit3 = psit.new(data=self.work_array2)

        wfs.subspace_diagonalize(Ht, dH, psit2, psit3)
        residuals = psit3  # will become (H-e*S)|psit> later

        proj = wfs.projections
        proj2 = proj.new()
        proj3 = proj.new()

        domain_comm = wfs.grid.comm
        band_comm = wfs.comm

        if domain_comm.rank == 0:
            eigs[:B] = wfs.eigs

        def me(a, b, **kwargs):
            return a.matrix_elements(b, domain_sum=False, out=M, **kwargs)

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
