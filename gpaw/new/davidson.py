from functools import partial

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.utilities.blas import axpy


def calculate_residuals(residuals, dH, dS, wfs, p1, p2):
    for r, e, p in zip(residuals.data, wfs.myeigs, wfs.wave_functions.data):
        axpy(-e, p, r)

    dH(wfs.projections, out=p1)
    p2.data[:] = wfs.projections.data * wfs.myeigs
    dS(p2, out=p2)
    p1.data -= p2.data
    wfs.projectors.add_to(residuals, p1)


def calculate_weights(convergence, wfs):
    """Calculate convergence weights for all eigenstates."""
    weight_un = np.zeros((len(wfs.kpt_u), self.bd.mynbands))

    if isinstance(self.nbands_converge, int):
        # Converge fixed number of bands:
        n = self.nbands_converge - self.bd.beg
        if n > 0:
            for weight_n, kpt in zip(weight_un, wfs.kpt_u):
                weight_n[:n] = kpt.weight
    elif self.nbands_converge == 'occupied':
        # Conveged occupied bands:
        for weight_n, kpt in zip(weight_un, wfs.kpt_u):
            if kpt.f_n is None:  # no eigenvalues yet
                weight_n[:] = np.inf
            else:
                # Methfessel-Paxton distribution can give negative
                # occupation numbers - so we take the absolute value:
                weight_n[:] = np.abs(kpt.f_n)
    else:
        # Converge state with energy up to CBM + delta:
        assert self.nbands_converge.startswith('CBM+')
        delta = float(self.nbands_converge[4:]) / Ha

        if wfs.kpt_u[0].f_n is None:
            weight_un[:] = np.inf  # no eigenvalues yet
        else:
            # Collect all eigenvalues and calculate band gap:
            efermi = np.mean(wfs.fermi_levels)
            eps_skn = np.array(
                [[wfs.collect_eigenvalues(k, spin) - efermi
                  for k in range(wfs.kd.nibzkpts)]
                 for spin in range(wfs.nspins)])
            if wfs.world.rank > 0:
                eps_skn = np.empty((wfs.nspins,
                                    wfs.kd.nibzkpts,
                                    wfs.bd.nbands))
            wfs.world.broadcast(eps_skn, 0)
            try:
                # Find bandgap + positions of CBM:
                gap, _, (s, k, n) = _bandgap(eps_skn,
                                             spin=None, direct=False)
            except ValueError:
                gap = 0.0

            if gap == 0.0:
                cbm = efermi
            else:
                cbm = efermi + eps_skn[s, k, n]

            ecut = cbm + delta

            for weight_n, kpt in zip(weight_un, wfs.kpt_u):
                weight_n[kpt.eps_n < ecut] = kpt.weight

            if (eps_skn[:, :, -1] < ecut - efermi).any():
                # We don't have enough bands!
                weight_un[:] = np.inf

    return weight_un


class Davidson:
    def __init__(self,
                 nbands,
                 basis,
                 band_comm,
                 preconditioner_factory,
                 niter=2,
                 blocksize=10,
                 convergence='occupied',
                 scalapack_parameters=None):
        self.niter = niter
        B = nbands
        domain_comm = basis.comm
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H = Matrix(2 * B, 2 * B, basis.dtype)
            self.S = Matrix(2 * B, 2 * B, basis.dtype)

        self.work_array1 = basis.empty(B, band_comm).data
        self.work_array2 = basis.empty(B, band_comm).data

        self.preconditioner = preconditioner_factory(blocksize)

    def iterate(self, ibz_wfs, Ht, dH, dS):
        for wfs in ibz_wfs:
            self.iterate1(wfs, Ht, dH, dS)

    def iterate1(self, wfs, Ht, dH, dS):
        H = self.H
        S = self.S
        M = H.new()

        psit = wfs.wave_functions
        psit2 = psit.new(data=self.work_array1)
        psit3 = psit.new(data=self.work_array2)

        B = psit.shape[0]  # number of bands
        eigs = np.empty(2 * B)

        wfs.subspace_diagonalize(Ht, dH, psit2.data, psit3)
        residuals = psit3  # will become (H-e*S)|psit> later

        proj = wfs.projections
        proj2 = proj.new()
        proj3 = proj.new()

        domain_comm = psit.grid.comm
        band_comm = psit.comm

        if domain_comm.rank == 0:
            eigs[:B] = wfs.eigs

        def me(a, b, **kwargs):
            return a.matrix_elements(b, domain_sum=False, out=M, **kwargs)

        calculate_residuals(residuals, dH, dS, wfs, proj2, proj3)

        for i in range(self.niter):
            if i == self.niter - 1:
                errors = residuals.norm2()

            self.preconditioner.apply(psit, residuals, out=psit2)

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

        return errors
