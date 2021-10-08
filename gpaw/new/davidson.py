from __future__ import annotations
from functools import partial
from typing import Callable

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.utilities.blas import axpy
from scipy.linalg import eigh
from gpaw.new.wave_functions import WaveFunctions
from gpaw.typing import Array1D
from gpaw.core.arrays import DistributedArrays as DA
from gpaw.core.atom_centered_functions import AtomArrays as AA

AAFunc = Callable[[AA, AA], AA]


def calculate_residuals(residuals: DA,
                        dH: AAFunc,
                        dS: AAFunc,
                        wfs: WaveFunctions,
                        p1: AA,
                        p2: AA) -> None:
    for r, e, p in zip(residuals.data, wfs.myeigs, wfs.wave_functions.data):
        axpy(-e, p, r)

    dH(wfs.projections, p1)
    p2.data[:] = wfs.projections.data * wfs.myeigs
    dS(p2, p2)
    p1.data -= p2.data
    wfs.projectors.add_to(residuals, p1)


def calculate_weights(converge: int | str, wfs: WaveFunctions) -> Array1D:
    """Calculate convergence weights for all eigenstates."""
    if converge == 'occupied':
        # Converge occupied bands:
        try:
            # Methfessel-Paxton distribution can give negative
            # occupation numbers - so we take the absolute value:
            return np.abs(wfs.occs)
        except ValueError:
            # No eigenvalues yet:
            return np.zeros(wfs.wave_functions.myshape) + np.inf

    1 / 0
    return np.zeros(42)

    """
    if isinstance(converge, int):
        # Converge fixed number of bands:
        n = self.nbands_converge - self.bd.beg
        if n > 0:
            for weight_n, kpt in zip(weight_un, wfs.kpt_u):
                weight_n[:n] = kpt.weight
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
    """


class Davidson:
    def __init__(self,
                 nbands,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 niter=2,
                 blocksize=10,
                 converge='occupied',
                 scalapack_parameters=None):
        self.niter = niter
        self.converge = converge

        B = nbands
        domain_comm = wf_grid.comm
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H = Matrix(2 * B, 2 * B, wf_grid.dtype)
            self.S = Matrix(2 * B, 2 * B, wf_grid.dtype)
            self.M = Matrix(B, B, wf_grid.dtype)

        self.work_array1 = wf_grid.empty(B, band_comm).data
        self.work_array2 = wf_grid.empty(B, band_comm).data

        self.preconditioner = preconditioner_factory(blocksize)

    def iterate(self, ibz_wfs, Ht, dH, dS) -> float:
        error = 0.0
        for wfs in ibz_wfs:
            e = self.iterate1(wfs, Ht, dH, dS)
            error += wfs.weight * e
        return error * ibz_wfs.spin_degeneracy

    def iterate1(self, wfs, Ht, dH, dS):
        H = self.H
        S = self.S
        M = self.M

        psit = wfs.wave_functions
        psit2 = psit.new(data=self.work_array1)
        psit3 = psit.new(data=self.work_array2)

        B = psit.shape[0]  # number of bands
        eigs = np.empty(2 * B)

        wfs.subspace_diagonalize(Ht, dH, work_array=psit2.data, Htpsit=psit3)
        residuals = psit3  # will become (H-e*S)|psit> later

        proj = wfs.projections
        proj2 = proj.new()
        proj3 = proj.new()

        domain_comm = psit.layout.comm
        band_comm = psit.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0 = M
        assert band_comm.size == 1

        if domain_comm.rank == 0:
            eigs[:B] = wfs.eigs

        def me(a, b, function=None):
            return a.matrix_elements(b, domain_sum=False, out=M,
                                     function=function)

        calculate_residuals(residuals, dH, dS, wfs, proj2, proj3)

        Ht = partial(Ht, out=residuals, spin=0)

        def copy(C_nn):
            domain_comm.sum(M.data, 0)
            if domain_comm.rank == 0:
                M.redist(M0)
                if band_comm.rank == 0:
                    C_nn[:] = M0.data

        for i in range(self.niter):
            if i == self.niter - 1:
                # Calulate error before we destroy residuals:
                weights = calculate_weights(self.converge, wfs)
                error = weights @ residuals.norm2()

            self.preconditioner(psit, residuals, out=psit2)

            # Calculate projections
            wfs.projectors.integrate(psit2, out=proj2)

            # <psi2 | H | psi2>
            me(psit2, psit2, function=Ht)
            dH(proj2, out=proj3)
            proj2.matrix.multiply(proj3, opa='C', symmetric=True, beta=1,
                                  out=M)
            copy(H.data[B:, B:])

            # <psi2 | H | psi>
            me(residuals, psit)
            proj3.matrix.multiply(proj, opa='C', beta=1.0, out=M)
            copy(H.data[B:, :B])

            # <psi2 | S | psi2>
            me(psit2, psit2)
            dS(proj2, out=proj3)
            proj2.matrix.multiply(proj3, opa='C', symmetric=True, beta=1,
                                  out=M)
            copy(S.data[B:, B:])

            # <psi2 | S | psi>
            me(psit2, psit)
            proj3.matrix.multiply(proj, opa='C', beta=1.0, out=M)
            copy(S.data[B:, :B])

            if is_domain_band_master:
                H.data[:B, :B] = np.diag(eigs[:B])
                S.data[:B, :B] = np.eye(B)
                if debug:
                    H.data[np.triu_indices(2 * B, 1)] = 42.0
                    S.data[np.triu_indices(2 * B, 1)] = 42.0

                eigs[:], H.data[:] = eigh(H.data, S.data,
                                          lower=True,
                                          check_finite=debug,
                                          overwrite_b=True)
                wfs._eigs = eigs[:B]

            if domain_comm.rank == 0:
                band_comm.broadcast(wfs.eigs, 0)
            domain_comm.broadcast(wfs.eigs, 0)

            if domain_comm.rank == 0:
                if band_comm.rank == 0:
                    M0.data[:] = H.data[:B, :B].T
                M0.redist(M)
            domain_comm.broadcast(M.data, 0)

            M.multiply(psit, out=residuals)
            proj.matrix.multiply(M, opb='T', out=proj3)

            if domain_comm.rank == 0:
                if band_comm.rank == 0:
                    M0.data[:] = H.data[B:, :B].T
                M0.redist(M)
            domain_comm.broadcast(M.data, 0)

            M.multiply(psit2, beta=1.0, out=residuals)
            proj2.matrix.multiply(M, opb='T', beta=1.0, out=proj3)
            psit.data[:] = residuals.data
            proj, proj3 = proj3, proj
            wfs._projections = proj

            if i < self.niter - 1:
                Ht(psit)
                calculate_residuals(residuals, dH, dS, wfs, proj2, proj3)

        return error
