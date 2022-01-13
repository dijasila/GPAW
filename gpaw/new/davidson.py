from __future__ import annotations
from functools import partial
from typing import Callable

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.utilities.blas import axpy
from scipy.linalg import eigh
from gpaw.new.wave_functions import WaveFunctions
from gpaw.typing import Array1D, Array2D
from gpaw.core.arrays import DistributedArrays as DA
from gpaw.core.atom_centered_functions import AtomArrays as AA

AAFunc = Callable[[AA, AA], AA]


def calculate_residuals(residuals_nX: DA,
                        dH: AAFunc,
                        dS: AAFunc,
                        wfs: WaveFunctions,
                        P1_ain: AA,
                        P2_ain: AA) -> None:
    for r, e, p in zip(residuals_nX.data, wfs.myeig_n, wfs.psit_nX.data):
        axpy(-e, p, r)

    dH(wfs.P_ain, P1_ain)
    P2_ain.data[:] = wfs.P_ain.data * wfs.myeig_n
    dS(P2_ain, P2_ain)
    P1_ain.data -= P2_ain.data
    wfs.pt_aiX.add_to(residuals_nX, P1_ain)


def calculate_weights(converge: int | str, wfs: WaveFunctions) -> Array1D:
    """Calculate convergence weights for all eigenstates."""
    if converge == 'occupied':
        # Converge occupied bands:
        try:
            # Methfessel-Paxton distribution can give negative
            # occupation numbers - so we take the absolute value:
            return np.abs(wfs.occ_n)
        except ValueError:
            # No eigenvalues yet:
            return np.zeros(wfs.psit_nX.mydims) + np.inf

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


class EmptyMatrix:
    data = np.empty((0, 0))


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
            self.H_NN = Matrix(2 * B, 2 * B, wf_grid.dtype)
            self.S_NN = Matrix(2 * B, 2 * B, wf_grid.dtype)
        else:
            self.H_NN = self.S_NN = EmptyMatrix()

        self.M_nn = Matrix(B, B, wf_grid.dtype,
                           dist=(band_comm, band_comm.size))

        self.work_array1 = wf_grid.empty(B, band_comm).data
        self.work_array2 = wf_grid.empty(B, band_comm).data

        self.preconditioner = preconditioner_factory(blocksize)

    def iterate(self, ibzwfs, Ht, dH, dS) -> float:
        error = 0.0
        for wfs in ibzwfs:
            e = self.iterate1(wfs, Ht, dH, dS)
            error += wfs.weight * e
        return ibzwfs.kpt_comm.sum(error) * ibzwfs.spin_degeneracy

    def iterate1(self, wfs, Ht, dH, dS):
        H_NN = self.H_NN
        S_NN = self.S_NN
        M_nn = self.M_nn

        psit_nX = wfs.psit_nX
        psit2_nX = psit_nX.new(data=self.work_array1)
        psit3_nX = psit_nX.new(data=self.work_array2)

        B = psit_nX.dims[0]  # number of bands
        eig_N = np.empty(2 * B)

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later

        P_ain = wfs.P_ain
        P2_ain = P_ain.new()
        P3_ain = P_ain.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn
        assert band_comm.size == 1

        if domain_comm.rank == 0:
            eig_N[:B] = wfs.eig_n

        def me(a, b, function=None):
            """Matrix elements"""
            return a.matrix_elements(b, domain_sum=False, out=M_nn,
                                     function=function)

        Ht = partial(Ht, out=residual_nX, spin=wfs.spin)
        dH = partial(dH, spin=wfs.spin)

        calculate_residuals(residual_nX, dH, dS, wfs, P2_ain, P3_ain)

        def copy(C_nn: Array2D) -> None:
            domain_comm.sum(M_nn.data, 0)
            if domain_comm.rank == 0:
                M_nn.redist(M0_nn)
                if band_comm.rank == 0:
                    C_nn[:] = M0_nn.data

        for i in range(self.niter):
            if i == self.niter - 1:
                # Calulate error before we destroy residuals:
                weights_n = calculate_weights(self.converge, wfs)
                error = weights_n @ residual_nX.norm2()

            self.preconditioner(psit_nX, residual_nX, out=psit2_nX)

            # Calculate projections
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ain)

            # <psi2 | H | psi2>
            me(psit2_nX, psit2_nX, function=Ht)
            dH(P2_ain, out=P3_ain)
            P2_ain.matrix.multiply(P3_ain, opa='C', symmetric=True, beta=1,
                                   out=M_nn)
            copy(H_NN.data[B:, B:])

            # <psi2 | H | psi>
            me(residual_nX, psit_nX)
            P3_ain.matrix.multiply(P_ain, opa='C', beta=1.0, out=M_nn)
            copy(H_NN.data[B:, :B])

            # <psi2 | S | psi2>
            me(psit2_nX, psit2_nX)
            dS(P2_ain, out=P3_ain)
            P2_ain.matrix.multiply(P3_ain, opa='C', symmetric=True, beta=1,
                                   out=M_nn)
            copy(S_NN.data[B:, B:])

            # <psi2 | S | psi>
            me(psit2_nX, psit_nX)
            P3_ain.matrix.multiply(P_ain, opa='C', beta=1.0, out=M_nn)
            copy(S_NN.data[B:, :B])

            if is_domain_band_master:
                H_NN.data[:B, :B] = np.diag(eig_N[:B])
                S_NN.data[:B, :B] = np.eye(B)
                if debug:
                    H_NN.data[np.triu_indices(2 * B, 1)] = 42.0
                    S_NN.data[np.triu_indices(2 * B, 1)] = 42.0

                eig_N[:], H_NN.data[:] = eigh(H_NN.data, S_NN.data,
                                              lower=True,
                                              check_finite=debug,
                                              overwrite_b=True)
                wfs._eig_n = eig_N[:B]

            if domain_comm.rank == 0:
                band_comm.broadcast(wfs.eig_n, 0)
            domain_comm.broadcast(wfs.eig_n, 0)

            if domain_comm.rank == 0:
                if band_comm.rank == 0:
                    M0_nn.data[:] = H_NN.data[:B, :B].T
                M0_nn.redist(M_nn)
            domain_comm.broadcast(M_nn.data, 0)

            M_nn.multiply(psit_nX, out=residual_nX)
            P_ain.matrix.multiply(M_nn, opb='T', out=P3_ain)

            if domain_comm.rank == 0:
                if band_comm.rank == 0:
                    M0_nn.data[:] = H_NN.data[B:, :B].T
                M0_nn.redist(M_nn)
            domain_comm.broadcast(M_nn.data, 0)

            M_nn.multiply(psit2_nX, beta=1.0, out=residual_nX)
            P2_ain.matrix.multiply(M_nn, opb='T', beta=1.0, out=P3_ain)
            psit_nX.data[:] = residual_nX.data
            P_ain, P3_ain = P3_ain, P_ain
            wfs._P_ain = P_ain

            if i < self.niter - 1:
                Ht(psit_nX)
                calculate_residuals(residual_nX, dH, dS, wfs, P2_ain, P3_ain)

        return error
