from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from gpaw import debug
from gpaw.core.arrays import DistributedArrays as DA
from gpaw.core.atom_centered_functions import AtomArrays as AA
from gpaw.core.matrix import Matrix
from gpaw.new.calculation import DFTState
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array1D, Array2D
from gpaw.utilities.blas import axpy
from gpaw.yml import obj2yaml as o2y
from scipy.linalg import eigh

AAFunc = Callable[[AA, AA], AA]


class Davidson(Eigensolver):
    def __init__(self,
                 nbands,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 niter=2,
                 blocksize=10,
                 converge_bands='occupied',
                 scalapack_parameters=None):
        self.niter = niter
        self.converge_bands = converge_bands

        B = nbands
        domain_comm = wf_grid.comm
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H_NN = Matrix(2 * B, 2 * B, wf_grid.dtype)
            self.S_NN = Matrix(2 * B, 2 * B, wf_grid.dtype)
        else:
            self.H_NN = self.S_NN = EmptyMatrix()

        self.M_nn = Matrix(B, B, wf_grid.dtype,
                           dist=(band_comm, band_comm.size))

        self.work_arrays: np.ndarray | None = None

        self.preconditioner = preconditioner_factory(blocksize)

    def __str__(self):
        return o2y(dict(name='Davidson',
                        niter=self.niter,
                        converge_bands=self.converge_bands))

    def iterate(self, state: DFTState, hamiltonian: Hamiltonian) -> float:
        """Iterate on state given fixed hamiltonian.

        Returns
        -------
        float:
            Weighted error of residuals:::

                   ~     ~ ~
              R = (H - ε S)ψ
               n        n   n
        """
        if self.work_arrays is None:
            # First time: allocate work-arrays
            shape = state.ibzwfs.get_max_shape()
            shape = (2, state.ibzwfs.nbands) + shape
            wfs = state.ibzwfs.wfs_qs[0][0]
            assert isinstance(wfs, PWFDWaveFunctions)
            dtype = wfs.psit_nX.data.dtype
            self.work_arrays = np.empty(shape, dtype)

        dS = state.ibzwfs.wfs_qs[0][0].setups.overlap_correction
        dH = state.potential.dH
        Ht = partial(hamiltonian.apply, state.potential.vt_sR)
        ibzwfs = state.ibzwfs
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
        psit2_nX = psit_nX.new(data=self.work_arrays[0])
        psit3_nX = psit_nX.new(data=self.work_arrays[1])

        B = psit_nX.dims[0]  # number of bands
        eig_N = np.empty(2 * B)

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn
        assert band_comm.size == 1

        if domain_comm.rank == 0:
            eig_N[:B] = wfs.eig_n

        def me(a, b, function=None):
            """Matrix elements"""
            return a.matrix_elements(b,
                                     domain_sum=False,
                                     out=M_nn,
                                     function=function,
                                     cc=True)

        Ht = partial(Ht, out=residual_nX, spin=wfs.spin)
        dH = partial(dH, spin=wfs.spin)

        calculate_residuals(residual_nX, dH, dS, wfs, P2_ani, P3_ani)

        def copy(C_nn: Array2D) -> None:
            domain_comm.sum(M_nn.data, 0)
            if domain_comm.rank == 0:
                M_nn.redist(M0_nn)
                if band_comm.rank == 0:
                    C_nn[:] = M0_nn.data

        for i in range(self.niter):
            if i == self.niter - 1:  # last iteration
                # Calulate error before we destroy residuals:
                weight_n = calculate_weights(self.converge_bands, wfs)
                if weight_n is None:
                    error = np.inf
                else:
                    error = weight_n @ residual_nX.norm2()
                    if wfs.ncomponents == 4:
                        error = error.sum()

            self.preconditioner(psit_nX, residual_nX, out=psit2_nX)

            # Calculate projections
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)

            # <psi2 | H | psi2>
            me(psit2_nX, psit2_nX, function=Ht)
            dH(P2_ani, out_ani=P3_ani)
            P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                   out=M_nn)
            copy(H_NN.data[B:, B:])

            # <psi2 | H | psi>
            me(residual_nX, psit_nX)
            P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
            copy(H_NN.data[B:, :B])

            # <psi2 | S | psi2>
            me(psit2_nX, psit2_nX)
            dS(P2_ani, out_ani=P3_ani)
            P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                   out=M_nn)
            copy(S_NN.data[B:, B:])

            # <psi2 | S | psi>
            me(psit2_nX, psit_nX)
            P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
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
                    M0_nn.data[:] = H_NN.data[:B, :B]
                M0_nn.redist(M_nn)
            domain_comm.broadcast(M_nn.data, 0)

            M_nn.multiply(psit_nX, opa='C', out=residual_nX)
            M_nn.multiply(P_ani, opa='C', out=P3_ani)

            if domain_comm.rank == 0:
                if band_comm.rank == 0:
                    M0_nn.data[:] = H_NN.data[B:, :B]
                M0_nn.redist(M_nn)
            domain_comm.broadcast(M_nn.data, 0)

            M_nn.multiply(psit2_nX, opa='C', beta=1.0, out=residual_nX)
            M_nn.multiply(P2_ani, opa='C', beta=1.0, out=P3_ani)
            psit_nX.data[:] = residual_nX.data
            P_ani, P3_ani = P3_ani, P_ani
            wfs._P_ani = P_ani

            if i < self.niter - 1:
                Ht(psit_nX)
                calculate_residuals(residual_nX, dH, dS, wfs, P2_ani, P3_ani)

        return error


def calculate_residuals(residuals_nX: DA,
                        dH: AAFunc,
                        dS: AAFunc,
                        wfs: PWFDWaveFunctions,
                        P1_ani: AA,
                        P2_ani: AA) -> None:
    for r, e, p in zip(residuals_nX.data, wfs.myeig_n, wfs.psit_nX.data):
        axpy(-e, p, r)

    dH(wfs.P_ani, P1_ani)
    if wfs.ncomponents < 4:
        subscripts = 'nI, n -> nI'
    else:
        subscripts = 'nsI, n -> nsI'
    np.einsum(subscripts, wfs.P_ani.data, wfs.myeig_n, out=P2_ani.data)
    dS(P2_ani, P2_ani)
    P1_ani.data -= P2_ani.data
    wfs.pt_aiX.add_to(residuals_nX, P1_ani)


def calculate_weights(converge_bands: int | str,
                      wfs: PWFDWaveFunctions) -> Array1D | None:
    """Calculate convergence weights for all eigenstates."""
    if converge_bands == 'occupied':
        # Converge occupied bands:
        try:
            # Methfessel-Paxton distribution can give negative
            # occupation numbers - so we take the absolute value:
            return np.abs(wfs.occ_n)
        except ValueError:
            # No eigenvalues yet:
            return None

    if isinstance(converge_bands, int):
        # Converge fixed number of bands:
        n = converge_bands
        if wfs.psit_nX.comm.size > 1:
            raise NotImplementedError
            n -= 117
        if n > 0:
            weight_n = np.zeros(wfs.psit_nX.mydims[0])
            weight_n[:n] = 1.0
            return weight_n

    else:
        assert False
        # Converge state with energy up to CBM + delta:
        assert converge_bands.startswith('CBM+')
        # delta = float(converge_bands[4:]) / Ha
        return None

        """
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
        """
    return None


class EmptyMatrix:
    data = np.empty((0, 0))
