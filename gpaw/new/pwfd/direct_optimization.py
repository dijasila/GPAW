from __future__ import annotations

from functools import partial
from typing import Generic, TypeVar, Callable, Optional, Tuple

import numpy as np

from gpaw.core import PWArray
from gpaw.core.arrays import DistributedArrays
from gpaw.new import zips
from gpaw.new.calculation import DFTState
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.pwfd import LBFGS, ArrayCollection
from gpaw.new.pwfd.davidson import calculate_weights
from gpaw.gpu import as_np

_TArray_co = TypeVar("_TArray_co", bound=DistributedArrays, covariant=True)


class DirectOptimizer(Eigensolver, Generic[_TArray_co]):
    def __init__(
        self,
        preconditioner_factory,
        blocksize=10,
        converge_bands='occupied',
        memory: int = 2,
        maxstep: float = 0.25,
    ):

        self.searchdir_algo = LBFGS(memory)
        self._maxstep = maxstep
        self.preconditioner: Optional[Callable] = None
        self.preconditioner_factory = preconditioner_factory
        self.blocksize = blocksize
        self.converge_bands = converge_bands
        assert converge_bands == 'occupied'
        # blocksize and precond were copied from Davidson

    def iterate(self, state: DFTState, hamiltonian: Hamiltonian) -> float:
        """In order to iterate, you need to get the weights and for this
        you need to know occupation numbers which are not initialized until
        we get the eigenvalues so the first iteration should be just
        the calculations of eigenvalues
        """

        kpt_comm = state.ibzwfs.kpt_comm
        xp = state.ibzwfs.xp

        assert state.ibzwfs.band_comm.size == 1, 'not implemented!'
        grad_u: ArrayCollection[_TArray_co]
        error: float
        weight_un: list = calculate_weights("occupied", state.ibzwfs)
        if weight_un[0] is None:
            # it's first iteration, eigenvalues and occupation numbers
            # are not available so just call grad_u to update eigenvalues
            # but don't move the wfs
            _, error = self.get_grad_u(state, hamiltonian, weight_un)
            return error

        grad_u, error = self.get_grad_u(
            state, hamiltonian, weight_un
        )

        if self.preconditioner is None:
            # we also initialize precond here,
            # don't know why this cannot be done in __init___
            # (the same in davidson)
            self.preconditioner = self.preconditioner_factory(
                self.blocksize, xp=xp
            )
        for wfs, grad_nX in zips(state.ibzwfs, grad_u):
            self.preconditioner(wfs.psit_nX, grad_nX, grad_nX)
            # the implemented preconditioner reverse the gradient
            # as well as it needs renormalization for spin degeneracy
            renormal_factor = -1 / (state.ibzwfs.spin_degeneracy)
            # if isinstance(wfs.psit_nX, PWArray):
            #     renormal_factor /= 2
            grad_nX.data *= renormal_factor

        self.searchdir_algo.update(grad_u, kpt_comm, xp)
        self.searchdir_algo.project_searchdir_vector(state.ibzwfs, weight_un)
        alpha: float = self.calc_step_length(state.ibzwfs)
        self.searchdir_algo.rescale_searchdir_vector(alpha)
        self.move_wave_functions(state.ibzwfs)

        return error

    def move_wave_functions(self, ibzwfs):
        for wfs, p_nG in zips(ibzwfs, self.searchdir_algo.search_dir_u):
            wfs.psit_nX.data += p_nG.data
            # wfs.pt_aiX.integrate(wfs.psit_nX, out=wfs._P_ani)
            wfs._P_ani = None  # is this how we update the projectors?
            wfs.orthonormalized = False
            wfs.orthonormalize()

    def calc_step_length(self, ibzwfs) -> float:
        norm = 0
        for p_nG in self.searchdir_algo.search_dir_u:
            for p_G in p_nG:
                norm += p_G.integrate(p_G).real

        norm = ibzwfs.kpt_comm.sum_scalar(norm)
        norm = ibzwfs.band_comm.sum_scalar(norm)
        norm = np.sqrt(norm)
        a_star = self._maxstep / norm if norm > self._maxstep else 1.0

        return a_star

    @staticmethod
    def get_grad_u(
        state, hamiltonian, weight_un
    ) -> Tuple["ArrayCollection[_TArray_co]", float]:
        dH = state.potential.dH
        Ht = partial(
            hamiltonian.apply,
            state.potential.vt_sR,
            state.potential.dedtaut_sR,
        )
        ibzwfs = state.ibzwfs
        wfs = ibzwfs.wfs_qs[0][0]
        dS_aii = wfs.setups.get_overlap_corrections(
            wfs.P_ani.layout.atomdist, wfs.xp
        )

        error = 0
        data_u: list[_TArray_co] = []
        for wfs, weight_n in zips(ibzwfs, weight_un):
            wfs.orthonormalize()

            # calc smooth part
            xp = wfs.xp
            Hpsi_nX = wfs.psit_nX.new(xp.empty_like(wfs.psit_nX.data))
            Ht(wfs.psit_nX, Hpsi_nX, wfs.spin)

            # calc paw part
            tmp_ani = wfs.P_ani.new()
            dH(wfs.P_ani, tmp_ani, wfs.spin)
            wfs.pt_aiX.add_to(Hpsi_nX, tmp_ani)

            # now need to project gradient on tangent plane
            # first apply overlap to psi.
            # we actually should apply overlap^-1 to Hpsi but when we do
            # that we get numerical instabilities for some reasons...
            wfs.P_ani.block_diag_multiply(dS_aii, out_ani=tmp_ani)
            Spsi_nX = wfs.psit_nX.copy()
            wfs.pt_aiX.add_to(Spsi_nX, tmp_ani)
            # projector matrix, which is also lagrange matrix to diagonalize
            # to get the eigenvalues
            psc_nn = Hpsi_nX.integrate(wfs.psit_nX)
            psc_nn = 0.5 * (psc_nn.conj() + psc_nn.T)

            eig_n = xp.linalg.eigvalsh(psc_nn)
            wfs.domain_comm.broadcast(eig_n, 0)
            wfs._eig_n = eig_n

            if weight_n is not None:
                psc_nn *= abs(
                    1 -
                    np.ceil(weight_n[:, np.newaxis]) -
                    np.ceil(weight_n[np.newaxis, :])
                )
            # project
            Hpsi_nX.data -= xp.tensordot(psc_nn, Spsi_nX.data, axes=1)

            # occupied only
            if weight_n is not None:
                Hpsi_nX.data *= np.ceil(weight_n[:, np.newaxis])

            data_u.append(Hpsi_nX)

            if weight_n is not None:
                tmp = weight_n @ as_np(Hpsi_nX.norm2())
                if wfs.ncomponents == 4:
                    tmp = tmp.sum()
            else:
                tmp = np.inf

            tmp = wfs.weight * tmp
            error += tmp

        error = ibzwfs.kpt_band_comm.sum_scalar(
            float(error)) * ibzwfs.spin_degeneracy

        return ArrayCollection(data_u), error

    def reset(self) -> None:
        """ sometimes you need to erase memory from LBFGS,
        for example when position of atoms moved. This is similar
        to reset in density mixing
        """
        if self.searchdir_algo is not None:
            self.searchdir_algo._local_iter = 0
