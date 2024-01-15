from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from gpaw.mpi import MPIComm, serial_comm

if TYPE_CHECKING:
    from gpaw.new.pwfd import ArrayCollection

from gpaw.core.arrays import DistributedArrays

_TArray_co = TypeVar("_TArray_co", bound=DistributedArrays, covariant=True)


class LBFGS:
    def __init__(
        self,
        grad_cur_u: ArrayCollection,
        memory: int = 2,
        kpt_comm: MPIComm = serial_comm,
        xp: ModuleType = np,
    ):

        self.grad_old_u: ArrayCollection = grad_cur_u.make_copy()
        self.search_dir_u: ArrayCollection = -grad_cur_u

        self.ds_mqs: list[ArrayCollection] = [grad_cur_u.empty()] * memory
        self.dy_mqs: list[ArrayCollection] = [grad_cur_u.empty()] * memory
        self.rho_m: list[float] = [0] * memory

        self._local_iter: int = 1
        self._memory: int = memory

        self.kpt_comm = kpt_comm
        self.xp = xp

    def update(
        self, grad_cur_u: ArrayCollection[_TArray_co]
    ) -> "ArrayCollection[_TArray_co]":

        m = self._local_iter % self._memory
        # ds = a_cur - a_old, which is search dir
        # but in some cases only the difference is known
        self.ds_mqs[m] = self.search_dir_u.make_copy()
        self.dy_mqs[m] = grad_cur_u - self.grad_old_u
        dyds = self.ds_mqs[m].dot(self.dy_mqs[m])
        dyds = self.kpt_comm.sum_scalar(dyds)
        if abs(dyds) > 1.0e-20:
            self.rho_m[m] = 1.0 / dyds
        else:
            self.rho_m[m] = 1.0e20

        if self.rho_m[m] < 0:
            # reset the optimizer
            self.grad_old_u = grad_cur_u.make_copy()
            self.search_dir_u = -grad_cur_u
            self.ds_mqs = [grad_cur_u.empty()] * self._memory
            self.dy_mqs = [grad_cur_u.empty()] * self._memory
            self.rho_m = [0] * self._memory
            self._local_iter = 1
            return self.search_dir_u

        q = grad_cur_u.make_copy()
        k = self._memory - 1
        alpha = [0.0] * self._memory

        while k > -1:
            c_ind = (k + m + 1) % self._memory
            k -= 1
            sq = q.dot(self.ds_mqs[c_ind])
            sq = self.kpt_comm.sum_scalar(sq)
            alpha[c_ind] = self.rho_m[c_ind] * sq
            q -= self.dy_mqs[c_ind].multiply_by_number(alpha[c_ind])

        yy = self.dy_mqs[m].dot(self.dy_mqs[m])
        yy = self.kpt_comm.sum_scalar(yy)

        devis: float = self.xp.maximum(self.rho_m[m] * yy, 1.0e-20)
        self.search_dir_u = q.multiply_by_number(1 / devis)

        for k in range(self._memory):
            if self._local_iter < self._memory:
                c_ind = k
            else:
                c_ind = (k + m + 1) % self._memory

            yr = self.search_dir_u.dot(self.dy_mqs[c_ind])
            yr = self.kpt_comm.sum_scalar(yr)

            beta = self.rho_m[c_ind] * yr
            self.search_dir_u += self.ds_mqs[c_ind].multiply_by_number(
                alpha[c_ind] - beta
            )

        self.grad_old_u = grad_cur_u.make_copy()
        self.search_dir_u.multiply_by_number_in_place(-1)
        self._local_iter += 1
        return self.search_dir_u

    def rescale_searchdir_vector(self, alpha: float) -> None:
        self.search_dir_u.multiply_by_number_in_place(alpha)
