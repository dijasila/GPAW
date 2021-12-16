from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import ArrayLike1D, ArrayLike2D, ArrayLike, Array2D
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpaw.core.arrays import DistributedArrays


def normalize_cell(cell: ArrayLike) -> Array2D:
    """...

    >>> normalize_cell([1, 2, 3])
    array([[1., 0., 0.],
           [0., 2., 0.],
           [0., 0., 3.]])
    """
    cell = np.array(cell, float)
    if cell.ndim == 2:
        return cell
    if len(cell) == 3:
        return np.diag(cell)
    raise ValueError


class Domain:
    def __init__(self,
                 cell: ArrayLike1D | ArrayLike2D,
                 pbc=(True, True, True),
                 kpt: ArrayLike1D = None,
                 comm: MPIComm = serial_comm,
                 dtype: DTypeLike = None):
        """"""
        self.cell_cv = normalize_cell(cell)
        self.pbc_c = np.array(pbc, bool)
        self.comm = comm

        assert dtype in [None, float, complex]

        if kpt is not None:
            if dtype is None:
                dtype = complex
            if dtype == float:
                raise ValueError(f'dtype must be complex for kpt={kpt}')
        else:
            if dtype is None:
                dtype = float
            kpt = (0.0, 0.0, 0.0)

        self.kpt_c = np.array(kpt, float)

        if self.kpt_c.any():
            for p, k in zip(pbc, kpt):
                if not p and k != 0:
                    raise ValueError(f'Bad k-point {kpt} for pbc={pbc}')

        self.dtype = np.dtype(dtype)

    @property
    def cell(self):
        return self.cell_cv.copy()

    @property
    def pbc(self):
        return self.pbc_c.copy()

    @property
    def kpt(self):
        return self.kpt_c.copy()

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> DistributedArrays:
        raise NotImplementedError

    def zeros(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> DistributedArrays:
        array = self.empty(shape, comm)
        array.data[:] = 0.0
        return array

    @property
    def icell(self):
        return np.linalg.inv(self.cell).T
