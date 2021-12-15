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
                 kpt: ArrayLike1D = (0.0, 0.0, 0.0),
                 comm: MPIComm = serial_comm,
                 dtype: DTypeLike = None):
        """"""
        self.cell_cv = normalize_cell(cell)
        self.pbc_c = np.array(pbc, bool)
        self.kpt_c = np.array(kpt, float)
        self.comm = comm

        assert dtype in [None, float, complex]
        if self.kpt_c.any():
            if dtype == float:
                raise ValueError
            dtype = complex
        else:
            dtype = dtype or float
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
