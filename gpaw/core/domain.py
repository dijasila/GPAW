from __future__ import annotations

import numpy as np

from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import ArrayLike1D, ArrayLike2D, ArrayLike, Array2D


def _normalize_cell(cell: ArrayLike) -> Array2D:
    cell = np.array(cell, float)
    if cell.ndim == 2:
        return cell
    if len(cell) == 3:
        return np.diag(cell)
    raise ValueError


class Domain:
    def __init__(self,
                 *,
                 cell: ArrayLike1D | ArrayLike2D,
                 pbc=(True, True, True),
                 kpt: ArrayLike1D = (0.0, 0.0, 0.0),
                 comm: MPIComm = serial_comm,
                 dtype=None):
        """"""
        self.cell = _normalize_cell(cell)
        self.pbc = np.array(pbc, bool)
        self.kpt = np.array(kpt, float)
        self.comm = comm

        assert dtype in [None, float, complex]
        if self.kpt.any():
            if dtype == float:
                raise ValueError
            dtype = complex
        else:
            dtype = dtype or float
        self.dtype = np.dtype(dtype)

    @property
    def icell(self):
        return np.linalg.inv(self.cell).T
