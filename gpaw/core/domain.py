from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
# from numpy.typing import DTypeLike
from ase.geometry.cell import cellpar_to_cell
from gpaw.fftw import get_efficient_fft_size
from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import Array2D, ArrayLike, ArrayLike1D, ArrayLike2D, Vector

if TYPE_CHECKING:
    from gpaw.core import UniformGrid
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
    return cellpar_to_cell(cell)


class Domain:
    def __init__(self,
                 cell: ArrayLike1D | ArrayLike2D,
                 pbc=(True, True, True),
                 kpt: Vector = None,
                 comm: MPIComm = serial_comm,
                 dtype=None):
        """"""
        if isinstance(pbc, int):
            pbc = (pbc,) * 3
        self.cell_cv = normalize_cell(cell)
        self.pbc_c = np.array(pbc, bool)
        self.comm = comm

        assert dtype in [None, float, complex]

        if kpt is not None:
            if dtype is None:
                dtype = complex
        else:
            if dtype is None:
                dtype = float
            kpt = (0.0, 0.0, 0.0)

        self.kpt_c = np.array(kpt, float)

        if self.kpt_c.any():
            if dtype == float:
                raise ValueError(f'dtype must be complex for kpt={kpt}')
            for p, k in zip(pbc, self.kpt_c):
                if not p and k != 0:
                    raise ValueError(f'Bad k-point {kpt} for pbc={pbc}')

        self.dtype = np.dtype(dtype)

    def __repr__(self):
        comm = self.comm
        if self.kpt_c.any():
            k = f', kpt={self.kpt_c.tolist()}'
        else:
            k = ''
        return (f'Domain(cell={self.cell_cv.tolist()}, '
                f'pbc={self.pbc_c.tolist()}, '
                f'comm={comm.rank}/{comm.size}, '
                f'dtype={self.dtype}{k})')

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

    def uniform_grid_with_grid_spacing(self,
                                       grid_spacing: float,
                                       n: int = 1,
                                       factors: Sequence[int] = (2, 3, 5, 7)
                                       ) -> UniformGrid:
        from gpaw.core import UniformGrid

        L_c = (np.linalg.inv(self.cell_cv)**2).sum(0)**-0.5
        size_c = np.maximum(n, (L_c / grid_spacing / n + 0.5).astype(int) * n)
        if factors:
            size_c = np.array([get_efficient_fft_size(N, n, factors)
                               for N in size_c])
        return UniformGrid(size=size_c,
                           cell=self.cell_cv,
                           pbc=self.pbc_c,
                           kpt=self.kpt_c,
                           dtype=self.dtype,
                           comm=self.comm)
