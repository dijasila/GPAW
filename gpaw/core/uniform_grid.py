from __future__ import annotations
import numpy as np
from gpaw.typing import ArrayLike1D, ArrayLike, Array2D, ArrayLike2D
from gpaw.mpi import serial_comm, MPIComm
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.grid import GridRedistributor
from gpaw.core.arrays import DistributedArrays
from typing import Sequence
from gpaw.core.layout import Layout


def _normalize_cell(cell: ArrayLike) -> Array2D:
    cell = np.array(cell, float)
    if cell.ndim == 2:
        return cell
    if len(cell) == 3:
        return np.diag(cell)
    ...


class UniformGrid(Layout):
    def __init__(self,
                 *,
                 cell: ArrayLike1D | ArrayLike2D,
                 size: ArrayLike1D,
                 pbc=(True, True, True),
                 kpt: ArrayLike1D = (0.0, 0.0, 0.0),
                 comm: MPIComm = serial_comm,
                 decomposition: Sequence[Sequence[int]] = None,
                 dtype=None):
        """"""
        self.cell = _normalize_cell(cell)
        self.size = np.array(size, int)
        self.pbc = np.array(pbc, bool)
        self.kpt = np.array(kpt, float)
        self.comm = comm

        if decomposition is None:
            gd = GridDescriptor(size, pbc_c=pbc, comm=comm)
            decomposition = gd.n_cp
        self.decomposition = decomposition

        self.myposition = np.unravel_index(comm.rank,
                                           [len(d) - 1 for d in decomposition])
        self.start = np.array([d[p]
                               for d, p
                               in zip(decomposition, self.myposition)])
        self.end = np.array([d[p + 1]
                             for d, p
                             in zip(decomposition, self.myposition)])
        self.mysize = self.end - self.start

        Layout.__init__(self, tuple(self.mysize))

        assert dtype in [None, float, complex]
        if self.kpt.any():
            if dtype == float:
                raise ValueError
            dtype = complex
        else:
            dtype = dtype or float
        self.dtype = dtype

    @property
    def icell(self):
        return np.linalg.inv(self.cell).T

    def new(self,
            kpt=None,
            comm=None) -> UniformGrid:
        if comm is None:
            decomposition = self.decomposition
        else:
            decomposition = None
        return UniformGrid(cell=self.cell,
                           size=self.size,
                           pbc=self.pbc,
                           kpt=kpt or self.kpt,
                           comm=comm or self.comm,
                           decomposition=decomposition)

    def empty(self,
              shape: int | tuple[int] = (),
              comm: MPIComm = serial_comm) -> UniformGridFunctions:
        return UniformGridFunctions(self, shape, comm)

    def redistributor(self, other):
        return Redistributor(self, other)

    @property
    def _gd(self):
        return GridDescriptor(self.size, pbc_c=self.pbc, comm=self.comm,
                              parsize_c=[len(d) - 1
                                         for d in self.decomposition])


class Redistributor:
    def __init__(self, ug1, ug2):
        self.ug2 = ug2
        comm = max(ug1.dist.comm, ug2.dist.comm, key=lambda comm: comm.size)
        self.redistributor = GridRedistributor(comm, serial_comm,
                                               ug1.gd, ug2.gd)

    def distribute(self, input, out=None):
        out = out or self.ug2.empty(input.shape)
        self.redistributor.distribute(input.data, out._data)
        return out


class UniformGridFunctions(DistributedArrays):
    def __init__(self,
                 grid: UniformGrid,
                 shape: int | tuple[int] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        DistributedArrays. __init__(self, grid, shape, comm, data)
        self.grid = grid

    def __getitem__(self, index):
        return UniformGridFunctions(data=self.data[index], grid=self.grid)

    def _arrays(self):
        return self.data.reshape((-1,) + self.data.shape[-3:])

    def xy(self, *axes):
        assert len(axes) == 3 + len(self.shape)
        index = [slice(0, None) if axis is ... else axis for axis in axes]
        y = self.data[index]
        assert y.ndim == 1
        n = axes[-3:].index(...)
        dx = (self.ug.cell[n]**2).sum()**0.5
        x = np.arange(self.ug.dist.start[n], self.ug.dist.end[n]) * dx
        return x, y

    def redistribute(self, other):
        self.ug.redistributer(other.ug).redistribute(self, out=other)

    def fft(self, plan=None, pws=None, out=None):
        if out is None:
            out = pws.empty(self.shape)
        if pws is None:
            pws = out.pws
        plan = plan or pws.fft_plans()[0]
        for input, output in zip(self._arrays(), out._arrays()):
            plan.in_R[:] = input
            plan.execute()
            output[:] = plan.out_R.ravel()[pws.indices]
        return out
