from __future__ import annotations
import numpy as np
from gpaw.typing import ArrayLike1D, ArrayLike, Array2D, ArrayLike2D
from gpaw.mpi import serial_comm
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.grid import GridRedistributor
from gpaw.core.distribution import create_shape_distributuion
from typing import Any

MPIComm = Any


def _normalize_cell(cell: ArrayLike) -> Array2D:
    cell = np.array(cell, float)
    if cell.ndim == 2:
        return cell
    if len(cell) == 3:
        return np.diag(cell)
    ...


class UniformGridDistribution:
    def __init__(self, comm, size, pbc, decomposition=None):
        self.comm = comm
        if decomposition is None:
            decomposition = GridDescriptor(size, pbc_c=pbc).n_cp
        self.decomposition = decomposition

        self.myposition = np.unravel_index(comm.rank,
                                           [len(d) - 1 for d in decomposition])
        self.start = tuple([d[p]
                            for d, p in zip(decomposition, self.myposition)])
        self.end = tuple([d[p + 1]
                          for d, p in zip(decomposition, self.myposition)])
        self.size = tuple([e - s for s, e in zip(self.start, self.end)])
        self.total_size = size


class UniformGrid:
    def __init__(self,
                 *
                 cell: ArrayLike1D | ArrayLike2D,
                 size: ArrayLike1D = None,
                 pbc=(True, True, True),
                 kpt: ArrayLike1D = None,
                 dist: MPIComm | UniformGridDistribution | None = None):
        """"""
        self.cell = _normalize_cell(cell)
        self.pbc = np.array(pbc, bool)
        self.kpt = kpt

        if isinstance(dist, UniformGridDistribution):
            assert size is None
            size = dist.total_size
        else:
            dist = dist or serial_comm
            dist = UniformGridDistribution(dist, size, pbc)
        self.dist = dist
        self.size = size

        self.dtype = float if kpt is None else complex
        self.icell = np.linalg.inv(self.cell).T

    def new(self, kpt='_default', dist='_default') -> UniformGrid:
        return UniformGrid(self.cell, self.size, self.pbc,
                           kpt=self.kpt if kpt == '_default' else kpt,
                           dist=self.dist if dist == '_default' else dist)

    def empty(self, shape=None, dist=None) -> UniformGridFunctions:
        dist = create_shape_distributuion(shape, dist)
        array = np.empty(dist.shape + self.dist.size, self.dtype)
        return UniformGridFunctions(array, self, dist)

    def zeros(self, shape=(), dist=None) -> UniformGridFunctions:
        funcs = self.empty(shape, dist)
        funcs.data[:] = 0.0
        return funcs

    def redistributor(self, other):
        return Redistributor(self, other)

    @property
    def gd(self):
        return GridDescriptor(self.size, pbc_c=self.pbc, comm=self.dist.comm,
                              parsize_c=[len(d) - 1
                                         for d in self.dist.decomposition])


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


class UniformGridFunctions:
    def __init__(self, data, ug, dist=None):
        self.ug = ug
        self.data = data
        self.shape = data.shape[:-3]

    def __getitem__(self, index):
        return UniformGridFunctions(self.data[index], self.ug)

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
