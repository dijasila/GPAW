from __future__ import annotations
import numpy as np
from gpaw.typing import ArrayLike1D, ArrayLike, Array2D, ArrayLike2D
from gpaw.mpi import serial_comm, MPIComm
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.grid import GridRedistributor
from gpaw.core.arrays import DistributedArrays
from typing import Sequence
from gpaw.core.layout import Layout
from gpaw.core.atom_centered_functions import UniformGridAtomCenteredFunctions


def _normalize_cell(cell: ArrayLike) -> Array2D:
    cell = np.array(cell, float)
    if cell.ndim == 2:
        return cell
    if len(cell) == 3:
        return np.diag(cell)
    raise ValueError


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

        self.dv = abs(np.linalg.det(self.cell)) / self.size.prod()

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

        Layout.__init__(self,
                        tuple(self.size - 1 + self.pbc),
                        tuple(self.mysize))

        assert dtype in [None, float, complex]
        if self.kpt.any():
            if dtype == float:
                raise ValueError
            dtype = complex
        else:
            dtype = dtype or float
        self.dtype = dtype

    def __str__(self):
        a, b, c = self.size
        comm = self.comm
        return (f'UniformGrid(size={a}*{b}*{c}, pbc={self.pbc}, '
                f'comm={comm.rank}/{comm.size}, dtype={self.dtype})')

    @property
    def icell(self):
        return np.linalg.inv(self.cell).T

    def new(self,
            size=None,
            pbc=None,
            kpt=None,
            comm=None,
            decomposition=None) -> UniformGrid:
        if decomposition is None and comm is None:
            if (size == self.size).all() and (pbc == self.pbc).all():
                decomposition = self.decomposition
        return UniformGrid(cell=self.cell,
                           size=self.size if size is None else size,
                           pbc=self.pbc if pbc is None else pbc,
                           kpt=self.kpt if kpt is None else kpt,
                           comm=comm or self.comm,
                           decomposition=decomposition)

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> UniformGridFunctions:
        return UniformGridFunctions(self, shape, comm)

    def redistributor(self, other):
        return Redistributor(self, other)

    def atom_centered_functions(self, functions, positions):
        return UniformGridAtomCenteredFunctions(functions, positions, self)

    def transformer(self, other):
        from gpaw.transformers import Transformer

        apply = Transformer(self._gd, other._gd, 3).apply

        def transform(functions, out=None):
            if out is None:
                out = other.empty(functions.shape, functions.comm)
            apply(functions.data, out.data)
            return out

        return transform

    @property
    def _gd(self):
        return GridDescriptor(self.size,
                              cell_cv=self.cell,
                              pbc_c=self.pbc,
                              comm=self.comm,
                              parsize_c=[len(d) - 1
                                         for d in self.decomposition])

    @classmethod
    def _from_gd_and_kpt_and_dtype(cls, gd, kpt, dtype):
        return UniformGrid(cell=gd.cell_cv,
                           size=gd.N_c,
                           pbc=gd.pbc_c,
                           comm=gd.comm,
                           dtype=dtype,
                           kpt=kpt,
                           decomposition=gd.n_cp)

    def random(self,
               shape: int | tuple[int, ...] = (),
               comm: MPIComm = serial_comm) -> UniformGridFunctions:
        functions = self.empty(shape, comm)
        seed = [functions.comm.rank, functions.grid.comm.rank]
        rng = np.random.default_rng(seed)
        a = functions.data.view(float)
        rng.random(a.shape, out=a)
        a -= 0.5
        return functions


class Redistributor:
    def __init__(self, grid1, grid2):
        self.grid2 = grid2
        comm = max(grid1.comm, grid2.comm, key=lambda comm: comm.size)
        self.redistributor = GridRedistributor(comm, serial_comm,
                                               grid1._gd, grid2._gd)

    def distribute(self, input, output):
        self.redistributor.distribute(input.data, output.data)


class UniformGridFunctions(DistributedArrays):
    def __init__(self,
                 grid: UniformGrid,
                 shape: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        DistributedArrays. __init__(self, grid, shape, comm, data)
        self.grid = grid

    def __repr__(self):
        txt = f'UniformGridFunctions(grid={self.grid}, shape={self.shape}'
        if self.comm.size > 1:
            txt += f', comm={self.comm.rank}/{self.comm.size}'

        def integrate(a_G):
            if wfs.collinear:
                return np.real(wfs.integrate(a_G, a_G, global_integral=False))
            return sum(
                np.real(wfs.integrate(b_G, b_G, global_integral=False))
                for b_G in a_G)
        return txt + ')'

    def new(self, data=None):
        if data is None:
            data = np.empty_like(self.data)
        return UniformGridFunctions(self.grid, self.shape, self.comm, data)

    def __getitem__(self, index):
        return UniformGridFunctions(data=self.data[index], grid=self.grid)

    def _arrays(self):
        return self.data.reshape((-1,) + self.data.shape[-3:])

    def xy(self, *axes):
        assert len(axes) == 3 + len(self.shape)
        index = tuple([slice(0, None) if axis is ... else axis
                       for axis in axes])
        y = self.data[index]
        n = axes[-3:].index(...)
        dx = (self.grid.cell[n]**2).sum()**0.5 / self.grid.size[n]
        x = np.arange(self.grid.start[n], self.grid.end[n]) * dx
        return x, y

    def redistribute(self, grid=None, out=None):
        if out is self:
            return out
        if out is None:
            out = grid.empty(self.shape, self.comm)
        if grid is None:
            grid = out.grid
        if self.grid.comm.size == 1 and grid.comm.size == 1:
            out.data[:] = self.data
            return out
        self.grid.redistributor(grid).redistribute(self, out)
        return out

    def fft(self, plan=None, pw=None, out=None):
        if out is None:
            out = pw.empty(self.shape)
        if pw is None:
            pw = out.pw
        plan = plan or pw.fft_plans()[0]
        for input, output in zip(self._arrays(), out._arrays()):
            plan.in_R[:] = input
            plan.execute()
            output[:] = plan.out_R.ravel()[pw.indices]
        return out
