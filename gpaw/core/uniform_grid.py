from __future__ import annotations

from typing import Sequence
from types import SimpleNamespace

import gpaw.fftw as fftw
import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_centered_functions import UniformGridAtomCenteredFunctions
from gpaw.core.layout import Layout
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import Array2D, ArrayLike, ArrayLike1D, ArrayLike2D, Array1D
from gpaw.utilities.grid import GridRedistributor
import _gpaw


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
        self.dtype = np.dtype(dtype)

        self._phase_factors = None

    def __eq__(self, other):
        return ((self.size == other.size).all() and
                (self.pbc == other.pbc).all())

    # @cached_property
    def phase_factors(self):
        if self._phase_factors is None:
            assert self.comm.size == 1
            disp = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
            self._phase_factors = np.exp(2j * np.pi *
                                         disp * self.kpt[:, np.newaxis])
        return self._phase_factors

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
            decomposition=None,
            dtype=None) -> UniformGrid:
        if decomposition is None and comm is None:
            if (size == self.size).all() and (pbc == self.pbc).all():
                decomposition = self.decomposition
        return UniformGrid(cell=self.cell,
                           size=self.size if size is None else size,
                           pbc=self.pbc if pbc is None else pbc,
                           kpt=self.kpt if kpt is None else kpt,
                           comm=comm or self.comm,
                           decomposition=decomposition,
                           dtype=self.dtype if dtype is None else dtype)

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> UniformGridFunctions:
        return UniformGridFunctions(self, shape, comm)

    def atom_centered_functions(self, functions, positions,
                                integral=None,
                                cut=False):
        return UniformGridAtomCenteredFunctions(functions, positions, self,
                                                integral=integral, cut=cut)

    def transformer(self, other):
        from gpaw.transformers import Transformer

        apply = Transformer(self._gd, other._gd, 3).apply

        def transform(functions, out=None, preserve_integral=False):
            if out is None:
                out = other.empty(functions.shape, functions.comm)
            apply(functions.data, out.data)
            if preserve_integral and not self.pbc.all():
                out.data *= functions.integrate() / out.integrate()
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

    def fft_plans(self, flags: int = fftw.MEASURE) -> tuple[fftw.FFTPlan,
                                                            fftw.FFTPlan]:
        size = tuple(self.size)
        if self.dtype == float:
            rsize = size[:2] + (size[2] // 2 + 1,)
            tmp1 = fftw.empty(rsize, complex)
            tmp2 = tmp1.view(float)[:, :, :size[2]]
        else:
            tmp1 = fftw.empty(size, complex)
            tmp2 = tmp1

        fftplan = fftw.create_plan(tmp2, tmp1, -1, flags)
        ifftplan = fftw.create_plan(tmp1, tmp2, 1, flags)
        return fftplan, ifftplan


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

    def distribute(self, grid=None, out=None):
        if out is self:
            return out
        if out is None:
            if grid is None:
                raise ValueError('You must spicify grid or out!')
            out = grid.empty(self.shape, self.comm)
        if grid is None:
            grid = out.grid
        if self.grid.comm.size == 1 and grid.comm.size == 1:
            out.data[:] = self.data
            return out

        bcast_comm = SimpleNamespace(
            size=grid.comm.size // self.grid.comm.size)
        redistributor = GridRedistributor(grid.comm, bcast_comm,
                                          self.grid._gd, grid._gd)
        redistributor.distribute(self.data, out.data)
        return out

    def collect(self, grid=None, out=None, broadcast=False):
        if out is self:
            return out
        if out is None and grid is None:
            grid = self.grid.new(comm=serial_comm)
        if out is None:
            out = grid.empty(self.shape, self.comm)
        if grid is None:
            grid = out.grid
        if self.grid.comm.size == 1 and grid.comm.size == 1:
            out.data[:] = self.data
            return out

        bcast_comm = SimpleNamespace(
            size=self.grid.comm.size // grid.comm.size,
            broadcast=lambda array, rank: None)
        redistributor = GridRedistributor(self.grid.comm,
                                          bcast_comm,
                                          grid._gd, self.grid._gd)
        redistributor.collect(self.data, out.data)
        if broadcast:
            self.grid.comm.broadcast(out.data, 0)
        return out

    def fft(self, plan=None, pw=None, out=None):
        assert self.shape == ()
        if out is None:
            assert pw is not None
            out = pw.empty()
        if pw is None:
            pw = out.pw
        input = self
        if self.grid.comm.size > 1:
            input = input.collect()
        if self.grid.comm.rank == 0:
            plan = plan or pw.grid.fft_plans()[0]
            plan.in_R[:] = input.data
            plan.execute()
            coefs = plan.out_R.ravel()[pw.indices]

        if pw.grid.comm.size > 1:
            out1 = pw.new(grid=input.grid).empty()
            if pw.grid.comm.rank == 0:
                out1.data[:] = coefs
            out1.distribute(out=out)
        else:
            out.data[:] = coefs

        return out

    def norm2(self):
        norms = []
        arrays = self._arrays()
        for a in arrays:
            norms.append(np.vdot(a, a).real * self.grid.dv)
        result = np.array(norms).reshape(self.myshape)
        self.grid.comm.sum(result)
        return result

    def integrate(self, other=None):
        if other is not None:
            assert self.grid.dtype == other.grid.dtype
            a = self._arrays()
            b = other._arrays()
            a = a.reshape((len(a), -1))
            b = b.reshape((len(b), -1))
            result = (a @ b.T.conj()).reshape(self.shape + other.shape)
        else:
            result = self.data.sum(axis=(-3, -2, -1))

        if result.ndim == 0:
            result = self.grid.comm.sum(result.item())
        else:
            self.grid.comm.sum(result)

        return result * self.grid.dv

    def fft_interpolate(self,
                        out: UniformGridFunctions,
                        fftplan: fftw.FFTPlan = None,
                        ifftplan: fftw.FFTPlan = None) -> None:
        size1 = self.grid.size
        size2 = out.grid.size
        if (size2 <= size1).any():
            raise ValueError('Too few points in target grid!')

        fftplan = fftplan or self.grid.fft_plans()[0]
        ifftplan = ifftplan or out.grid.fft_plans()[1]

        fftplan.in_R[:] = self.data
        fftplan.execute()

        a_Q = fftplan.out_R
        b_Q = ifftplan.in_R

        e0, e1, e2 = 1 - size1 % 2  # even or odd size
        a0, a1, a2 = size2 // 2 - size1 // 2
        b0, b1, b2 = size1 + (a0, a1, a2)

        if self.grid.dtype == float:
            b2 = (b2 - a2) // 2 + 1
            a2 = 0
            axes = [0, 1]
        else:
            axes = [0, 1, 2]

        b_Q[:] = 0.0
        b_Q[a0:b0, a1:b1, a2:b2] = np.fft.fftshift(a_Q, axes=axes)

        if e0:
            b_Q[a0, a1:b1, a2:b2] *= 0.5
            b_Q[b0, a1:b1, a2:b2] = b_Q[a0, a1:b1, a2:b2]
            b0 += 1
        if e1:
            b_Q[a0:b0, a1, a2:b2] *= 0.5
            b_Q[a0:b0, b1, a2:b2] = b_Q[a0:b0, a1, a2:b2]
            b1 += 1
        if self.grid.dtype == complex:
            if e2:
                b_Q[a0:b0, a1:b1, a2] *= 0.5
                b_Q[a0:b0, a1:b1, b2] = b_Q[a0:b0, a1:b1, a2]
        else:
            if e2:
                b_Q[a0:b0, a1:b1, b2 - 1] *= 0.5

        b_Q[:] = np.fft.ifftshift(b_Q, axes=axes)
        ifftplan.execute()
        out.data[:] = ifftplan.out_R
        out.data *= (1.0 / self.data.size)

    def fft_restrict(self,
                     out: UniformGridFunctions,
                     fftplan: fftw.FFTPlan = None,
                     ifftplan: fftw.FFTPlan = None) -> None:
        size1 = self.grid.size
        size2 = out.grid.size

        fftplan = fftplan or self.grid.fft_plans()[0]
        ifftplan = ifftplan or out.grid.fft_plans()[1]

        fftplan.in_R[:] = self.data
        a_Q = ifftplan.in_R
        b_Q = fftplan.out_R

        e0, e1, e2 = 1 - size2 % 2  # even or odd size
        a0, a1, a2 = size1 // 2 - size2 // 2
        b0, b1, b2 = size2 // 2 + size1 // 2 + 1

        if self.grid.dtype == float:
            b2 = size2[2] // 2 + 1
            a2 = 0
            axes = [0, 1]
        else:
            axes = [0, 1, 2]

        fftplan.execute()
        b_Q[:] = np.fft.fftshift(b_Q, axes=axes)

        if e0:
            b_Q[a0, a1:b1, a2:b2] += b_Q[b0 - 1, a1:b1, a2:b2]
            b_Q[a0, a1:b1, a2:b2] *= 0.5
            b0 -= 1
        if e1:
            b_Q[a0:b0, a1, a2:b2] += b_Q[a0:b0, b1 - 1, a2:b2]
            b_Q[a0:b0, a1, a2:b2] *= 0.5
            b1 -= 1
        if self.grid.dtype == complex and e2:
            b_Q[a0:b0, a1:b1, a2] += b_Q[a0:b0, a1:b1, b2 - 1]
            b_Q[a0:b0, a1:b1, a2] *= 0.5
            b2 -= 1

        a_Q[:] = b_Q[a0:b0, a1:b1, a2:b2]
        a_Q[:] = np.fft.ifftshift(a_Q, axes=axes)
        ifftplan.execute()
        out.data[:] = ifftplan.out_R
        out.data *= (1.0 / self.data.size)

    def abs_square(self,
                   weights: Array1D,
                   out: UniformGridFunctions = None) -> None:
        assert out is not None
        for f, psit in zip(weights, self.data):
            # Same as density.data += f * abs(psit)**2, but much faster:
            _gpaw.add_to_density(f, psit, out.data)
