from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import _gpaw
import gpaw.fftw as fftw
import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_centered_functions import UniformGridAtomCenteredFunctions
from gpaw.core.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import Array1D, Array4D, ArrayLike1D, ArrayLike2D
from gpaw.utilities.grid import GridRedistributor
from gpaw.new import cached_property


class UniformGrid(Domain):
    def __init__(self,
                 *,
                 cell: ArrayLike1D | ArrayLike2D,
                 size: ArrayLike1D,
                 pbc=(True, True, True),
                 kpt: ArrayLike1D = None,
                 comm: MPIComm = serial_comm,
                 decomp: Sequence[Sequence[int]] = None,
                 dtype=None):
        """"""
        self.size_c = np.array(size, int)

        if decomp is None:
            gd = GridDescriptor(size, pbc_c=pbc, comm=comm)
            decomp = gd.n_cp
        self.decomp_cp = decomp

        self.mypos_c = np.unravel_index(comm.rank,
                                        [len(d_p) - 1
                                         for d_p in self.decomp_cp])
        self.start_c = np.array([d_p[p]
                                 for d_p, p
                                 in zip(self.decomp_cp, self.mypos_c)])
        self.end_c = np.array([d_p[p + 1]
                               for d_p, p
                               in zip(self.decomp_cp, self.mypos_c)])
        self.mysize_c = self.end_c - self.start_c

        Domain.__init__(self, cell, pbc, kpt, comm, dtype)
        self.myshape = tuple(self.mysize_c)

        self.dv = abs(np.linalg.det(self.cell_cv)) / self.size_c.prod()

    def __repr__(self):
        return Domain.__repr__(self).replace(
            'Domain(',
            f'UniformGrid(size={self.size_c.tolist()}, ')

    @cached_property
    def phase_factors_cd(self):
        assert self.comm.size == 1
        disp_cd = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
        return np.exp(2j * np.pi *
                      disp_cd *
                      self.kpt_c[:, np.newaxis])

    def new(self,
            size=None,
            pbc=None,
            kpt=None,
            comm='inherit',
            decomp=None,
            dtype=None) -> UniformGrid:
        if decomp is None and comm == 'inherit':
            if size is None and pbc is None:
                decomp = self.decomp_cp
        comm = self.comm if comm == 'inherit' else comm
        return UniformGrid(cell=self.cell_cv,
                           size=self.size_c if size is None else size,
                           pbc=self.pbc_c if pbc is None else pbc,
                           kpt=(self.kpt_c if self.kpt_c.any() else None)
                           if kpt is None else kpt,
                           comm=comm or serial_comm,
                           decomp=decomp,
                           dtype=self.dtype if dtype is None else dtype)

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> UniformGridFunctions:
        return UniformGridFunctions(self, shape, comm)

    def xyz(self) -> Array4D:
        indices_Rc = np.indices(self.mysize_c).transpose((1, 2, 3, 0))
        indices_Rc += self.start_c
        return indices_Rc @ (self.cell_cv.T / self.size_c)

    def atom_centered_functions(self, functions, positions,
                                integral=None,
                                cut=False):
        return UniformGridAtomCenteredFunctions(functions, positions, self,
                                                integral=integral, cut=cut)

    def transformer(self, other: UniformGrid):
        from gpaw.transformers import Transformer

        apply = Transformer(self._gd, other._gd, nn=3).apply

        def transform(functions, out=None):
            if out is None:
                out = other.empty(functions.dims, functions.comm)
            for input, output in zip(functions._arrays(), out._arrays()):
                apply(input, output)
            return out

        return transform

    @property
    def _gd(self):
        return GridDescriptor(self.size_c,
                              cell_cv=self.cell_cv,
                              pbc_c=self.pbc_c,
                              comm=self.comm,
                              parsize_c=[len(d_p) - 1
                                         for d_p in self.decomp_cp])

    @classmethod
    def _from_gd_and_kpt_and_dtype(cls, gd, kpt, dtype):
        return UniformGrid(cell=gd.cell_cv,
                           size=gd.N_c,
                           pbc=gd.pbc_c,
                           comm=gd.comm,
                           dtype=dtype,
                           kpt=kpt,
                           decomp=gd.n_cp)

    def random(self,
               shape: int | tuple[int, ...] = (),
               comm: MPIComm = serial_comm) -> UniformGridFunctions:
        functions = self.empty(shape, comm)
        seed = [functions.comm.rank, functions.desc.comm.rank]
        rng = np.random.default_rng(seed)
        a = functions.data.view(float)
        rng.random(a.shape, out=a)
        a -= 0.5
        return functions

    def fft_plans(self, flags: int = fftw.MEASURE) -> tuple[fftw.FFTPlan,
                                                            fftw.FFTPlan]:
        size = tuple(self.size_c)
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


class UniformGridFunctions(DistributedArrays[UniformGrid]):
    def __init__(self,
                 grid: UniformGrid,
                 dims: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        DistributedArrays. __init__(self, dims, grid.myshape,
                                    comm, grid.comm, data, grid.dv,
                                    grid.dtype, transposed=False)
        self.desc = grid

    def __repr__(self):
        txt = f'UniformGridFunctions(grid={self.desc}, shape={self.dims}'
        if self.comm.size > 1:
            txt += f', comm={self.comm.rank}/{self.comm.size}'
        return txt + ')'

    def new(self, data=None):
        if data is None:
            data = np.empty_like(self.data)
        return UniformGridFunctions(self.desc, self.dims, self.comm, data)

    def __getitem__(self, index):
        return UniformGridFunctions(data=self.data[index], grid=self.desc)

    def _arrays(self):
        return self.data.reshape((-1,) + self.data.shape[-3:])

    def xy(self, *axes):
        assert len(axes) == 3 + len(self.dims)
        index = tuple([slice(0, None) if axis is ... else axis
                       for axis in axes])
        y = self.data[index]
        c = axes[-3:].index(...)
        grid = self.desc
        dx = (grid.cell_cv[c]**2).sum()**0.5 / grid.size_c[c]
        x = np.arange(grid.start_c[c], grid.end_c[c]) * dx
        return x, y

    def distribute(self, grid=None, out=None):
        if out is self:
            return out
        if out is None:
            if grid is None:
                raise ValueError('You must spicify grid or out!')
            out = grid.empty(self.dims, self.comm)
        if grid is None:
            grid = out.desc
        if self.desc.comm.size == 1 and grid.comm.size == 1:
            out.data[:] = self.data
            return out

        bcast_comm = SimpleNamespace(
            size=grid.comm.size // self.desc.comm.size)
        redistributor = GridRedistributor(grid.comm, bcast_comm,
                                          self.desc._gd, grid._gd)
        redistributor.distribute(self.data, out.data)
        return out

    def collect(self, grid=None, out=None, broadcast=False):
        if out is self:
            return out
        if out is None and grid is None:
            grid = self.desc.new(comm=serial_comm)
        if out is None:
            out = grid.empty(self.dims, self.comm)
        if grid is None:
            grid = out.desc
        if self.desc.comm.size == 1 and grid.comm.size == 1:
            out.data[:] = self.data
            return out

        bcast_comm = SimpleNamespace(
            size=self.desc.comm.size // grid.comm.size,
            broadcast=lambda array, rank: None)
        redistributor = GridRedistributor(self.desc.comm,
                                          bcast_comm,
                                          grid._gd, self.desc._gd)
        redistributor.collect(self.data, out.data)
        if broadcast:
            self.desc.comm.broadcast(out.data, 0)
        return out

    def fft(self, plan=None, pw=None, out=None):
        assert self.dims == ()
        if out is None:
            assert pw is not None
            out = pw.empty()
        if pw is None:
            pw = out.desc
        input = self
        if self.desc.comm.size > 1:
            input = input.collect()
        if self.desc.comm.rank == 0:
            plan = plan or self.desc.fft_plans()[0]
            plan.in_R[:] = input.data
            plan.execute()
            coefs = pw.cut(plan.out_R) * (1 / plan.in_R.size)

        if pw.comm.size > 1:
            out1 = pw.new(comm=serial_comm).empty()
            if pw.comm.rank == 0:
                out1.data[:] = coefs
            out1.distribute(out=out)
        else:
            out.data[:] = coefs

        return out

    def norm2(self):
        norm_x = []
        arrays_xR = self._arrays()
        for a_R in arrays_xR:
            norm_x.append(np.vdot(a_R, a_R).real * self.desc.dv)
        result = np.array(norm_x).reshape(self.mydims)
        self.desc.comm.sum(result)
        return result

    def integrate(self, other=None):
        if other is not None:
            assert self.desc.dtype == other.desc.dtype
            a_xR = self._arrays()
            b_yR = other._arrays()
            a_xR = a_xR.reshape((len(a_xR), -1))
            b_yR = b_yR.reshape((len(b_yR), -1))
            result = (a_xR @ b_yR.T.conj()).reshape(self.dims + other.dims)
        else:
            result = self.data.sum(axis=(-3, -2, -1))

        if result.ndim == 0:
            result = self.desc.comm.sum(result.item())
        else:
            self.desc.comm.sum(result)

        return result * self.desc.dv

    def fft_interpolate(self,
                        out: UniformGridFunctions,
                        fftplan: fftw.FFTPlan = None,
                        ifftplan: fftw.FFTPlan = None) -> None:
        size1_c = self.desc.size_c
        size2_c = out.desc.size_c
        if (size2_c <= size1_c).any():
            raise ValueError('Too few points in target grid!')

        fftplan = fftplan or self.desc.fft_plans()[0]
        ifftplan = ifftplan or out.desc.fft_plans()[1]

        fftplan.in_R[:] = self.data
        fftplan.execute()

        a_Q = fftplan.out_R
        b_Q = ifftplan.in_R

        e0, e1, e2 = 1 - size1_c % 2  # even or odd size
        a0, a1, a2 = size2_c // 2 - size1_c // 2
        b0, b1, b2 = size1_c + (a0, a1, a2)

        if self.desc.dtype == float:
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
        if self.desc.dtype == complex:
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
                     ifftplan: fftw.FFTPlan = None,
                     indices=None) -> None:
        size1_c = self.desc.size_c
        size2_c = out.desc.size_c

        fftplan = fftplan or self.desc.fft_plans()[0]
        ifftplan = ifftplan or out.desc.fft_plans()[1]

        fftplan.in_R[:] = self.data
        a_Q = ifftplan.in_R
        b_Q = fftplan.out_R

        e0, e1, e2 = 1 - size2_c % 2  # even or odd size
        a0, a1, a2 = size1_c // 2 - size2_c // 2
        b0, b1, b2 = size2_c // 2 + size1_c // 2 + 1

        if self.desc.dtype == float:
            b2 = size2_c[2] // 2 + 1
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
        if self.desc.dtype == complex and e2:
            b_Q[a0:b0, a1:b1, a2] += b_Q[a0:b0, a1:b1, b2 - 1]
            b_Q[a0:b0, a1:b1, a2] *= 0.5
            b2 -= 1

        a_Q[:] = b_Q[a0:b0, a1:b1, a2:b2]
        a_Q[:] = np.fft.ifftshift(a_Q, axes=axes)
        if indices is not None:
            coefs = a_Q.ravel()[indices]
        else:
            coefs = None
        ifftplan.execute()
        out.data[:] = ifftplan.out_R
        out.data *= (1.0 / self.data.size)
        return coefs

    def abs_square(self,
                   weights: Array1D,
                   out: UniformGridFunctions = None) -> None:
        assert out is not None
        for f, psit_R in zip(weights, self.data):
            # Same as out.data += f * abs(psit_R)**2, but much faster:
            _gpaw.add_to_density(f, psit_R, out.data)

    def symmetrize(self, rotation_scc, translation_sc):
        if len(rotation_scc) == 1:
            return
        a_xR = self.collect()
        b_xR = a_xR.new()
        if self.desc.comm.rank == 0:
            t_sc = (translation_sc * self.desc.size_c).round().astype(int)
            offset_c = 1 - self.desc.pbc_c
            for a_R, b_R in zip(a_xR._arrays(), b_xR._arrays()):
                b_R[:] = 0.0
                for r_cc, t_c in zip(rotation_scc, t_sc):
                    _gpaw.symmetrize_ft(a_R, b_R, r_cc, t_c, offset_c)
        b_xR.distribute(out=self)
        self.data *= 1.0 / len(rotation_scc)
