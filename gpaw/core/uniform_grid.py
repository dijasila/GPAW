from __future__ import annotations

from typing import Sequence

import _gpaw
import gpaw.fftw as fftw
import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_centered_functions import UniformGridAtomCenteredFunctions
from gpaw.core.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import MPIComm, serial_comm
from gpaw.typing import Array1D, Array4D, ArrayLike1D, ArrayLike2D, Vector
from gpaw.new import cached_property


class UniformGrid(Domain):
    def __init__(self,
                 *,
                 cell: ArrayLike1D | ArrayLike2D,
                 size: ArrayLike1D,
                 pbc=(True, True, True),
                 kpt: Vector = None,
                 comm: MPIComm = serial_comm,
                 decomp: Sequence[Sequence[int]] = None,
                 dtype=None):
        """"""
        self.size_c = np.array(size, int)

        if decomp is None:
            gd = GridDescriptor(size, pbc_c=pbc, comm=comm)
            decomp = gd.n_cp
        self.decomp_cp = [np.asarray(d) for d in decomp]

        self.parsize_c = np.array([len(d_p) - 1 for d_p in self.decomp_cp])
        self.mypos_c = np.unravel_index(comm.rank, self.parsize_c)

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
        delta_d = np.array([1, -1])
        disp_cd = np.empty((3, 2))
        for pos, pbc, size, disp_d in zip(self.mypos_c, self.pbc_c,
                                          self.parsize_c, disp_cd):
            disp_d[:] = (pos + delta_d) // size
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

    def blocks(self, data):
        s0, s1, s2 = self.parsize_c
        d0_p, d1_p, d2_p = (d_p - d_p[0] for d_p in self.decomp_cp)
        for p0 in range(s0):
            b0, e0 = d0_p[p0:p0 + 2]
            for p1 in range(s1):
                b1, e1 = d1_p[p1:p1 + 2]
                for p2 in range(s2):
                    b2, e2 = d2_p[p2:p2 + 2]
                    yield data[..., b0:e0, b1:e1, b2:e2]

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

    def scatter_from(self, data=None):
        if isinstance(data, UniformGridFunctions):
            data = data.data

        comm = self.desc.comm
        if comm.size == 1:
            self.data[:] = data
            return

        if comm.rank != 0:
            comm.receive(self.data, 0, 42)
            return

        requests = []
        for rank, block in enumerate(self.desc.blocks(data)):
            if rank != 0:
                block = block.copy()
                request = comm.send(block, rank, 42, False)
                # Remember to store a reference to the
                # send buffer (block) so that is isn't
                # deallocated:
                requests.append((request, block))
            else:
                self.data[:] = block

        for request, _ in requests:
            comm.wait(request)

    def gather(self, broadcast=False):
        comm = self.desc.comm
        if comm.size == 1:
            return self

        if broadcast or comm.rank == 0:
            grid = self.desc.new(comm=serial_comm)
            out = grid.empty(self.dims)

        if comm.rank != 0:
            # There can be several sends before the corresponding receives
            # are posted, so use syncronous send here
            comm.ssend(self.data, 0, 301)
            if broadcast:
                comm.broadcast(out.data, 0)
                return out
            return

        # Put the subdomains from the slaves into the big array
        # for the whole domain:
        for rank, block in enumerate(self.desc.blocks(out.data)):
            if rank != 0:
                buf = np.empty_like(block)
                comm.receive(buf, rank, 301)
                block[:] = buf
            else:
                block[:] = self.data

        if broadcast:
            comm.broadcast(out.data, 0)

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
            input = input.gather()
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

        a_xR = self.gather()

        if a_xR is None:
            b_xR = None
        else:
            b_xR = a_xR.new()
            t_sc = (translation_sc * self.desc.size_c).round().astype(int)
            offset_c = 1 - self.desc.pbc_c
            for a_R, b_R in zip(a_xR._arrays(), b_xR._arrays()):
                b_R[:] = 0.0
                for r_cc, t_c in zip(rotation_scc, t_sc):
                    _gpaw.symmetrize_ft(a_R, b_R, r_cc, t_c, offset_c)

        self.scatter_from(b_xR)

        self.data *= 1.0 / len(rotation_scc)
