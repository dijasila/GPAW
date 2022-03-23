from __future__ import annotations

from math import pi

import _gpaw
import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.domain import Domain
from gpaw.core.matrix import Matrix
from gpaw.core.pwacf import PlaneWaveAtomCenteredFunctions
from gpaw.core.uniform_grid import UniformGridFunctions
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import prod
from gpaw.new import zip_strict as zip
from gpaw.pw.descriptor import pad
from gpaw.typing import (Array1D, Array2D, Array3D, ArrayLike1D, ArrayLike2D,
                         Vector)


class PlaneWaves(Domain):
    def __init__(self,
                 *,
                 ecut: float,
                 cell: ArrayLike1D | ArrayLike2D,
                 kpt: Vector = None,
                 comm: MPIComm = serial_comm,
                 dtype=None):
        """Description of 3-D plane-wave expansion.

        parameters
        ----------
        ecut:
            Cutoff energy for kinetic energy of plane waves.
        cell:
            Unit cell given as three floats (orthorhombic grid), six floats
            (three lengths and the angles in degrees) or a 3x3 matrix.
        comm:
            Communicator for distribution of plane-waves.
        kpt:
            K-point for Block-boundary conditions specified in units of the
            reciprocal cell.
        dtype:
            Data-type (float or complex).
        """
        self.ecut = ecut
        Domain.__init__(self, cell, (True, True, True), kpt, comm, dtype)

        G_plus_k_Gv, ekin_G, self.indices_cG = find_reciprocal_vectors(
            ecut, self.cell_cv, self.kpt_c, self.dtype)

        # Find distribution:
        S = comm.size
        ng = len(ekin_G)
        self.maxmysize = (ng + S - 1) // S
        ng1 = comm.rank * self.maxmysize
        ng2 = ng1 + self.maxmysize

        # Distribute things:
        self.ekin_G = ekin_G[ng1:ng2].copy()
        self.ekin_G.flags.writeable = False
        # self.myindices_cG = self.indices_cG[:, ng1:ng2]
        self.G_plus_k_Gv = G_plus_k_Gv[ng1:ng2].copy()

        self.shape = (ng,)
        self.myshape = (len(self.ekin_G),)

        self.dv = abs(np.linalg.det(self.cell_cv))

        self._indices_cache: dict[tuple[int, ...], Array1D] = {}

    def __repr__(self) -> str:
        return Domain.__repr__(self).replace(
            'Domain(',
            f'PlaneWaves(ecut={self.ecut}, ')

    def global_shape(self) -> tuple[int, ...]:
        """Tuple with one element: number of plane waves."""
        return self.shape

    def reciprocal_vectors(self) -> Array2D:
        """Returns reciprocal lattice vectors, G + k, in xyz coordinates."""
        return self.G_plus_k_Gv

    def kinetic_energies(self) -> Array1D:
        """Kinetic energy of plane waves.

        :::

             _ _ 2
            |G+k| / 2

        """
        return self.ekin_G

    def empty(self,
              dims: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> PlaneWaveExpansions:
        """Create new PlaneWaveExpanions object.

        parameters
        ----------
        dims:
            Extra dimensions.
        comm:
            Distribute dimensions along this communicator.
        """
        return PlaneWaveExpansions(self, dims, comm)

    def new(self,
            ecut: float = None,
            kpt=None,
            comm: MPIComm | str = 'inherit') -> PlaneWaves:
        """Create new plane-wave expansion description."""
        comm = self.comm if comm == 'inherit' else comm
        return PlaneWaves(ecut=ecut or self.ecut,
                          cell=self.cell_cv,
                          kpt=self.kpt_c if kpt is None else kpt,
                          dtype=self.dtype,
                          comm=comm or serial_comm)

    def indices(self, shape: tuple[int, ...]) -> Array1D:
        """Return indices into FFT-grid."""
        Q_G = self._indices_cache.get(shape)
        if Q_G is None:
            Q_G = np.ravel_multi_index(self.indices_cG, shape,  # type: ignore
                                       mode='wrap').astype(np.int32)
            self._indices_cache[shape] = Q_G
        return Q_G

    def cut(self, array_Q: Array3D) -> Array1D:
        """Cut out G-vectors with (G+k)^2/2<E_kin."""
        return array_Q.ravel()[self.indices(array_Q.shape)]

    def paste(self, coef_G: Array1D, array_Q: Array3D) -> None:
        """Paste G-vectors with (G+k)^2/2<E_kin into 3-D FFT grid and
        zero-pad."""
        Q_G = self.indices(array_Q.shape)
        # array_Q[:] = 0.0
        # array_Q.ravel()[Q_G] = coef_G
        _gpaw.pw_insert(coef_G, Q_G, 1.0, array_Q)

    def map_indices(self, other):
        """Map from one set of plane waves to another."""
        size_c = tuple(self.indices_cG.ptp(axis=1) + 1)
        Q_G = self.indices(size_c)
        i_Q = np.empty(np.prod(size_c), int)
        i_Q[Q_G] = np.arange(len(Q_G))
        return i_Q[other.indices(size_c)]

    def atom_centered_functions(self,
                                functions,
                                positions,
                                atomdist=None,
                                integral=None,
                                cut=False):
        """Create PlaneWaveAtomCenteredFunctions object."""
        return PlaneWaveAtomCenteredFunctions(functions, positions, self)


class PlaneWaveExpansions(DistributedArrays[PlaneWaves]):
    def __init__(self,
                 pw: PlaneWaves,
                 dims: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        """Object for storing function(s) as a plane-wave expansions.

        parameters
        ----------
        pw:
            Description of plane-waves.
        dims:
            Extra dimensions.
        comm:
            Distribute plane-waves along this communicator.
        data:
            Data array for storage.
        """
        DistributedArrays. __init__(self, dims, pw.myshape,
                                    comm, pw.comm,
                                    data, pw.dv, complex,
                                    transposed=False)
        self.desc = pw
        self._matrix: Matrix | None

    def __repr__(self):
        txt = f'PlaneWaveExpansions(pw={self.desc}, dims={self.dims}'
        if self.comm.size > 1:
            txt += f', comm={self.comm.rank}/{self.comm.size}'
        return txt + ')'

    def __getitem__(self, index: int) -> PlaneWaveExpansions:
        data = self.data[index]
        return PlaneWaveExpansions(self.desc, data.shape[:-1], data=data)

    def __iter__(self):
        for data in self.data:
            yield PlaneWaveExpansions(self.desc, data=data)

    def new(self, data=None):
        """Create new PlaneWaveExpansions object of same kind.

        Parameters
        ----------
        data:
            Array to use for storage.
        """
        if data is None:
            data = np.empty_like(self.data)
        else:
            # Number of plane-waves depends on the k-point.  We therfore
            # allow for data to be bigger than needed:
            data = data.ravel()[:self.data.size].reshape(self.data.shape)
        return PlaneWaveExpansions(self.desc, self.dims, self.comm, data)

    def copy(self):
        """Create a copy (surprise!)."""
        a = self.new()
        a.data[:] = self.data
        return a

    def _arrays(self):
        shape = self.data.shape
        return self.data.reshape((np.prod(shape[:-1], dtype=int), shape[-1]))

    @property
    def matrix(self) -> Matrix:
        """Matrix view of data."""
        if self._matrix is not None:
            return self._matrix

        shape = (np.prod(self.dims), self.myshape[0])
        myshape = (np.prod(self.mydims), self.myshape[0])
        dist = (self.comm, -1, 1)
        data = self.data.reshape(myshape)

        if self.desc.dtype == float:
            data = data.view(float)
            shape = (shape[0], shape[1] * 2)

        self._matrix = Matrix(*shape, data=data, dist=dist)
        return self._matrix

    def ifft(self, *, plan=None, grid=None, out=None):
        """Do inverse FFT to uniform grid.

        Parameters
        ----------
        plan:
            Plan for inverse FFT.
        grid:
            Target grid.
        out:
            Target UniformGridFunctions object.
        """
        if out is None:
            out = grid.empty(self.dims)
        assert self.desc.dtype == out.desc.dtype
        assert out.desc.pbc_c.all()

        comm = self.desc.comm

        this = self.gather()
        if this is not None:
            arrays_iG = this._arrays()
            plan = plan or out.desc.fft_plans()[1]
        for i, out1 in enumerate(out.flat()):
            if comm.rank == 0:
                coef_G = arrays_iG[i]
                self.desc.paste(coef_G, plan.in_R)
                if self.desc.dtype == float:
                    t = plan.in_R[:, :, 0]
                    n, m = (s // 2 - 1 for s in out.desc.size_c[:2])
                    t[0, -m:] = t[0, m:0:-1].conj()
                    t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
                    t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
                    t[-n:, 0] = t[n:0:-1, 0].conj()
                plan.execute()
                out1.scatter_from(plan.out_R)
            else:
                out1.scatter_from(None)

        out.multiply_by_eikr()

        return out

    interpolate = ifft

    def gather(self, out=None, broadcast=False):
        """Gather coefficients on master."""
        comm = self.desc.comm

        if comm.size == 1:
            if out is None:
                return self
            out.data[:] = self.data
            return out

        if out is None:
            if comm.rank == 0 or broadcast:
                pw = self.desc.new(comm=serial_comm)
                out = pw.empty(self.dims)
            else:
                out = Empty(self.dims)

        if comm.rank == 0:
            data = np.empty(self.desc.maxmysize * comm.size, complex)
        else:
            data = None

        for input, output in zip(self._arrays(), out._arrays()):
            mydata = pad(input, self.desc.maxmysize)
            comm.gather(mydata, 0, data)
            if comm.rank == 0:
                output[:] = data[:len(output)]

        if broadcast:
            comm.broadcast(out.data, 0)

        return out if not isinstance(out, Empty) else None

    def scatter_from(self, data: Array1D = None) -> None:
        """Scatter data from rank-0 to all ranks."""
        comm = self.desc.comm
        if comm.size == 1:
            self.data[:] = data
            return

        assert self.dims == ()

        if comm.rank == 0:
            data = pad(data, comm.size * self.desc.maxmysize)
            comm.scatter(data, self.data, 0)
        else:
            buf = np.empty(self.desc.maxmysize, complex)
            comm.scatter(None, buf, 0)
            self.data[:] = buf[:len(self.data)]

    def integrate(self, other: PlaneWaveExpansions = None) -> np.ndarray:
        """Integral of self or self time cc(other)."""
        if other is not None:
            assert self.comm.size == 1
            assert self.desc.dtype == other.desc.dtype
            a = self._arrays()
            b = other._arrays()
            dv = self.dv
            if self.desc.dtype == float:
                a = a.view(float)
                b = b.view(float)
                dv *= 2
            result = a @ b.T.conj()
            if self.desc.dtype == float and self.desc.comm.rank == 0:
                result -= 0.5 * np.outer(a[:, 0], b[:, 0])
            self.desc.comm.sum(result)
            result.shape = self.dims + other.dims
        else:
            dv = self.dv
            if self.desc.comm.rank == 0:
                result = self.data[..., 0]
            else:
                result = np.empty(self.mydims, complex)
            self.desc.comm.broadcast(result, 0)

        if self.desc.dtype == float:
            result = result.real
        return result * dv

    def _matrix_elements_correction(self,
                                    M1: Matrix,
                                    M2: Matrix,
                                    out: Matrix,
                                    symmetric: bool) -> None:
        if self.desc.dtype == float:
            out.data *= 2.0
            if self.desc.comm.rank == 0:
                correction = np.outer(M1.data[:, 0],
                                      M2.data[:, 0]) * self.dv
                if symmetric:
                    correction *= 0.5
                    out.data -= correction
                    out.data -= correction.T
                else:
                    out.data -= correction

    def norm2(self, kind: str = 'normal') -> np.ndarray:
        r"""Calculate integral over cell.

        For kind='normal' we calculate:::

          /   _  2 _   --    2
          ||a(r)| dr = > |c | V,
          /            --  G
                        G

        where V is the volume of the unit cell.

        And for kind='kinetic':::

           1  --    2  2
          --- > |c |  G V,
           2  --  G
               G

        """
        a_xG = self._arrays().view(float)
        if kind == 'normal':
            result_x = np.einsum('xG, xG -> x', a_xG, a_xG)
        elif kind == 'kinetic':
            a_xG.shape = (len(a_xG), -1, 2)
            result_x = np.einsum('xGi, xGi, G -> x',
                                 a_xG, a_xG, self.desc.ekin_G)
        else:
            1 / 0
        if self.desc.dtype == float:
            result_x *= 2
            if self.desc.comm.rank == 0 and kind == 'normal':
                result_x -= a_xG[:, 0]**2
        self.desc.comm.sum(result_x)
        result_x.shape = self.mydims
        return result_x * self.dv

    def abs_square(self,
                   weights: Array1D,
                   out: UniformGridFunctions = None) -> None:
        """Add weighted absolute square of data to output array."""
        assert out is not None
        tmp_R = out.desc.new(dtype=self.desc.dtype).empty()
        for f, psit_G in zip(weights, self):
            # Same as (but much faster):
            # out.data += f * abs(psit.ifft().data)**2
            psit_G.ifft(out=tmp_R)
            _gpaw.add_to_density(f, tmp_R.data, out.data)

    def to_pbc_grid(self):
        return self


class Empty:
    def __init__(self, dims):
        self.dims = dims

    def _arrays(self):
        for _ in range(prod(self.dims)):
            yield


def find_reciprocal_vectors(ecut: float,
                            cell: Array2D,
                            kpt=np.zeros(3),
                            dtype=complex) -> tuple[Array2D,
                                                    Array1D,
                                                    Array2D]:
    """Find reciprocal lattice vectors inside sphere.

    >>> cell = np.eye(3)
    >>> ecut = 0.5 * (2 * pi)**2
    >>> G, e, i = find_reciprocal_vectors(ecut, cell)
    >>> G
    array([[ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  6.28318531],
           [ 0.        ,  0.        , -6.28318531],
           [ 0.        ,  6.28318531,  0.        ],
           [ 0.        , -6.28318531,  0.        ],
           [ 6.28318531,  0.        ,  0.        ],
           [-6.28318531,  0.        ,  0.        ]])
    >>> e
    array([ 0.       , 19.7392088, 19.7392088, 19.7392088, 19.7392088,
           19.7392088, 19.7392088])
    >>> i
    array([[ 0,  0,  0,  0,  0,  1, -1],
           [ 0,  0,  0,  1, -1,  0,  0],
           [ 0,  1, -1,  0,  0,  0,  0]])
    """
    Gcut = (2 * ecut)**0.5
    n = Gcut * (cell**2).sum(axis=1)**0.5 / (2 * pi) + abs(kpt)
    size = 2 * n.astype(int) + 4

    if dtype == float:
        size[2] = size[2] // 2 + 1
        i_Qc = np.indices(size).transpose((1, 2, 3, 0))
        i_Qc[..., :2] += size[:2] // 2
        i_Qc[..., :2] %= size[:2]
        i_Qc[..., :2] -= size[:2] // 2
    else:
        i_Qc = np.indices(size).transpose((1, 2, 3, 0))  # type: ignore
        half = [s // 2 for s in size]
        i_Qc += half
        i_Qc %= size
        i_Qc -= half

    # Calculate reciprocal lattice vectors:
    B_cv = 2.0 * pi * np.linalg.inv(cell).T
    # i_Qc.shape = (-1, 3)
    G_plus_k_Qv = (i_Qc + kpt) @ B_cv

    ekin = 0.5 * (G_plus_k_Qv**2).sum(axis=3)
    mask = ekin <= ecut

    assert not mask[size[0] // 2].any()
    assert not mask[:, size[1] // 2].any()
    if dtype == complex:
        assert not mask[:, :, size[2] // 2].any()
    else:
        assert not mask[:, :, -1].any()

    if dtype == float:
        mask &= ((i_Qc[..., 2] > 0) |
                 (i_Qc[..., 1] > 0) |
                 ((i_Qc[..., 0] >= 0) & (i_Qc[..., 1] == 0)))

    indices = i_Qc[mask]
    ekin = ekin[mask]
    G_plus_k = G_plus_k_Qv[mask]

    return G_plus_k, ekin, indices.T


x = '''
class PWMapping:
    def __init__(self, pw1: PlaneWaves, pw2: PlaneWaves):
        """Mapping from pd1 to pd2."""
        N_c = pw1.grid.size
        N2_c = pw2.grid.size
        assert pw1.grid.dtype == pw2.grid.dtype
        if pw1.grid.dtype == float:
            N_c = N_c.copy()
            N_c[2] = N_c[2] // 2 + 1
            N2_c = N2_c.copy()
            N2_c[2] = N2_c[2] // 2 + 1

        Q1_G = pw1.myindices
        Q1_Gc = np.empty((len(Q1_G), 3), int)
        Q1_Gc[:, 0], r_G = divmod(Q1_G, N_c[1] * N_c[2])
        Q1_Gc.T[1:] = divmod(r_G, N_c[2])
        if pw1.grid.dtype == float:
            C = 2
        else:
            C = 3
        Q1_Gc[:, :C] += N_c[:C] // 2
        Q1_Gc[:, :C] %= N_c[:C]
        Q1_Gc[:, :C] -= N_c[:C] // 2
        Q1_Gc[:, :C] %= N2_c[:C]
        Q2_G = Q1_Gc[:, 2] + N2_c[2] * (Q1_Gc[:, 1] + N2_c[1] * Q1_Gc[:, 0])
        G2_Q = np.empty(N2_c, int).ravel()
        G2_Q[:] = -1
        G2_Q[pw2.myindices] = np.arange(len(pw2.myindices))
        G2_G1 = G2_Q[Q2_G]

        if pw1.grid.comm.size == 1:
            self.G2_G1 = G2_G1
            self.G1 = None
        else:
            mask_G1 = (G2_G1 != -1)
            self.G2_G1 = G2_G1[mask_G1]
            self.G1 = np.arange(pw1.maxmysize)[mask_G1]

        self.pw1 = pw1
        self.pw2 = pw2

    def add_to1(self, a_G1, b_G2):
        """Do a += b * scale, where a is on pd1 and b on pd2."""
        scale = self.pd1.tmp_R.size / self.pd2.tmp_R.size

        if self.pd1.gd.comm.size == 1:
            a_G1 += b_G2[self.G2_G1] * scale
            return

        b_G1 = self.pd1.tmp_G
        b_G1[:] = 0.0
        b_G1[self.G1] = b_G2[self.G2_G1]
        self.pd1.gd.comm.sum(b_G1)
        ng1 = self.pd1.gd.comm.rank * self.pd1.maxmyng
        ng2 = ng1 + self.pd1.myng_q[0]
        a_G1 += b_G1[ng1:ng2] * scale

    def add_to2(self, a2, b1):
        """Do a += b * scale, where a is on pd2 and b on pd1."""
        myb = b1.data * (self.pw2.grid.shape[0] / self.pw1.grid.shape[0])
        if self.desc1.grid.comm.size == 1:
            a2.data[self.G2_G1] += myb
        else:
            1 / 0
'''
