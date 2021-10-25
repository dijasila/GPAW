from __future__ import annotations

from math import pi

import _gpaw
import numpy as np
from gpaw.core.arrays import DistributedArrays
from gpaw.core.atom_centered_functions import PlaneWaveAtomCenteredFunctions
from gpaw.core.layout import Layout
from gpaw.core.matrix import Matrix
from gpaw.core.uniform_grid import UniformGrid, UniformGridFunctions
from gpaw.mpi import MPIComm, serial_comm
from gpaw.pw.descriptor import pad
from gpaw.typing import Array1D, Array2D


class PlaneWaves(Layout):
    def __init__(self,
                 ecut: float,
                 grid: UniformGrid):
        self.ecut = ecut
        self.grid = grid
        assert grid.pbc.all()

        self.comm = grid.comm
        self.dtype = grid.dtype
        G_plus_k, ekin, self.indices = find_reciprocal_vectors(ecut, grid)

        # Find distribution:
        S = grid.comm.size
        ng = len(self.indices)
        self.maxmysize = (ng + S - 1) // S
        ng1 = grid.comm.rank * self.maxmysize
        ng2 = ng1 + self.maxmysize

        # Distribute things:
        self.ekin = ekin[ng1:ng2].copy()
        self.ekin.flags.writeable = False
        self.myindices = self.indices[ng1:ng2]
        self.G_plus_k = G_plus_k[ng1:ng2]

        Layout.__init__(self, (ng,), (len(self.myindices),))

        self.dv = grid.dv / grid.size.prod()

    def __str__(self) -> str:
        a, b, c = self.grid.size
        comm = self.grid.comm
        txt = f'PlaneWaves(ecut={self.ecut}, grid={a}*{b}*{c}'
        if comm.size > 1:
            txt += f', comm={comm.rank}/{comm.size}'
        return txt + ')'

    def reciprocal_vectors(self) -> Array2D:
        """Returns reciprocal lattice vectors, G + k,
        in xyz coordinates."""
        return self.G_plus_k

    def kinetic_energies(self) -> Array1D:
        return self.ekin

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> PlaneWaveExpansions:
        return PlaneWaveExpansions(self, shape, comm)

    def atom_centered_functions(self,
                                functions,
                                positions,
                                atomdist=None,
                                integral=None):
        return PlaneWaveAtomCenteredFunctions(functions, positions, self,
                                              atomdist)


class PlaneWaveExpansions(DistributedArrays):
    def __init__(self,
                 pw: PlaneWaves,
                 shape: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        DistributedArrays. __init__(self, pw, shape, comm, data, complex)
        self.pw = pw

    def __repr__(self):
        txt = f'PlaneWaveExpansions(pw={self.pw}, shape={self.shape}'
        if self.comm.size > 1:
            txt += f', comm={self.comm.rank}/{self.comm.size}'
        return txt + ')'

    def __getitem__(self, index: int) -> PlaneWaveExpansions:
        return PlaneWaveExpansions(self.pw, data=self.data[index])

    def __iter__(self):
        for data in self.data:
            yield PlaneWaveExpansions(self.pw, data=data)

    def new(self, data=None):
        if data is None:
            data = np.empty_like(self.data)
        return PlaneWaveExpansions(self.pw, self.shape, self.comm, data)

    def _arrays(self):
        return self.data.reshape((-1,) + self.data.shape[-1:])

    def ifft(self, plan=None, out=None):
        out = out or self.pw.grid.empty(self.shape)
        plan = plan or self.pw.grid.fft_plans()[1]
        scale = 1.0 / plan.out_R.size
        for input, output in zip(self._arrays(), out._arrays()):
            _gpaw.pw_insert(input, self.pw.indices, scale, plan.in_R)
            if self.pw.grid.dtype == float:
                t = plan.in_R[:, :, 0]
                n, m = (s // 2 - 1 for s in self.pw.grid.size[:2])
                t[0, -m:] = t[0, m:0:-1].conj()
                t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
                t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
                t[-n:, 0] = t[n:0:-1, 0].conj()
            plan.execute()
            output[:] = plan.out_R

        return out

    def collect(self, out=None):
        """Gather coefficients on master."""
        comm = self.pw.grid.comm

        if comm.size == 1:
            if out is None:
                return self
            out.data[:] = self.data
            return out

        if out is None:
            if comm.rank == 0:
                pw = PlaneWaves(ecut=self.pw.ecut,
                                grid=self.pw.grid.new(comm=serial_comm))
                out = pw.empty(self.shape)
            else:
                out = Empty()

        if comm.rank == 0:
            data = np.empty(self.pw.maxmysize * comm.size, complex)
        else:
            data = None

        for input, output in zip(self._arrays(), out._arrays()):
            mydata = pad(input, self.maxmysize)
            comm.gather(mydata, 0, data)
            if comm.rank == 0:
                output[:] = data[:self.grid.size]

        return out if comm.rank == 0 else None

    def _distribute(self, data, out)
        """Scatter coefficients from master to all cores."""
        comm = self.gd.comm
        if comm.size == 1:
            return a_G

        mya_G = np.empty(self.maxmyng, complex)
        comm.scatter(pad(a_G, self.maxmyng * comm.size), mya_G, 0)
        return mya_G[:self.myng_q[q or 0]]
    def integrate(self, other: PlaneWaveExpansions = None) -> np.ndarray:
        if other is not None:
            assert self.pw.grid.dtype == other.pw.grid.dtype
            a = self._arrays()
            b = other._arrays()
            dv = self.pw.dv
            if self.pw.dtype == float:
                a = a.view(float)
                b = b.view(float)
                dv *= 2
            result = a @ b.T.conj()
            if self.pw.grid.dtype == float and self.pw.grid.comm.rank == 0:
                result -= 0.5 * np.outer(a[:, 0], b[:, 0])
            self.pw.comm.sum(result)
            result.shape = self.shape + other.shape
        else:
            dv = self.pw.grid.dv
            result = self.data[..., 0]
            if self.pw.grid.comm.rank > 0:
                result = np.empty_like(result)
            self.pw.grid.comm.broadcast(result[np.newaxis], 0)

        if self.pw.grid.dtype == float:
            result = result.real
        return result * dv

    def _matrix_elements_correction(self,
                                    M1: Matrix,
                                    M2: Matrix,
                                    out: Matrix,
                                    symmetric: bool) -> None:
        if self.pw.dtype == float:
            out.data *= 2.0
            if self.pw.comm.rank == 0:
                correction = np.outer(M1.data[:, 0],
                                      M2.data[:, 0]) * self.pw.dv
                if symmetric:
                    correction *= 0.5
                    out.data -= correction
                    out.data -= correction.T
                else:
                    out.data -= correction

    def norm2(self, kind: str = 'normal') -> np.ndarray:
        a = self._arrays().view(float)
        if kind == 'normal':
            result = np.einsum('ig, ig -> i', a, a)
        elif kind == 'kinetic':
            a.shape = (len(a), -1, 2)
            result = np.einsum('igx, igx, g -> i', a, a, self.pw.ekin)
        else:
            1 / 0
        if self.pw.dtype == float:
            result *= 2
            if self.pw.comm.rank == 0 and kind == 'normal':
                result -= a[:, 0] * a[:, 0]
        self.pw.comm.sum(result)
        result.shape = self.myshape
        return result * self.pw.dv

    def abs_square(self,
                   weights: Array1D,
                   out: UniformGridFunctions = None) -> None:
        assert out is not None
        for f, psit in zip(weights, self):
            # Same as (but much faster):
            # out.data += f * abs(psit.ifft().data)**2
            _gpaw.add_to_density(f, psit.ifft().data, out.data)


class Empty:
    def _arrays(self):
        while True:
            yield


def find_reciprocal_vectors(ecut: float,
                            grid: UniformGrid) -> tuple[Array2D,
                                                        Array1D,
                                                        Array1D]:
    size = grid.size

    if grid.dtype == float:
        Nr_c = list(size)
        Nr_c[2] = size[2] // 2 + 1
        i_Qc = np.indices(Nr_c).transpose((1, 2, 3, 0))
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
    B_cv = 2.0 * pi * grid.icell
    i_Qc.shape = (-1, 3)
    G_plus_k_Qv = np.dot(i_Qc + grid.kpt, B_cv)

    # Map from vectors inside sphere to fft grid:
    Q_Q = np.arange(len(i_Qc), dtype=np.int32)

    G2_Q = (G_plus_k_Qv**2).sum(axis=1)
    mask_Q = (G2_Q <= 2 * ecut)

    if grid.dtype == float:
        mask_Q &= ((i_Qc[:, 2] > 0) |
                   (i_Qc[:, 1] > 0) |
                   ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))

    indices = Q_Q[mask_Q]
    ekin = 0.5 * G2_Q[indices]
    G_plus_k = G_plus_k_Qv[mask_Q]

    return G_plus_k, ekin, indices


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
        if self.pw1.grid.comm.size == 1:
            a2.data[self.G2_G1] += myb
        else:
            1 / 0
