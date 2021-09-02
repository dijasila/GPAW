from __future__ import annotations
from math import pi
import numpy as np
from gpaw.pw.lfc import PWLFC
from gpaw.spline import Spline
from gpaw.matrix import Matrix
from gpaw.typing import ArrayLike1D, ArrayLike, Array2D, ArrayLike2D
from gpaw.mpi import serial_comm
from gpaw.pw.descriptor import PWDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
import gpaw.fftw as fftw
import _gpaw
from gpaw.utilities.grid import GridRedistributor

from typing import Any

__all__ = ['Matrix']

MPIComm = Any


def gaussian(l=0, alpha=3.0, rcut=4.0):
    r = np.linspace(0, rcut, 41)
    return Spline(l, rcut, np.exp(-alpha * r**2))


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


class UniformGrid:
    def __init__(self,
                 cell: ArrayLike1D | ArrayLike2D,
                 size: ArrayLike1D,
                 pbc=(True, True, True),
                 kpt: ArrayLike1D = None,
                 dist: MPIComm | UniformGridDistribution | None = None):
        """"""
        self.cell = _normalize_cell(cell)
        self.size = size
        self.pbc = np.array(pbc, bool)
        self.kpt = kpt

        dist = dist or serial_comm
        if not isinstance(dist, UniformGridDistribution):
            dist = UniformGridDistribution(dist, size, pbc)
        self.dist = dist

        self.dtype = float if kpt is None else complex
        self.icell = np.linalg.inv(self.cell).T

    def new(self, kpt='_default', dist='_default') -> UniformGrid:
        return UniformGrid(self.cell, self.size, self.pbc,
                           kpt=self.kpt if kpt == '_default' else kpt,
                           dist=self.dist if dist == '_default' else dist)

    def empty(self, shape=(), dist=None) -> UniformGrids:
        if isinstance(shape, int):
            shape = (shape,)
        array = np.empty(shape + self.dist.size, self.dtype)
        return UniformGrids(array, self, dist)

    def zeros(self, shape=(), dist=None) -> UniformGrids:
        funcs = self.empty(shape, dist)
        funcs._data[:] = 0.0
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
        self.redistributor.distribute(input._data, out._data)
        return out


class UniformGridFunctions:
    def __init__(self, data, ug, dist=None):
        self.ug = ug
        self._data = data
        self.shape = data.shape[:-3]

    def __getitem__(self, index):
        return UniformGridFunctions(self._data[index], self.ug)

    def _arrays(self):
        return self._data.reshape((-1,) + self._data.shape[-3:])

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


def find_reciprocal_vectors(ecut, ug):
    size = ug.size

    if ug.kpt is None:
        Nr_c = list(size)
        Nr_c[2] = size[2] // 2 + 1
        i_Qc = np.indices(Nr_c).transpose((1, 2, 3, 0))
        i_Qc[..., :2] += size[:2] // 2
        i_Qc[..., :2] %= size[:2]
        i_Qc[..., :2] -= size[:2] // 2
    else:
        i_Qc = np.indices(size).transpose((1, 2, 3, 0))
        half = [s // 2 for s in size]
        i_Qc += half
        i_Qc %= size
        i_Qc -= half

    # Calculate reciprocal lattice vectors:
    B_cv = 2.0 * pi * ug.icell
    i_Qc.shape = (-1, 3)
    G_plus_k_Qv = np.dot(i_Qc + ug.kpt, B_cv)

    # Map from vectors inside sphere to fft grid:
    Q_Q = np.arange(len(i_Qc), dtype=np.int32)

    G2_Q = (G_plus_k_Qv**2).sum(axis=1)
    mask_Q = (G2_Q <= 2 * ecut)

    if ug.kpt is None:
        mask_Q &= ((i_Qc[:, 2] > 0) |
                   (i_Qc[:, 1] > 0) |
                   ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))

    indices = Q_Q[mask_Q]
    ekin = 0.5 * G2_Q[indices]
    G_plus_k = G_plus_k_Qv[mask_Q]

    return G_plus_k, ekin, indices


class PlaneWaves:
    def __init__(self, ecut: float, ug):
        self.ug = ug
        self.ecut = ecut

        G_plus_k, ekin, self.all_indices = find_reciprocal_vectors(ecut, ug)

        # Distribute things:
        S = ug.dist.comm.size
        ng = len(self.all_indices)
        myng = (ng + S - 1) // S
        ng1 = ug.dist.comm.rank * myng
        ng2 = ng1 + myng

        self.ekin = ekin[ng1:ng2].copy()
        self.ekin.flags.writeable = False
        self.indices = self.all_indices[ng1:ng2]
        self.G_plus_k = G_plus_k[ng1:ng2]
        self.size = len(self.indices)

    def reciprocal_vectors(self):
        """Returns reciprocal lattice vectors, G + k,
        in xyz coordinates."""
        return self.G_plus_k

    def kinetic_energies(self):
        return self.ekin

    def empty(self, shape=(), dist=None) -> PlaneWaveExpansions:
        if isinstance(shape, int):
            shape = (shape,)
        array = np.empty(shape + (self.size,), complex)
        return PlaneWaveExpansions(array, self, dist)

    def zeros(self, shape=(), dist=None) -> PlaneWaveExpansions:
        funcs = self.empty(shape, dist)
        funcs._data[:] = 0.0
        return funcs

    def fft_plans(self, flags=fftw.MEASURE):
        size = self.ug.size
        if self.ug.kpt is None:
            rsize = size[:2] + (size[2] // 2 + 1,)
            tmp1 = fftw.empty(rsize, complex)
            tmp2 = tmp1.view(float)[:, :, :size[2]]
        else:
            tmp1 = fftw.empty(size, complex)
            tmp2 = tmp1

        fftplan = fftw.FFTPlan(tmp2, tmp1, -1, flags)
        ifftplan = fftw.FFTPlan(tmp1, tmp2, 1, flags)
        return fftplan, ifftplan


class PlaneWaveExpansions:
    def __init__(self, data, pw, dist=None):
        self.pw = pw
        self._data = data
        self.shape = data.shape[:-1]

    def __getitem__(self, index):
        return PlaneWaveExpansions(self._data[index], self.pw)

    def _arrays(self):
        return self._data.reshape((-1,) + self._data.shape[-1:])

    def ifft(self, plan=None, out=None):
        out = out or self.pw.ug.empty(self.shape)
        plan = plan or self.pw.fft_plans()[1]
        scale = 1.0 / plan.out_R.size
        for input, output in zip(self._arrays(), out._arrays()):
            _gpaw.pw_insert(input, self.pw.indices, scale, plan.in_R[:])
            if self.pw.ug.kpt is None:
                t = plan.in_R[:, :, 0]
                n, m = (s // 2 - 1 for s in self.pw.ug.size[:2])
                t[0, -m:] = t[0, m:0:-1].conj()
                t[n:0:-1, -m:] = t[-n:, m:0:-1].conj()
                t[-n:, -m:] = t[n:0:-1, m:0:-1].conj()
                t[-n:, 0] = t[n:0:-1, 0].conj()
            plan.execute()
            output[:] = plan.out_R

        return out


class AtomCenteredFunctions:
    def __init__(self, functions, positions=None, kpts=None):
        self.functions = functions
        self.positions = np.array(positions)
        if kpts is None:
            self.kpt2index = {}
        else:
            self.kpt2index = {tuple(kpt): index
                              for index, kpt in enumerate(kpts)}


class ReciprocalSpaceAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, positions=None, kpts=None):
        AtomCenteredFunctions.__init__(self, functions, positions, kpts)
        self.lfc = None

    def _lazy_init(self, pw):
        if self.lfc is not None:
            return
        gd = GridDescriptor(pw.ug.size, pw.ug.cell, pw.ug.pbc)
        kd = KPointDescriptor(np.array(list(self.kpt2index)))
        pd = PWDescriptor(pw.ecut, gd, kd=kd)
        self.lfc = PWLFC(self.functions, pd)
        self.lfc.set_positions(self.positions)

    def add(self, coefs, functions):
        self._lazy_init(functions.pw)
        index = self.kpt2index[functions.pw.ug.kpt]
        self.lfc.add(functions._data, coefs, q=index)
