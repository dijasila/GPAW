from __future__ import annotations
import numpy as np
from math import pi
import gpaw.fftw as fftw
import _gpaw
from gpaw.core.layout import Layout
from gpaw.core.uniform_grid import UniformGrid
from gpaw.typing import Array1D, Array2D
from gpaw.mpi import MPIComm, serial_comm
from gpaw.core.arrays import DistributedArrays
from gpaw.pw.descriptor import pad


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
        i_Qc = np.indices(size).transpose((1, 2, 3, 0))
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

    if grid.kpt is None:
        mask_Q &= ((i_Qc[:, 2] > 0) |
                   (i_Qc[:, 1] > 0) |
                   ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))

    indices = Q_Q[mask_Q]
    ekin = 0.5 * G2_Q[indices]
    G_plus_k = G_plus_k_Qv[mask_Q]

    return G_plus_k, ekin, indices


class PlaneWaves(Layout):
    def __init__(self,
                 ecut: float,
                 grid: UniformGrid,
                 comm: MPIComm = serial_comm):
        self.ecut = ecut
        self.grid = grid.new(pbc=(True, True, True))
        self.pbc = grid.pbc

        self.dtype = complex

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

        Layout.__init__(self, (len(self.myindices),))

    def __str__(self):
        a, b, c = self.grid.size
        comm = self.grid.comm
        return (f'PlaneWaves(ecut={self.ecut}, grid={a}*{b}*{c}, '
                f'comm={comm.rank}/{comm.size})')

    def new(self,
            comm=None) -> PlaneWaves:
        return PlaneWaves(ecut=self.ecut,
                          grid=self.grid,
                          comm=comm or self.comm)

    def reciprocal_vectors(self):
        """Returns reciprocal lattice vectors, G + k,
        in xyz coordinates."""
        return self.G_plus_k

    def kinetic_energies(self):
        return self.ekin

    def empty(self,
              shape: int | tuple[int, ...] = (),
              comm: MPIComm = serial_comm) -> PlaneWaveExpansions:
        return PlaneWaveExpansions(self, shape, comm)

    def fft_plans(self, flags=fftw.MEASURE):
        size = self.grid.size
        if self.grid.kpt is None:
            rsize = size[:2] + (size[2] // 2 + 1,)
            tmp1 = fftw.empty(rsize, complex)
            tmp2 = tmp1.view(float)[:, :, :size[2]]
        else:
            tmp1 = fftw.empty(size, complex)
            tmp2 = tmp1

        fftplan = fftw.FFTPlan(tmp2, tmp1, -1, flags)
        ifftplan = fftw.FFTPlan(tmp1, tmp2, 1, flags)
        return fftplan, ifftplan


class PlaneWaveExpansions(DistributedArrays):
    def __init__(self,
                 pw: PlaneWaves,
                 shape: int | tuple[int, ...] = (),
                 comm: MPIComm = serial_comm,
                 data: np.ndarray = None):
        DistributedArrays. __init__(self, pw, shape, comm, data)
        self.pw = pw

    def __repr__(self):
        return (f'PlaneWaveExpansions(pw={self.pw}, shape={self.shape}, '
                f'comm={self.comm.rank}/{self.comm.size})')

    def __getitem__(self, index):
        return PlaneWaveExpansions(self.pw, data=self.data[index])

    def _arrays(self):
        return self.data.reshape((-1,) + self.data.shape[-1:])

    def ifft(self, plan=None, out=None):
        out = out or self.pw.grid.empty(self.shape)
        plan = plan or self.pw.fft_plans()[1]
        scale = 1.0 / plan.out_R.size
        for input, output in zip(self._arrays(), out._arrays()):
            _gpaw.pw_insert(input, self.pw.indices, scale, plan.in_R[:])
            if self.pw.grid.kpt is None:
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
        if out is None:
            out = self.pw.empty(self.shape)
        comm = self.pw.grid.comm

        if comm.size == 1:
            if out is not self:
                out.data[:] = self.data
            return out

        if comm.rank == 0:
            data = np.empty(self.maxmysize * comm.size, complex)
        else:
            data = None

        for input, output in zip(self._arrays(), out._arrays()):
            mydata = pad(input, self.maxmyng)
            comm.gather(mydata, 0, data)
            if comm.rank == 0:
                output[:] = data[:self.grid.size]

        return out
