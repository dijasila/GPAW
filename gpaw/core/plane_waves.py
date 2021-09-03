from __future__ import annotations
import numpy as np
from math import pi
import gpaw.fftw as fftw
import _gpaw


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
    def __init__(self, ecut: float,
                 ug,
                 dist):
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
        funcs.data[:] = 0.0
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
        self.data = data
        self.shape = data.shape[:-1]

    def __getitem__(self, index):
        return PlaneWaveExpansions(self.data[index], self.pw)

    def _arrays(self):
        return self.data.reshape((-1,) + self._data.shape[-1:])

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
