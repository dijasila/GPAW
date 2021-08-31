from __future__ import annotations
from math import pi
import numpy as np
from gpaw.pw.lfc import PWLFC
from gpaw.spline import Spline
from gpaw.matrix import Matrix

__all__ = ['Matrix']


def gaussian(l=0, alpha=3.0, rcut=4.0):
    r = np.linspace(0, rcut, 41)
    return Spline(l, rcut, np.exp(-alpha * r**2))


class UniformGrid:
    def __init__(self, cell,
                 size,
                 pbc=(True, True, True),
                 kpt=(0.0, 0.0, 0.0),
                 dist=None):
        self.cell = np.array(cell)
        self.size = size
        self.pbc = np.array(pbc, bool)
        self.kpt = kpt

    def new(self, kpt=None) -> UniformGrid:
        return UniformGrid(self.cell, self.size, self.pbc,
                           kpt=self.kpt if kpt is None else kpt)

    def empty(self, shape=()) -> UniformGrids:
        if isinstance(shape, int):
            shape = (shape,)
        return UniformGrids(np.empty(shape + self.size), self)

    def zeros(self, shape=()) -> UniformGrids:
        funcs = self.empty(shape)
        funcs._data[:] = 0.0
        return funcs


class UniformGrids:
    def __init__(self, data, ug, dist=None):
        self.ug = ug
        self._data = data

    def __index__(self, index):
        return UniformGrids(self._data[index], self.ug)


class PlaneWaves:
    def __init__(self, ecut: float, ug):
        self.ug = ug
        self.ecut = ecut

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
        self.G_Qv = np.dot(i_Qc, B_cv)

        self.K_qv = np.dot(kd.ibzk_qc, B_cv)

        # Map from vectors inside sphere to fft grid:
        self.Q_qG = []
        G2_qG = []
        Q_Q = np.arange(len(i_Qc), dtype=np.int32)

        self.ng_q = []
        for q, K_v in enumerate(self.K_qv):
            G2_Q = ((self.G_Qv + K_v)**2).sum(axis=1)
            mask_Q = (G2_Q <= 2 * ecut)

            if self.dtype == float:
                mask_Q &= ((i_Qc[:, 2] > 0) |
                           (i_Qc[:, 1] > 0) |
                           ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))
            Q_G = Q_Q[mask_Q]
            self.Q_qG.append(Q_G)
            G2_qG.append(G2_Q[Q_G])
            ng = len(Q_G)
            self.ng_q.append(ng)

        self.ngmin = min(self.ng_q)
        self.ngmax = max(self.ng_q)

        if kd is not None:
            self.ngmin = kd.comm.min(self.ngmin)
            self.ngmax = kd.comm.max(self.ngmax)

        # Distribute things:
        S = gd.comm.size
        self.maxmyng = (self.ngmax + S - 1) // S
        ng1 = gd.comm.rank * self.maxmyng
        ng2 = ng1 + self.maxmyng

        self.G2_qG = []
        self.myQ_qG = []
        self.myng_q = []
        for q, G2_G in enumerate(G2_qG):
            G2_G = G2_G[ng1:ng2].copy()
            G2_G.flags.writeable = False
            self.G2_qG.append(G2_G)
            myQ_G = self.Q_qG[q][ng1:ng2]
            self.myQ_qG.append(myQ_G)
            self.myng_q.append(len(myQ_G))

        if S > 1:
            self.tmp_G = np.empty(self.maxmyng * S, complex)
        else:
            self.tmp_G = None

    def get_reciprocal_vectors(self, q=0, add_q=True):
        """Returns reciprocal lattice vectors plus q, G + q,
        in xyz coordinates."""

        if add_q:
            q_v = self.K_qv[q]
            return self.G_Qv[self.myQ_qG[q]] + q_v
        return self.G_Qv[self.myQ_qG[q]]

    def zeros(self, x=(), dtype=None, q=None, global_array=False):
        """Return zeroed array.

        The shape of the array will be x + (ng,) where ng is the number
        of G-vectors for on this core.  Different k-points will have
        different values for ng.  Therefore, the q index must be given,
        unless we are describibg a real-valued function."""

        a_xG = self.empty(x, dtype, q, global_array)
        a_xG.fill(0.0)
        return a_xG

    def empty(self, x=(), dtype=None, q=None, global_array=False):
        """Return empty array."""
        if dtype is not None:
            assert dtype == self.dtype
        if isinstance(x, numbers.Integral):
            x = (x,)
        if q is None:
            assert self.only_one_k_point
            q = 0
        if global_array:
            shape = x + (self.ng_q[q],)
        else:
            shape = x + (self.myng_q[q],)
        return np.empty(shape, complex)


class AtomCenteredFunctions:
    def __init__(self, functions, positions=None, kpts=None):
        self.functions = functions
        self.positions = positions
        self.kpts = kpts
        if kpts is None:
            self.kpt2index = {}
        else:
            self.kpt2index = {tuple(kpt): index
                              for index, kpt in enumerate(kpts)}


class ReciprocalSpaceAtomCenteredFunctions(AtomCenteredFunctions):
    def __init__(self, functions, positions=None, kpts=None):
        AtomCenteredFunctions.__init__(self, functions, positions, kpts)
        self.lfc = PWLFC()
        self.lfc.set_positions(positions)

    def add(self, coefs, functions):
        index = self.kpt2index[functions.kpt]
        self.lfc.add(functions._data._data, coefs, q=index)
