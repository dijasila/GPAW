from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from ase.utils import plural
from gpaw.mpi import MPIComm
from gpaw.typing import Array1D
if TYPE_CHECKING:
    from gpaw.new.symmetry import Symmetries


class BZPoints:
    def __init__(self, points):
        self.kpt_Kc = points
        self.gamma_only = len(self.kpt_Kc) == 1 and not self.kpt_Kc.any()

    def __len__(self):
        """Number of k-points in the IBZ."""
        return len(self.kpt_Kc)

    def __repr__(self):
        return f'BZPoints([<{len(self)} points>])'

    def __str__(self):
        if self.gamma_only:
            return '1 k-point (Gamma)'
        return f'{len(self)} k-points'


class MonkhorstPackKPoints(BZPoints):
    def __init__(self, size, shift=(0, 0, 0)):
        self.size_c = size
        self.shift_c = np.array(shift)
        super().__init__(monkhorst_pack(size) + shift)

    def __repr__(self):
        return f'MonkhorstPackKPoints({self.size_c}, shift={self.shift_c})'

    def __str__(self):
        if self.gamma_only:
            return '1 k-point (Gamma)'

        a, b, c = self.size_c
        s = f'{len(self)} k-points: {a} x {b} x {c} Monkhorst-Pack grid'

        if self.shift_c.any():
            s += ' + ['
            for x in self.shift_c:
                if x != 0 and abs(round(1 / x) - 1 / x) < 1e-12:
                    s += '1/%d,' % round(1 / x)
                else:
                    s += f'{x:f},'
            s = s[:-1] + ']'

        return s


class IBZ:
    def __init__(self,
                 symmetries: Symmetries,
                 bz: BZPoints,
                 ibz2bz, bz2ibz, weights):
        self.symmetries = symmetries
        self.bz = bz
        self.weight_k = weights
        self.kpt_kc = bz.kpt_Kc[ibz2bz]
        self.ibz2bz_k = ibz2bz
        self.bz2ibz_K = bz2ibz

        # self.bz2bz_Ks = []  # later ...

    def __len__(self):
        """Number of k-points in the IBZ."""
        return len(self.kpt_kc)

    def __repr__(self):
        return f'IBZ(<{plural(len(self), "point")}>)'

    def __str__(self):
        s = ''
        # if -1 in self.bz2bz_Ks:
        #    s += 'Note: your k-points are not as symmetric as your crystal!\n'
        N = len(self)
        s += str(self.bz)
        nk = plural(N, 'k-point')
        s += f'\n{nk} in the irreducible part of the Brillouin zone\n'

        if isinstance(self.bz, MonkhorstPackKPoints):
            w_k = (self.weight_k * len(self.bz)).round().astype(int)

        s += '          k-point in crystal coordinates           weight\n'
        for k, (a, b, c) in enumerate(self.kpt_kc):
            if k >= 10 and k < N - 1:
                continue
            elif k == 10:
                s += '          ...\n'
            s += f'{k:4}:   {a:12.8f}  {b:12.8f}  {c:12.8f}     '
            if isinstance(self.bz, MonkhorstPackKPoints):
                s += f'{w_k[k]}/{len(self.bz)}\n'
            else:
                s += f'{self.weight_k[k]:.8f}\n'
        return s

    def ranks(self, comm: MPIComm) -> Array1D:
        """Distribute k-points over MPI-communicator."""
        return ranks(comm.size, len(self))


def ranks(N, K) -> Array1D:
    """Distribute k-points over MPI-communicator.

    >>> ranks(4, 6)
    array([0, 1, 2, 2, 3, 3])
    """
    n, x = divmod(K, N)

    rnks = np.empty(K, int)
    k0 = K - x * (n + 1)
    for k in range(k0):
        rnks[k] = k // n
    for k in range(k0, K):
        rnks[k] = (k - k0) // (n + 1) + x
    return rnks
