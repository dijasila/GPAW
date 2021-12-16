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
        self.kpt_kc = points
        self.gamma_only = len(self.kpt_kc) == 1 and not self.kpt_kc.any()

    def __len__(self):
        """Number of k-points in the IBZ."""
        return len(self.kpt_kc)

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
        self.weight_i = weights
        self.kpt_ic = bz.kpt_kc[ibz2bz]
        self.ibz2bz_i = ibz2bz
        self.bz2ibz_k = bz2ibz

        self.bz2bz_ks = []  # later ...

    def __len__(self):
        """Number of k-points in the IBZ."""
        return len(self.kpt_ic)

    def __repr__(self):
        return f'IBZ(<{plural(len(self), "point")}>)'

    def __str__(self):
        s = ''
        if -1 in self.bz2bz_ks:
            s += 'Note: your k-points are not as symmetric as your crystal!\n'
        N = len(self)
        s += str(self.bz)
        nk = plural(N, 'k-point')
        s += f'\n{nk} in the irreducible part of the Brillouin zone\n'

        if isinstance(self.bz, MonkhorstPackKPoints):
            w_i = (self.weight_i * len(self.bz)).round().astype(int)

        s += '          k-points in crystal coordinates           weights\n'
        for i, (a, b, c) in enumerate(self.kpt_ic):
            if i >= 10 and i < N - 1:
                continue
            elif i == 10:
                s += '          ...\n'
            s += f'{i:4}:   {a:12.8f}  {b:12.8f}  {c:12.8f}     '
            if isinstance(self.bz, MonkhorstPackKPoints):
                s += f'{w_i[i]}/{N}\n'
            else:
                s += f'{self.weight_i[i]:.8f}\n'
        return s

    def ranks(self, comm: MPIComm) -> Array1D:
        """Distribute k-points over MPI-communicator.

        Example (6 k-points and 4 cores)::

            [0, 0, 1, 1, 2, 3]
        """
        nibzk = len(self)
        n = nibzk // comm.size
        x = nibzk - comm.size * n
        assert x * (n + 1) + (comm.size - x) * n == nibzk

        rnks = np.empty(nibzk, int)
        for k in range(x * (n + 1)):
            rnks[k] = k // (n + 1)
        for k in range(x * (n + 1), nibzk):
            rnks[k] = (k - x * (n + 1)) // n + x
        return rnks
