from time import time

import numpy as np
import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.gpu import cupy as cp
from gpaw.new.pw.hamiltonian import PWHamiltonian


def apply(a: float,  # lattice constant
          N: int,  # number of grid-points
          B: int,  # number of bands
          xp,
          slow=False) -> float:
    """Calculate density from wave functions."""
    grid = UGDesc(cell=[a, a, a], size=[N, N, N])
    ecut = 0.5 * (np.pi * N / a)
    pw = PWDesc(ecut=ecut, cell=grid.cell, dtype=complex)
    psit_nG = pw.zeros(B, xp=xp)
    psit_nG.data[:, 0] = 1.0
    vt_R = grid.zeros(xp=xp)
    vt_R.data[:] = -2.0
    ham = PWHamiltonian(grid, pw, xp)
    t = time()
    ham.apply_local_potential(vt_R, psit_nG, psit_nG.new())
    t = time() - t
    return t


@pytest.mark.parametrize('xp',
                         [np,
                          pytest.param(cp, marks=pytest.mark.gpu)])
@pytest.mark.parametrize('nbands', [2, 17])
def test_apply_ham(xp, nbands):
    apply(a=2.5, N=6, B=nbands, xp=xp)


def main():
    """Test speedup for larger system."""
    for _ in range(2):
        t = apply(6.0, 32, 100, cp)
        print(t)


if __name__ == '__main__':
    main()
