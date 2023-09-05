from time import time

import numpy as np
import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.gpu import cupy as cp


def abs_square(a: float,  # lattice constant
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
    weight_n = np.linspace(1, 0, B)
    nt_R = grid.zeros(xp=xp)

    t = time()
    psit_nG.abs_square(weight_n, nt_R, _slow=slow)
    t = time() - t

    assert nt_R.integrate() == pytest.approx(a**3 * weight_n.sum())

    return t


@pytest.mark.parametrize('xp',
                         [np,
                          pytest.param(cp, marks=pytest.mark.gpu)])
@pytest.mark.parametrize('nbands', [2, 17])
def test_pw_abs_square(xp, nbands):
    abs_square(a=2.5, N=6, B=nbands, xp=xp)


def main():
    """Test speedup for larger system."""
    abs_square(6.0, 32, 100, cp)  # GPU-warmup
    t = abs_square(6.0, 32, 100, cp)
    print('Fast:', t)
    t = abs_square(6.0, 32, 100, cp, slow=True)
    print('Slow:', t)


if __name__ == '__main__':
    main()
