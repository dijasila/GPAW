from time import time

import numpy as np
import pytest

from gpaw.core import PlaneWaves, UniformGrid
from gpaw.gpu import cupy as cp


def abs_square(a: float,  # lattice constanc
               N: int,  # number of grid-points
               B: int,  # number of bands
               xp,
               slow=False) -> float:
    grid = UniformGrid(cell=[a, a, a], size=[N, N, N])
    ecut = 0.5 * (np.pi * N / a)
    pw = PlaneWaves(ecut=ecut, cell=grid.cell, dtype=complex)
    psit_nG = pw.zeros(B, xp=xp)
    psit_nG.data[:, 0] = 1.0
    weight_n = np.linspace(1, 0, B)
    nt_R = grid.zeros(xp=xp)
    t = time()
    psit_nG.abs_square(weight_n, nt_R, _slow=slow)
    t = time() - t
    assert nt_R.integrate() == pytest.approx(a**3 * weight_n.sum())
    return t


@pytest.mark.parametrize('xp', [np, cp])
def test_pw_abs_square(xp):
    abs_square(a=2.5, N=10, B=17, xp=xp)


if __name__ == '__main__':
    t = abs_square(6.0, 32, 100, cp)  # warmup
    t = abs_square(6.0, 32, 100, cp)
    print('Fast:', t)
    abs_square(6.0, 32, 100, cp, slow=True)
    t = abs_square(6.0, 32, 100, cp, slow=True)
    print('Slow:', t)
