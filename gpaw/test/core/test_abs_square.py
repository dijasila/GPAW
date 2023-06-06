from math import pi

import numpy as np
import pytest
from gpaw import SCIPY_VERSION
from gpaw.core import PlaneWaves, UniformGrid
from gpaw.core.plane_waves import find_reciprocal_vectors
from gpaw.gpu import cupy as cp
from gpaw.mpi import world


@pytest.mark.parametrize('xp', [np, cp])
def test_pw_abs_square(xp):
    a = 2.5
    N = 10
    B = 17
    grid = UniformGrid(cell=[a, a, a], size=[N, N, N])
    pw = PlaneWaves(ecut=10, cell=grid.cell, dtype=complex)
    psit_nG = pw.zeros(B, xp=xp)
    psit_nG.data[:, 0] = 1.0
    weight_n = np.linspace(1, 0, B)
    nt_R = grid.zeros(xp=xp)
    psit_nG.abs_square(weight_n, nt_R)
    assert nt_R.integrate() == pytest.approx(a**3 * weight_n.sum())
