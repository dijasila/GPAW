from gpaw.densities import add
from gpaw.spline import Spline
from gpaw.core import UniformGrid
import numpy as np


def test_ae_density():
    n = 10
    n_sR = UniformGrid(cell=[1, 1, 1], size=[n, n, n], pbc=True).zeros(1)
    D_sii = np.zeros((1, 4, 4))
    D_sii[0, 0, 0] = 1.0
    D_sii[0, 1, 1] = 1.0
    rc = 0.5
    phi0 = Spline(0, rc, [1, 0])
    phit0 = Spline(0, rc, [2, 0])
    phi1 = Spline(1, rc, [1, 0])
    phit1 = Spline(1, rc, [2, 0])
    nc = Spline(0, rc, [0, 0])
    add([0.5, 0.5, 0.5],
        n_sR,
        [phi0, phi1], [phit0, phit1],
        nc, nc,
        rc, D_sii)
    y, v = n_sR.xy(0, 5, ..., 5)
    v *= 4 * np.pi
    y = abs(y - 0.5)
    v0 = -9 * phi1.map(y)**2 * y**2 - 3 * phi0.map(y)**2
    assert abs(v - v0).max() < 1e-14
