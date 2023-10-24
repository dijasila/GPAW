from gpaw.grid_descriptor import GridDescriptor
from gpaw.poisson import FDPoissonSolver
import pytest
import numpy as np

@pytest.mark.gpu
def test_poisson(gpu):
    import cupy
    rhos = []
    for xp in [np, cupy]:
        lat = 8.0
        gd = GridDescriptor((80, 80, 88), (lat, lat, lat),
                            pbc_c=[False, False, False])
        # Use Gaussian as input
        x, y, z = gd.get_grid_point_coordinates()
        x, y, z = xp.asarray(x), xp.asarray(y), xp.asarray(z)
        sigma = 1.5
        mu = lat / 2.0

        rho = gd.zeros(xp=xp)
        rho[:] = xp.exp(-((x - mu)**2 + (y - mu)**2 + (z - mu)**2) / (2.0 * sigma))
        phi = gd.zeros(xp=xp)

        poisson = FDPoissonSolver(xp=xp)
        poisson.set_grid_descriptor(gd)
        poisson.solve(phi, rho)
        print(phi[4,5,6])
        rhos.append(rho)
    cupy.allclose(rhos[0], rhos[1], rtol=1e-10)
