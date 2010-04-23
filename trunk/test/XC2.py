from math import pi
from gpaw.grid_descriptor import RadialGridDescriptor, GridDescriptor
from gpaw.domain import Domain
from gpaw.xc_functional import XC3DGrid, XCRadialGrid
import numpy as npy
from gpaw.utilities import equal


for name in ['LDA', 'PBE']:
    r = 0.01 * npy.arange(100)
    dr = 0.01 * npy.ones(100)
    rgd = RadialGridDescriptor(r, dr)
    xc = XCRadialGrid(name, rgd)
    n = npy.exp(-r**2)
    v = npy.zeros(100)
    E = xc.get_energy_and_potential(n, v)
    print E
    n2 = 1.0 * n
    i = 23
    n2[i] += 0.000001
    x = v[i] * rgd.dv_g[i]
    E2 = xc.get_energy_and_potential(n2, v)
    x2 = (E2 - E) / 0.000001
    print i, x, x2, x - x2
    equal(x, x2, 2e-8)

    N = 20
    a = 1.0
    gd = GridDescriptor(Domain((a, a, a)), (N, N, N))
    xc = XC3DGrid(name, gd)
    n = 0.02 * npy.ones((N, N, N))
    n += 0.01 * npy.sin(npy.arange(N) * 2 * pi / N)
    v = 0.0 * n
    E = xc.get_energy_and_potential(n, v)

    n2 = 1.0 * n
    i = 17
    n2[i, i, i] += 0.000001
    x = v[i, i, i] * gd.dv
    E2 = xc.get_energy_and_potential(n2, v)
    x2 = (E2 - E) / 0.000001
    print i, x, x2, x - x2
    equal(x, x2, 2e-8)
