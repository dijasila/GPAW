from math import pi
from gpaw.grid_descriptor import RadialGridDescriptor, GridDescriptor
from gpaw.xc.functional import XC
import numpy as np
from gpaw.test import equal


for name in ['LDA', 'PBE']:
    r = 0.01 * np.arange(100)
    dr = 0.01 * np.ones(100)
    rgd = RadialGridDescriptor(r, dr)
    xc = XC(name)
    n = np.exp(-r**2)[np.newaxis]
    v = np.zeros((1, 100))
    E = xc.calculate_spherical(rgd, n, v)
    print E
    i = 23
    x = v[0, i] * rgd.dv_g[i]
    n[0, i] += 0.000001
    Ep = xc.calculate_spherical(rgd, n, v)
    n[0, i] -= 0.000002
    Em = xc.calculate_spherical(rgd, n, v)
    x2 = (Ep - Em) / 0.000002
    print i, x, x2, x - x2
    equal(x, x2, 1e-11)

    N = 20
    a = 1.0
    gd = GridDescriptor((N, N, N), (a, a, a))

    n = gd.empty(1)
    n.fill(0.02)
    n += 0.01 * np.sin(np.arange(gd.beg_c[2], gd.end_c[2]) * 2 * pi / N)
    v = 0.0 * n
    E = xc.calculate(gd, n, v)

    here = (gd.beg_c[0] <= 1 < gd.end_c[0] and
            gd.beg_c[1] <= 2 < gd.end_c[1] and
            gd.beg_c[2] <= 3 < gd.end_c[2])
    if here:
        x = v[0, 1, 2, 3] * gd.dv
        n[0, 1, 2, 3] += 0.000001
    Ep = xc.calculate(gd, n, v)
    if here:
        n[0, 1, 2, 3] -= 0.000002
    Em = xc.calculate(gd, n, v)
    x2 = (Ep - Em) / 0.000002
    if here:
        print x, x2, x - x2
        equal(x, x2, 1e-11)
