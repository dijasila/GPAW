from math import sqrt
from gpaw.poisson import PoissonSolver
import numpy as npy
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain

L = 2.87 / 0.529177
def f(n):
    N = 2 * n
    domain = Domain((L, L, L))
    gd = GridDescriptor(domain, (N, N, N))
    a = gd.zeros()
    p = PoissonSolver(nn=1, relax='J')
    p.initialize(gd)
    cut = N / 2.0 * 0.9
    C = N // 2
    for x in range(N):
        for y in range(N):
            for z in range(N):
                r = sqrt((x-C)**2 + (y-C)**2 + (z-C)**2) / cut
                if r < 1:
                    a[x, y, z] = 1 - (3 - 2 * r) * r**2
    for x in range(1-C, C+1):
        for y in range(1-C, C+1):
            for z in range(1-C, C+1):
                r = sqrt(x**2 + y**2 + z**2) / cut
                if r < 1:
                    a[x, y, z] = 1 - (3 - 2 * r) * r**2

    #print max(npy.fabs((a[:C,:C,:C]-a[C:,C:,C:]).ravel()))
    I0 = gd.integrate(a)
    a -= gd.integrate(a) / L**3

    I = gd.integrate(a)
    b = gd.zeros()
    np = p.solve(b, a)#, eps=1e-20)
    return b[0,0,0]-b[C,C,C]

assert f(8) == 0
