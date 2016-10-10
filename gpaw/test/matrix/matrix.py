import functools
import numpy as np
from gpaw.mpi import world
from gpaw.fd_operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.matrix import Matrix, UniformGridMatrix

gd = GridDescriptor([2, 3, 4], [2, 3, 4])
dt = complex
ph = np.ones((3, 2), complex)
T = functools.partial(Laplace(gd, -0.5, 1, dt).apply, phase_cd=ph)
N = 2
a = UniformGridMatrix(N, gd, dt, dist=(world, world.size))
a.a[:] = 1
a.A[0, 0, world.rank] = -1j
c = Matrix(N, N, dt, dist=(world, world.size))

c[:] = (a | a)
print(c.a)
c.cholesky()
c.inv()
print(c.a)
b = a.new()
b[:] = c.T * a
a[:] = b
c[:] = (a | a)
print(c.a)
a.apply(T, b)
c[:] = (a | b)
print('H:', c.a)
eps = np.empty(2)
c.eigh(eps)
print(eps, c.a)
d = a.new()
d[:] = c.T * a
d.apply(T, b)
c[:] = (d | b)
print('H2:', c.a)
