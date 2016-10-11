import functools
import numpy as np
from gpaw.mpi import world
from gpaw.fd_operators import Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.matrix import Matrix, UniformGridMatrix, ProjectionMatrix

gd = GridDescriptor([2, 3, 4], [2, 3, 4])
dt = complex
ph = np.ones((3, 2), complex)
T = functools.partial(Laplace(gd, -0.5, 1, dt).apply, phase_cd=ph)
N = 2
a = UniformGridMatrix(N, gd, dt, dist=(world, world.size))
a.a[:] = 1
a.array[0, 0, world.rank] = -1j
c = Matrix(N, N, dt, dist=(world, world.size))
p = {0: np.arange(10).reshape((2, 5)) * 0.1,
     1: np.arange(10).reshape((2, 5)) * 0.2}
P = ProjectionMatrix(N, world, float, p, ())
P2 = P.new()
P2.a[:]=1
m = Matrix(N, N, float, dist=(world, world.size))
m.a[:]=0
m += P.C * P2.T
print(m.a)
adflkjh
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
