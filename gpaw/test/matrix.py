from gpaw.mpi import world
from gpaw.grid_descriptor import GridDescriptor
gd = GridDescriptor([2, 3, 4], [2, 3, 4])
N = 2
a = RealSpaceMatrix(N, gd, float, dist=(world, world.size))
a.data[:] = 1
a.data[0, 0, world.rank] = 0
c = Matrix(N, N, dist=(world, world.size))

def f(x, y):
    y.x[:] = np.dot([[2, 1], [1, 3.5]], x.x)

c[:] = (a | a)
print(c.data)
c.inverse_cholesky()
print(c.data)
b = a.new()
b[:] = c.T * a
a[:] = b
c[:] = (a | a)
print(c.data)
b[:] = f * a
c[:] = (a | b)
print(c)
eps = np.empty(2)
c.eigh(eps)
print(eps, c)
d = a.new()
d[:] = c.T * a
b[:] = f * d
c[:] = (d | b)
print(c)
