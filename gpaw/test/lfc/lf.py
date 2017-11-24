from gpaw.test import equal
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
import gpaw.mpi as mpi
from gpaw.lfc import LocalizedFunctionsCollection as LFC

s = Spline(0, 1.0, [1.0, 0.5, 0.0])
n = 40
a = 8.0
gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)
c = LFC([[s], [s], [s]],
        [(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]],
        gd)
b = gd.zeros()
c.add(b)
x = gd.integrate(b)

gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)
c = LFC([[s], [s], [s]],
        [(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]],
        gd)
b = gd.zeros()
c.add(b)
y = gd.integrate(b)
equal(x, y, 1e-13)
