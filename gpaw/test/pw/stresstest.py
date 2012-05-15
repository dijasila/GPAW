import numpy as np

from gpaw.test import equal
from gpaw.grid_descriptor import GridDescriptor
from gpaw.spline import Spline
import gpaw.mpi as mpi
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions.pw import PWDescriptor, PWLFC, ft
from gpaw.kpt_descriptor import KPointDescriptor


x = 2.0
rc = 3.5
r = np.linspace(0, rc, 100)

n = 40
a = 8.0
gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)

a_R = gd.empty()
z = np.linspace(0, a, n, endpoint=False)
a_R[:] = 2 + np.sin(2 * np.pi * z / a)

spos_ac = np.array([(0.15, 0.45, 0.95)])

pd = PWDescriptor(45, gd)
a_G = pd.fft(a_R)

s = Spline(0, rc, 2 * x**1.5 / np.pi * np.exp(-x * r**2))
p = Spline(1, rc, 2 * x**1.5 / np.pi * np.exp(-x * r**2))

lfc = PWLFC([[s, p]], pd)
lfc.set_positions(spos_ac)
b_LG = pd.zeros(4)
lfc.add(b_LG, {0: np.eye(4)})
e1 = pd.integrate(a_G, b_LG)
assert abs(lfc.integrate(a_G)[0] - e1).max() < 1e-11

s1 = -3 * e1 + lfc.stress_tensor_contribution(a_G)[0]

eps = 1e-6
a *= 1 + eps
gd = GridDescriptor((n, n, n), (a, a, a), comm=mpi.serial_comm)

pd = PWDescriptor(45, gd)
a_G /= (1 + eps)**3
lfc = PWLFC([[s, p]], pd)
lfc.set_positions(spos_ac)
b_LG = pd.zeros(4)
lfc.add(b_LG, {0: np.eye(4)})
e2 = pd.integrate(a_G, b_LG)
assert abs((e2-e1)/eps - s1).max() < 2e-5

