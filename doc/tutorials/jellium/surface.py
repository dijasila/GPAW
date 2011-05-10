from math import pi
from ase import Atoms
from ase.units import Bohr
from gpaw.jellium import JelliumSurfacePoissonSolver
from gpaw import GPAW

rs = 5.0    # Wigner-Seitz radius
h = 0.2     # grid-spacing
a = 8 * h
v = 3 * a   # vacuum
L = 10 * a  # thickness
k = 12      # number of k-points (k*k*1)

ps = JelliumSurfacePoissonSolver(z1=v, z2=v + L)
ne = a**2 * L / (4 * pi / 3 * (rs * Bohr)**3)
surf = Atoms(pbc=(True, True, False),
             cell=(a, a, v + L + v))
surf.calc = GPAW(poissonsolver=ps,
                 xc='LDA_X+LDA_C_WIGNER',
                 eigensolver='cg',
                 charge=-ne,
                 kpts=[k, k, 1],
                 h=h,
                 nbands=int(ne / 2) + 5,
                 txt='surface.txt')
e = surf.get_potential_energy()
surf.calc.write('surface.gpw')
