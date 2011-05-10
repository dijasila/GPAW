from math import pi

import numpy as np
from math import pi
from ase import Atoms
from ase.units import Bohr
from gpaw.jellium import JelliumPoissonSolver
from gpaw import GPAW

rs = 5.0    # Wigner-Seitz radius
h = 0.2     # grid-spacing
a = 8 * h
k = 12      # number of k-points (k*k*k)

ne = a**3 / (4 * pi / 3 * (rs * Bohr)**3)

bulk = Atoms(pbc=True, cell=(a, a, a))
bulk.calc = GPAW(poissonsolver=JelliumPoissonSolver(),
                 xc='LDA_X+LDA_C_WIGNER',
                 charge=-ne,
                 nbands=5,
                 kpts=[k, k, k],
                 h=h,
                 txt='bulk.txt')
e0 = bulk.get_potential_energy()
