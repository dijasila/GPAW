from ase import Atoms
from gpaw import GPAW
a = Atoms('OO', [[0, 0, 0], [0, 0, 1.1]], magmoms=[1, 1])
a.center(vacuum=2.5)
a.calc = GPAW(mode='pw',
              experimental={'magmoms': [[0, 0.5, 0.5], [0, 0, 1]]})
a.get_forces()
