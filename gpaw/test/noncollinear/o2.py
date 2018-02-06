from ase import Atoms
from gpaw import GPAW
a = Atoms('OO', [[0, 0, 0], [0, 0, 1.1]], magmoms=[1, 1], pbc=(1, 0, 0))
a.center(vacuum=2.5)
a.calc = GPAW(mode='pw',
              kpts=(2, 1, 1),
              symmetry='off',
              experimental={'magmoms': [[0, 0.5, 0.5], [0, 0, 1]]})
a.get_forces()
a.calc.write('o2.gpw')
a.calc.write('o2b.gpw', 'all')
GPAW('o2.gpw')
