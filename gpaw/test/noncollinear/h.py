from ase import Atoms
from gpaw import GPAW
a = Atoms('H', cell=[2, 2, 2], magmoms=[1], pbc=(1, 0, 0))
a.calc = GPAW(mode='pw',
              kpts=(4, 1, 1),
              symmetry={'do_not_symmetrize_the_density': True},
              experimental={'magmoms': [[0.5, 0.5, 0]]}
              #experimental={'magmoms': [[1, 0, 0]]}
              )
a.get_potential_energy()
a.calc.write('h.gpw', mode='all')
