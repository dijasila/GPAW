from ase import Atoms
from gpaw import GPAW
a = Atoms('H', cell=[2, 2, 2], magmoms=[1])
a.calc = GPAW(mode='pw', experimental={'magmoms': [[0, 0, 1]]})
a.get_potential_energy()
