from ase import Atoms
from gpaw import GPAW
a = Atoms('H', cell=[2, 2, 2])
a.calc = GPAW(mode='pw', experimental={'magmoms': [[1, 0, 0]]})
a.get_potential_energy()
