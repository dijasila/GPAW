import numpy as np
from ase.build import mx2
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
a = mx2('MoS2')
a.center(vacuum=3, axis=2)
a.calc = GPAW(mode='pw',
              experimental={'magmoms': np.zeros((3, 3)), 'soc': True},
              kpts={'size': (6, 6, 1), 'gamma': True})
a.get_potential_energy()
e1 = a.calc.get_eigenvalues(kpt=4)

a.calc = GPAW(mode='pw',
              kpts={'size': (6, 6, 1), 'gamma': True})
a.get_potential_energy()
e2 = get_spinorbit_eigenvalues(a.calc)[:, 4]
print(e1[24:28])
print(e2[24:28])
