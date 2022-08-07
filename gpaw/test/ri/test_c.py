from ase.build import bulk
from gpaw import GPAW

k = 4
atoms = bulk('C', 'diamond')
atoms.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                  mode='lcao',
                  xc='HSE06:backend=ri')
atoms.get_potential_energy()
