from ase.build import bulk
from ase.parallel import paropen
from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues
from gpaw import GPAW, PW

k = 4
atoms = bulk('C', 'diamond')
atoms.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                  mode='lcao', #PW(200),
                  xc='HSE06:backend=ri')
atoms.get_potential_energy()
