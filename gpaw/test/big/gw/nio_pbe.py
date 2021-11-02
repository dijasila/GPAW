"""NiO in an antiferromagnetic configuration."""
from ase import Atoms
from gpaw import GPAW

a = 4.1767  # lattice constants
b = a / 2
m = 2.0

atoms = Atoms('Ni2O2',
              pbc=True,
              cell=[(b, b, 0),
                    (0, b, b),
                    (a, 0, a)],
              positions=[(0, 0, 0),
                         (b, 0, b),
                         (b, 0, 0),
                         (a, 0, b)],
              magmoms=[m, -m, 0, 0])

atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 600},
                  txt='nio.txt',
                  kpts={'density': 3.0, 'gamma': True},
                  xc='PBE')
e = atoms.get_potential_energy()
atoms.write('nio.json')

calc = atoms.calc.fixed_density(nbands=100)
calc.write('nio.gpw', mode='all')
