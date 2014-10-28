from ase import Atoms
from gpaw import GPAW
from gpaw import PW

H2 = Atoms('H2', positions=[(0, 0, 0), (1, 0, 0)])
H2.center(vacuum=1)
calc = GPAW(convergence={'forces': 1e-4,
                         'density': 100,
                         'energy': 100,
                         'eigenstates': 100})
H2.set_calculator(calc)
H2.get_potential_energy()
assert 17 < calc.iter < 23
