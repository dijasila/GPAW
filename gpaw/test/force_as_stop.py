from ase import Atoms
from gpaw import GPAW

H2 = Atoms('H2', positions=[(0, 0, 0), (1, 0, 0)])
H2.center(vacuum=2)
calc = GPAW(spinpol=True,
            convergence={'forces': 1e-6,
                         'density': 100,
                         'energy': 100,
                         'eigenstates': 100})
H2.set_calculator(calc)
H2.get_potential_energy()
assert 15 < calc.iter < 20
