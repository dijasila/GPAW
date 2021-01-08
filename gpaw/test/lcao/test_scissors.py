from ase import Atoms
from gpaw import GPAW
from gpaw.lcao.scissors import Scissors

h2 = Atoms('2H2', [[0, 0, 0], [0, 0, 0.74],
                   [4, 0, 0], [4, 0, 0.74]])
h2.center(vacuum=3.0)
h2.calc = GPAW(mode='lcao',
               basis='sz(dzp)',
               eigensolver=Scissors([(-1, 1, 2)]),
               txt='2h2.txt')
h2.get_potential_energy()
