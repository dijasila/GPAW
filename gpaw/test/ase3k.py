from ase import Atom, Atoms
from ase.io import read
from gpaw import GPAW
from gpaw.test import equal
a = 2.0
calc = GPAW(gpts=(12, 12, 12), txt='H.txt')
H = Atoms([Atom('H')],
          cell=(a, a, a),
          pbc=True,
          calculator=calc)
e1 = H.get_potential_energy()
H = read('H.txt')
e2 = H.get_potential_energy()
assert abs(e1 - e2) < 1e-6
assert calc.get_xc_functional() == 'LDA'
