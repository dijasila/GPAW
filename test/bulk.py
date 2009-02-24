from ase import *
from gpaw import GPAW
bulk = Atoms([Atom('Li')], pbc=True)
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2)
bulk.set_calculator(calc)
e = []
A = [2.6, 2.65, 2.7, 2.75, 2.8]
for a in A:
    bulk.set_cell((a, a, a))
    e.append(bulk.get_potential_energy())
print e

import numpy as np
a = np.roots(np.polyder(np.polyfit(A, e, 2), 1))[0]
print 'a =', a
assert abs(a - 2.6430) < 0.0001
