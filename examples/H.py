#!/usr/bin/env python
from ase import Atom, Atoms
from gpaw import GPAW

a = 5.0
H = Atoms([Atom('H',(a/2, a/2, a/2), magmom=1)],
           pbc=False,
           cell=(a, a, a))

H.set_calculator(GPAW(nbands=1, h=0.2, convergence={'eigenstates': 1e-3}))
e = H.get_potential_energy()
