#!/usr/bin/env python
from ase import *
from gpaw import GPAW

a = 10
b = a / 2
mol = Atoms([Atom('O',(b, b, 0.1219 + b)),
             Atom('H',(b, 0.7633 + b, -0.4876 + b)),
             Atom('H',(b, -0.7633 + b, -0.4876 + b))],
            pbc=False, cell=[a, a, a])
calc = GPAW(nbands=4, h=0.2, mode='lcao')
mol.set_calculator(calc)
e = mol.get_potential_energy()
