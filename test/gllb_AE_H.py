#!/usr/bin/env python
from ase import *
from gpaw import Calculator

SS = Atoms( [Atom('H')], cell=(5,5,5), pbc=False)
SS.center()
calc = Calculator(h=0.1, xc='GLLB', setups='ae')
SS.set_calculator(calc)
SS.get_potential_energy()
