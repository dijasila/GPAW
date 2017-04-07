from __future__ import print_function
from ase import Atoms
from gpaw import GPAW, Davidson
from gpaw.test import equal

s = Atoms('Cl')
s.center(vacuum=3)
c = GPAW(xc='PBE', nbands=-4, charge=-1, h=0.3,
         eigensolver=Davidson(4))
s.set_calculator(c)

e = s.get_potential_energy()
niter = c.get_number_of_iterations()

print(e, niter)
energy_tolerance = 0.0003
niter_tolerance = 0
equal(e, -2.88774625225, energy_tolerance)
