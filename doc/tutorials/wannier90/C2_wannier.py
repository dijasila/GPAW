import os
import gpaw.wannier90 as w90
from gpaw import GPAW

seed = 'C2'

calc = GPAW(seed + '.gpw', txt=None)

w90.write_input(calc, orbitals_ai=[range(2), range(2)], bands=range(4))

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, orbitals_ai=[range(2), range(2)])
w90.write_eigenvalues(calc)
w90.write_overlaps(calc)

os.system('wannier90.x ' + seed)
