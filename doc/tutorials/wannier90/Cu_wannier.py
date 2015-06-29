import os 
import gpaw.wannier90 as w90
from gpaw import GPAW

seed = 'Cu'

calc = GPAW(seed + '.gpw', txt=None)

w90.write_input(calc, orbitals_ai=[[0, 1] + range(4, 9)])

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, orbitals_ai=[[0, 1] + range(4, 9)])
w90.write_eigenvalues(calc)
w90.write_overlaps(calc)

os.system('wannier90.x ' + seed)
