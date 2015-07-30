import os 
import gpaw.wannier90 as w90
from gpaw import GPAW

seed = 'Si2'

calc = GPAW(seed + '.gpw', txt=None)

w90.write_input(calc)

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, bands=range(12))
w90.write_eigenvalues(calc)
w90.write_overlaps(calc)

os.system('wannier90.x ' + seed)
