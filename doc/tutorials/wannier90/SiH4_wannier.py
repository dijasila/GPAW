import os 
import gpaw.wannier90 as w90
from gpaw import GPAW

seed = 'SiH4'

calc = GPAW(seed + '.gpw', txt=None)

w90.write_input(calc, seed=seed, orbitals_ai=[range(4), [], [], [], []])

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, seed=seed, orbitals_ai=[range(4), [], [], [], []])
w90.write_eigenvalues(calc, seed=seed)
w90.write_overlaps(calc, seed=seed)

os.system('wannier90.x ' + seed)
