import os 
import gpaw.wannier90 as w90
from gpaw import GPAW

seed = 'BaTiO3'

calc = GPAW(seed + '.gpw', txt=None)

orbitals_ai = [[], [], range(1,4), range(1,4), range(1,4)]
w90.write_input(calc, seed=seed, orbitals_ai=orbitals_ai, bands=range(11, 20))

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, seed=seed, orbitals_ai=orbitals_ai)
w90.write_eigenvalues(calc, seed=seed)
w90.write_overlaps(calc, seed=seed)

os.system('wannier90.x ' + seed)
