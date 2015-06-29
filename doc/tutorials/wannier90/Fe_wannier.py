import os 
import gpaw.wannier90 as w90
from gpaw import GPAW

spin = 1
assert spin in [0, 1]

if spin == 0:
    seed = 'Fe_up'
else:
    seed = 'Fe_down'
    
calc = GPAW('Fe.gpw', txt=None)

w90.write_input(calc, seed=seed)

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, seed=seed, spin=spin)
w90.write_eigenvalues(calc, seed=seed, spin=spin)
w90.write_overlaps(calc, seed=seed, spin=spin)

os.system('wannier90.x ' + seed)
