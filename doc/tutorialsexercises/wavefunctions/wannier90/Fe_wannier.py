import os
import gpaw.wannier90 as w90
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates

seed = 'Fe'

calc = GPAW('Fe.gpw')
soc = soc_eigenstates(calc)

w90.write_input(calc,
                bands=range(30),
                spinors=True,
                num_iter=200,
                dis_num_iter=500,
                dis_mix_ratio=1.0,
                dis_froz_max=15.0,
                seed=seed)

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc,
                      seed=seed, soc=soc)
w90.write_eigenvalues(calc, seed=seed, soc=soc)
w90.write_overlaps(calc, seed=seed, soc=soc)

os.system('wannier90.x ' + seed)
