import os
import gpaw.wannier90 as w90
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates

seed = 'Fe'

calc = GPAW('Fe.gpw', txt=None)

soc = soc_eigenstates(calc, return_wfs=True)
e_mk = soc['eigenvalues'].T
v_knm = soc['eigenstates']

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
                      v_knm=v_knm,
                      seed=seed)
w90.write_eigenvalues(calc, e_km=e_mk.T, seed=seed)
w90.write_overlaps(calc, v_knm=v_knm, seed=seed)

os.system('wannier90.x ' + seed)
