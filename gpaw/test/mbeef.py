from ase import *
from ase.dft.bee import BEEF_Ensemble
from gpaw import GPAW, PW
import numpy as np

xc = 'mBEEF'
pw = 600.
d = 1.09

# N2 molecule
n2 = Atoms('N2',[[0.,0.,0.],[0.,0.,d]])
n2.center(vacuum=3.)
cell = n2.get_cell()
calc = GPAW(xc=xc, mode=PW(pw))
n2.set_calculator(calc)
e_n2 = n2.get_potential_energy()
f = n2.get_forces()
ens = BEEF_Ensemble(n2)
de_n2 = ens.get_ensemble_energies()
del n2, calc, ens

# N atom
n = Atoms('N')
n.set_cell(cell)
n.center()
calc = GPAW(xc=xc, mode=PW(pw), hund=True)
n.set_calculator(calc)
e_n = n.get_potential_energy()
ens = BEEF_Ensemble(n)
de_n = ens.get_ensemble_energies()
del n, calc, ens

# forces
f0 = f[0].sum()
f1 = f[1].sum()
assert abs(f0 + f1) < 1.e-10
assert (f1 - 0.469) < 1.e-2

# binding energy
E_bind = 2*e_n - e_n2
dE_bind = 2*de_n[:] - de_n2[:]
dE_bind = np.std(dE_bind)
assert abs(E_bind - 9.70) < 1.e-2
assert abs(dE_bind - 0.41) < 1.e-2
