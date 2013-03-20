from ase import *
from ase.dft.bee import BEEF_Ensemble
from gpaw import GPAW, PW
from gpaw.test import equal
import numpy as np

xc = 'mBEEF'
pw = 600.
d = 1.09
tol1 = 1.e-10
tol2 = 1.e-2
tol3 = 1.e-1

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
equal(f0, -f1, tol1)
equal(f0, -0.469, tol2)

# binding energy
E_bind = 2*e_n - e_n2
dE_bind = 2*de_n[:] - de_n2[:]
dE_bind = np.std(dE_bind)
equal(E_bind, 9.700, tol2)
equal(dE_bind, 0.42, tol3)
