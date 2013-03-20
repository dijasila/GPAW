from ase import *
from ase.dft.bee import BEEF_Ensemble
from gpaw import GPAW, PW
from gpaw.test import equal
import numpy as np

xc = 'mBEEF'
conv = {'eigenstates':1.e-6, 'density':1.e-6, 'energy':1.e-6}
d = 1.09
tol1 = 1.e-9
tol2 = 1.e-1

# N2 molecule
n2 = Atoms('N2',[[0.,0.,0.],[0.,0.,d]])
n2.center(vacuum=3.)
cell = n2.get_cell()
calc = GPAW(xc='PBE', convergence=conv)
n2.set_calculator(calc)
n2.get_potential_energy()
n2.calc.set(xc=xc)
e_n2 = n2.get_potential_energy()
f = n2.get_forces()
ens = BEEF_Ensemble(n2)
de_n2 = ens.get_ensemble_energies()
del n2, calc, ens

# N atom
n = Atoms('N')
n.set_cell(cell)
n.center()
calc = GPAW(xc='PBE', convergence=conv, hund=True)
n.set_calculator(calc)
n.get_potential_energy()
n.calc.set(xc=xc)
e_n = n.get_potential_energy()
ens = BEEF_Ensemble(n)
de_n = ens.get_ensemble_energies()
del n, calc, ens

# forces
f0 = f[0].sum()
f1 = f[1].sum()
equal(f0, -f1, tol1)
equal(f0, -0.25, tol2)

# binding energy
E_bind = 2*e_n - e_n2
dE_bind = 2*de_n[:] - de_n2[:]
dE_bind = np.std(dE_bind)
equal(E_bind, 9.75, tol2)
equal(dE_bind, 0.42, tol2)
