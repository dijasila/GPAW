from ase import *
from gpaw import GPAW
from ase.dft.bee import BEEFEnsemble
xc = 'BEEF-vdW'
h2 = Atoms('H2',[[0.,0.,0.],[0.,0.,0.75]])
h2.center(vacuum=3)
cell = h2.get_cell()
calc = GPAW(xc=xc)
h2.set_calculator(calc)
e_h2 = h2.get_potential_energy()
ens = BEEFEnsemble(calc)
de_h2 = ens.get_ensemble_energies()
del h2, calc, ens
h = Atoms('H')
h.set_cell(cell)
h.center()
calc = GPAW(xc=xc)
h.set_calculator(calc)
e_h = h.get_potential_energy()
ens = BEEFEnsemble(calc)
de_h = ens.get_ensemble_energies()
E_bind = 2*e_h - e_h2
dE_bind = 2*de_h[:] - de_h2[:]
dE_bind = dE_bind.std()
