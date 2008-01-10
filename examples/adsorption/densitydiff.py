from ase import *
from gpaw import Calculator

calc = Calculator('relax.gpw')
HAl_density = calc.get_pseudo_density()

atoms = calc.get_atoms()
HAl = atoms.copy()

H = atoms.pop()
calc.set(txt='Al.txt')
atoms.get_potential_energy()
Al_density = calc.get_pseudo_density()

atoms += H
del atoms[:-1]
calc.set(txt='H.txt', nbands=1)
atoms.get_potential_energy()
Al_density = calc.get_pseudo_density()

diff = HAl_density - H_density - Al_density
write('diff.cube', HAl, data=diff)
write('diff.plt', HAl, data=diff)
