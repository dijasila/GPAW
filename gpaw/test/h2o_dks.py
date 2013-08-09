from ase.structure import molecule
from ase.parallel import rank, barrier

from gpaw import GPAW, FermiDirac
from gpaw.test import equal
from gpaw.atom.generator2 import generate

# Generate setup for oxygen with a core-hole:
generate(['O', '--core-hole=1s,1', '-wt', 'fch', '-f', 'PBE'])

atoms = molecule('H2O')
atoms.center(vacuum=2.5)

calc = GPAW(xc='PBE')
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy() + calc.get_reference_energy()
niter1 = calc.get_number_of_iterations()

atoms[0].magmom = 1
calc.set(charge=-1,
         setups={'O': './fch'},
         occupations=FermiDirac(0.0, fixmagmom=True))
e2 = atoms.get_potential_energy() + calc.get_reference_energy()

atoms[0].magmom = 0
calc.set(charge=0,
         setups={'O': './fch'},
         occupations=FermiDirac(0.0, fixmagmom=True),
         spinpol=True)
e3 = atoms.get_potential_energy() + calc.get_reference_energy()

print 'Energy difference %.3f eV' % (e2 - e1)
print 'XPS %.3f eV' % (e3 - e1)

print e2 - e1
print e3 - e1
equal(e2 - e1, 534.4, 0.5)
equal(e3 - e2, 5.50, 0.05)
