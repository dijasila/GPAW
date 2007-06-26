# This test takes approximately 0.4 seconds
from ASE import Atom, ListOfAtoms
from ASE.Calculators.CheckForce import CheckForce
from gpaw.utilities import equal
from gpaw import Calculator

a = 6.0
atoms = ListOfAtoms([Atom('C', [a / 2 + 0.0234,
                                a / 2 + 0.0345,
                                a / 2 + 0.0456], magmom=2)],
                    cell=(a, a, a))
calc = Calculator(nbands=6, h=0.2, hund=0, tolerance=1e-11)
atoms.SetCalculator(calc)
for i in range(3):
    f1, f2 = CheckForce(atoms, 0, i, 0.012)
    equal(f1, f2, 0.0022)
