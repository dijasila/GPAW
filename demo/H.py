from ASE import Atom, ListOfAtoms
from gridpaw import Calculator

a = 5.0
H = ListOfAtoms([Atom('H',(a/2, a/2, a/2), magmom=0)],
                periodic=0,
                cell=(a, a, a))
calc = Calculator(nbands=1, h=0.20, onohirose=5, tolerance=0.001, softgauss=0)
H.SetCalculator(calc)
print H.GetPotentialEnergy()

