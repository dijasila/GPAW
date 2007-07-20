from ASE import Crystal, Atom
from gpaw import Calculator
from gpaw.utilities import equal


a = 4.0
n = 16
hydrogen = Crystal([Atom('H', (0.0, 0.0, 0.0))], cell=(a, a, a))
calc = Calculator(gpts=(n, n, n), nbands=1, tolerance=1e-10)
hydrogen.SetCalculator(calc)
e1 = hydrogen.GetPotentialEnergy()
calc.Set(spinpol=True)
hydrogen[0].SetMagneticMoment(1.0)
e2 = hydrogen.GetPotentialEnergy()
de = e1 - e2
print de
equal(de, 0.7929, 1e-3)
