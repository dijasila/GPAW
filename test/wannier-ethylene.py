import os
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import equal, center
from gpaw.wannier import Wannier
import numpy as npy

# GPAW wannier example for ethylene corresponding to the ASE Wannier
# tutorial.

if not os.path.isfile('ethylene.gpw'):
    a = 6.0  # Size of unit cell (Angstrom)

    ethylene = ListOfAtoms([
                       Atom('H', (-1.235,-0.936 , 0 )),
                       Atom('H', (-1.235, 0.936 , 0 )),
                       Atom('C', (-0.660, 0.000 , 0 )),
                       Atom('C', ( 0.660, 0.000 , 0 )),
                       Atom('H', ( 1.235,-0.936 , 0 )),
                       Atom('H', ( 1.235, 0.936 , 0 ))],
                       cell=(a, a, a), periodic=True)
    center(ethylene)
    calc = Calculator(nbands=8, h=0.20, convergence={'eigenstates': 1e-6})
    ethylene.SetCalculator(calc)
    ethylene.GetPotentialEnergy()
    calc.write('ethylene.gpw', 'all')
else:
    calc = Calculator('ethylene.gpw', txt=None)

wannier = Wannier(numberofwannier=6,
                  calculator=calc,
                  numberoffixedstates=[6])
wannier.Localize(tolerance=1e-5)

centers = wannier.GetCenters()
print centers
expected = [[1.950, 2.376, 3.000],
            [1.950, 3.624, 3.000],
            [3.000, 3.000, 2.671],
            [3.000, 3.000, 3.329],
            [4.050, 2.376, 3.000],
            [4.050, 3.624, 3.000]]
equal(13.7995, wannier.GetFunctionalValue(), 0.016)
for center in centers:
    i = 0
    while npy.sum((expected[i] - center['pos'])**2) > 0.01:
        i += 1
        if i == len(expected):
            raise RuntimeError, 'Correct center not found'
    expected.pop(i)    

os.remove('ethylene.gpw')

## for i in range(6):
##     wannier.WriteCube(i, 'ethylene%s.cube' % i, real=True)

## from ASE.Visualization.PrimiPlotter import PrimiPlotter, X11Window
## ethylene.extend(wannier.GetCentersAsAtoms())
## plot = PrimiPlotter(ethylene)
## plot.SetOutput(X11Window())
## plot.SetRadii(.2)
## plot.SetRotation([15, 0, 0])
## plot.Plot()
