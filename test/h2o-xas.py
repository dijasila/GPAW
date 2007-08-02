import os
from math import pi, cos, sin
from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities import center
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
from gpaw.corehole import xas, plot_xas

# Generate setup for oxygen with half a core-hole:
g = Generator('O', scalarrel=True, corehole=(1, 0, 0.5), nofiles=True)
g.run(name='hch1s', **parameters['O'])
setup_paths.insert(0, '.')

a = 5.0
d = 0.9575
t = pi / 180 * 104.51
H2O = ListOfAtoms([Atom('O', (0, 0, 0)),
                   Atom('H', (d, 0, 0)),
                   Atom('H', (d * cos(t), d * sin(t), 0))],
                  cell=(a, a, a), periodic=False)
center(H2O)
calc = Calculator(nbands=10, h=0.2, setups={'O': 'hch1s'})
H2O.SetCalculator(calc)
e = H2O.GetPotentialEnergy()

a, b = xas(calc)
e, w, x, y = plot_xas(a, b[0])

calc.write('h2o-xas.gpw', mode='No wave functions, please!')

calc = Calculator('h2o-xas.gpw', txt=None)
a, b = xas(calc)
e2, w, x, y = plot_xas(a, sum(b))

de = e[1] - e[0]
de2 = e2[1] - e2[0]

print de, de2
print e, w

assert de == de2
assert abs(de - 2.052) < 0.001
assert abs(w[1] / w[0] - 2.19) < 0.01

os.remove('h2o-xas.gpw')
os.remove('O.hch1s.LDA')
del setup_paths[0]
