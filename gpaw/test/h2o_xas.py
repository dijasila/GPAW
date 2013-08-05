import os
import numpy as np
from math import pi, cos, sin
from ase import Atom, Atoms
from ase.parallel import rank, barrier
from gpaw import GPAW
from gpaw.xas import XAS
from gpaw.test import equal
from gpaw.atom.generator2 import generate

# Generate setup for oxygen with half a core-hole:
generate(['O', '--core-hole=1s,0.5', '-wt', 'hch'])

a = 5.0
d = 0.9575
t = pi / 180 * 104.51
H2O = Atoms([Atom('O', (0, 0, 0)),
             Atom('H', (d, 0, 0)),
             Atom('H', (d * cos(t), d * sin(t), 0))],
            cell=(a, a, a), pbc=False)
H2O.center()
calc = GPAW(nbands=10, h=0.2, setups={'O': './hch'})
H2O.set_calculator(calc)
e = H2O.get_potential_energy()
niter = calc.get_number_of_iterations()

import gpaw.mpi as mpi

if mpi.size == 1: #
    xas = XAS(calc)
    x, y = xas.get_spectra()
    e1_n = xas.eps_n
    de1 = e1_n[1] - e1_n[0]

calc.write('h2o-xas.gpw')

if mpi.size == 1:
    calc = GPAW('h2o-xas.gpw', txt=None)
    calc.initialize()
    xas = XAS(calc)
    x, y = xas.get_spectra()
    e2_n = xas.eps_n
    w_n = np.sum(xas.sigma_cn.real**2, axis=0)
    de2 = e2_n[1] - e2_n[0]

    equal(de2, 2.08, 0.005)
    equal(w_n[1] / w_n[0], 2.18, 0.01)
    assert de1 == de2

if 0:
    import pylab as p
    p.plot(x, y[0])
    p.plot(x, sum(y))
    p.show()
