from math import sqrt

from ase import Atoms

from gpaw import GPAW
from gpaw.test import equal
from gpaw.xc.vdw import FFTVDWFunctional

vdw = FFTVDWFunctional('vdW-DF', verbose=1)
d = 3.9
x = d / sqrt(3)
L = 3.0 + 2 * 4.0
dimer = Atoms('Ar2', [(0, 0, 0), (x, x, x)], cell=(L, L, L))
dimer.center()
calc = GPAW(h=0.2, xc='revPBE')
dimer.set_calculator(calc)
e2 = dimer.get_potential_energy()
calc.write('Ar2.gpw')
e2vdw = calc.get_xc_difference(vdw)
e2vdwb = GPAW('Ar2.gpw').get_xc_difference(vdw)
print e2vdwb - e2vdw
assert abs(e2vdwb - e2vdw) < 1e-9
del dimer[1]
e = dimer.get_potential_energy()
evdw = calc.get_xc_difference(vdw)

E = 2 * e - e2
Evdw = E + 2 * evdw - e2vdw
equal(E, -0.005, 0.001)
equal(Evdw, 0.023, 0.001)
