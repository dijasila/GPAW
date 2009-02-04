from ase import *
from gpaw import GPAW
from gpaw.vdw import FFTVDWFunctional

vdw = FFTVDWFunctional(verbose=1)
d = 3.9
x = d / sqrt(3)
L = 3.0 + 2 * 4.0
dimer = Atoms('Ar2', [(0, 0, 0), (x, x, x)], cell=(L, L, L))
dimer.center()
calc = GPAW(h=0.2, xc='revPBE')
dimer.set_calculator(calc)
e2 = dimer.get_potential_energy()
e2vdw = calc.get_xc_difference(vdw)
del dimer[1]
e = dimer.get_potential_energy()
evdw = calc.get_xc_difference(vdw)

E = 2 * e - e2
Evdw = E + 2 * evdw - e2vdw
print E, Evdw
assert abs(E - -0.0048) < 1e-4
assert abs(E - +0.0223) < 1e-4
