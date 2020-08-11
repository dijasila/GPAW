import numpy as np
from gpaw import GPAW
from gpaw.spinorbit import get_anisotropy

theta = np.linspace(0, np.pi, 21)
calc = GPAW('gs_Co.gpw', txt=None)
E_so = [get_anisotropy(calc, theta=t, phi=0.0) for t in theta]
dE = E_so[11] - E_so[0]
print(dE, E_so)
assert abs(dE - 60e-6) < 1e-6
with open('anisotropy.dat', 'w') as fd:
    for t, e in zip(theta, E_so):
        print(t, e, file=fd)
