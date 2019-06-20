import numpy as np
from ase import Atoms
from gpaw import GPAW, PW
a = 4.0
h2 = Atoms('H2', [[0, 0, 0], [0, 0, 0.8]],
           cell=[a, a, a], pbc=1)
E = np.linspace(100, 1000, 10)
E = [200]
A = []
for ec in E:
    h2.calc = GPAW(mode=PW(ec), txt='sc.txt')
    e = h2.get_potential_energy()
    A.append(e)
b = a / 2**0.5
h2 = Atoms('H2', [[0, 0, 0], [0, 0, 0.8]],
           cell=[[a, 0, 0], [a, a, 0], [0, 0, a]], pbc=1)
B = []
for ec in E:
    h2.calc = GPAW(mode=PW(ec), txt='fcc.txt')
    e = h2.get_potential_energy()
    B.append(e)

import matplotlib.pyplot as plt
plt.plot(E, A)
plt.plot(E, B, label='fcc')
plt.legend()
plt.show()
