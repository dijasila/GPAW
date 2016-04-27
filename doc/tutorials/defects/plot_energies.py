# creates: energies.png
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

indata = np.loadtxt('energy_differences.dat')

Nat = indata[:, 0]
Ediff = indata[:, 1]
Ediffcor = indata[:, 2]
plt.plot(Nat**(-1 / 3), Ediff, 'o', label='No corrections')
plt.plot(Nat**(-1 / 3), Ediffcor, 'p', label='FNV corrected')

plt.xlabel('$N_\mathrm{at}^{-1/3}$', fontsize=18)
plt.ylabel('Energy differences (eV)', fontsize=18)
plt.xlim(xmin=0.06)
plt.legend(loc='lower left')
plt.savefig('energies.png')
