import pickle
import numpy as np

from ase.parallel import paropen
from ase.lattice import bulk

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0

direct_gap = pickle.load(paropen('direct_gap_GW.pckl', 'r'))

import matplotlib.pyplot as plt

plt.figure(1)
plt.figure(figsize=(6.5, 4.5))

ecuts = np.array([50, 100, 150, 200])

for j, k in enumerate([3, 5, 7, 9]):
    print direct_gap[:,j]
    plt.plot(ecuts, direct_gap[:,j], 'o-', label='(%sx%sx%s) k-points' % (k, k, k))

plt.xlabel('$E_{\mathrm{cut}}$ (eV)')
plt.ylabel('Direct band gap (eV)')
#plt.xlim([0., 250.])
#plt.ylim([7.5, 10.])
plt.title('non-selfconsistent G0W0')
plt.legend(loc='upper right')
plt.savefig('Si_GW_new.png')
