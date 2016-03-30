import pickle
import numpy as np

from ase.parallel import paropen
from ase.lattice import bulk

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0

import matplotlib.pyplot as plt

plt.figure(figsize=(6.5, 4.5))

ecuts = np.array([50, 100, 150, 200])
color = ['ro-', 'bo-', 'go-', 'ko-']
direct_gap = np.zeros(4)

for j, k in enumerate([3, 5, 7, 9]):
    for i, ecut in enumerate([50,100,150,200]):
        fil = pickle.load(paropen('Si-g0w0_GW_k%s_ecut%s_results.pckl' %(k, ecut), 'r'))
        direct_gap[i] = fil['qp'][0,0,1] - fil['qp'][0,0,0]
    plt.plot(ecuts, direct_gap, color[j], label='(%sx%sx%s) k-points' % (k, k, k))

plt.xlabel('Cutoff energy (eV)')
plt.ylabel('Direct band gap (eV)')
plt.title('non-selfconsistent G0W0@LDA')
plt.legend(loc='lower right')
plt.savefig('Si_GW.png')
plt.show()
