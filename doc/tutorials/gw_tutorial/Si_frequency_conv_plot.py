import numpy as np
from ase.lattice import bulk
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0
import pickle

import matplotlib.pyplot as plt

#plt.figure(1)
#plt.figure(figsize=(6.5, 4.5))
f = plt.figure()
xomega2 = np.array([1, 5, 10, 15, 20, 25])
color = ['ro-', 'go-', 'bo-', 'ko-', 'mo-', 'co-']
data = np.zeros((6,6))
for i, domega0 in enumerate([0.005, 0.01, 0.02, 0.03, 0.04, 0.05]):
    for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
        filename = 'Si_g0w0_domega0_%s_omega2_%s_results.pckl' % (domega0, omega2)
        results = pickle.load(open(filename,'r'))

        data[i,j] = results['qp'][0,0,1] - results['qp'][0,0,0]

print data

for j, k in enumerate([0.005, 0.01, 0.02, 0.03, 0.04, 0.05]):
    plt.plot(xomega2, data[j,:], color[j], label='domega0 = %s' % (k))

plt.xlabel('omega2 (eV)')
plt.ylabel('Direct band gap (eV)')
#plt.xlim([0., 250.])
#plt.ylim([7.5, 10.])
plt.title('$G_0W_0$@LDA')
plt.legend(loc='upper left')
#plt.savefig('Si_freq_new.png')
plt.show()
