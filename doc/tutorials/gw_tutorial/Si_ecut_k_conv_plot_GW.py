import pickle
import numpy as np
from ase.parallel import paropen
import matplotlib.pyplot as plt

plt.figure(figsize=(6.5, 4.5))

ecuts = np.array([50, 100, 150, 200])
color = ['ro-', 'bo-', 'go-', 'ko-', 'co-', 'mo-', 'yo-']
direct_gap = np.zeros(4)

for j, k in enumerate([4, 6, 8, 10]):
    for i, ecut in enumerate([50, 100, 150, 200]):
        fil = pickle.load(paropen('Si-g0w0_k%s_ecut%s_results.pckl' %
                                  (k, ecut), 'r'))
        direct_gap[i] = fil['qp'][0, 0, 1] - fil['qp'][0, 0, 0]
    plt.plot(ecuts, direct_gap, color[j],
             label='(%sx%sx%s) k-points' % (k, k, k))

plt.xlabel('Cutoff energy (eV)')
plt.ylabel('Direct band gap (eV)')
plt.title('non-selfconsistent G0W0@LDA')
plt.legend(loc='lower right')
plt.savefig('Si_GW.png')
plt.show()
