import numpy as np
from ase.lattice import bulk
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0
import pickle

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

calc = GPAW(
            mode=PW(200),
            kpts=(3,3,3),
            xc='LDA',
            eigensolver='rmm-diis',
            occupations=FermiDirac(0.001),
#            parallel={'band': 1},
            txt='Si_groundstate_freq.txt'
           )

#atoms.set_calculator(calc)
#atoms.get_potential_energy()

#calc.diagonalize_full_hamiltonian()
#calc.write('Si_groundstate_freq.gpw','all')

data = np.zeros((6,6))

for i, domega0 in enumerate([0.005, 0.01, 0.02, 0.03, 0.04, 0.05]):
    for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
        gw = G0W0(calc='Si_groundstate_freq.gpw',
                  nbands=30,
                  bands=(3,5),
                  kpts=[0],
                  ecut=20,
                  filename='Si_g0w0_domega0_%s_omega2_%s' % (domega0, omega2)
                  )

        results = gw.calculate()

        data[i,j] = results['qp'][0,0,1] - results['qp'][0,0,0]


pickle.dump(data, paropen('frequency_data.pckl', 'w'))

#import matplotlib.pyplot as plt

#plt.figure(1)
#plt.figure(figsize=(6.5, 4.5))

#omega2 = np.array([1, 5, 10, 15, 20, 25])
#color = ['ro-', 'go-', 'bo-', 'ko-', 'mo-', 'co-']

#for j, k in enumerate([0.005, 0.01, 0.02, 0.03, 0.04, 0.05]):
#    plt.plot(omega2, direct_gap[j,:], color[j], label='domega0 = %s' % (k))

#plt.xlabel('omega2 (eV)')
#plt.ylabel('Direct band gap (eV)')
##plt.xlim([0., 250.])
##plt.ylim([7.5, 10.])
#plt.title('$G_0W_0$@LDA')
#plt.legend(loc='upper left')
#plt.savefig('Si_freq_new.png')
