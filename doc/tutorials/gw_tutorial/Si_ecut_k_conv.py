import pickle
import numpy as np

from ase.parallel import paropen
from ase.lattice import bulk

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.g0w0 import G0W0

direct_gap = np.zeros(4,4)

a = 5.431
atoms = bulk('Si', 'diamond', a=a)

for i, ecut in enumerate([50, 100, 150, 200]):
    for j, k in enumerate([3, 5, 7, 9]):
        calc = GPAW(mode=PW(350),                  
                    kpts={'size': (k, k, k), 'gamma': True},
                    dtype=complex,
                    xc='LDA',
                    occupations=FermiDirac(0.001),
                    txt='Si_groundstate.txt'
                    )

        atoms.set_calculator(calc)
        atoms.get_potential_energy()

        calc.diagonalize_full_hamiltonian()       
        calc.write('Si_groundstate.gpw','all')    

        gw = G0W0(calc='Si_groundstate.gpw',
                  nbands=100,                
                  bands=(0,5),               
                  ecut=ecut,                  
                  filename='Si-g0w0_k%s_ecut%s' %(k, ecut)
                  )

        gw.calculate_ks_xc_contribution()
        gw.calculate_exact_exchange()

        eps_skn = gw.eps_sin
        vxc_skn = gw.vxc_sin
        exx_skn = gw.exx_sin

        exx_gap = eps_skn - vxc_skn + exx_skn

        direct_gap[i,j] = exx_gap[0,0,-1] - exx_gap[0,0,-2]

pickle.dump(direct_gap, paropen('direct_gap.pckl'))

import matplotlib.pyplot as plt

plt.figure(1)
plt.figure(figsize=(6.5, 4.5))

ecuts = np.array([50, 100, 150, 200])

for j, k in enumerate([3, 5, 7, 9]):
    plt.plot(ecuts, direct_gap[:,j], 'o-', label='(%sx%sx%s) k-points' % (k, k, k))

plt.xlabel('$E_{\mathrm{cut}}$ (eV)')
plt.ylabel('Direct band gap (eV)')
plt.xlim([0., 250.])
plt.ylim([7.5, 10.])
plt.title('non-selfconsistent Hartree-Fock')
plt.legend(loc='upper right')
plt.savefig('Si_EXX_new.png')
