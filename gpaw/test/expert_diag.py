from __future__ import print_function

from time import time
from ase.lattice import bulk
from gpaw import GPAW, PW

import numpy as np

from gpaw.response.chi0 import Chi0
from gpaw.test import equal

# This test is asserting whether the expert diagonalization
# routine gives the same results as the non-expert version
# in terms of eigenvalues and wavefunctions

strings = []
wfs_e = []

for i, expert in enumerate([True, False]):
    si = bulk('Si')
    name = 'si_{0:d}'.format(i)
    si.center()
    calc = GPAW(mode=PW(400), kpts=(4, 4, 4), txt=name + '.txt')
    si.set_calculator(calc)
    si.get_potential_energy()
    t1 = time()
    # Choosing 51 as bands in order not to cut over a degenerate band
    calc.diagonalize_full_hamiltonian(expert=expert, nbands=51)
    t2 = time() - t1
    strings.append('expert={0}, time={1:.2f}sec'.format(str(expert), t2))

    string = name + '.gpw'
    calc.write(string, 'all')

    wfs_e.append(calc.wfs)

while len(strings) > 1:
    string = strings.pop()
    wfs = wfs_e.pop()
    print(string)
    for stringtmp, wfstmp in zip(strings, wfs_e):
        print(stringtmp)
        for kpt, kpttmp in zip(wfs.kpt_u, wfstmp.kpt_u):
            for m, (psi_G, eps) in enumerate(zip(kpt.psit_nG, kpt.eps_n)):
                # Have to do like this if bands are degenerate
                booleanarray = np.abs(kpttmp.eps_n - eps) < 1e-10
                inds = np.argwhere(booleanarray)
                count = len(inds)
                assert count > 0, 'Difference between eigenvalues!'

                psitmp_nG = kpttmp.psit_nG[inds][:, 0, :]
                fidelity = 0
                for psitmp_G in psitmp_nG:
                    fidelity += (np.abs(np.dot(psitmp_G.conj(), psi_G))**2 /
                                 np.dot(psitmp_G.conj(), psitmp_G) /
                                 np.dot(psi_G.conj(), psi_G))

                equal(fidelity, 1, 1e-10, 'Difference between wfs!')
