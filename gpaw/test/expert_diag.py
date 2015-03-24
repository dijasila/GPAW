from __future__ import print_function

import numpy as np

from ase.lattice import bulk
from gpaw import GPAW, PW
from gpaw.test import equal
from gpaw.mpi import rank

# This test is asserting that the expert diagonalization
# routine gives the same result as the non-expert version
# in terms of eigenvalues and wavefunctions

wfs_e = []
for i, expert in enumerate([True, False]):
    si = bulk('Si')
    name = 'si_{0:d}'.format(i)
    si.center()
    calc = GPAW(mode=PW(400), kpts=(1, 1, 8),
                symmetry='off', txt=name + '.txt')
    si.set_calculator(calc)
    si.get_potential_energy()
    calc.diagonalize_full_hamiltonian(expert=expert, nbands=50)
    string = name + '.gpw'
    calc.write(string, 'all')
    wfs_e.append(calc.wfs)

# Test against values from rev #12394
wfsold_G = np.array([-7.03242443e+01 - 4.00102028e-13j,
                     -1.37801953e+02 - 9.82133133e-14j,
                     9.63846708e+00 - 1.45030653e-14j,
                     6.22404541e-01 + 8.10551006e-15j,
                     1.09715123e-02 - 8.54339549e-15j])
epsn_n = np.array([-0.15883479, -0.04404914, 0.16348702,
                   0.16348703, 0.25032194])


kpt_u = wfs_e[0].kpt_u
for kpt in kpt_u:
    if kpt.k == 0:
        assert np.allclose(epsn_n, kpt.eps_n[0:5], 1e-5), \
            'Eigenvalues have changed since rev. #12394'
        assert np.allclose(wfsold_G, kpt.psit_nG[1, 0:5], 1e-5), \
            'Wavefunctions have changed rev. #12394'

# Check that expert={True, False} give
# same result
while len(wfs_e) > 1:
    wfs = wfs_e.pop()
    for wfstmp in wfs_e:
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
