# Refer to Chulkov and Echenique, PRB 67, 245402 (2003) for comparison of
# results
import numpy as np
import sys

from ase.visualize import view
from ase.lattice.surface import hcp0001
from gpaw import GPAW
from gpaw.mpi import rank
from gpaw.utilities import devnull
from gpaw.response.df import DF
from gpaw.test import findpeak, equal

if rank != 0:
    sys.stdout = devnull

GS = 1
EELS = 1
check = 1

nband = 30

if GS:
    kpts = (64, 64, 1)
    atoms = hcp0001('Be', size=(1, 1, 1))
    atoms.cell[2][2] = (21.2)
    atoms.set_pbc(True)
    atoms.center(axis=2)
    view(atoms)

    calc = GPAW(
                gpts=(12, 12, 108),
                xc='LDA',
                txt='be.txt',
                kpts=kpts,
                basis='dzp',
                nbands=nband + 5,
                parallel={'domain': 1,
                          'band': 1},
                convergence={'bands': nband},
                eigensolver='cg',
                width=0.1)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()

if EELS:
    for i in range(1, 2):
        w = np.linspace(0, 15, 301)
        q = np.array([-i / 64.0, i / 64.0, 0.0])  # Gamma - K
        ecut = 40 + i * 10
        df = DF(calc=calc, q=q, w=w, eta=0.05, ecut=ecut,
                      txt='df_' + str(i) + '.out')
        df.get_surface_response_function(z0=21.2 / 2, filename='be_EELS')
        df.get_EELS_spectrum()
        df.check_sum_rule()
        df.write('df_' + str(i) + '.pckl')

if check:
    d = np.loadtxt('be_EELS')
    dw = d[1, 0]
    wpeak1, e1 = findpeak(d[:, 1], dw)
    print(wpeak1, e1)
    equal(wpeak1, 2.56, 0.1)
    equal(e1, 10.4, 0.2)

    d[:int(1.5 * wpeak1 / dw)] = 0
    wpeak2, e2 = findpeak(d[:, 1], dw)
    print(wpeak2, e2)
    equal(wpeak2, 10.03, 0.1)
    equal(e1, 2.4, 0.2)
