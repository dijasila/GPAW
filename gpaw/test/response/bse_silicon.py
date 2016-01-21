from __future__ import print_function
import os
import numpy as np
from ase.lattice import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.bse_new import BSE
from gpaw.mpi import rank
from gpaw.test import findpeak, equal

GS = 1
bse = 1
check = 1
delfiles = 1

if GS:
    a = 5.431 # From PRB 73,045112 (2006)
    atoms = bulk('Si', 'diamond', a=a)
    atoms.positions -= a/8
    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.001),
                #symmetry='off',
                nbands=12,
                convergence={'bands':-4})
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Si.gpw','all')

if bse:
    eshift = 0.8
    bse = BSE('Si.gpw',
              ecut=50.,
              valence_bands=range(4),
              conduction_bands=range(4,8),
              eshift=eshift,
              nbands=8,
              write_h=False,
              write_v=False)
    w_w, eps_w = bse.get_dielectric_function(filename='Si_bse.csv',
                                             eta=0.2,
                                             w_w=np.linspace(0,10,2001),
                                             )

if check:
    w_ = 2.5460
    I_ = 421.2425
    d = np.loadtxt('Si_bse.csv', delimiter=',')
    w, I = findpeak(d[:, 0], d[:, 2])
    print(w, I)
    equal(w, w_, 0.01)
    equal(I, I_, 0.1)

if delfiles and rank == 0:
    os.remove('Si.gpw')
    os.remove('pair.txt')
    os.remove('chi0.txt')
