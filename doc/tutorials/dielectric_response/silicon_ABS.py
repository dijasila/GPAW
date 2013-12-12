### Refer to G. Kresse, Phys. Rev. B 73, 045112 (2006)
### for comparison of macroscopic and microscopic dielectric constant 
### and absorption peaks. 

import os
import sys
import numpy as np

from ase.units import Bohr
from ase.lattice import bulk
from ase.parallel import paropen
from gpaw import GPAW, FermiDirac
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull
from gpaw.response.df2 import DielectricFunction


if rank != 0:
    sys.stdout = devnull

GS = 1
ABS = 1

if GS:

    # Ground state calculation
    a = 5.431 #10.16 * Bohr 
    atoms = bulk('Si', 'diamond', a=a)

    calc = GPAW(mode='pw',
            kpts=(12,12,12),
            xc='LDA',
            basis='dzp',
            txt='si_gs.txt',
            nbands=80,
            eigensolver='cg',
            occupations=FermiDirac(0.001),
            convergence={'bands':70})
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('si.gpw','all')


if ABS:
            
    w = np.linspace(0, 24, 481)

    # getting macroscopic constant
    df = DielectricFunction(calc='si.gpw', frequencies=w, eta=0.0001,
                            ecut=150, txt='df_1.out')

    df.get_macroscopic_dielectric_constant()

    #getting absorption spectrum
    df = DielectricFunction(calc='si.gpw', frequencies=w, eta=0.1,
                            ecut=150, txt='df_2.out')

    df.get_polarizability(filename='si_abs.csv')

