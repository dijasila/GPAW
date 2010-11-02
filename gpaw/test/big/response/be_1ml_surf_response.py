### Refer to Chulkov and Echenique, PRB 67, 245402 (2003) for comparison of results ###
import numpy as np
import sys
import time

from math import sqrt
from ase import Atoms, Atom
from ase.visualize import view
from ase.units import Bohr
from ase.lattice.surface import *
from ase.parallel import paropen
from gpaw import GPAW
from gpaw.mpi import serial_comm, rank, size
from gpaw.utilities import devnull
from gpaw.response.df import DF

if rank != 0:
  sys.stdout = devnull 

GS = 1
EELS = 1
check = 1

nband = 30

if GS:

    kpts = (54,54,1)
    atoms = hcp0001('Be',size=(1,1,1))
    atoms.cell[2][2] = (21.2)
    atoms.set_pbc(True)
    atoms.center(axis=2)
    view(atoms)

    calc = GPAW(
                gpts=(12,12,108),
                xc='LDA',
                txt='be.txt',
                kpts=kpts,
                basis='dzp',
                nbands=nband+5,
                convergence={'bands':nband},
                eigensolver = 'cg',
                width=0.1)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Be.gpw','all')


if EELS:

    calc = GPAW('Be.gpw',communicator=serial_comm, txt=None)
                
    for i in range(1, 2):
        w = np.linspace(0, 15, 301)
#        q = np.array([i/50., 0., 0.]) # Gamma - M
        q = np.array([-i/54., i/54., 0.]) # Gamma - K
	ecut = 40 + i*10
        df = DF(calc=calc, q=q, w=w, eta=0.05, ecut = ecut,
                      txt='df_' + str(i) + '.out')  
        df.get_surface_response_function(z0=21.2/2, filename='be_EELS')  
        df.get_EELS_spectrum()    
        df.check_sum_rule()
        df.write('df_' + str(i) + '.pckl')

if check:
    d = np.loadtxt('be_EELS')

    wpeak1 = 2.55 # eV
    wpeak2 = 9.95
    Nw1 = 51
    Nw2 = 199

    if (d[Nw1, 2] > d[Nw1-1, 2] and d[Nw1, 2] > d[Nw1+1, 2] and  
       d[Nw2, 2] > d[Nw2-1, 2] and d[Nw2, 2] > d[Nw2+1, 2]):
        pass
    else:
        raise ValueError('Plasmon peak not correct ! ')

    if (np.abs(d[Nw1, 2] - 10.6346557215) > 1e-5
        or np.abs(d[Nw2, 2] - 2.6228166295 ) > 1e-5):
        print d[Nw1, 2], d[Nw, 2]
        raise ValueError('Please check spectrum strength ! ')
