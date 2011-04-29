from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation
import numpy as np

calc = GPAW('Kr_gs.gpw', communicator=serial_comm, txt=None)
rpa = RPACorrelation(calc, txt='rpa_Kr.txt')

for ecut in [150, 175, 200, 225, 250, 275, 300]:
    E_rpa = rpa.get_rpa_correlation_energy(ecut=ecut, 
                                           kcommsize=8, 
                                           directions=[[0, 1.]])

    f = paropen('rpa_Kr.dat', 'a')
    print >> f, ecut, rpa.nbands, E_rpa
    f.close()
