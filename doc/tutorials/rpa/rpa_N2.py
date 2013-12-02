from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa import RPACorrelation

calc1 = GPAW('N.gpw', communicator=serial_comm, txt=None)
calc2 = GPAW('N2.gpw', communicator=serial_comm, txt=None)

rpa1 = RPACorrelation(calc1, txt='rpa_N.txt')    
rpa2 = RPACorrelation(calc2, txt='rpa_N2.txt')

f = paropen('rpa_N2.dat', 'w')

for ecut in [100, 150, 200, 250, 300, 350, 400]:
    E1 = rpa1.calculate(ecut=[ecut])
    E2 = rpa2.calculate(ecut=[ecut])
    print >> f, ecut, E2 - 2*E1

f.close()
