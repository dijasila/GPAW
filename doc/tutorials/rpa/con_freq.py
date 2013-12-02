from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation

calc = GPAW('N2.gpw', communicator=serial_comm, txt=None)

f = paropen('con_freq.dat', 'w')
for N in [4, 6, 8, 12, 16, 24, 32]:
    rpa = RPACorrelation(calc, txt='rpa_N2_frequencies.txt', nfrequencies=N)
    E = rpa.calculate(ecut=[200])
    print >> f, N, E
f.close()
