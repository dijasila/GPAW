from ase import *
from ase.parallel import paropen
from gpaw import *
from gpaw.mpi import serial_comm
from gpaw.xc.rpa_correlation_energy import RPACorrelation

calc = GPAW('N2.gpw', communicator=serial_comm, txt=None)

rpa = RPACorrelation(calc2, txt='rpa_N2_frequencies.txt')

f = paropen('con_freq.dat', 'w')
for N in [2, 4, 6, 8, 12, 16, 24, 32]:
    E2 = rpa.get_rpa_correlation_energy(ecut=200,
                                        gauss_legendre=N,
                                        directions=[[0, 2/3.], [2, 1/3.]])
    print >> f, N, E2 - 2*E1
f.close()
