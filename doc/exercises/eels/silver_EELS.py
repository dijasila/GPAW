import numpy as np
from gpaw import GPAW
from gpaw.response.df import DielectricFunction

calc = GPAW('Ag_GLLBSC.gpw')

# Diagonalize Hamiltonian, converges all bands
calc.diagonalize_full_hamiltonian(nbands=30)
calc.write('Ag_GLLBSC_full.gpw', 'all')

# Set up dielectric function
df = DielectricFunction(calc='Ag_GLLBSC_full.gpw',  # Ground state input
                        domega0=0.05)

# Momentum transfer, must be the difference between two kpoints!
q_c = [1.0 / 10, 0, 0]

# By default, a file called 'eels.csv' is generated
df.get_eels_spectrum(q_c=q_c)

# Plot spectrum
from numpy import genfromtxt
import pylab as p
data = genfromtxt('eels.csv', delimiter=',')
omega = data[:, 0]
eels = data[:, 2]
p.plot(omega, eels)
p.xlabel('Energy (eV)')
p.ylabel('Loss spectrum')
p.xlim(0, 20)
p.show()
