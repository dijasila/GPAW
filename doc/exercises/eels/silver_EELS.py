import numpy as np
from gpaw import GPAW
from gpaw.response.df import DielectricFunction

calc = GPAW('Ag_GLLBSC.gpw'
            )

calc.diagonalize_full_hamiltonian(nbands = 30) # Diagonalize Hamiltonian, converges all bands
calc.write('Ag_GLLBSC_full.gpw', 'all')



# Set up dielectric function 
df = DielectricFunction(calc='Ag_GLLBSC_full.gpw',  # Ground state gpw file as input
                        alpha=0.0,
                        domega0 = 0.05)

q_c = np.array([1./10., 0, 0])          # Momentum transfer, must be the difference between two kpoints !
df.get_eels_spectrum(q_c=q_c)           # By default, a file called 'eels.csv' is generated


# Plot spectrum
from numpy import genfromtxt
import pylab as p
data = genfromtxt('eels.csv', delimiter=','  )
omega = data[:,0]
eels = data[:,2]
p.plot(omega, eels)
p.xlabel('Energy (eV)')
p.ylabel('Loss spectrum')
p.xlim(0,20)
p.show()


