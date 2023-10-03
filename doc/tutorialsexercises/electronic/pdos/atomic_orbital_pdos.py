import numpy as np
import pylab as plt
from gpaw import GPAW

calc = GPAW('au.gpw', txt=None)
energy, pdos = calc.get_orbital_ldos(a=0, angular='d')
I = np.trapz(pdos, energy)
center = np.trapz(pdos * energy, energy) / I
width = np.sqrt(np.trapz(pdos * (energy - center)**2, energy) / I)
plt.plot(energy, pdos)
plt.xlabel('Energy (eV)')
plt.ylabel('d-projected DOS on atom 0')
plt.title('d-band center = %s eV, d-band width = %s eV' % (center, width))
plt.show()
