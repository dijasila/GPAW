from ase import *
from gpaw import *

calc = Calculator('ferro.gpw', txt=None)

# Plot s, p, d projected LDOS:
import pylab as p
for c in 'spd':
    energies, ldos = calc.get_orbital_ldos(a=0, spin=0, angular=c, width=0.4)
    p.plot(energies, ldos, label=c + '-up')
    
    energies, ldos = calc.get_orbital_ldos(a=0, spin=1, angular=c, width=0.4)
    p.plot(energies, ldos, label=c + '-down')

p.legend()
p.show()


