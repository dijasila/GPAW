"""Calculate the vibrational modes of a H2O molecule."""
from ase import *
from ase.vibrations import Vibrations
from gpaw import Calculator

h2o = Calculator('h2o0.gpw', txt=None).get_atoms()
vib = Vibrations(h2o)
vib.run()
vib.summary()

#Make trajectory files to visualize normal modes
#for all 9 modes
for mode in range(9):
    vib.write_mode(mode)

print 'Zero-point energy = %1.2f eV' % vib.get_zero_point_energy()
