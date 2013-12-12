import numpy as np
from ase.lattice import bulk
from gpaw import GPAW
from gpaw.response.df2 import DielectricFunction

# Part 1: Ground state calculation
atoms = bulk('Al', 'fcc', a=4.043)      # Generate fcc crystal structure for aluminum
calc = GPAW(mode='pw', kpts=(4,4,4))    # GPAW calculator initialization

atoms.set_calculator(calc)
atoms.get_potential_energy()            # Ground state calculation is performed
calc.write('Al.gpw','all')              # Use 'all' option to write wavefunctions

# Part 2: Spectrum calculation
w = np.linspace(0, 24, 241)             # The energies (eV) for spectrum: from 0-24 eV with 0.1 eV spacing
df = DielectricFunction(calc='Al.gpw' , # Ground state gpw file as input
                         frequencies=w) # Energies as input

q = np.array([1./4., 0, 0])             # Momentum transfer, must be the difference between two kpoints !
df.get_eels_spectrum(q_c=q)             # By default, a file called 'eels.csv' is generated
