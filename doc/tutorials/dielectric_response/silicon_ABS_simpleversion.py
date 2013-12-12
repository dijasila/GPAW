import numpy as np
from ase.lattice import bulk
from gpaw import GPAW
from gpaw.response.df2 import DielectricFunction

# Part 1: Ground state calculation
atoms = bulk('Si', 'diamond', a=5.431)   # Generate diamond crystal structure for silicon
calc = GPAW(mode='pw', kpts=(4,4,4))        # GPAW calculator initialization
 
atoms.set_calculator(calc)               
atoms.get_potential_energy()             # Ground state calculation is performed
calc.write('si.gpw','all')               # Use 'all' option to write wavefunction

# Part 2 : Spectrum calculation          # DF: dielectric function object
df = DielectricFunction(calc='si.gpw',                   # Ground state gpw file (with wavefunction) as input
         frequencies=np.linspace(0,14,141))         # The Energies (eV) for spectrum: from 0-14 eV with 0.1 eV spacing
#        optical_limit=True)              # Indicates that its a optical spectrum calculation

df.get_polarizability()             # By default, a file called 'Absorption.dat' is generated
