import numpy as np
from ase.lattice import bulk
from gpaw import GPAW
from gpaw.response.df2 import DielectricFunction

# Part 1: Ground state calculation
atoms = bulk('Si', 'diamond', a=5.431)   # Generate diamond crystal structure for silicon
calc = GPAW(mode='pw', kpts=(4,4,4))     # GPAW calculator initialization
 
atoms.set_calculator(calc)               
atoms.get_potential_energy()             # Ground state calculation is performed
calc.write('si.gpw','all')               # Use 'all' option to write wavefunction

# Part 2 : Spectrum calculation          # DF: dielectric function object
w = np.linspace(0,14,141)                # The Energies (eV) for spectrum: from 0-14 eV with 0.1 eV spacing
df = DielectricFunction(calc='si.gpw',   # Ground state gpw file (with wavefunction) as input
                        frequencies=w)   # Energies as input

df.get_polarizability()                  # By default, a file called 'absorption.csv' is generated
