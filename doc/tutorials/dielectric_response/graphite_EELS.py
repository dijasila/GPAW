from __future__ import print_function

from math import sqrt
import numpy as np
from ase import Atoms
from ase.units import Bohr
from ase.parallel import paropen
from gpaw import GPAW, FermiDirac, PW
from gpaw.response.df import DielectricFunction

# Part 1: Ground state calculation
a=1.42
c=3.355
atoms = Atoms('C4',[                  # Generate graphite AB-stack structure.
              (1/3.0,1/3.0,0),
              (2/3.0,2/3.0,0),
              (0.   ,0.   ,0.5),
              (1/3.0,1/3.0,0.5)
              ],
              pbc=(1,1,1))
atoms.set_cell([(sqrt(3)*a/2.0,3/2.0*a,0),
                (-sqrt(3)*a/2.0,3/2.0*a,0),
                (0.,0.,2*c)],
               scale_atoms=True)

# Part 2: Find ground state density and diagonalize full hamiltonian
calc = GPAW(mode=PW(500),
            kpts=(6, 6, 3),                # The result should be converged with respect to kpoints.
            occupations=FermiDirac(0.05))  # Use smaller Fermi-Dirac smearing to avoid intraband transitions 

atoms.set_calculator(calc)       
atoms.get_potential_energy()          

calc.set(kpts=(20, 20, 7), fixdensity=True)
atoms.get_potential_energy()
calc.diagonalize_full_hamiltonian(nbands=60)  # The result should also be converged with respect to bands.
calc.write('graphite.gpw', 'all')

# Part 2: Spectra calculations            
f = paropen('graphite_q_list', 'w')  # Write down q.

for i in range(1, 6):  # Loop over different q.   
    df = DielectricFunction(calc='graphite.gpw',       
                            domega0=0.01,
                            eta=0.2,  # Broadening parameter.
                            ecut=100,  # In general, larger q requires larger planewave cutoff energy.
                            txt='out_df_%d.txt' %(i))  # Write different output for different q.

    q_c = np.array([i/20., 0., 0.])  # Gamma - M excitation
    #q_c = np.array([i/20., -i/20., 0.])  # Gamma - K excitation
    
    df.get_eels_spectrum(q_c=q_c,
                         filename='graphite_EELS_%d' % i)  # Use different filenames for different q

    # Calculate cartesian momentum vector
    cell_cv = atoms.get_cell()
    bcell_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
    q_v = np.dot(q_c, bcell_cv)

    print(sqrt(np.inner(q_v, q_v)), file=f)

f.close()



