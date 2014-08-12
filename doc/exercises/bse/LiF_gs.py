import numpy as np
from ase.lattice.spacegroup import crystal
from gpaw import GPAW, FermiDirac, PW
from ase.dft.kpoints import monkhorst_pack
from ase.visualize import view


####################################################################################################
# GROUND STATE DFT
####################################################################################################

#-------------------------------------
# Structure
#-------------------------------------
a = 4.0351     # Experimental lattice constant in \AA

Ecut = 250       # Energy cut off for PW calculation
kgrid=5          # Number of kpoints per each direction


LiF = crystal(['Li', 'F'], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
              cellpar=[a, a, a, 90, 90, 90])                                #This gives the typical NaCl structure

kpts = monkhorst_pack((kgrid,kgrid,kgrid))


#-------------------------------------
# Self consistent calculation
#-------------------------------------

calc = GPAW(mode=PW(Ecut),
           xc='LDA',
           kpts=kpts,
           txt='LiF_out_gs.txt')

LiF.set_calculator(calc)       
LiF.get_potential_energy() 

#-------------------------------------
# Full diagonalization
#-------------------------------------
# With the full diagonalization we calculate all the single particle states needed for the response calculation

calc.diagonalize_full_hamiltonian(nbands=100)  
calc.write('LiF_fulldiag.gpw','all')
