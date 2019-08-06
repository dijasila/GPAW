from ase import Atoms
from gpaw import GPAW, PW
import numpy as np
from gpaw import Mixer


atoms = Atoms("SiH4", positions=[[0.5, 0.5, 0.5],
                                 [0.534179, 465824, 0.534176],
                                 [0.465824, 0.534176, 0.534176],
                                 [0.465824, 0.465824, 0.465824],
                                 [0.534176, 0.534176, 0.465824]], cell=25.0000413052*np.identity(3))

atoms.center()
        
ecut_Ry = 45
ecut_eV = ecut_Ry * 13.6056981
calc = GPAW(mode=PW(ecut_eV), nbands=50, symmetry="off", xc="PBE", mixer=Mixer(0.02, 5, 100))
# TODO Do diagonalize_full_hamiltonian(nbands=??) in server
atoms.set_calculator(calc)
atoms.get_potential_energy()


import sys
fname = sys.argv[1]
calc.write(fname, mode="all")
